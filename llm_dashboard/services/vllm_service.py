import json
import subprocess
import time
import os
import psutil
import requests
from typing import Dict
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

from .vector_service.token_aware_vector import TokenAwareVectorService
from ..models import LLMModel, LLMRequest
from .map_reduce_service import MapReduceService
import logging

logger = logging.getLogger(__name__)


class VLLMService:
    def __init__(self):
        """
        Manages initialization of process storage and channel layer instance.

        This class is responsible for maintaining a dictionary to store
        information about processes associated with models. It also initializes
        a channel layer for handling asynchronous communication between
        different components.

        Attributes:
            processes (dict): A dictionary to store process information for each model.
            channel_layer (ChannelsLayer): A channel layer instance used for communication.
        """
        self.processes = {}  # Store process info for each model
        self.channel_layer = get_channel_layer()

    def _append_log(self, model: LLMModel, message: str):
        """
        Appends a log entry to the model's loading logs, maintains size constraints on the log,
        and optionally sends a real-time WebSocket update if available.

        :param model: The LLMModel instance whose logs are being updated.
        :type model: LLMModel
        :param message: The log message to append with a timestamp.
        :type message: str
        :return: None
        """
        try:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"[{timestamp}] {message}\n"

            # Update model logs
            if model.loading_logs:
                model.loading_logs += log_entry
            else:
                model.loading_logs = log_entry

            # Keep only last 10000 characters to prevent unbounded growth
            if len(model.loading_logs) > 10000:
                model.loading_logs = model.loading_logs[-10000:]

            model.save(update_fields=['loading_logs'])

            # Send real-time update via WebSocket
            if self.channel_layer:
                group_name = f"model_loading_{model.id}"
                async_to_sync(self.channel_layer.group_send)(group_name, {
                    "type": "loading_update",
                    "message": message
                })

        except Exception as e:
            logger.error(f"Error appending log: {e}")

    def _detect_best_attention_backend(self, model: LLMModel) -> str:
        """
        Detects the best attention backend available for the provided model.

        The function attempts to determine the most appropriate backend for attention
        operations by trying different modules in the following order:
        1. XFormers (preferred)
        2. FlashInfer
        3. PyTorch SDPA (default fallback)

        The detection is done dynamically, and the selected backend is logged for
        tracking purposes.

        :param model: The large language model instance used for attention
            backend detection.
        :type model: LLMModel
        :return: A string indicating the detected attention backend. Possible values are
            'XFORMERS', 'FLASHINFER', and 'TORCH_SDPA'.
        :rtype: str
        """
        self._append_log(model, "🔍 Detecting best attention backend...")

        # Try XFormers first (preferred)
        try:
            import xformers
            self._append_log(model, f"✅ XFormers detected (version: {xformers.__version__})")
            return 'XFORMERS'
        except ImportError:
            self._append_log(model, "⚠️ XFormers not available")

        # Try FlashInfer (if available and working)
        try:
            import flashinfer
            self._append_log(model, f"✅ FlashInfer detected")
            return 'FLASHINFER'
        except ImportError:
            self._append_log(model, "⚠️ FlashInfer not available")

        # Fall back to PyTorch SDPA
        self._append_log(model, "📍 Using PyTorch SDPA (default)")
        return 'TORCH_SDPA'

    def _check_cuda_environment(self, model: LLMModel):
        """
        Checks the CUDA environment and configures relevant settings for optimal GPU utilization.

        This method verifies the presence of CUDA devices and automatically configures system
        and library-wide environment variables to enhance the stability and performance of
        CUDA-based computations. It also identifies and sets the best attention backend
        (e.g., XFormers) if applicable. Logs are appended throughout the process to track
        progress. Returns whether the CUDA environment check was successful or failed.

        :param model: An instance of `LLMModel` containing parameters and configurations
            required for the CUDA setup.
        :return: A boolean value indicating whether the CUDA environment initialization
            was successful (True) or encountered an error (False).
        """
        self._append_log(model, "🔧 Checking CUDA environment...")

        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            self._append_log(model, f"✅ Found {gpu_count} CUDA devices")

            # Detect best attention backend
            attention_backend = self._detect_best_attention_backend(model)

            # Set CUDA environment variables for stability
            cuda_env = {
                'CUDA_VISIBLE_DEVICES': ','.join(str(i) for i in range(min(gpu_count, model.tensor_parallel_size))),
                'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1',
                'VLLM_ATTENTION_BACKEND': attention_backend,
                'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128,expandable_segments:True',
                'TOKENIZERS_PARALLELISM': 'false',
                'CUDA_LAUNCH_BLOCKING': '0'
            }

            # Additional XFormers optimizations
            if attention_backend == 'XFORMERS':
                cuda_env.update({
                    'XFORMERS_MORE_DETAILS': '1',
                    'XFORMERS_DISABLED_OPS': '',  # Enable all ops
                })
                self._append_log(model, "🚀 XFormers optimizations enabled")

            for key, value in cuda_env.items():
                os.environ[key] = value
                self._append_log(model, f"   {key}={value}")

            return True

        except Exception as e:
            self._append_log(model, f"❌ CUDA environment check failed: {e}")
            return False

    def _build_vllm_command(self, model: LLMModel) -> list:
        """
        Builds the command to execute the vLLM server with specified parameters of the
        provided LLMModel instance. The command includes details about the model path,
        network configuration, tensor parallelization, and GPU memory utilization.
        Additional optimizations for the XFormers attention backend are included if
        enabled via environment settings.

        :param model: The LLMModel instance containing all configurations for the vLLM
                      server execution.
        :type model: LLMModel
        :return: A list of strings representing the command to start the vLLM server.
        :rtype: list
        """
        cmd = [
            'python', '-m', 'vllm.entrypoints.openai.api_server',
            '--model', model.model_path,
            '--host', '0.0.0.0',
            '--port', str(model.vllm_port),
            '--tensor-parallel-size', str(model.tensor_parallel_size),
            '--gpu-memory-utilization', str(model.gpu_memory_utilization),
            '--max-model-len', str(model.max_tokens),
            '--dtype', model.dtype,
            '--disable-log-requests',
            '--enforce-eager'
        ]

        # Add XFormers-specific optimizations
        if os.environ.get('VLLM_ATTENTION_BACKEND') == 'XFORMERS':
            cmd.extend([
                '--enable-chunked-prefill',
                '--max-num-batched-tokens', '8192',
                '--max-num-seqs', '256'
            ])
            self._append_log(model, "🚀 Added XFormers performance optimizations")

        return cmd

    def start_vllm_server(self, model: LLMModel):
        """
        Starts the vLLM server for the given model and manages its life cycle. This
        includes checking server status, cleaning up GPU memory, building and
        executing the vLLM command, monitoring server output, and ensuring readiness.

        :param model: The LLMModel object representing the model for which the vLLM
            server will be started.
        :type model: LLMModel
        :return: True if the server starts successfully and becomes ready, otherwise
            False.
        :rtype: bool
        :raises Exception: Captures and logs any exception that occurs during the
            process, and updates the model status to 'ERROR'.
        """
        try:
            self._append_log(model, f"🚀 Starting vLLM server for {model.name}")

            # Check if already running
            if self.is_server_running(model):
                self._append_log(model, "⚠️ Server already running")
                # Update memory usage even if already running
                model.memory_usage = self._calculate_gpu_memory_usage(model)
                model.status = 'LOADED'
                model.save()
                return True

            # Update model status
            model.status = 'LOADING'
            model.save()

            # Check CUDA environment
            if not self._check_cuda_environment(model):
                model.status = 'ERROR'
                model.save()
                return False

            # Force cleanup before starting
            self._append_log(model, "🧹 Cleaning up GPU memory...")
            self._force_gpu_memory_cleanup()

            # Build vLLM command
            cmd = self._build_vllm_command(model)

            self._append_log(model, f"📋 Command: {' '.join(cmd)}")

            # Start process
            self._append_log(model, "⏳ Starting vLLM process...")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Store process info
            self.processes[model.id] = {
                'process': process,
                'start_time': time.time(),
                'port': model.vllm_port
            }

            # Update model with process PID
            model.loading_pid = process.pid
            model.save()

            self._append_log(model, f"🔄 Process started with PID: {process.pid}")

            # Monitor process output in background
            import threading
            monitor_thread = threading.Thread(
                target=self._monitor_process_output,
                args=(model, process)
            )
            monitor_thread.daemon = True
            monitor_thread.start()

            # Wait for server to be ready
            if self._wait_for_server_ready(model):
                # Calculate and update memory usage
                memory_usage = self._calculate_gpu_memory_usage(model)
                model.memory_usage = memory_usage
                model.status = 'LOADED'
                model.save()

                self._append_log(model, f"✅ Server ready! Memory usage: {memory_usage:.2f} GB")
                return True
            else:
                model.status = 'ERROR'
                model.save()
                return False

        except Exception as e:
            self._append_log(model, f"❌ Error starting server: {e}")
            model.status = 'ERROR'
            model.save()
            return False

    def _monitor_process_output(self, model: LLMModel, process):
        """
        Monitors the output of a provided process and appends it to a log. This function listens to the
        `stdout` stream of the given process line by line until the process is terminated. Any errors
        encountered during monitoring are appended to the log as well.

        :param model: The LLMModel to associate the monitored output with.
        :type model: LLMModel
        :param process: The process whose output is to be monitored.
        :return: None
        """
        try:
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    self._append_log(model, output.strip())

        except Exception as e:
            self._append_log(model, f"Error monitoring process: {e}")

    def _wait_for_server_ready(self, model: LLMModel, timeout: int = 300):
        """
        Waits for the server to be ready for the specified model within the given timeout period.

        This method continuously checks if the server associated with the given model is ready by
        querying its health endpoint. If the server becomes ready within the given timeout period,
        the model's status is updated to 'LOADED' and the method returns True. Otherwise, it logs
        that the server did not become ready and returns False.

        :param model: The LLMModel instance representing the model for which the server readiness is to be checked.
        :param timeout: An optional integer specifying the maximum time in seconds to wait for the server
            to become ready. Defaults to 300.
        :return: A boolean indicating whether the server became ready within the timeout period.
        """
        self._append_log(model, "⏳ Waiting for server to be ready...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f'http://localhost:{model.vllm_port}/health', timeout=5)
                if response.status_code == 200:
                    # Update model status immediately when server is ready
                    model.status = 'LOADED'
                    model.save()
                    return True
            except:
                pass

            time.sleep(5)
            elapsed = int(time.time() - start_time)
            self._append_log(model, f"⏳ Still waiting... ({elapsed}s elapsed)")

        self._append_log(model, "❌ Server did not become ready within timeout")
        return False

    def _calculate_gpu_memory_usage(self, model: LLMModel) -> float:
        """
        Calculates GPU memory usage across multiple devices up to the model's tensor
        parallel size. Utilizes the NVIDIA Management Library (NVML) to retrieve memory
        usage information. If any error occurs during the calculation, a warning log is
        appended, and a value of 0.0 is returned.

        :param model: The language model object containing tensor parallel size.
        :type model: LLMModel
        :return: The total GPU memory usage in gigabytes. Returns 0.0 if the calculation
            fails due to an exception.
        :rtype: float
        """
        try:
            import pynvml
            pynvml.nvmlInit()

            total_memory_used = 0
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(min(device_count, model.tensor_parallel_size)):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_memory_used += info.used

            # Convert to GB
            return total_memory_used / (1024 ** 3)
        except Exception as e:
            self._append_log(model, f"Warning: Could not calculate memory usage: {e}")
            return 0.0

    def stop_vllm_server(self, model: LLMModel):
        """
        Stops the vLLM server associated with the provided model instance. This method
        attempts a graceful shutdown of the model's associated process, falling back to
        a forced termination if necessary. It also performs cleanup actions including
        removal of process information and GPU memory cleanup, and updates the model's
        status and persistent storage.

        :param model: The LLMModel instance for which the vLLM server needs to be stopped.
        :type model: LLMModel
        :return: Boolean flag indicating whether the vLLM server was stopped successfully.
        :rtype: bool
        """
        try:
            self._append_log(model, f"🛑 Stopping vLLM server for {model.name}")

            # Get process info
            process_info = self.processes.get(model.id)
            if process_info:
                process = process_info['process']

                # Try graceful shutdown first
                self._append_log(model, "🔄 Attempting graceful shutdown...")
                process.terminate()

                # Wait for graceful shutdown
                try:
                    process.wait(timeout=30)
                    self._append_log(model, "✅ Process terminated gracefully")
                except subprocess.TimeoutExpired:
                    self._append_log(model, "⚠️ Graceful shutdown timed out, forcing...")
                    process.kill()
                    process.wait()
                    self._append_log(model, "✅ Process killed")

                # Clean up
                del self.processes[model.id]

            # Force cleanup any remaining processes
            self._force_gpu_memory_cleanup()

            # Update model status
            model.status = 'UNLOADED'
            model.loading_pid = None
            model.save()

            self._append_log(model, "✅ Server stopped successfully")
            return True

        except Exception as e:
            self._append_log(model, f"❌ Error stopping server: {e}")
            return False

    def _force_gpu_memory_cleanup(self):
        """
        Attempts to free GPU memory by performing several cleanup operations.

        This method is used to ensure GPU resources are released, especially after
        running processes that utilize the GPU. It tries to stop any active processes
        associated with `vllm` and clears the GPU memory cache if possible.

        :return: None
        """
        try:
            # Kill any vLLM processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['cmdline'] and any('vllm' in arg for arg in proc.info['cmdline']):
                        proc.terminate()
                        proc.wait(timeout=5)
                except:
                    pass

            # Force GPU memory cleanup
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except:
                pass

        except Exception as e:
            logger.error(f"Error in force cleanup: {e}")

    def stop_loading(self, model: LLMModel):
        """
        Stops the loading process of a given model by terminating its associated process
        if it is still running. If termination is unsuccessful within a timeout period,
        it forcefully kills the process. Updates the model's status to reflect the unloading
        operation and logs the action. Returns a boolean indicating the success of the
        operation.

        :param model: The LLMModel object whose loading process is to be stopped.
        :type model: LLMModel
        :return: A boolean value indicating whether the loading process was successfully
                 stopped. Returns False if an exception occurred during the process.
        :rtype: bool
        """
        try:
            if model.loading_pid:
                import psutil
                try:
                    # Use psutil for better process management
                    process = psutil.Process(model.loading_pid)
                    process.terminate()
                    process.wait(timeout=5)  # Wait up to 5 seconds for graceful shutdown
                except psutil.TimeoutExpired:
                    process.kill()  # Force kill if it doesn't terminate gracefully
                except psutil.NoSuchProcess:
                    pass  # Process already dead

                # Clear the PID
                model.loading_pid = None

            # Update model status
            model.status = 'UNLOADED'
            self._append_log(model, "Loading process stopped by user")
            model.save()

            return True

        except Exception as e:
            self._append_log(model, f"Error stopping loading process: {str(e)}")
            model.save()
            return False

    def is_server_running(self, model: LLMModel) -> bool:
        """
        Checks if the server corresponding to the provided model is running by
        sending a health check request.

        This function makes an HTTP GET request to the health endpoint of the
        server using the `vllm_port` specified in the model. If the request
        is successful and the server responds with a status code 200, the server
        is considered to be running. In case of an exception or a different
        response code, the server is considered not running.

        :param model: The LLMModel instance, representing the model whose
            server's status is to be checked.
        :type model: LLMModel
        :return: True if the server is running (responds with status code 200),
            otherwise False.
        :rtype: bool
        """
        try:
            response = requests.get(f'http://localhost:{model.vllm_port}/health', timeout=5)
            return response.status_code == 200
        except:
            return False

    def generate_text(self, model: LLMModel, prompt: str, max_tokens: int = 100,
                      temperature: float = 0.7, top_p: float = 0.9) -> Dict:
        """
        Generates text using the specified large language model and prompt. This method
        prepares and sends a request to the language model server, waits for a
        response, and processes the returned data. The method can handle various
        configurable parameters for text generation, such as maximum tokens,
        temperature, and nucleus sampling probability.

        Raises an exception if the model server is not running or if the API response
        indicates a failure.

        :param model: The language model instance that provides details like the
                      model path and server port.
        :type model: LLMModel
        :param prompt: The text prompt for generating a continuation.
        :type prompt: str
        :param max_tokens: The maximum number of tokens that can be generated.
                           Defaults to 100.
        :type max_tokens: int
        :param temperature: The sampling temperature, controlling creativity in
                            generation. Defaults to 0.7.
        :type temperature: float
        :param top_p: The cumulative probability threshold for nucleus (top-p)
                      sampling. Defaults to 0.9.
        :type top_p: float
        :return: A dictionary containing the generated text, the number
                 of tokens used in the prompt and generation, and the
                 generation duration.
        :rtype: Dict
        """
        try:
            if not self.is_server_running(model):
                raise Exception("Model server is not running")

            start_time = time.time()

            # Prepare request
            data = {
                "model": model.model_path,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": None
            }

            # Make request
            response = requests.post(
                f'http://localhost:{model.vllm_port}/v1/completions',
                json=data,
                timeout=120
            )

            if response.status_code != 200:
                raise Exception(f"API request failed: {response.text}")

            result = response.json()
            duration = time.time() - start_time

            # Extract response
            generated_text = result['choices'][0]['text']
            tokens_generated = result['usage']['completion_tokens']
            tokens_prompt = result['usage']['prompt_tokens']

            return {
                'response': generated_text,
                'tokens_generated': tokens_generated,
                'tokens_prompt': tokens_prompt,
                'duration': duration
            }

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise

    def get_active_model(self):
        """
        Get the currently active and running model.

        This method retrieves the model that is marked as 'LOADED' in the database.
        It ensures that the model's server is running before returning it. If no
        model is loaded or the server is not operational, it returns None instead.

        :return: The active model if found and running, otherwise None.
        :rtype: Optional[LLMModel]
        """
        try:
            # Find the model that is currently loaded
            active_model = LLMModel.objects.filter(status='LOADED').first()

            if active_model and self.is_server_running(active_model):
                return active_model

            # If no loaded model found or server not running, return None
            return None
        except Exception as e:
            logger.error(f"Error getting active model: {e}")
            return None

    async def generate_streaming_text(self, model: LLMModel, prompt: str, max_tokens: int = 100,
                                      temperature: float = 0.7, top_p: float = 0.9):
        """
        Generates streaming text completions based on the provided model and prompt.

        This method performs a streaming request to the model server to generate
        text tokens incrementally in response to the specified input prompt. It
        uses Server-Sent Events (SSE) to stream tokens, allowing for real-time
        text generation.

        :param model: The language model instance to use for text generation.
                      The model must have its server running and accessible.
        :param prompt: The input text to base the text generation on.
                       This is the starting point for generating responses.
        :param max_tokens: The maximum number of tokens to generate in the response.
                           Defaults to 100.
        :param temperature: Controls the randomness of the text generation.
                            Lower values make the response more deterministic,
                            while higher values introduce more randomness.
                            Defaults to 0.7.
        :param top_p: Implements nucleus sampling, where the model considers the
                      smallest set of tokens whose cumulative probability exceeds
                      the `top_p` threshold. Defaults to 0.9.

        :return: A streaming generator producing dictionaries with the following keys:
                 - `token`: The generated text token.
                 - `is_final`: A boolean indicating whether the end of the stream
                   is reached.
                 - `total_tokens`: The cumulative count of tokens generated so far.

        :raises Exception: If the model server is not running or the streaming
                           request fails, an exception is raised.
        """
        try:
            if not self.is_server_running(model):
                raise Exception("Model server is not running")

            # Prepare request for streaming
            data = {
                "model": model.model_path,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": True,  # Enable streaming
                "stop": None
            }

            # Make streaming request
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        f'http://localhost:{model.vllm_port}/v1/completions',
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=300)
                ) as response:

                    if response.status != 200:
                        raise Exception(f"API request failed: {response.status}")

                    total_tokens = 0
                    async for line in response.content:
                        line = line.decode('utf-8').strip()

                        # Skip empty lines and non-data lines
                        if not line or not line.startswith('data: '):
                            continue

                        # Parse SSE data
                        data_content = line[6:]  # Remove 'data: ' prefix

                        # Check for end of stream
                        if data_content == '[DONE]':
                            yield {
                                'token': '',
                                'is_final': True,
                                'total_tokens': total_tokens
                            }
                            break

                        try:
                            # Parse JSON data
                            token_data = json.loads(data_content)

                            # Extract token from response
                            choices = token_data.get('choices', [])
                            if choices:
                                token = choices[0].get('text', '')
                                total_tokens += 1

                                yield {
                                    'token': token,
                                    'is_final': False,
                                    'total_tokens': total_tokens
                                }

                        except json.JSONDecodeError:
                            # Skip malformed JSON
                            continue

        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            raise

    def generate_text_with_smart_rag(self, model: LLMModel, prompt: str,
                                     document_id: str, max_tokens: int = 100,
                                     temperature: float = 0.7, top_p: float = 0.9,
                                     strategy: str = None) -> Dict:
        """
        Generates text using Smart RAG (Retrieve-and-Generate) approach. The function utilizes different
        retrieval and response generation strategies including map-reduce and standard methods.

        :param model:
            The language model object used for text generation.
        :param prompt:
            The textual input prompt used for context generation or response generation.
        :param document_id:
            The identifier of the document to retrieve context from.
        :param max_tokens:
            The maximum number of tokens to generate in the final response.
        :param temperature:
            Parameter to control randomness in the output; higher values generate more diverse responses.
        :param top_p:
            Parameter for nucleus sampling that determines the probability mass for considering tokens.
        :param strategy:
            Specifies the RAG strategy to use, such as 'map_reduce' or other pre-defined strategies.
            If not specified, the model's default strategy is used.

        :return:
            A dictionary containing the generated response, strategy used, context information, number
            of tokens generated, total context tokens used, and request ID for tracking.
        """

        # Determine strategy
        if strategy is None:
            strategy = model.rag_strategy

        # Create request record
        request = LLMRequest.objects.create(
            model=model,
            prompt=prompt,
            document_id=document_id,
            rag_enabled=True,
            strategy_used=strategy,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )

        try:
            # Calculate available tokens for context
            prompt_tokens = len(model.get_tokenizer().encode(prompt))
            available_context_tokens = (
                    model.get_available_context_tokens() -
                    prompt_tokens -
                    max_tokens -
                    100  # Buffer for formatting
            )

            if available_context_tokens < 100:
                raise ValueError("Not enough context tokens available")

            # Get context based on strategy
            if strategy == 'map_reduce':
                context_result = self._process_map_reduce_rag(
                    model, document_id, prompt, available_context_tokens
                )
            else:
                context_result = self._process_standard_rag(
                    model, document_id, prompt, available_context_tokens, strategy
                )

            # Generate response
            if strategy == 'map_reduce' and context_result.get('reduce_result', {}).get('final_answer'):
                # For map-reduce, we already have the answer
                generation_result = {
                    'response': context_result['reduce_result']['final_answer'],
                    'tokens_generated': context_result['reduce_result'].get('tokens_used', 0),
                    'duration': 0  # Would need timing
                }
            else:
                # Standard generation with context
                enhanced_prompt = self._build_enhanced_prompt(prompt, context_result)
                generation_result = self.generate_text(
                    model=model,
                    prompt=enhanced_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )

            # Update request record
            request.response = generation_result.get('response', '')
            request.duration = generation_result.get('duration', 0)
            request.tokens_generated = generation_result.get('tokens_generated', 0)
            request.total_context_tokens = context_result.get('total_tokens', 0)
            request.chunks_processed = context_result.get('chunks_used', 0)
            request.status = 'COMPLETED'

            if strategy == 'map_reduce':
                request.map_reduce_steps = context_result.get('map_results', [])

            request.save()

            # Build complete result
            result = {
                'response': generation_result.get('response', ''),
                'strategy_used': strategy,
                'context_info': context_result,
                'tokens_generated': generation_result.get('tokens_generated', 0),
                'total_context_tokens': context_result.get('total_tokens', 0),
                'request_id': request.id
            }

            return result

        except Exception as e:
            request.status = 'ERROR'
            request.error_message = str(e)
            request.save()
            raise

    def _process_standard_rag(self, model: LLMModel, document_id: str,
                              query: str, max_tokens: int, strategy: str) -> Dict:
        """
        Processes a query using the standard RAG (retrieval-augmented generation) approach
        by retrieving relevant chunks from a document within the specified token limit.

        This method uses a token-aware vector service to fetch document chunks that fit
        the token length constraints, ensuring efficient query processing.

        :param model: The language model to be used for processing.
        :type model: LLMModel
        :param document_id: The unique identifier for the document to be queried.
        :type document_id: str
        :param query: The input query to retrieve relevant information from the document.
        :type query: str
        :param max_tokens: The maximum number of tokens allowed for the query response.
        :type max_tokens: int
        :param strategy: The retrieval strategy to apply when fetching document chunks.
        :type strategy: str
        :return: A dictionary containing the retrieved document chunks within the token limit.
        :rtype: Dict
        """

        vector_service = TokenAwareVectorService(model)

        return vector_service.get_chunks_within_token_limit(
            document_id=document_id,
            query=query,
            max_tokens=max_tokens,
            strategy=strategy
        )

    def _process_map_reduce_rag(self, model: LLMModel, document_id: str,
                                query: str, max_tokens: int) -> Dict:
        """
        Processes a given document using the Map-Reduce approach with the provided query
        and language model. This method initializes a MapReduceService instance with
        the provided language model and uses it to process the document specified
        by its identifier.

        :param model: The language model instance to use for processing.
        :type model: LLMModel
        :param document_id: The identifier of the document to process.
        :type document_id: str
        :param query: The query to execute for processing the document.
        :type query: str
        :param max_tokens: The maximum number of tokens allowed during document processing.
        :type max_tokens: int
        :return: A dictionary containing the processed output as per the Map-Reduce approach.
        :rtype: Dict
        """

        map_reduce_service = MapReduceService(model, self)

        return map_reduce_service.process_document_with_map_reduce(
            document_id=document_id,
            query=query,
            max_tokens=max_tokens
        )

    def _build_enhanced_prompt(self, original_prompt: str, context_result: Dict) -> str:
        """
        Constructs an enhanced prompt string by incorporating context information from
        the provided context result. It first determines the strategy for processing the
        context (e.g., map-reduce or chunk). For the map-reduce strategy, summaries are
        utilized as context information. For other strategies, chunks of content are
        used. The method formats the context appropriately and appends it to the original
        prompt.

        :param original_prompt: The original prompt/question provided by the user.
        :type original_prompt: str
        :param context_result: A dictionary containing context information, which may
            include strategy type, summaries, or content chunks. The structure of this
            dictionary determines how the context is processed and incorporated into
            the enhanced prompt.
        :type context_result: Dict
        :return: A formatted prompt string that includes the original question and any
            relevant context extracted from the context result.
        :rtype: str
        """

        if context_result.get('strategy') == 'map_reduce':
            # For map-reduce, use summaries
            summaries = context_result.get('map_results', [])
            if summaries:
                context_text = "\n\n".join([
                    f"Summary {i + 1}: {summary['summary']}"
                    for i, summary in enumerate(summaries)
                    if summary.get('summary')
                ])
            else:
                context_text = "No relevant context found."
        else:
            # For other strategies, use chunks
            chunks = context_result.get('chunks', [])
            if chunks:
                context_text = "\n\n".join([
                    f"Context {i + 1}: {chunk['content']}"
                    for i, chunk in enumerate(chunks)
                ])
            else:
                context_text = "No relevant context found."

        return f"""Based on the following context information, please answer the question:

Context Information:
{context_text}

Question: {original_prompt}

Please provide a comprehensive answer based on the context provided above."""

    def get_model_info(self, model: LLMModel) -> Dict:
        """Get model information"""
        try:
            if not self.is_server_running(model):
                return {'error': 'Model server not running'}

            response = requests.get(f'http://localhost:{model.vllm_port}/v1/models')
            return response.json()

        except Exception as e:
            return {'error': str(e)}


# Global instance
vllm_service = VLLMService()