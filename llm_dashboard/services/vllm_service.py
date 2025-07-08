import os
import sys
import subprocess
import logging
import threading
import time
import requests
import psutil
import signal
from typing import Dict, Optional, Any
from django.utils import timezone
from ..models import LLMModel

logger = logging.getLogger(__name__)


class VLLMService:
    """Simplified vLLM service with enhanced CUDA stability"""

    def __init__(self):
        self.processes: Dict[int, subprocess.Popen] = {}

    def _append_log(self, model: LLMModel, message: str, level: str = "INFO"):
        """Append log message to model's loading logs"""
        timestamp = timezone.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"

        # Refresh model from database to avoid conflicts
        model.refresh_from_db()

        if model.loading_logs:
            model.loading_logs += log_entry
        else:
            model.loading_logs = log_entry

        # Keep logs under reasonable size (last 50KB)
        if len(model.loading_logs) > 50000:
            lines = model.loading_logs.split('\n')
            model.loading_logs = '\n'.join(lines[-500:])  # Keep last 500 lines

        model.save(update_fields=['loading_logs'])
        logger.info(f"Model {model.name}: {message}")

    def _check_cuda_environment(self, model: LLMModel) -> bool:
        """Check CUDA environment and provide diagnostics"""
        try:
            self._append_log(model, "Checking CUDA environment...")

            # Check if CUDA is available
            import torch
            if not torch.cuda.is_available():
                self._append_log(model, "CUDA is not available in PyTorch", "ERROR")
                return False

            gpu_count = torch.cuda.device_count()
            self._append_log(model, f"CUDA available with {gpu_count} GPU(s)")

            if model.tensor_parallel_size > gpu_count:
                self._append_log(model, f"Requested {model.tensor_parallel_size} GPUs but only {gpu_count} available",
                                 "ERROR")
                return False

            # Check individual GPU properties
            for i in range(min(model.tensor_parallel_size, gpu_count)):
                try:
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / (1024 ** 3)
                    self._append_log(model, f"GPU {i}: {props.name} ({memory_gb:.1f}GB)")
                except Exception as e:
                    self._append_log(model, f"Error checking GPU {i}: {str(e)}", "WARNING")

            return True

        except Exception as e:
            self._append_log(model, f"CUDA environment check failed: {str(e)}", "ERROR")
            return False

    def start_vllm_server(self, model: LLMModel) -> bool:
        """Start vLLM server with enhanced CUDA stability"""
        try:
            self._append_log(model, f"Starting vLLM server for model: {model.name}")

            # Check CUDA environment first
            if not self._check_cuda_environment(model):
                self._append_log(model, "CUDA environment check failed", "ERROR")
                model.status = 'ERROR'
                model.save(update_fields=['status'])
                return False

            # Validate model path
            model_path = model.model_path.strip()
            if not model_path:
                raise ValueError(f"Model {model.name} has no valid model_path")

            self._append_log(model, f"Model path: {model_path}")

            # Check if path exists (for local models)
            if os.path.exists(model_path):
                if os.path.isdir(model_path):
                    self._append_log(model, f"Local model directory found: {model_path}")
                else:
                    self._append_log(model, f"Model path exists but is not a directory: {model_path}", "WARNING")
            else:
                self._append_log(model, f"Assuming Hugging Face model ID: {model_path}")

            # Choose the appropriate backend and configuration
            if model.tensor_parallel_size > 1:
                # Multi-GPU setup - use ray backend which is more stable
                cmd = [
                    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                    "--model", model_path,
                    "--host", "0.0.0.0",
                    "--port", str(model.vllm_port),
                    "--tensor-parallel-size", str(model.tensor_parallel_size),
                    "--gpu-memory-utilization", str(model.gpu_memory_utilization),
                    "--max-model-len", str(model.max_tokens),
                    "--dtype", getattr(model, 'dtype', 'auto'),
                    "--served-model-name", model.name,
                    "--trust-remote-code",
                    "--disable-log-requests",
                    "--disable-log-stats",
                    "--enforce-eager",  # Disable CUDA graphs
                    "--distributed-executor-backend", "mp",
                    "--disable-custom-all-reduce",  # Disable custom all-reduce
                ]
                self._append_log(model, "Using Ray backend for multi-GPU setup")
            else:
                # Single GPU setup - simpler configuration
                cmd = [
                    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                    "--model", model_path,
                    "--host", "0.0.0.0",
                    "--port", str(model.vllm_port),
                    "--gpu-memory-utilization", str(model.gpu_memory_utilization),
                    "--max-model-len", str(model.max_tokens),
                    "--dtype", getattr(model, 'dtype', 'auto'),
                    "--served-model-name", model.name,
                    "--trust-remote-code",
                    "--disable-log-requests",
                    "--disable-log-stats",
                    "--enforce-eager",  # Disable CUDA graphs
                ]
                self._append_log(model, "Using single GPU configuration")

            self._append_log(model, f"Command: {' '.join(cmd)}")

            # Set up environment with comprehensive CUDA fixes
            env = os.environ.copy()

            # Core CUDA environment variables
            if model.tensor_parallel_size > 1:
                # Multi-GPU: explicitly set visible devices
                gpu_ids = ','.join(str(i) for i in range(model.tensor_parallel_size))
                env['CUDA_VISIBLE_DEVICES'] = gpu_ids
                self._append_log(model, f"Multi-GPU: CUDA_VISIBLE_DEVICES={gpu_ids}")
            else:
                # Single GPU: use first available GPU only
                env['CUDA_VISIBLE_DEVICES'] = '0'
                self._append_log(model, "Single GPU: CUDA_VISIBLE_DEVICES=0")

            # Ray configuration for multi-GPU
            if model.tensor_parallel_size > 1:
                env['RAY_DEDUP_LOGS'] = '0'
                env['RAY_DISABLE_IMPORT_WARNING'] = '1'
                env['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE'] = '1'

            # PyTorch and CUDA optimization
            env['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
            env['NCCL_TIMEOUT'] = '1800'
            env['NCCL_IB_DISABLE'] = '1'  # Disable InfiniBand
            env['NCCL_P2P_DISABLE'] = '1'  # Disable P2P
            env['CUDA_LAUNCH_BLOCKING'] = '0'  # Don't block CUDA launches

            # vLLM specific optimizations
            env['VLLM_USE_MODELSCOPE'] = 'False'
            env['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'
            env['VLLM_USE_TRITON_FLASH_ATTN'] = 'False'
            env['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = 'True'
            env['VLLM_DISABLE_ASYNC_OUTPUT_PROC'] = 'True'

            # Memory management
            env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            env['OMP_NUM_THREADS'] = '1'  # Prevent CPU oversubscription

            self._append_log(model, "Environment configured with CUDA stability fixes")

            # Start the process with a clean environment
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                preexec_fn=os.setsid,  # Create new process group
                cwd=os.path.expanduser("~")  # Start in home directory
            )

            self.processes[model.id] = process
            model.loading_pid = process.pid
            model.save(update_fields=['loading_pid'])

            self._append_log(model, f"vLLM process started with PID: {process.pid}")

            # Start output monitoring
            def monitor_output():
                try:
                    server_ready_detected = False
                    startup_timeout = 600  # 10 minutes for large models
                    start_time = time.time()

                    for line in iter(process.stdout.readline, ''):
                        if line:
                            line = line.strip()

                            # Log all important information
                            if any(keyword in line.lower() for keyword in [
                                'error', 'warning', 'loaded', 'serving', 'started',
                                'model', 'gpu', 'memory', 'ready', 'application startup complete',
                                'uvicorn running', 'cuda', 'runtime', 'ray', 'worker'
                            ]):
                                self._append_log(model, f"vLLM: {line}")

                            # Check for successful startup
                            if not server_ready_detected and (
                                    "Application startup complete" in line or
                                    "Uvicorn running" in line or
                                    "started server process" in line.lower()
                            ):
                                server_ready_detected = True
                                # Start server readiness check
                                ready_thread = threading.Thread(
                                    target=lambda: self._wait_for_server_ready(model),
                                    daemon=True
                                )
                                ready_thread.start()

                            # Check for startup timeout
                            if time.time() - start_time > startup_timeout:
                                self._append_log(model, f"Startup timeout after {startup_timeout}s", "ERROR")
                                break

                    # Process completed
                    return_code = process.wait()

                    if return_code != 0:
                        model.refresh_from_db()
                        if model.status != 'UNLOADED':
                            model.status = 'ERROR'
                            self._append_log(model, f"vLLM process exited with code: {return_code}", "ERROR")
                            model.save(update_fields=['status'])

                except Exception as e:
                    self._append_log(model, f"Output monitoring error: {str(e)}", "ERROR")
                finally:
                    if model.id in self.processes:
                        del self.processes[model.id]

            monitor_thread = threading.Thread(target=monitor_output, daemon=True)
            monitor_thread.start()

            return True

        except Exception as e:
            self._append_log(model, f"Failed to start vLLM server: {str(e)}", "ERROR")
            model.status = 'ERROR'
            model.save(update_fields=['status'])
            return False

    def _wait_for_server_ready(self, model: LLMModel, timeout: int = 600):
        """Wait for the vLLM server to be ready"""
        self._append_log(model, "Waiting for vLLM server to be ready...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"http://localhost:{model.vllm_port}/health",
                    timeout=5
                )
                if response.status_code == 200:
                    self._append_log(model, "vLLM server is ready and accepting requests!", "SUCCESS")
                    model.status = 'LOADED'
                    model.loaded_at = timezone.now()
                    model.save(update_fields=['status', 'loaded_at'])
                    return True

            except requests.exceptions.RequestException:
                pass

            time.sleep(5)  # Check every 5 seconds

        # Timeout
        self._append_log(model, f"Server readiness timeout after {timeout}s", "ERROR")
        model.status = 'ERROR'
        model.save(update_fields=['status'])
        return False

    def stop_vllm_server(self, model: LLMModel) -> bool:
        """Stop vLLM server for a model and free GPU memory"""
        try:
            self._append_log(model, "Stopping vLLM server...")

            if model.id in self.processes:
                process = self.processes[model.id]

                try:
                    # Kill entire process group
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    self._append_log(model, f"Sent termination signal to process group {process.pid}")

                    # Wait for graceful termination
                    try:
                        process.wait(timeout=20)
                        self._append_log(model, "vLLM server terminated gracefully")
                    except subprocess.TimeoutExpired:
                        # Force kill
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        self._append_log(model, "vLLM server force killed", "WARNING")

                        # Extra cleanup for Ray processes
                        try:
                            subprocess.run(["ray", "stop", "--force"],
                                           timeout=10, capture_output=True)
                            self._append_log(model, "Ray processes cleaned up")
                        except:
                            pass

                except (ProcessLookupError, OSError):
                    self._append_log(model, "vLLM server process was already terminated")

                del self.processes[model.id]

            # Force GPU memory cleanup
            self._force_gpu_memory_cleanup(model)

            model.status = 'UNLOADED'
            model.loading_pid = None
            model.save(update_fields=['status', 'loading_pid'])
            self._append_log(model, "vLLM server stopped successfully", "SUCCESS")
            return True

        except Exception as e:
            logger.error(f"Failed to stop vLLM server for model {model.name}: {str(e)}")
            self._append_log(model, f"Error stopping vLLM server: {str(e)}", "ERROR")
            return False

    def _force_gpu_memory_cleanup(self, model: LLMModel):
        """Force GPU memory cleanup after stopping vLLM server"""
        try:
            self._append_log(model, "Forcing GPU memory cleanup...")

            # Clear PyTorch cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    self._append_log(model, "PyTorch GPU cache cleared")
            except ImportError:
                pass

            # Kill any remaining CUDA processes
            try:
                # Find and kill any remaining processes using GPU
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                        capture_output=True, text=True, timeout=10)

                if result.returncode == 0 and result.stdout.strip():
                    gpu_pids = [int(pid.strip()) for pid in result.stdout.strip().split('\n') if pid.strip()]

                    for pid in gpu_pids:
                        try:
                            # Check if this PID belongs to our model process or its children
                            proc = psutil.Process(pid)
                            if 'python' in proc.name().lower() and ('vllm' in ' '.join(proc.cmdline()).lower() or
                                                                    str(model.vllm_port) in ' '.join(proc.cmdline())):
                                proc.terminate()
                                self._append_log(model, f"Terminated GPU process PID {pid}")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass

            except (subprocess.TimeoutExpired, FileNotFoundError):
                self._append_log(model, "nvidia-smi not available for GPU cleanup")

            # Additional cleanup for multi-GPU setups
            if hasattr(model, 'tensor_parallel_size') and model.tensor_parallel_size > 1:
                try:
                    # Stop Ray if it was used for distributed inference
                    subprocess.run(["ray", "stop", "--force"], timeout=10, capture_output=True)
                    self._append_log(model, "Ray cluster stopped for multi-GPU cleanup")
                except:
                    pass

            # Force garbage collection
            import gc
            gc.collect()

            self._append_log(model, "GPU memory cleanup completed")

        except Exception as e:
            self._append_log(model, f"GPU memory cleanup failed: {str(e)}", "WARNING")

    def stop_loading(self, model: LLMModel) -> bool:
        """Stop loading a model"""
        return self.stop_vllm_server(model)

    def is_server_running(self, model: LLMModel) -> bool:
        """Check if the vLLM server is running and responding"""
        try:
            if model.id not in self.processes:
                return False

            process = self.processes[model.id]
            if process.poll() is not None:
                return False

            response = requests.get(
                f"http://localhost:{model.vllm_port}/health",
                timeout=3
            )
            return response.status_code == 200

        except Exception:
            return False

    def generate_text(self, model: LLMModel, prompt: str, max_tokens: int = 512,
                      temperature: float = 0.7, top_p: float = 0.9) -> Dict[str, Any]:
        """Generate text using the vLLM OpenAI-compatible API"""
        try:
            response = requests.post(
                f"http://localhost:{model.vllm_port}/v1/completions",
                json={
                    "model": model.name,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stream": False
                },
                timeout=60
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")

        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            raise e

    def get_model_info(self, model: LLMModel) -> Dict[str, Any]:
        """Get model information from the vLLM server"""
        try:
            response = requests.get(
                f"http://localhost:{model.vllm_port}/v1/models",
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to get model info: {response.status_code}"}

        except Exception as e:
            return {"error": str(e)}


# Global service instance
vllm_service = VLLMService()