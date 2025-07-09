import subprocess
import time
import os
import signal
import psutil
import requests
import json
from typing import Dict, Any, Optional
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

from .vector_service.token_aware_vector import TokenAwareVectorService
from ..models import LLMModel, LLMRequest, Document
from .map_reduce_service import MapReduceService
import logging

logger = logging.getLogger(__name__)

class VLLMService:
    def __init__(self):
        self.processes = {}  # Store process info for each model
        self.channel_layer = get_channel_layer()

    def _append_log(self, model: LLMModel, message: str):
        """Append log message to model's loading logs"""
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

    def _check_cuda_environment(self, model: LLMModel):
        """Check and configure CUDA environment"""
        self._append_log(model, "🔧 Checking CUDA environment...")
        
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            self._append_log(model, f"✅ Found {gpu_count} CUDA devices")
            
            # Set CUDA environment variables for stability
            cuda_env = {
                'CUDA_VISIBLE_DEVICES': ','.join(str(i) for i in range(min(gpu_count, model.tensor_parallel_size))),
                'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1',
                'VLLM_ATTENTION_BACKEND': 'FLASHINFER',
                'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
                'TOKENIZERS_PARALLELISM': 'false',
                'CUDA_LAUNCH_BLOCKING': '0'
            }
            
            for key, value in cuda_env.items():
                os.environ[key] = value
                self._append_log(model, f"   {key}={value}")
            
            return True
            
        except Exception as e:
            self._append_log(model, f"❌ CUDA environment check failed: {e}")
            return False

    def start_vllm_server(self, model: LLMModel):
        """Start vLLM server for the given model"""
        try:
            self._append_log(model, f"🚀 Starting vLLM server for {model.name}")
            
            # Check if already running
            if self.is_server_running(model):
                self._append_log(model, "⚠️ Server already running")
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
            self._monitor_process_output(model, process)
            
            # Wait for server to be ready
            if self._wait_for_server_ready(model):
                model.status = 'LOADED'
                model.save()
                self._append_log(model, "✅ Server is ready and responding!")
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
        """Monitor process output and log it"""
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
        """Wait for vLLM server to be ready"""
        self._append_log(model, "⏳ Waiting for server to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f'http://localhost:{model.vllm_port}/health', timeout=5)
                if response.status_code == 200:
                    return True
            except:
                pass
            
            time.sleep(5)
            elapsed = int(time.time() - start_time)
            self._append_log(model, f"⏳ Still waiting... ({elapsed}s elapsed)")
        
        self._append_log(model, "❌ Server did not become ready within timeout")
        return False

    def stop_vllm_server(self, model: LLMModel):
        """Stop vLLM server for the given model"""
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
        """Force cleanup of GPU memory and processes"""
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
        """Stop loading process"""
        try:
            if model.loading_pid:
                os.kill(model.loading_pid, signal.SIGTERM)
        except:
            pass

    def is_server_running(self, model: LLMModel) -> bool:
        """Check if vLLM server is running and responding"""
        try:
            response = requests.get(f'http://localhost:{model.vllm_port}/health', timeout=5)
            return response.status_code == 200
        except:
            return False

    def generate_text(self, model: LLMModel, prompt: str, max_tokens: int = 100, 
                     temperature: float = 0.7, top_p: float = 0.9) -> Dict:
        """Generate text using the loaded model"""
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

    def generate_text_with_smart_rag(self, model: LLMModel, prompt: str, 
                                   document_id: str, max_tokens: int = 100,
                                   temperature: float = 0.7, top_p: float = 0.9,
                                   strategy: str = None) -> Dict:
        """Generate text using smart RAG with token management"""
        
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
        """Process standard RAG (sliding window, hybrid, direct)"""
        
        vector_service = TokenAwareVectorService(model)
        
        return vector_service.get_chunks_within_token_limit(
            document_id=document_id,
            query=query,
            max_tokens=max_tokens,
            strategy=strategy
        )
    
    def _process_map_reduce_rag(self, model: LLMModel, document_id: str, 
                              query: str, max_tokens: int) -> Dict:
        """Process map-reduce RAG"""
        
        map_reduce_service = MapReduceService(model, self)
        
        return map_reduce_service.process_document_with_map_reduce(
            document_id=document_id,
            query=query,
            max_tokens=max_tokens
        )
    
    def _build_enhanced_prompt(self, original_prompt: str, context_result: Dict) -> str:
        """Build enhanced prompt with context"""
        
        if context_result.get('strategy') == 'map_reduce':
            # For map-reduce, use summaries
            summaries = context_result.get('map_results', [])
            if summaries:
                context_text = "\n\n".join([
                    f"Summary {i+1}: {summary['summary']}"
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
                    f"Context {i+1}: {chunk['content']}"
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