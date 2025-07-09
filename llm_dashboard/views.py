from django.views.generic import TemplateView
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .models import LLMModel, LLMRequest, Document
from .services.vector_service.token_aware_vector import TokenAwareVectorService
from .services.vllm_service import vllm_service
from django.utils import timezone
from datetime import timedelta
from django.db.models import Avg
import pynvml
import psutil
import json
import os
import threading


@csrf_exempt
@require_http_methods(["GET"])
def debug_model_status(request):
    """Debug endpoint to check model status"""
    try:
        models = LLMModel.objects.all()
        debug_info = []

        for model in models:
            debug_info.append({
                'id': model.id,
                'name': model.name,
                'status': model.status,
                'model_path': model.model_path,
                'path_exists': os.path.exists(model.model_path) if model.model_path else False,
                'loading_pid': model.loading_pid,
                'vllm_port': model.vllm_port,
                'loading_logs': model.loading_logs[-1000:] if model.loading_logs else "No logs",  # Last 1000 chars
                'process_running': model.loading_pid and psutil.pid_exists(
                    model.loading_pid) if model.loading_pid else False,
                'server_health': vllm_service.is_server_running(model) if model.status == 'LOADED' else False
            })

        return JsonResponse({'models': debug_info})
    except Exception as e:
        return JsonResponse({'error': str(e)})


@csrf_exempt
@require_http_methods(["GET"])
def browse_directories(request):
    """Browse directories for model selection"""
    try:
        # Get the requested path, default to current directory or models folder
        path = request.GET.get('path', os.path.abspath('./models'))

        # Security check - ensure we don't go outside allowed directories
        # You might want to configure this based on your setup
        allowed_base_paths = [
            os.path.abspath('./models'),
            os.path.abspath('/home'),
            os.path.abspath('/opt'),
            os.path.abspath('/data'),
            os.path.expanduser('~')  # User home directory
        ]

        # Normalize the path
        normalized_path = os.path.abspath(path)

        # Check if path is within allowed directories
        if not any(normalized_path.startswith(base) for base in allowed_base_paths):
            return JsonResponse({'error': 'Access denied to this directory'}, status=403)

        if not os.path.exists(normalized_path):
            return JsonResponse({'error': 'Directory does not exist'}, status=404)

        if not os.path.isdir(normalized_path):
            return JsonResponse({'error': 'Path is not a directory'}, status=400)

        try:
            entries = []

            # Add parent directory entry (if not at root)
            parent_dir = os.path.dirname(normalized_path)
            if normalized_path != parent_dir and any(parent_dir.startswith(base) for base in allowed_base_paths):
                entries.append({
                    'name': '..',
                    'path': parent_dir,
                    'type': 'directory',
                    'is_parent': True
                })

            # List directory contents
            for item in sorted(os.listdir(normalized_path)):
                item_path = os.path.join(normalized_path, item)

                if os.path.isdir(item_path):
                    # Check if this looks like a model directory
                    is_model = is_model_directory(item_path)

                    entries.append({
                        'name': item,
                        'path': item_path,
                        'type': 'directory',
                        'is_parent': False,
                        'is_model': is_model,
                        'size': get_directory_size(item_path) if is_model else None
                    })

            return JsonResponse({
                'current_path': normalized_path,
                'entries': entries
            })

        except PermissionError:
            return JsonResponse({'error': 'Permission denied'}, status=403)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def is_model_directory(path):
    """Check if directory contains model files"""
    try:
        files = os.listdir(path)
        # Look for common model files
        model_indicators = [
            'config.json',
            'pytorch_model.bin',
            'model.safetensors',
            'tokenizer.json',
            'tokenizer_config.json'
        ]

        return any(indicator in files for indicator in model_indicators)
    except:
        return False


def get_directory_size(path):
    """Get directory size in GB"""
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return round(total_size / (1024 ** 3), 2)  # Convert to GB
    except:
        return 0


@csrf_exempt
@require_http_methods(["POST"])
def stop_loading(request, model_id):
    """Stop loading a model"""
    try:
        model = LLMModel.objects.get(id=model_id)
        success = vllm_service.stop_loading(model)

        if success:
            return JsonResponse({'success': True})
        else:
            return JsonResponse({'success': False, 'error': 'Failed to stop loading'})
    except LLMModel.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Model not found'}, status=404)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)



@csrf_exempt
@require_http_methods(["GET"])
def get_loading_logs(request, model_id):
    """Get loading logs for a model"""
    try:
        model = LLMModel.objects.get(id=model_id)
        return JsonResponse({
            'logs': model.loading_logs,
            'status': model.status,
            'model_name': model.name
        })
    except LLMModel.DoesNotExist:
        return JsonResponse({'error': 'Model not found'})


# NEW API ENDPOINTS FOR ENHANCED LOGS FUNCTIONALITY
@require_http_methods(["GET"])
def get_model_logs(request, model_id):
    """API endpoint to fetch model loading logs"""
    try:
        model = LLMModel.objects.get(id=model_id)

        return JsonResponse({
            'success': True,
            'logs': model.loading_logs or '',
            'status': model.status,
            'model_name': model.name,
            'last_updated': model.loaded_at.isoformat() if hasattr(model, 'loaded_at') and model.loaded_at else None
        })
    except LLMModel.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Model not found'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@require_http_methods(["POST"])
def clear_model_logs(request, model_id):
    """API endpoint to clear model loading logs"""
    try:
        # Require explicit confirmation to prevent accidental clearing
        data = json.loads(request.body) if request.body else {}
        confirm = data.get('confirm', False)

        if not confirm:
            return JsonResponse({
                'success': False,
                'error': 'Confirmation required. Send {"confirm": true} to clear logs.'
            }, status=400)

        model = LLMModel.objects.get(id=model_id)

        # Archive current logs with timestamp before clearing
        timestamp = timezone.now().strftime("%Y-%m-%d %H:%M:%S")
        archived_logs = f"[{timestamp}] === LOGS MANUALLY CLEARED BY USER ===\n"
        if model.loading_logs:
            archived_logs += f"Previous logs length: {len(model.loading_logs)} characters\n"
        archived_logs += "=== END OF CLEARED LOGS ===\n\n"

        model.loading_logs = archived_logs
        model.save()

        return JsonResponse({
            'success': True,
            'message': 'Logs cleared successfully (with archive marker)'
        })
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON data'
        }, status=400)
    except LLMModel.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Model not found'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


def get_gpu_stats():
    try:
        pynvml.nvmlInit()
        gpu_stats = []
        deviceCount = pynvml.nvmlDeviceGetCount()

        for i in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

            memory_total = info.total / (1024 ** 3)  # Convert to GB
            memory_used = info.used / (1024 ** 3)

            # Handle both string and bytes return types for GPU name
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode('utf-8')

            gpu_stats.append({
                'id': i,
                'name': gpu_name,
                'memory_total': memory_total,
                'memory_used': memory_used,
                'memory_free': info.free / (1024 ** 3),
                'memory_percentage': round((memory_used / memory_total) * 100, 1) if memory_total > 0 else 0,
                'temperature': temperature,
                'gpu_utilization': utilization.gpu,
                'memory_utilization': utilization.memory
            })

        return gpu_stats
    except Exception as e:
        return [{'error': str(e)}]
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass


def get_available_models():
    """Get list of available models from models directory"""
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    available_models = []

    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path):
                # Check if it's a valid model directory (contains config.json or similar)
                config_files = ['config.json', 'model.json', 'pytorch_model.bin', 'model.safetensors']
                if any(os.path.exists(os.path.join(item_path, f)) for f in config_files):
                    available_models.append({
                        'name': item,
                        'path': item_path,
                        'size_gb': get_directory_size(item_path)
                    })

    return available_models


class DashboardView(TemplateView):
    template_name = 'llm_dashboard/dashboard.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # Get the first model (should only be one active model)
        context['model'] = LLMModel.objects.filter(status__in=['LOADING', 'LOADED', 'ERROR']).first()

        context['recent_requests'] = LLMRequest.objects.order_by('-created_at')[:10]
        context['available_models'] = get_available_models()

        # Get GPU count safely
        gpu_stats = get_gpu_stats()
        context['gpu_count'] = len([gpu for gpu in gpu_stats if not gpu.get('error')])
        context['gpu_stats'] = gpu_stats

        # Calculate stats
        last_hour = timezone.now() - timedelta(hours=1)
        hourly_requests = LLMRequest.objects.filter(created_at__gte=last_hour)

        context['stats'] = {
            'total_requests': LLMRequest.objects.count(),
            'hourly_requests': hourly_requests.count(),
            'avg_duration': hourly_requests.aggregate(Avg('duration'))['duration__avg'],
            'error_rate': (hourly_requests.filter(status='ERROR').count() /
                           max(hourly_requests.count(), 1)) * 100
        }

        return context


@csrf_exempt
@require_http_methods(["POST"])
def load_model(request):
    """Load a model into vLLM"""
    try:
        data = json.loads(request.body)
        model_name = data.get('name')  # Changed from 'model_name' to match frontend
        model_path = data.get('model_path')
        tensor_parallel_size = data.get('tensor_parallel_size', 1)
        gpu_memory_utilization = data.get('gpu_memory_utilization', 0.9)
        max_tokens = data.get('max_tokens', 2048)
        dtype = data.get('dtype', 'auto')

        if not model_name or not model_path:
            return JsonResponse({'error': 'Model name and path are required'}, status=400)

        # Check if any model is currently loading or loaded
        existing_model = LLMModel.objects.filter(status__in=['LOADING', 'LOADED']).first()
        if existing_model:
            return JsonResponse({
                'error': f'Model "{existing_model.name}" is already {existing_model.status.lower()}. Please unload it first.'},
                status=400)

        # Check if there's an existing model with the same name and path
        existing_model_same_config = LLMModel.objects.filter(
            name=model_name,
            model_path=model_path
        ).first()

        if existing_model_same_config:
            # Reuse existing model and preserve logs
            model = existing_model_same_config
            model.tensor_parallel_size = tensor_parallel_size
            model.gpu_memory_utilization = gpu_memory_utilization
            model.max_tokens = max_tokens
            model.dtype = dtype
            model.status = 'LOADING'

            # PRESERVE existing logs and add separator
            if model.loading_logs:
                timestamp = timezone.now().strftime("%Y-%m-%d %H:%M:%S")
                model.loading_logs += f"\n{'=' * 60}\n[{timestamp}] === RESTARTING MODEL LOADING ===\n{'=' * 60}\n"
            else:
                model.loading_logs = f"Initializing model loading for {model_name}...\n"
            model.save()
        else:
            # Remove any old unloaded models (but preserve errored ones for debugging)
            LLMModel.objects.filter(status='UNLOADED').delete()

            # Create new model instance with initial log
            model = LLMModel.objects.create(
                name=model_name,
                model_path=model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_tokens=max_tokens,
                dtype=dtype,
                status='LOADING',
                loading_logs=f"Initializing model loading for {model_name}...\n"
            )

        # Start vLLM server in background thread
        def start_server():
            try:
                success = vllm_service.start_vllm_server(model)
                if not success:
                    # Refresh model from database to get latest logs
                    model.refresh_from_db()
                    if model.status != 'ERROR':
                        model.status = 'ERROR'
                        # Use the new logging method to preserve logs
                        vllm_service._append_log(model, "Failed to start vLLM server", "ERROR")
            except Exception as e:
                model.status = 'ERROR'
                # Use the new logging method to preserve logs
                vllm_service._append_log(model, f"Exception during loading: {str(e)}", "ERROR")

        thread = threading.Thread(target=start_server)
        thread.daemon = True
        thread.start()

        return JsonResponse({
            'success': True,
            'message': 'Model loading started',
            'model_id': model.id
        })

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def unload_model(request, model_id):
    """Unload a model from vLLM"""
    try:
        model = get_object_or_404(LLMModel, id=model_id)

        # Stop vLLM server
        success = vllm_service.stop_vllm_server(model)

        if success:
            return JsonResponse({'success': True, 'message': 'Model unloaded successfully'})
        else:
            return JsonResponse({'error': 'Failed to unload model'}, status=500)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def force_cleanup_all_gpus(request):
    """Force cleanup of all GPUs - kill all vLLM processes and clear GPU memory"""
    try:
        from .services.vllm_service import vllm_service

        # Get all models that might be running
        models = LLMModel.objects.all()

        cleanup_results = []

        # First, try to gracefully stop all known models
        for model in models:
            try:
                if model.status in ['LOADED', 'LOADING']:
                    success = vllm_service.stop_vllm_server(model)
                    cleanup_results.append(
                        f"Model {model.name}: {'Stopped' if success else 'Failed to stop gracefully'}")
            except Exception as e:
                cleanup_results.append(f"Model {model.name}: Error - {str(e)}")

        # Then do comprehensive GPU cleanup
        try:
            vllm_service.cleanup_gpu_memory()
            cleanup_results.append("GPU memory cleanup completed")
        except Exception as e:
            cleanup_results.append(f"GPU cleanup error: {str(e)}")

        # Force kill any remaining vLLM processes
        import psutil
        import signal
        killed_processes = []

        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'python' in proc.info['name'].lower():
                        cmdline = ' '.join(proc.info['cmdline'] or [])
                        if any(keyword in cmdline.lower() for keyword in ['vllm', 'api_server', 'openai.api_server']):
                            proc.terminate()
                            killed_processes.append(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            if killed_processes:
                cleanup_results.append(f"Killed vLLM processes: {killed_processes}")
        except Exception as e:
            cleanup_results.append(f"Process cleanup error: {str(e)}")

        # Reset all model statuses
        models.update(status='UNLOADED', loading_pid=None)
        cleanup_results.append("All model statuses reset to UNLOADED")

        # Additional Ray cleanup
        try:
            import subprocess
            result = subprocess.run(["ray", "stop", "--force"], timeout=15, capture_output=True, text=True)
            cleanup_results.append("Ray cluster stopped")
        except Exception as e:
            cleanup_results.append(f"Ray cleanup: {str(e)}")

        # Clear PyTorch cache on all GPUs
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                cleanup_results.append(f"Cleared PyTorch cache on {torch.cuda.device_count()} GPUs")
        except Exception as e:
            cleanup_results.append(f"PyTorch cleanup error: {str(e)}")

        return JsonResponse({
            'success': True,
            'message': 'Force GPU cleanup completed',
            'details': cleanup_results
        })

    except Exception as e:
        return JsonResponse({
            'error': f'Force cleanup failed: {str(e)}',
            'details': []
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def test_model(request):
    """Test model with a prompt"""
    try:
        data = json.loads(request.body)
        prompt = data.get('prompt', '')

        if not prompt:
            return JsonResponse({'error': 'Prompt is required'}, status=400)

        # Get loaded model
        model = LLMModel.objects.filter(status='LOADED').first()
        if not model:
            return JsonResponse({'error': 'No loaded model available'}, status=400)

        # Create request record
        llm_request = LLMRequest.objects.create(
            model=model,
            prompt=prompt,
            temperature=data.get('temperature', 0.7),
            top_p=data.get('top_p', 0.9),
            max_tokens=data.get('max_tokens', 512),
            status='PENDING'
        )

        start_time = timezone.now()

        try:
            # Generate text using vLLM
            result = vllm_service.generate_text(
                model=model,
                prompt=prompt,
                max_tokens=data.get('max_tokens', 512),
                temperature=data.get('temperature', 0.7),
                top_p=data.get('top_p', 0.9)
            )

            # Extract response
            response_text = result['choices'][0]['text']
            tokens_generated = result['usage']['completion_tokens']
            tokens_prompt = result['usage']['prompt_tokens']

            # Update request record
            llm_request.response = response_text
            llm_request.tokens_generated = tokens_generated
            llm_request.tokens_prompt = tokens_prompt
            llm_request.duration = (timezone.now() - start_time).total_seconds()
            llm_request.status = 'COMPLETED'
            llm_request.save()

            return JsonResponse({
                'success': True,
                'response': response_text,
                'tokens_generated': tokens_generated,
                'tokens_prompt': tokens_prompt,
                'duration': llm_request.duration
            })

        except Exception as e:
            llm_request.status = 'ERROR'
            llm_request.error_message = str(e)
            llm_request.duration = (timezone.now() - start_time).total_seconds()
            llm_request.save()
            return JsonResponse({'success': False, 'error': str(e)}, status=500)

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def generate_text(request):
    """Generate text using loaded model"""
    try:
        data = json.loads(request.body)
        prompt = data.get('prompt', '')
        model_id = data.get('model_id')

        if not prompt:
            return JsonResponse({'error': 'Prompt is required'}, status=400)

        # Get model
        if model_id:
            model = get_object_or_404(LLMModel, id=model_id)
        else:
            model = LLMModel.objects.filter(status='LOADED').first()
            if not model:
                return JsonResponse({'error': 'No loaded model available'}, status=400)

        # Create request record
        llm_request = LLMRequest.objects.create(
            model=model,
            prompt=prompt,
            temperature=data.get('temperature', 0.7),
            top_p=data.get('top_p', 0.9),
            max_tokens=data.get('max_tokens', 512),
            status='PENDING'
        )

        start_time = timezone.now()

        try:
            # Generate text using vLLM
            result = vllm_service.generate_text(
                model=model,
                prompt=prompt,
                max_tokens=data.get('max_tokens', 512),
                temperature=data.get('temperature', 0.7),
                top_p=data.get('top_p', 0.9)
            )

            # Extract response
            response_text = result['choices'][0]['text']
            tokens_generated = result['usage']['completion_tokens']
            tokens_prompt = result['usage']['prompt_tokens']

            # Update request record
            llm_request.response = response_text
            llm_request.tokens_generated = tokens_generated
            llm_request.tokens_prompt = tokens_prompt
            llm_request.duration = (timezone.now() - start_time).total_seconds()
            llm_request.status = 'COMPLETED'
            llm_request.save()

            return JsonResponse({
                'response': response_text,
                'tokens_generated': tokens_generated,
                'tokens_prompt': tokens_prompt,
                'duration': llm_request.duration
            })

        except Exception as e:
            llm_request.status = 'ERROR'
            llm_request.error_message = str(e)
            llm_request.duration = (timezone.now() - start_time).total_seconds()
            llm_request.save()
            raise e

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def model_stats(request):
    model = LLMModel.objects.first()
    gpu_stats = get_gpu_stats()

    # Handle case where no model exists in database
    if model is None:
        return JsonResponse({
            'error': 'No LLM model found in database',
            'name': None,
            'status': 'NOT_FOUND',
            'memory_usage': 0,
            'uptime': 0,
            'gpu_stats': gpu_stats
        })

    # Check if server is actually running
    if model.status == 'LOADED' and not vllm_service.is_server_running(model):
        model.status = 'ERROR'
        model.save()

    return JsonResponse({
        'name': model.name,
        'status': model.status,
        'memory_usage': model.memory_usage,
        'uptime': (timezone.now() - model.loaded_at).total_seconds(),
        'gpu_stats': gpu_stats,
        'tensor_parallel_size': model.tensor_parallel_size,
        'model_size_gb': model.get_model_size_gb()
    })


@csrf_exempt
@require_http_methods(["POST"])
def generate_with_document_rag(request, model_id):
    """Generate text using RAG with a specific document ID"""
    try:
        model = get_object_or_404(LLMModel, id=model_id)
        data = json.loads(request.body)

        document_id = data.get('document_id')
        if not document_id:
            return JsonResponse({'error': 'document_id is required'}, status=400)

        # Verify document exists
        document = get_object_or_404(Document, id=document_id, model=model)

        result = vllm_service.generate_text_with_smart_rag(
            model=model,
            prompt=data.get('prompt', ''),
            document_id=document_id,
            max_tokens=data.get('max_tokens', 100),
            temperature=data.get('temperature', 0.7),
            top_p=data.get('top_p', 0.9),
            strategy=data.get('strategy')
        )

        return JsonResponse(result)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def list_documents(request, model_id):
    """List all documents for a model"""
    model = get_object_or_404(LLMModel, id=model_id)
    documents = Document.objects.filter(model=model).values(
        'id', 'title', 'created_at', 'updated_at', 'is_indexed', 'estimated_tokens'
    )

    return JsonResponse({'documents': list(documents)})


@require_http_methods(["GET"])
def get_document_chunks(request, model_id, document_id):
    """Get all chunks for a specific document"""
    model = get_object_or_404(LLMModel, id=model_id)
    vector_service = TokenAwareVectorService(model)

    chunks = vector_service.get_document_chunks_by_id(document_id)

    return JsonResponse({'chunks': chunks})


@csrf_exempt
@require_http_methods(["POST"])
def search_document_chunks(request, model_id, document_id):
    """Search for relevant chunks within a specific document"""
    try:
        model = get_object_or_404(LLMModel, id=model_id)
        data = json.loads(request.body)

        query = data.get('query', '')
        k = data.get('k', 5)

        vector_service = TokenAwareVectorService(model)
        chunks = vector_service.search_document_chunks(document_id, query, k)

        return JsonResponse({'chunks': chunks})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def add_document_api(request, model_id):
    """Add a document via API"""
    try:
        model = get_object_or_404(LLMModel, id=model_id)
        data = json.loads(request.body)

        title = data.get('title', '')
        content = data.get('content', '')

        if not title or not content:
            return JsonResponse({'error': 'title and content are required'}, status=400)

        # Create document
        document = Document.objects.create(
            model=model,
            title=title,
            content=content,
            metadata=data.get('metadata', {})
        )

        # Add to vector store
        vector_service = TokenAwareVectorService(model)
        success = vector_service.add_document_to_vector_store(
            document=document,
            target_chunk_tokens=data.get('chunk_tokens', 300),
            overlap_tokens=data.get('overlap_tokens', 50)
        )

        if success:
            return JsonResponse({
                'success': True,
                'document_id': str(document.id),
                'message': 'Document added successfully'
            })
        else:
            return JsonResponse({'error': 'Failed to add document to vector store'}, status=500)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
