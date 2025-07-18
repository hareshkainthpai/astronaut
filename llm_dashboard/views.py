import asyncio

from django.views.generic import TemplateView
from django.http import JsonResponse, HttpResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .models import LLMModel, LLMRequest, Document, DocumentChunk
from .services.vector_service.document_vector import DocumentVectorStoreService
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
    """
    Handles an HTTP GET request to return the debug status of all models stored in the database.
    The method collects the information about model details, such as its name, status, model path,
    loading process, and server health. The response contains a list of dictionaries, with each
    entry representing the debug status of a model.

    :param request: Django HttpRequest object representing the incoming HTTP GET request.
    :type request: HttpRequest
    :return: HttpResponse containing the JSON representation of the models' debug status or an error message.
    :rtype: JsonResponse
    """
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
    """
    Processes a request to browse directories and fetch their content. The function
    validates the input path, ensures it falls within predefined allowed directories,
    and generates details of the requested directory's contents, including parent
    directory links and information about nested directories.

    :param request: HTTP request object, expected to contain 'path' as a query
                    parameter to specify the directory to be listed.
    :type request: django.http.HttpRequest

    :return: A JSON response containing metadata about the files and directories,
             including the current path, entries (name, type, and other details),
             or an error message if the request parameters are invalid.
    :rtype: django.http.JsonResponse
    """
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
    """
    Checks if the given directory path contains files that typically indicate the presence
    of a machine learning model. The function looks for specific filenames commonly used
    in frameworks such as PyTorch or Hugging Face to determine if the directory is a
    model directory.

    :param path: str
        The path to the directory to check.

    :return: bool
        Returns True if the directory contains any of the expected model indicator files,
        otherwise returns False.
    """
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
    """
    Calculate the total size of all files in a given directory and its subdirectories.

    This function traverses through the directory tree starting from the given path
    and aggregates the total size of files in gigabytes (GB). If an error occurs during
    the computation (e.g., due to invalid paths or permission issues), it safely returns 0.

    :param path: The directory path whose total file size needs to be calculated.
    :type path: str

    :return: The total size of files in the directory and its subdirectories in gigabytes,
        rounded to two decimal places. Returns 0 if an error occurs.
    :rtype: float
    """
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
    """
    Handles stopping the loading process of a specified model through a POST request.

    This function is designed to interact with the `vllm_service` to stop the loading
    process of a specific model identified by its unique ID. It handles different
    errors such as when the model does not exist or any unexpected internal
    exceptions.

    :param request: HTTP request object.
    :type request: HttpRequest
    :param model_id: Unique identifier of the model to stop its loading process.
    :type model_id: int
    :return: JSON response indicating whether the operation was successful or failed.
    :rtype: JsonResponse
    :raises LLMModel.DoesNotExist: Indicates the specified model ID does not exist.
    :raises Exception: Handles unexpected exceptions that occur during the process.
    """
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
def api_gpu_stats(request):
    """
    Handles API requests to fetch GPU stats. The view supports only GET requests and is
    exempt from CSRF verification. It interacts with the GPU monitoring system to gather
    data and returns it as a JSON response. If an exception occurs during this process,
    an appropriate error response is returned.

    :param request: HTTP request object.
    :type request: HttpRequest
    :return: JSON response containing GPU stats, count, and timestamp or an error message.
    :rtype: JsonResponse
    """
    try:
        gpu_stats = get_gpu_stats()
        gpu_count = len([gpu for gpu in gpu_stats if not gpu.get('error')])

        return JsonResponse({
            'success': True,
            'gpu_stats': gpu_stats,
            'gpu_count': gpu_count,
            'timestamp': timezone.now().isoformat()
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def get_loading_logs(request, model_id):
    """
    Fetches and returns the loading logs, status, and model name for a given model.

    This view function retrieves details of a model identified by its ID. If the model
    exists, it returns the loading logs, status, and name via a JSON response. If no
    such model exists, it returns an error message in JSON format.

    :param request: Django HttpRequest object for the incoming request.
    :type request: HttpRequest
    :param model_id: ID of the model to retrieve loading logs for.
    :type model_id: int
    :return: A JsonResponse containing the loading logs, status, and model name if
             the model exists, or an error message if the model does not exist.
    :rtype: JsonResponse
    """
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
    """
    Fetches the logs and status information of a specified model by its ID.

    This function handles HTTP GET requests and retrieves logs, model status,
    name, and last updated timestamp for a model identified by the provided
    `model_id`. If the model does not exist, it returns a JSON response with an
    error message and a 404 status code. If any other exceptions occur, it will
    respond with an error message and a 500 status code.

    :param request: The HTTP request object.
    :param model_id: The ID of the model whose logs and status are to
        be retrieved.
    :type model_id: int
    :return: A JSON response containing the model logs, status, name,
        last updated timestamp, or an error message.
    :rtype: JsonResponse
    """
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
    """
    Clear the logs of a specified LLMModel after an explicit confirmation. This function
    requires a POST HTTP request and validates the confirmation from the request body
    before proceeding to clear the logs. The previous logs are archived with a
    timestamp for reference.

    :param request: The HTTP request object containing the body payload with
        confirmation to clear logs.
    :type request: HttpRequest
    :param model_id: The ID of the LLMModel whose logs need to be cleared.
    :type model_id: int
    :return: A JsonResponse indicating the success or failure of the log clearing
        process. It also provides appropriate messages in case of errors.
    :rtype: JsonResponse
    :raises JSONDecodeError: Raised if the request body contains invalid JSON format.
    :raises LLMModel.DoesNotExist: Raised if the specified model ID does not correspond
        to any existing LLMModel instance.
    :raises Exception: Handles any unexpected server-side issues during the process.
    """
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
    """
    Gets the statistics of GPUs on the system using the PyNVML library. This function retrieves
    detailed information for each GPU including memory details, utilization rates, temperature,
    and GPU name. If an exception occurs, it includes the error message in the returned data.
    Ensure that the PyNVML library is properly installed and supported by the hardware.

    :return: A list of dictionaries containing GPU statistics. Each dictionary includes:
             - id (int): GPU index.
             - name (str): Name of the GPU.
             - memory_total (float): Total memory of the GPU in GB.
             - memory_used (float): Memory currently in use on the GPU in GB.
             - memory_free (float): Free memory on the GPU in GB.
             - memory_percentage (float): The percentage of memory currently used.
             - temperature (int): Temperature of the GPU in Celsius.
             - gpu_utilization (int): GPU utilization percentage.
             - memory_utilization (int): Memory utilization percentage.
             If an error occurs, a dictionary with an 'error' key containing the error message
             will be returned.
    :rtype: list[dict]
    """
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
    """
    Retrieve a list of available models from the 'models' directory. A model is considered
    available if it resides within a subdirectory that contains at least one of the specified
    configuration files (e.g., config.json, model.json, etc.).

    :return: A list of dictionaries, where each dictionary contains details about a valid
             model directory, including its name, path, and size (in GB).
    :rtype: list[dict]
    """
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
    """
    Represents the dashboard view, used to render detailed information about the
    current state of LLM models, requests, and system statistics.

    This class provides a framework to display a comprehensive dashboard where
    relevant details such as the loaded LLM model, GPU usage, system statistics, and
    recent requests are easily accessible and actively updated.

    :ivar template_name: Path to the template used for rendering the dashboard view.
    :type template_name: str
    """
    template_name = 'llm_dashboard/dashboard.html'

    def get_context_data(self, **kwargs):
        """
        Generate context data for the view, including model details, recent requests, available models,
        GPU statistics, and various calculated stats related to LLM requests. The generated context
        provides comprehensive data to be consumed in templates and other components.

        :param kwargs: Additional context arguments passed by the parent view.
        :return: A dictionary containing the context for rendering the template, including detailed model,
            GPU, and request statistics.
        :rtype: dict
        """
        context = super().get_context_data(**kwargs)

        # Get the first model (should only be one active model)
        context['model'] = LLMModel.objects.filter(status__in=['LOADING', 'LOADED', 'ERROR']).first()

        # Add recent requests with all needed fields
        recent_requests = LLMRequest.objects.select_related('model').order_by('-created_at')[:10]
        context['recent_requests'] = recent_requests

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
    """
    Handles the loading of a machine learning model based on the provided request
    data. Checks for any existing models currently being loaded or already loaded.
    Initializes a new instance or updates an existing one with the requested
    configuration and starts the vLLM server in a background thread.

    :param request: The HTTP request object, expected to contain a JSON payload with
        the model loading configuration. The payload should include:
        - 'name': The name of the model (required, str).
        - 'model_path': The path to the model (required, str).
        - 'tensor_parallel_size': The degree of tensor parallelization (optional, int).
        - 'gpu_memory_utilization': The desired GPU memory utilization (optional, float).
        - 'max_tokens': The maximum tokens allowed (optional, int).
        - 'dtype': The desired data type for the model (optional, str).

    :return: A JsonResponse indicating the result of the operation. Possible
        responses include:
        - Success response with a message of successful initialization and the
          model ID (status 200).
        - Error response indicating missing parameters, conflicting model statuses,
          or issues during initialization (status 400 or 500).
    """
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
    """
    Handles the unloading of a machine learning model from the vLLM server.

    This function allows the user to stop the vLLM server associated with
    a specific machine learning model. If the server stops successfully,
    a success message is returned. Otherwise, an error message is returned.

    :param request:
        The HTTP request object, expected to be a POST request.
    :param model_id:
        The ID of the machine learning model to be unloaded.
    :type model_id: int
    :return:
        A JSON response indicating the success or failure of the unload action.
    :rtype: JsonResponse
    """
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
    """
    Forcefully cleans up GPU resources and resets the application state for all
    models and associated processes. This includes tasks such as stopping vLLM
    servers, clearing GPU memory, force-killing processes, resetting model statuses,
    and performing additional cleanup of Ray clusters and PyTorch caches.

    :param request: Django HTTP request object.
    :type request: django.http.HttpRequest
    :return: JSON response containing success status, message, and cleanup details.
    :rtype: django.http.JsonResponse
    """
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
    """
    Handles the generation of text using a pre-loaded language model. This function interacts with a
    loaded language model to generate text based on the provided prompt and optional parameters
    like max_tokens, temperature, and top_p. The function supports the logging and tracking of the
    request's execution details, including response, total tokens generated, and duration. In case
    of any errors during processing, appropriate error messages are returned.

    :param request: The HTTP POST request containing the JSON payload with input data. The
                    payload must include a 'prompt' field, with optional 'temperature',
                    'top_p', and 'max_tokens' for parameter configuration.
    :return: A JsonResponse containing the generated text, token usage statistics, or an
             error message in case of failure.
    """
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
    """
    Handles text generation requests by accepting input parameters, validating them,
    and generating text using a preloaded language model. Results, including response
    text, token usage, and processing duration, are logged and returned in JSON format.

    :param request: Django HttpRequest object containing the POST request data.
    :type request: HttpRequest
    :return: JsonResponse containing the generated text, token counts, and duration,
        or an error message if the request fails.
    :rtype: JsonResponse
    :raises Http404: If the specified LLM model is not found.
    :raises JsonResponse: If a required parameter is missing or an error occurs.
    """
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

        # Add initial log
        llm_request.add_log(f"Request started - Model: {model.name}")
        llm_request.add_log(
            f"Parameters: temp={llm_request.temperature}, top_p={llm_request.top_p}, max_tokens={llm_request.max_tokens}")

        start_time = timezone.now()

        try:
            llm_request.add_log("Starting text generation...")

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
    """
    Handles retrieval of the state and statistics of an LLM model from the database
    including its memory usage, runtime status, GPU statistics, and other details.

    :param request: The incoming HTTP request
    :type request: HttpRequest

    :return: A JsonResponse containing details of the LLM model such as name,
        status, memory usage, uptime, GPU statistics, tensor parallel size,
        and model size in GB. If no model exists, an error response is returned.
    :rtype: JsonResponse
    """
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

    # Check if server is actually running and update status accordingly
    if model.status == 'LOADED' and not vllm_service.is_server_running(model):
        model.status = 'ERROR'
        model.memory_usage = 0.0
        model.save()
    elif model.status == 'LOADING' and vllm_service.is_server_running(model):
        # Server is ready but status wasn't updated
        model.status = 'LOADED'
        # Calculate memory usage
        try:
            memory_usage = vllm_service._calculate_gpu_memory_usage(model)
            model.memory_usage = memory_usage
        except:
            model.memory_usage = 0.0
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


def refresh_model_status(request):
    """
    Refreshes the status of the language model by verifying if the associated server
    is running and updating status information accordingly.

    The function handles a POST request and interacts with an instance of `LLMModel`.
    If no model is found, or if the server is unresponsive, the status and memory
    usage attributes of the model are updated in the database. It responds with JSON
    data indicating success or failure and includes updated status and memory usage
    when applicable.

    :param request: The HTTP request object containing data for processing the operation.
    :type request: HttpRequest
    :return: A JSON response indicating the success or failure of the status refresh,
             along with related details such as updated status and memory usage if
             applicable.
    :rtype: JsonResponse
    """
    if request.method == 'POST':
        try:
            model = LLMModel.objects.first()
            if not model:
                return JsonResponse({'success': False, 'error': 'No model found'})

            # Check if server is actually running
            if vllm_service.is_server_running(model):
                model.status = 'LOADED'
                # Calculate memory usage
                try:
                    memory_usage = vllm_service._calculate_gpu_memory_usage(model)
                    model.memory_usage = memory_usage
                except:
                    model.memory_usage = 0.0
                model.save()

                return JsonResponse({
                    'success': True,
                    'status': model.status,
                    'memory_usage': model.memory_usage
                })
            else:
                model.status = 'ERROR'
                model.memory_usage = 0.0
                model.save()

                return JsonResponse({
                    'success': False,
                    'error': 'Server not responding',
                    'status': model.status
                })

        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})

    return JsonResponse({'success': False, 'error': 'Invalid request method'})


@csrf_exempt
@require_http_methods(["POST"])
def generate_with_document_rag(request, model_id):
    """
    Handles a POST request for generating text using a Retrieval-Augmented Generation (RAG) approach
    with a specific document and language model. Validates input to ensure necessary parameters are
    provided and the specified document exists. Generates text using the provided model and RAG strategy.

    :param request: Django HttpRequest object containing the POST request data
    :param model_id: Unique identifier for the LLMModel instance
    :type model_id: int
    :return: JsonResponse containing the RAG text generation result or an error message
    :rtype: JsonResponse
    :raises Http404: If the specified model or document cannot be found
    """
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
    """
    Handles the retrieval of a list of documents associated with a specific model.

    This view function processes HTTP GET requests and fetches document data
    associated with the given model ID. The response includes document details
    such as ID, title, creation and update timestamps, indexing status, and
    estimated token count.

    :param request: The HTTP request object.
    :type request: HttpRequest
    :param model_id: The identifier of the model whose documents are to be listed.
    :type model_id: int

    :return: A JSON response containing a list of documents associated with
             the specified model.
    :rtype: JsonResponse
    """
    model = get_object_or_404(LLMModel, id=model_id)
    documents = Document.objects.filter(model=model).values(
        'id', 'title', 'created_at', 'updated_at', 'is_indexed', 'estimated_tokens'
    )

    return JsonResponse({'documents': list(documents)})


@require_http_methods(["GET"])
def get_document_chunks(request, model_id, document_id):
    """
    Retrieves document chunks for a specific model and document.

    This view handles incoming HTTP GET requests to fetch the chunks of a
    document associated with a particular machine learning model. The
    document chunks are retrieved from a vectorization service that is bound
    to the model.

    :param request: The HTTP GET request object received from the client.
    :type request: HttpRequest
    :param model_id: The unique identifier for the machine learning model
        to which the document belongs.
    :type model_id: int
    :param document_id: The unique identifier for the document whose chunks
        are to be retrieved.
    :type document_id: int
    :return: A JSON response containing the chunks of the specified document.
    :rtype: JsonResponse
    """
    model = get_object_or_404(LLMModel, id=model_id)
    vector_service = TokenAwareVectorService(model)

    chunks = vector_service.get_document_chunks_by_id(document_id)

    return JsonResponse({'chunks': chunks})


@csrf_exempt
@require_http_methods(["POST"])
def search_document_chunks(request, model_id, document_id):
    """
    Handles POST requests to search for document chunks based on a query and model.
    This function interacts with a language model to search relevant chunks within
    a specified document and returns the results in JSON format.

    :param request: HttpRequest object containing metadata and data related
        to the client's request.
    :type request: HttpRequest
    :param model_id: The unique identifier of the language model.
    :type model_id: int
    :param document_id: The unique identifier of the document to be searched.
    :type document_id: int
    :return: JsonResponse containing a list of document chunks if successful,
        or an error message in case of failure.
    :rtype: JsonResponse
    """
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
    """
    Handles the API request for adding a new document to a specific model.

    This function accepts a POST request containing details of a document to be added
    to the specified model. The document is created, stored, and linked to a vector
    store for further processing.

    :param request: The HTTP request object containing the document data, expected to
        be in JSON format.
    :type request: HttpRequest
    :param model_id: The ID of the model to which the document is to be associated.
    :type model_id: int
    :return: A JSON response indicating success or failure of the document addition
        operation, including the newly created document ID on success.
    :rtype: JsonResponse
    """
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


@require_http_methods(["GET"])
def get_request_details(request, request_id):
    """
    Retrieves and returns detailed information about a specific LLM request based on the provided request ID.
    The response contains metadata, processing information, and contextual details for the requested LLM interaction.

    :param request: HTTP request object representing the current client request.
    :type request: HttpRequest
    :param request_id: ID of the LLM request to retrieve details for.
    :type request_id: int
    :return: JSON response containing the detailed LLM request data on success or an error message on failure.
    :rtype: JsonResponse
    :raises Http404: Raised if the LLM request with the provided ID does not exist.
    """
    try:
        llm_request = get_object_or_404(LLMRequest, id=request_id)

        # Prepare detailed response data
        data = {
            'success': True,
            'request': {
                'id': llm_request.id,
                'created_at': llm_request.created_at.isoformat(),
                'model_name': llm_request.get_model_name(),
                'prompt': llm_request.prompt,
                'response': llm_request.response,
                'duration': llm_request.duration,
                'tokens_generated': llm_request.tokens_generated,
                'tokens_prompt': llm_request.tokens_prompt,
                'status': llm_request.status,
                'error_message': llm_request.error_message,
                'temperature': llm_request.temperature,
                'top_p': llm_request.top_p,
                'max_tokens': llm_request.max_tokens,
                'rag_enabled': llm_request.rag_enabled,
                'document_id': str(llm_request.document_id) if llm_request.document_id else None,
                'context_chunks_used': llm_request.context_chunks_used,
                'total_context_tokens': llm_request.total_context_tokens,
                'strategy_used': llm_request.strategy_used,
                'chunks_processed': llm_request.chunks_processed,
                'map_reduce_steps': llm_request.map_reduce_steps,
                'logs': llm_request.logs or 'No detailed logs available for this request.',
            }
        }

        return JsonResponse(data)

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@require_http_methods(["GET"])
def export_request_data(request, request_id):
    """
    Exports request data in JSON format for the specified request ID. The export
    includes detailed request information and associated metadata as a downloadable
    file response. The endpoint is accessible via an HTTP GET request.

    :param request: HttpRequest object representing the HTTP request made by the
        client
    :type request: HttpRequest
    :param request_id: ID of the LLMRequest object to be exported
    :type request_id: int
    :return: An HttpResponse containing the exported data as a downloadable JSON
        file, or a JsonResponse with an error message in case of a failure
    :rtype: HttpResponse or JsonResponse
    """
    try:
        llm_request = get_object_or_404(LLMRequest, id=request_id)

        # Prepare export data
        export_data = {
            'export_info': {
                'exported_at': timezone.now().isoformat(),
                'exported_by': 'LLM Dashboard',
                'request_id': llm_request.id,
            },
            'request_data': {
                'id': llm_request.id,
                'created_at': llm_request.created_at.isoformat(),
                'model_name': llm_request.get_model_name(),
                'model_id': llm_request.model.id if llm_request.model else None,
                'prompt': llm_request.prompt,
                'response': llm_request.response,
                'duration': llm_request.duration,
                'tokens_generated': llm_request.tokens_generated,
                'tokens_prompt': llm_request.tokens_prompt,
                'status': llm_request.status,
                'error_message': llm_request.error_message,
                'temperature': llm_request.temperature,
                'top_p': llm_request.top_p,
                'max_tokens': llm_request.max_tokens,
                'rag_enabled': llm_request.rag_enabled,
                'document_id': str(llm_request.document_id) if llm_request.document_id else None,
                'context_chunks_used': llm_request.context_chunks_used,
                'total_context_tokens': llm_request.total_context_tokens,
                'strategy_used': llm_request.strategy_used,
                'chunks_processed': llm_request.chunks_processed,
                'map_reduce_steps': llm_request.map_reduce_steps,
                'logs': llm_request.logs,
            }
        }

        # Create JSON response
        response = HttpResponse(
            json.dumps(export_data, indent=2),
            content_type='application/json'
        )
        response[
            'Content-Disposition'] = f'attachment; filename="request_{request_id}_{timezone.now().strftime("%Y%m%d_%H%M%S")}.json"'

        return response

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def generate_streaming_text(request):
    """
    Handles the generation of streaming text responses using the given prompt and model configuration.
    Allows parameterized text generation through a POST request, providing real-time tokens via
    Server-Sent Events (SSE).

    :param request: HTTP POST request containing JSON data with generation parameters such as prompt,
                    max_tokens, temperature, and top_p. The expected format for the payload is:
                    {
                        "prompt": str,
                        "max_tokens": int (optional, default=100),
                        "temperature": float (optional, default=0.7),
                        "top_p": float (optional, default=0.9)
                    }

    :return: A `StreamingHttpResponse` object that streams generated text data in the form of
             Server-Sent Events (SSE). Each streamed event provides token-level information including:
             - "token": the generated token as a string
             - "is_final": a boolean indicating whether the token ends the generation
             - "total_tokens": the total number of tokens generated so far
             - "model_name": the name of the active model used for generation
             - "model_id": the unique identifier of the active model
    """
    try:
        # Parse request data
        data = json.loads(request.body)
        prompt = data.get('prompt')
        max_tokens = data.get('max_tokens', 100)
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 0.9)

        # Validate parameters
        if not prompt:
            return JsonResponse({
                'error': 'Missing prompt'
            }, status=400)

        # Get active model automatically
        model = vllm_service.get_active_model()
        if not model:
            return JsonResponse({
                'error': 'No active model found. Please load a model first.'
            }, status=404)

        # Create streaming response
        def stream_generator():
            try:
                # Run async streaming in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                async def async_stream():
                    async for token_data in vllm_service.generate_streaming_text(
                            model=model,
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p
                    ):
                        # Format as Server-Sent Events
                        event_data = json.dumps({
                            'token': token_data.get('token', ''),
                            'is_final': token_data.get('is_final', False),
                            'total_tokens': token_data.get('total_tokens', 0),
                            'model_name': model.name,
                            'model_id': model.id
                        })
                        yield f"data: {event_data}\n\n"

                        if token_data.get('is_final'):
                            break

                # Run the async generator
                async_gen = async_stream()
                try:
                    while True:
                        chunk = loop.run_until_complete(async_gen.__anext__())
                        yield chunk
                except StopAsyncIteration:
                    pass
                finally:
                    loop.close()

            except Exception as e:
                print(f"Streaming error: {e}")
                error_data = json.dumps({'error': str(e)})
                yield f"data: {error_data}\n\n"

        # Return streaming response
        response = StreamingHttpResponse(
            stream_generator(),
            content_type='text/event-stream'
        )
        response['Cache-Control'] = 'no-cache'
        response['Connection'] = 'keep-alive'
        response['Access-Control-Allow-Origin'] = '*'

        return response

    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'Invalid JSON format'
        }, status=400)
    except Exception as e:
        print(f"Error in streaming endpoint: {e}")
        return JsonResponse({
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def get_active_model_status(request):
    """
    Handles the request to retrieve the status of the currently active model.
    This function checks if there is an active model being managed by the VLLM
    service and, if available, returns its details as a JSON response. If no
    active model is found, it returns a response indicating such.

    :param request: Django HttpRequest object.
    :type request: HttpRequest
    :return: JSON response with the details of the active model or an error
        message.
    :rtype: JsonResponse
    :raises Exception: Raises an exception if there is an error in retrieving
        the active model status.
    """
    try:
        active_model = vllm_service.get_active_model()

        if not active_model:
            return JsonResponse({
                'active_model': None,
                'message': 'No active model found'
            })

        return JsonResponse({
            'active_model': {
                'id': active_model.id,
                'name': active_model.name,
                'model_path': active_model.model_path,
                'status': active_model.status,
                'memory_usage': active_model.memory_usage,
                'max_tokens': active_model.max_tokens,
                'loaded_at': active_model.loaded_at.isoformat() if active_model.loaded_at else None
            },
            'server_running': vllm_service.is_server_running(active_model)
        })

    except Exception as e:
        print(f"Error getting active model status: {e}")
        return JsonResponse({
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def generate_with_active_model_rag(request):
    """
    Handles POST request to generate text using the active model with smart
    retrieval-augmented generation (RAG). Ensures an active model is set, validates
    the existence of the specified document, and returns generated text based on
    the provided prompt and parameters. Adds the active model's metadata in the
    response.

    :param request: HTTP request object containing a JSON body with required
        parameters:
        - document_id (str): ID of the document to base text generation.
        - prompt (str, optional): Text prompt for generation. Defaults to an
          empty string.
        - max_tokens (int, optional): Maximum number of tokens to generate.
          Defaults to 100.
        - temperature (float, optional): Sampling temperature for randomness,
          typically in the range [0.0, 1.0]. Defaults to 0.7.
        - top_p (float, optional): Nucleus sampling parameter to adjust token
          selection. Defaults to 0.9.
        - strategy (str, optional): Custom strategy for text generation.
          Providing this is optional.

    :return: JsonResponse
        - If successful: JSON object containing text generation result and
          metadata on the active model.
        - On errors: JSON object detailing the error with relevant HTTP
          status codes.
    """
    try:
        # Get the currently active model
        active_model = vllm_service.get_active_model()

        if not active_model:
            return JsonResponse({
                'error': 'No active model found. Please load a model first.'
            }, status=400)

        data = json.loads(request.body)

        document_id = data.get('document_id')
        if not document_id:
            return JsonResponse({'error': 'document_id is required'}, status=400)

        # Verify document exists and belongs to the active model
        try:
            document = Document.objects.get(id=document_id, model=active_model)
        except Document.DoesNotExist:
            return JsonResponse({
                'error': f'Document with ID {document_id} not found for the active model'
            }, status=404)

        result = vllm_service.generate_text_with_smart_rag(
            model=active_model,
            prompt=data.get('prompt', ''),
            document_id=document_id,
            max_tokens=data.get('max_tokens', 100),
            temperature=data.get('temperature', 0.7),
            top_p=data.get('top_p', 0.9),
            strategy=data.get('strategy')
        )

        # Add active model info to the response
        result['active_model'] = {
            'id': active_model.id,
            'name': active_model.name,
            'status': active_model.status
        }

        return JsonResponse(result)

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def generate_with_active_model(request):
    """
    Handles the generation of text using the currently active model.

    This endpoint processes a POST request containing the input prompt and related
    generation parameters, utilizes the active model to generate text, and
    returns the result in JSON format. If no active model is found or there is an
    error decoding the input, appropriate error responses are returned.

    :param request: Django HttpRequest object containing details of the HTTP
        request, including the body with prompt and generation parameters.
    :type request: HttpRequest
    :return: A JSON response containing the generated text and additional
        information about the active model, or an error message if an issue
        occurs during the process.
    :rtype: JsonResponse
    :raises JSONDecodeError: Indicated when the JSON provided in the request
        body is invalid or malformatted.
    :raises Exception: A generic exception capturing any unforeseen errors
        during generation or processing.

    """
    try:
        # Get the currently active model
        active_model = vllm_service.get_active_model()

        if not active_model:
            return JsonResponse({
                'error': 'No active model found. Please load a model first.'
            }, status=400)

        data = json.loads(request.body)

        result = vllm_service.generate_text(
            model=active_model,
            prompt=data.get('prompt', ''),
            max_tokens=data.get('max_tokens', 100),
            temperature=data.get('temperature', 0.7),
            top_p=data.get('top_p', 0.9),
            stop=data.get('stop', [])
        )

        # Add active model info to the response
        result['active_model'] = {
            'id': active_model.id,
            'name': active_model.name,
            'status': active_model.status
        }

        return JsonResponse(result)

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def get_active_model_documents(request):
    """
    Handles the retrieval of documents associated with the currently active model.

    This function processes HTTP GET requests to retrieve metadata for all documents
    associated with the active model in the system. If no active model is found,
    a 400 HTTP response is returned with an error message. If an error occurs during
    processing, a 500 HTTP response is returned.

    :param request: The HTTP request object.
    :type request: HttpRequest
    :return: A JSON response containing a list of documents and information about
        the active model, or an error message on failure.
    :rtype: JsonResponse
    """
    try:
        # Get the currently active model
        active_model = vllm_service.get_active_model()

        if not active_model:
            return JsonResponse({
                'error': 'No active model found. Please load a model first.'
            }, status=400)

        documents = Document.objects.filter(model=active_model).values(
            'id', 'title', 'filename', 'file_size', 'is_indexed', 'created_at'
        )

        return JsonResponse({
            'documents': list(documents),
            'active_model': {
                'id': active_model.id,
                'name': active_model.name,
                'status': active_model.status
            }
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def search_active_model_document_chunks(request):
    """
    Handles the search for document chunks associated with the active model based on
    a provided query. The function verifies the active model's existence, the specified
    document's association with it, and performs the search.

    :param request: The HTTP request object containing GET parameters:
                    - `document_id` (str): The ID of the document to search.
                    - `query` (str): The search query.
                    - `k` (int): Optional. The maximum number of results to return.
                      Default is 5.
    :return: A JsonResponse containing:
             - `chunks` (list): The search result chunks.
             - `document_title` (str): The title of the document.
             - `active_model` (dict): Information about the active model:
               - `id` (int): The active model's ID.
               - `name` (str): The active model's name.
               - `status` (str): The current status of the active model.
    """
    try:
        # Get the currently active model
        active_model = vllm_service.get_active_model()

        if not active_model:
            return JsonResponse({
                'error': 'No active model found. Please load a model first.'
            }, status=400)

        document_id = request.GET.get('document_id')
        query = request.GET.get('query', '')
        k = int(request.GET.get('k', 5))

        if not document_id:
            return JsonResponse({'error': 'document_id parameter is required'}, status=400)

        if not query:
            return JsonResponse({'error': 'query parameter is required'}, status=400)

        # Verify document exists and belongs to the active model
        try:
            document = Document.objects.get(id=document_id, model=active_model)
        except Document.DoesNotExist:
            return JsonResponse({
                'error': f'Document with ID {document_id} not found for the active model'
            }, status=404)

        # Search chunks
        vector_service = DocumentVectorStoreService(active_model)
        chunks = vector_service.search_document_chunks(document_id, query, k)

        return JsonResponse({
            'chunks': chunks,
            'document_title': document.title,
            'active_model': {
                'id': active_model.id,
                'name': active_model.name,
                'status': active_model.status
            }
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def add_document_from_file_path(request, model_id):
    """
    Handles the addition of a document from a specified file path, associates it
    with a given model, and processes it for indexing in the vector store. The
    function ensures that the file is accessible, readable, and properly processed
    while also managing document creation and storage in the vector framework.

    :param request: HTTP request object containing the request data.
    :type request: HttpRequest
    :param model_id: Primary key of the LLMModel to which the document is associated.
    :type model_id: int
    :return: JSON response indicating the success or failure of the operation,
             including details of the document and the result of the vector
             store indexing.
    :rtype: JsonResponse
    :raises JsonDecodeError: If the request body contains invalid JSON.
    :raises JsonResponse: For different error conditions including missing
                          file_path, inaccessible or unreadable file, and other
                          processing exceptions.
    """
    try:
        model = get_object_or_404(LLMModel, id=model_id)
        data = json.loads(request.body)

        file_path = data.get('file_path', '')
        if not file_path:
            return JsonResponse({'error': 'file_path is required'}, status=400)

        # Check if file exists
        if not os.path.exists(file_path):
            return JsonResponse({'error': f'File not found: {file_path}'}, status=404)

        # Check if file is readable
        if not os.access(file_path, os.R_OK):
            return JsonResponse({'error': f'File not readable: {file_path}'}, status=403)

        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
            except Exception as e:
                return JsonResponse({'error': f'Error reading file: {str(e)}'}, status=400)
        except Exception as e:
            return JsonResponse({'error': f'Error reading file: {str(e)}'}, status=400)

        # Get file info
        file_size = os.path.getsize(file_path)
        filename = os.path.basename(file_path)
        title = data.get('title', filename)

        # Create document
        document = Document.objects.create(
            model=model,
            title=title,
            filename=filename,
            content=content,
            file_size=file_size,
            metadata=data.get('metadata', {'source_path': file_path})
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
                'title': document.title,
                'filename': document.filename,
                'file_size': document.file_size,
                'chunks_created': DocumentChunk.objects.filter(document=document).count(),
                'message': 'Document added and indexed successfully'
            })
        else:
            # Clean up document if vector store failed
            document.delete()
            return JsonResponse({'error': 'Failed to add document to vector store'}, status=500)

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def add_document_from_file_path_active_model(request):
    """
    Adds a document to the active model from a specified file path. This function retrieves
    the currently active model, reads the content of the provided file path, and stores
    it as a document in the database. If successful, the document is indexed in the vector
    store associated with the active model. The process ensures that the provided file
    exists, is readable, and can be processed, reporting appropriate errors otherwise.

    :param request: The HTTP request object, expected to contain a JSON body with file
        details such as `file_path`, and optionally `title`, `metadata`, `chunk_tokens`,
        and `overlap_tokens`.
    :type request: HttpRequest
    :return: A JSON response indicating whether the operation was successful or failure
        details in case of an error.
    :rtype: JsonResponse
    """
    try:
        # Get the currently active model
        active_model = vllm_service.get_active_model()

        if not active_model:
            return JsonResponse({
                'error': 'No active model found. Please load a model first.'
            }, status=400)

        data = json.loads(request.body)

        file_path = data.get('file_path', '')
        if not file_path:
            return JsonResponse({'error': 'file_path is required'}, status=400)

        # Check if file exists
        if not os.path.exists(file_path):
            return JsonResponse({'error': f'File not found: {file_path}'}, status=404)

        # Check if file is readable
        if not os.access(file_path, os.R_OK):
            return JsonResponse({'error': f'File not readable: {file_path}'}, status=403)

        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
            except Exception as e:
                return JsonResponse({'error': f'Error reading file: {str(e)}'}, status=400)
        except Exception as e:
            return JsonResponse({'error': f'Error reading file: {str(e)}'}, status=400)

        # Get file info
        file_size = os.path.getsize(file_path)
        filename = os.path.basename(file_path)
        title = data.get('title', filename)

        # Create document
        document = Document.objects.create(
            model=active_model,
            title=title,
            filename=filename,
            content=content,
            file_size=file_size,
            metadata=data.get('metadata', {'source_path': file_path})
        )

        # Add to vector store
        vector_service = TokenAwareVectorService(active_model)
        success = vector_service.add_document_to_vector_store(
            document=document,
            target_chunk_tokens=data.get('chunk_tokens', 300),
            overlap_tokens=data.get('overlap_tokens', 50)
        )

        if success:
            return JsonResponse({
                'success': True,
                'document_id': str(document.id),
                'title': document.title,
                'filename': document.filename,
                'file_size': document.file_size,
                'chunks_created': DocumentChunk.objects.filter(document=document).count(),
                'active_model': {
                    'id': active_model.id,
                    'name': active_model.name,
                    'status': active_model.status
                },
                'message': 'Document added and indexed successfully'
            })
        else:
            # Clean up document if vector store failed
            document.delete()
            return JsonResponse({'error': 'Failed to add document to vector store'}, status=500)

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON in request body'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def add_global_document_from_file_path(request):
    """
    Adds a global document from a specified file path and processes it for the global
    vector store. The file content is read, metadata and document details are recorded,
    and chunks are created and added to the vector store.

    :param request:
        The HTTP POST request object containing the JSON payload to specify the
        file path, document title (optional), and other optional parameters like
        metadata, target chunk tokens, and overlap tokens.

    :return:
        JsonResponse indicating the success or failure of the operation. On success,
        returns a response containing details such as document id, title, filename,
        file size, number of chunks created, and a success message. On failure,
        returns an appropriate error message with an HTTP status code.
    """
    try:
        data = json.loads(request.body)

        file_path = data.get('file_path', '')
        if not file_path:
            return JsonResponse({'error': 'file_path is required'}, status=400)

        # Check if file exists
        if not os.path.exists(file_path):
            return JsonResponse({'error': f'File not found: {file_path}'}, status=404)

        # Read file content (same logic as before)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
            except Exception as e:
                return JsonResponse({'error': f'Error reading file: {str(e)}'}, status=400)

        # Get file info
        file_size = os.path.getsize(file_path)
        filename = os.path.basename(file_path)
        title = data.get('title', filename)

        # Create global document (no model association)
        document = Document.objects.create(
            model=None,  # Global document
            title=title,
            filename=filename,
            content=content,
            file_size=file_size,
            metadata=data.get('metadata', {'source_path': file_path})
        )

        # Add to global vector store
        vector_service = TokenAwareVectorService(model=None)  # No model = global
        success = vector_service.add_document_to_vector_store(
            document=document,
            target_chunk_tokens=data.get('chunk_tokens', 300),
            overlap_tokens=data.get('overlap_tokens', 50)
        )

        if success:
            return JsonResponse({
                'success': True,
                'document_id': str(document.id),
                'title': document.title,
                'filename': document.filename,
                'file_size': document.file_size,
                'chunks_created': DocumentChunk.objects.filter(document=document).count(),
                'is_global': True,
                'message': 'Global document added successfully'
            })
        else:
            document.delete()
            return JsonResponse({'error': 'Failed to add document to vector store'}, status=500)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def list_global_documents(request):
    """
    Handles the retrieval of globally available documents from the database where the `model`
    field is null. This endpoint supports only the GET HTTP method. Documents are ordered
    by their creation date in descending order and returned as a list with metadata.

    :param request: The HTTP request object.
    :type request: HttpRequest

    :return: A JSON response containing a list of global documents with their metadata
        and the total count, or an error message in case of failure.
    :rtype: JsonResponse
    """
    try:
        documents = Document.objects.filter(model__isnull=True).order_by('-created_at')

        documents_data = []
        for doc in documents:
            documents_data.append({
                'id': str(doc.id),
                'title': doc.title,
                'filename': doc.filename,
                'file_size': doc.file_size,
                'created_at': doc.created_at.isoformat(),
                'is_indexed': doc.is_indexed,
                'chunks_count': doc.chunks.count(),
                'is_global': True
            })

        return JsonResponse({
            'documents': documents_data,
            'count': len(documents_data)
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
