import json
import asyncio
import uuid
from channels.generic.websocket import AsyncWebsocketConsumer
from .models import LLMModel, LLMRequest
from channels.db import database_sync_to_async
from asgiref.sync import sync_to_async
from .services.vllm_service import vllm_service
import logging

logger = logging.getLogger(__name__)


class ModelLoadingConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        # FIXED: Add proper error handling for model_id extraction
        try:
            model_id_str = self.scope['url_route']['kwargs']['model_id']
            # Ensure model_id is properly converted to int
            self.model_id = int(model_id_str)
        except (KeyError, ValueError, TypeError) as e:
            # Log the error and reject connection
            print(f"WebSocket connection error: Invalid model_id - {e}")
            print(f"URL route kwargs: {self.scope.get('url_route', {}).get('kwargs', {})}")
            await self.close(code=4000)  # Close with custom error code
            return

        self.group_name = f"model_loading_{self.model_id}"

        try:
            # Join the model loading group
            await self.channel_layer.group_add(self.group_name, self.channel_name)
            await self.accept()
        except Exception as e:
            print(f"WebSocket group_add error: {e}")
            await self.close(code=4001)

    async def disconnect(self, close_code):
        # Leave the model loading group with error handling
        try:
            await self.channel_layer.group_discard(self.group_name, self.channel_name)
        except Exception as e:
            print(f"WebSocket group_discard error: {e}")

    async def receive(self, text_data):
        try:
            # Handle incoming messages (e.g., request for current logs)
            data = json.loads(text_data)
            if data.get('type') == 'get_logs':
                logs = await self.get_model_logs()
                await self.send(text_data=json.dumps({
                    'type': 'current_logs',
                    'logs': logs
                }))
        except Exception as e:
            print(f"WebSocket receive error: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Error processing request: {str(e)}'
            }))

    async def loading_update(self, event):
        try:
            # Send loading update to WebSocket
            await self.send(text_data=json.dumps({
                'type': 'loading_update',
                'message': event['message']
            }))
        except Exception as e:
            print(f"WebSocket loading_update error: {e}")

    @database_sync_to_async
    def get_model_logs(self):
        try:
            model = LLMModel.objects.get(id=self.model_id)
            return model.loading_logs or ""
        except LLMModel.DoesNotExist:
            return ""
        except Exception as e:
            print(f"get_model_logs error: {e}")
            return f"Error retrieving logs: {str(e)}"


class LLMStreamConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.request_id = None
        self.active_requests = set()

    async def connect(self):
        try:
            # Generate unique connection ID
            self.connection_id = str(uuid.uuid4())

            # Join the global streaming group
            await self.channel_layer.group_add("llm_stream", self.channel_name)
            await self.accept()

            # Send connection confirmation
            await self.send(text_data=json.dumps({
                'type': 'connection_established',
                'connection_id': self.connection_id
            }))

        except Exception as e:
            logger.error(f"LLMStreamConsumer connect error: {e}")
            await self.close(code=4002)

    async def disconnect(self, close_code):
        try:
            # Cancel any active requests
            for request_id in self.active_requests:
                await self.cancel_request(request_id)

            await self.channel_layer.group_discard("llm_stream", self.channel_name)
        except Exception as e:
            logger.error(f"LLMStreamConsumer disconnect error: {e}")

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            message_type = data.get('type')

            if message_type == 'start_generation':
                await self.handle_start_generation(data)
            elif message_type == 'cancel_generation':
                await self.handle_cancel_generation(data)
            elif message_type == 'ping':
                await self.send(text_data=json.dumps({'type': 'pong'}))
            else:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': f'Unknown message type: {message_type}'
                }))

        except json.JSONDecodeError:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Invalid JSON format'
            }))
        except Exception as e:
            logger.error(f"LLMStreamConsumer receive error: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Error processing request: {str(e)}'
            }))

    async def handle_start_generation(self, data):
        """Handle start generation request - automatically uses active model"""
        try:
            # Extract parameters (no model_id needed)
            prompt = data.get('prompt')
            max_tokens = data.get('max_tokens', 100)
            temperature = data.get('temperature', 0.7)
            top_p = data.get('top_p', 0.9)

            # Generate unique request ID
            request_id = str(uuid.uuid4())
            self.active_requests.add(request_id)

            # Validate parameters
            if not prompt:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'request_id': request_id,
                    'message': 'Missing prompt'
                }))
                return

            # Get the active model automatically
            model = await self.get_active_model()
            if not model:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'request_id': request_id,
                    'message': 'No active model found. Please load a model first.'
                }))
                return

            # Send generation started confirmation
            await self.send(text_data=json.dumps({
                'type': 'generation_started',
                'request_id': request_id,
                'model_id': model.id,
                'model_name': model.name
            }))

            # Start streaming generation in background
            asyncio.create_task(self.stream_generation(
                request_id, model, prompt, max_tokens, temperature, top_p
            ))

        except Exception as e:
            logger.error(f"Error in handle_start_generation: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'request_id': request_id,
                'message': str(e)
            }))

    async def handle_cancel_generation(self, data):
        """Handle cancel generation request"""
        request_id = data.get('request_id')
        if request_id in self.active_requests:
            await self.cancel_request(request_id)
            await self.send(text_data=json.dumps({
                'type': 'generation_cancelled',
                'request_id': request_id
            }))

    async def stream_generation(self, request_id, model, prompt, max_tokens, temperature, top_p):
        """Stream text generation tokens"""
        try:
            # Create request record
            request = await self.create_request_record(
                model, prompt, max_tokens, temperature, top_p
            )

            # Stream tokens using the vLLM service
            async for token_data in self.generate_streaming_tokens(
                    model, prompt, max_tokens, temperature, top_p
            ):
                # Check if request was cancelled
                if request_id not in self.active_requests:
                    break

                # Send token to client
                await self.send(text_data=json.dumps({
                    'type': 'token',
                    'request_id': request_id,
                    'token': token_data.get('token', ''),
                    'is_final': token_data.get('is_final', False),
                    'total_tokens': token_data.get('total_tokens', 0)
                }))

                # Update request record with token
                await self.update_request_with_token(request.id, token_data.get('token', ''))

            # Send completion message
            if request_id in self.active_requests:
                await self.send(text_data=json.dumps({
                    'type': 'generation_complete',
                    'request_id': request_id
                }))

                # Update request status
                await self.mark_request_complete(request.id)

        except Exception as e:
            logger.error(f"Error in stream_generation: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'request_id': request_id,
                'message': str(e)
            }))
        finally:
            # Clean up
            self.active_requests.discard(request_id)

    async def generate_streaming_tokens(self, model, prompt, max_tokens, temperature, top_p):
        """Generate streaming tokens using vLLM service"""
        try:
            # Use the existing vLLM service but with streaming
            response = await sync_to_async(vllm_service.generate_streaming_text)(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )

            # Yield tokens from the streaming response
            async for token_data in response:
                yield token_data

        except Exception as e:
            logger.error(f"Error generating streaming tokens: {e}")
            raise

    async def cancel_request(self, request_id):
        """Cancel an active request"""
        self.active_requests.discard(request_id)

    # Event handler for group messages
    async def llm_response(self, event):
        try:
            await self.send(text_data=json.dumps({
                "type": "llm_response",
                "request_id": event["request_id"],
                "token": event["token"]
            }))
        except Exception as e:
            logger.error(f"LLMStreamConsumer llm_response error: {e}")

    # Database operations
    @database_sync_to_async
    def get_active_model(self):
        """Get the currently active model"""
        try:
            return vllm_service.get_active_model()
        except Exception as e:
            logger.error(f"Error getting active model: {e}")
            return None

    @database_sync_to_async
    def create_request_record(self, model, prompt, max_tokens, temperature, top_p):
        return LLMRequest.objects.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            status='STREAMING'
        )

    @database_sync_to_async
    def update_request_with_token(self, request_id, token):
        try:
            request = LLMRequest.objects.get(id=request_id)
            request.response = (request.response or '') + token
            request.save()
        except LLMRequest.DoesNotExist:
            pass

    @database_sync_to_async
    def mark_request_complete(self, request_id):
        try:
            request = LLMRequest.objects.get(id=request_id)
            request.status = 'COMPLETED'
            request.save()
        except LLMRequest.DoesNotExist:
            pass