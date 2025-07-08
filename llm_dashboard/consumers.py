import json
from channels.generic.websocket import AsyncWebsocketConsumer
from .models import LLMModel
from channels.db import database_sync_to_async


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
    async def connect(self):
        try:
            await self.channel_layer.group_add("llm_stream", self.channel_name)
            await self.accept()
        except Exception as e:
            print(f"LLMStreamConsumer connect error: {e}")
            await self.close(code=4002)

    async def disconnect(self, close_code):
        try:
            await self.channel_layer.group_discard("llm_stream", self.channel_name)
        except Exception as e:
            print(f"LLMStreamConsumer disconnect error: {e}")

    async def receive(self, text_data):
        pass

    async def llm_response(self, event):
        try:
            await self.send(text_data=json.dumps({
                "type": "llm_response",
                "request_id": event["request_id"],
                "token": event["token"]
            }))
        except Exception as e:
            print(f"LLLStreamConsumer llm_response error: {e}")