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
    """
    Handles WebSocket communication for a model loading process. This includes connection
    management, group updates, processing incoming messages, and fetching/loading model
    specific updates or logs.

    This class is designed as an asynchronous WebSocket consumer that interacts with a model
    loading process, facilitating real-time communication and updates. It leverages Django
    Channels and the `AsyncWebsocketConsumer` base class for handling WebSocket events such
    as connection, reception of messages, and disconnection.

    :ivar model_id: Identifier of the model for which loading updates are managed.
    :type model_id: int
    :ivar group_name: Name of the WebSocket group associated with the model loading updates.
    :type group_name: str
    """
    async def connect(self):
        """
        Handles the WebSocket connection process including validation of `model_id`,
        joining a specific group, and properly managing errors and connection states.

        :raises KeyError: if 'model_id' is not found in the provided URL routing kwargs.
        :raises ValueError: if 'model_id' cannot be converted to an integer.
        :raises TypeError: if the provided 'model_id' is not of expected data type.
        :raises Exception: for any exceptions encountered while adding to a group or accepting the connection.

        :return: None
        """
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
        """
        Disconnects the WebSocket client and removes it from the group. This method ensures
        the client is correctly removed from the group even if an error occurs.

        :param close_code: Code indicating the reason for the disconnection
        :type close_code: int
        :return: None
        """
        # Leave the model loading group with error handling
        try:
            await self.channel_layer.group_discard(self.group_name, self.channel_name)
        except Exception as e:
            print(f"WebSocket group_discard error: {e}")

    async def receive(self, text_data):
        """
        Handles the incoming WebSocket text data, processes the incoming message, and sends
        an appropriate response back through the WebSocket.

        :param text_data: The incoming WebSocket message as a JSON-encoded string.
        :type text_data: str
        :return: None
        :rtype: None
        """
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
        """
        Handles sending a loading update message through a WebSocket connection. The method takes an
        event containing the loading update message and sends it using the WebSocket protocol.

        :param event: The event containing the loading update data. It must be a dictionary
            with a 'message' key holding the update message as a string.
        :return: None
        :raises Exception: If an error occurs during the WebSocket communication process.
        """
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
        """
        Asynchronously retrieves loading logs for a specific model instance using the
        model's unique identifier.

        It fetches the model instance identified by the given `model_id`. If the model
        is found, it returns the associated loading logs. If the model does not exist,
        an empty string is returned. If an unexpected error occurs, a corresponding
        error message is generated.

        :raises LLMModel.DoesNotExist: If the model with the specified `model_id`
            does not exist in the database.

        :return: A string containing the loading logs of the model, an empty string if
            the model does not exist, or an error message on exceptions other than
            `LLMModel.DoesNotExist`.
        :rtype: str
        """
        try:
            model = LLMModel.objects.get(id=self.model_id)
            return model.loading_logs or ""
        except LLMModel.DoesNotExist:
            return ""
        except Exception as e:
            print(f"get_model_logs error: {e}")
            return f"Error retrieving logs: {str(e)}"


class LLMStreamConsumer(AsyncWebsocketConsumer):
    """
    Represents a WebSocket consumer that enables live interaction with a large
    language model (LLM) for processing and responding to user requests.

    This class is used to handle WebSocket connections, manage active requests
    associated with a connection, and perform operations like initiating
    text generation, canceling generation, or responding to client pings.
    It allows real-time streaming of generated tokens and provides an interface
    for extended user interactions with generative models.

    :ivar request_id: A unique identifier for the current request. Initializes
        as None.
    :type request_id: str | None
    :ivar active_requests: Tracks active generation request IDs for the current
        connection.
    :type active_requests: set
    """
    def __init__(self, *args, **kwargs):
        """
        Represents an object that tracks active requests and a unique request identifier.

        This class is initialized with any arguments and keyword arguments passed to its
        superclass. It manages a `request_id` to uniquely identify a request and maintains
        a set of `active_requests` to track ongoing operations.

        Attributes:
            request_id:
                A unique identifier for the current request. Initially set to None.
            active_requests:
                A set of currently active requests managed by the instance.

        :param args: Positional arguments passed to the superclass.
        :type args: tuple
        :param kwargs: Keyword arguments passed to the superclass.
        :type kwargs: dict
        """
        super().__init__(*args, **kwargs)
        self.request_id = None
        self.active_requests = set()

    async def connect(self):
        """
        Establishes a WebSocket connection, generating a unique connection ID, adding the
        websocket to a global group, and sending a confirmation message. If any error occurs,
        the connection is closed with a specific error code.

        :return: None
        :rtype: None
        """
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
        """
        Disconnects the current consumer and performs cleanup tasks. This includes
        canceling any active requests and removing the consumer from the associated
        group in the channel layer.

        :param close_code: The close code indicating the reason for disconnection.
        :type close_code: int
        :return: None
        """
        try:
            # Cancel any active requests
            for request_id in self.active_requests:
                await self.cancel_request(request_id)

            await self.channel_layer.group_discard("llm_stream", self.channel_name)
        except Exception as e:
            logger.error(f"LLMStreamConsumer disconnect error: {e}")

    async def receive(self, text_data):
        """
        Handles incoming WebSocket text messages by identifying the message type,
        parsing the data, and orchestrating appropriate responses or actions
        based on the extracted message type. Supports operations such as initiating
        generation, canceling generation, responding to ping messages, and handling errors.

        The function operates asynchronously, ensuring non-blocking execution for
        each incoming message and handling different failure scenarios, like invalid
        JSON format or unhandled exceptions, with appropriate error responses.

        :param text_data: The raw text message received over the WebSocket. Must be
            a JSON-encoded string containing at least a 'type' key.
        :type text_data: str
        :return: None
        """
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
        """
        Handles the start of a text generation task based on the provided parameters. Validates
        the required data, retrieves the active model, and begins streaming the generation process
        asynchronously. If any errors occur, appropriate error messages are sent back to the client.

        :param data: Input data containing configuration for the generation task. Must include a
                     `prompt`. Additional optional parameters are `max_tokens`, `temperature`,
                     and `top_p`.
        :type data: dict
        :return: None
        """
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
        """
        Handles the cancellation of a generation request.

        This asynchronous method cancels an ongoing generation request if the provided
        request ID matches an active request. Once canceled, it sends a confirmation
        response indicating that the generation has been successfully canceled.

        :param data: A dictionary containing the cancellation request data,
                     including the `request_id` of the generation to cancel.
        :type data: dict

        :return: None
        """
        request_id = data.get('request_id')
        if request_id in self.active_requests:
            await self.cancel_request(request_id)
            await self.send(text_data=json.dumps({
                'type': 'generation_cancelled',
                'request_id': request_id
            }))

    async def stream_generation(self, request_id, model, prompt, max_tokens, temperature, top_p):
        """
        Streams data token by token for a text generation request, using
        a language model. The method creates a request record, streams
        tokens from a language model in real time, and sends them back
        to the client. It also handles request completion or cancellation
        and updates the request record with generated token data.

        :param request_id: The unique identifier for the text generation
                           request
        :type request_id: str
        :param model: Name or identifier of the language model
        :type model: str
        :param prompt: The input text prompt for the model to generate
                       a response from
        :type prompt: str
        :param max_tokens: The maximum number of tokens to generate
        :type max_tokens: int
        :param temperature: Sampling temperature to control text variability
                            during generation
        :type temperature: float
        :param top_p: Nucleus sampling top-p value for controlling the
                      probability mass
        :type top_p: float
        :return: None
        """
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
        """
        Generates streaming tokens by utilizing a vLLM service with streaming capabilities.
        This method is asynchronous and yields tokens from the streaming response
        one by one. The method also handles exceptions that may occur during the
        generation process, logging the error before raising it.

        :param model: The model name or identifier to use for text generation.
        :type model: str
        :param prompt: The prompt text to guide the text generation.
        :type prompt: str
        :param max_tokens: The maximum number of tokens to generate.
        :type max_tokens: int
        :param temperature: The sampling temperature to use for token generation.
        :type temperature: float
        :param top_p: The p-value threshold for nucleus sampling.
        :type top_p: float
        :return: A generator yielding token data from the streaming response.
        :rtype: AsyncGenerator[dict, None]
        :raises Exception: If an error occurs during token generation.
        """
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
        """
        Cancels a previously active request.

        This method removes a specific request from the set of active requests.
        It ensures the request is no longer registered as active.

        :param request_id: Unique identifier of the request to be canceled.
        :type request_id: str
        :return: None
        """
        self.active_requests.discard(request_id)

    # Event handler for group messages
    async def llm_response(self, event):
        """
        Handles the response for a language model (LLM) event by sending a message
        to the connected client. The message contains the event type and associated
        information such as request ID and token. In case of an error, logs the
        error details.

        :param event: Contains the details needed to construct the response. Includes
                      fields such as "request_id" and "token".
        :type event: dict
        :return: None
        """
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
        """
        Fetches the currently active model asynchronously by interacting with the vllm_service.
        Logs any errors that occur during the execution and returns None in such cases.

        :return: The active model if successfully retrieved, or None if any exception occurs.
        :rtype: Any or None
        """
        try:
            return vllm_service.get_active_model()
        except Exception as e:
            logger.error(f"Error getting active model: {e}")
            return None

    @database_sync_to_async
    def create_request_record(self, model, prompt, max_tokens, temperature, top_p):
        """
        Creates a record for a request to an LLM (Large Language Model) in the database.

        The method is asynchronous and meant to be used in an async context. It records
        the details of the request, including the model used, the input prompt, token
        limits, and additional parameters to control generation characteristics.

        :param model: The name of the LLM model being used.
        :type model: str
        :param prompt: The input text or query sent to the LLM.
        :type prompt: str
        :param max_tokens: The maximum number of tokens the LLM can generate in response.
        :type max_tokens: int
        :param temperature: A value controlling randomness in the generated output. Higher
            values lead to more randomness.
        :type temperature: float
        :param top_p: A value for nucleus sampling where the model considers the result
            from the smallest set of tokens whose probability is greater than or equal
            to the given value.
        :type top_p: float
        :return: The newly created LLMRequest object representing the recorded request details.
        :rtype: LLMRequest
        """
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
        """
        Updates the request object in the database with the provided token. If the request
        does not exist, the operation will silently fail. Appends the token value to the
        request's existing response. The modification is saved to the database.

        :param request_id: Unique identifier of the request.
        :type request_id: int
        :param token: Token string to append to the request response.
        :type token: str
        :return: None
        """
        try:
            request = LLMRequest.objects.get(id=request_id)
            request.response = (request.response or '') + token
            request.save()
        except LLMRequest.DoesNotExist:
            pass

    @database_sync_to_async
    def mark_request_complete(self, request_id):
        """
        Marks a given request as completed in the database.

        This method updates the status of a request in the database
        to 'COMPLETED' based on the provided request ID. If the
        request does not exist, it silently handles the case
        without raising errors.

        :param request_id: The unique ID of the request to mark as completed
        :type request_id: int
        :return: None
        """
        try:
            request = LLMRequest.objects.get(id=request_id)
            request.status = 'COMPLETED'
            request.save()
        except LLMRequest.DoesNotExist:
            pass