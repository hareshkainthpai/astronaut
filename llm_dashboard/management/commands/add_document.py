from django.core.management.base import BaseCommand
from llm_dashboard.models import LLMModel, Document
import uuid

from llm_dashboard.services.vector_service.token_aware_vector import TokenAwareVectorService


class Command(BaseCommand):
    """
    Command to add a document to the vector store.

    This command allows users to upload a document to a vector store by specifying details such as
    the associated model, document title, content file, and tokenization parameters. It also
    supports optional configuration for a custom document ID. The command handles file reading,
    document metadata creation, and communication with the vector store service for proper indexing.

    :ivar help: Short description of what the command does, shown in the help text for the command.
    :type help: str
    """
    help = 'Add a document to the vector store'

    def add_arguments(self, parser):
        parser.add_argument('--model-id', type=int, required=True, help='LLM Model ID')
        parser.add_argument('--title', type=str, required=True, help='Document title')
        parser.add_argument('--content-file', type=str, required=True, help='Path to content file')
        parser.add_argument('--document-id', type=str, help='Custom document ID (UUID)')
        parser.add_argument('--chunk-tokens', type=int, default=300, help='Target tokens per chunk')
        parser.add_argument('--overlap-tokens', type=int, default=50, help='Overlap tokens between chunks')

    def handle(self, *args, **options):
        """
        Handles the addition of a document to the system by creating the
        document, storing its metadata, and pushing the content to a
        vector store for further processing. This method retrieves an
        LLM model instance, reads document content from a specified file,
        and processes the document by using the vector service. The document
        metadata includes the source file path.

        :param args: Positional arguments. Not used directly.
        :type args: tuple
        :param options: Dictionary of options that includes:
            - model_id (str): ID of the LLM Model to retrieve.
            - content_file (str): The file containing document content to process.
            - document_id (optional, str): ID of the document to create. If omitted,
              it will be automatically generated.
            - title (str): Title of the document to create.
            - chunk_tokens (int): Target size in tokens for chunks when adding
              the document to the vector store.
            - overlap_tokens (int): Number of overlapping tokens between chunks.
        :type options: dict
        :return: None
        """
        try:
            model = LLMModel.objects.get(id=options['model_id'])

            # Read content from file
            with open(options['content_file'], 'r', encoding='utf-8') as f:
                content = f.read()

            # Create document
            document_id = options.get('document_id')
            if document_id:
                document_id = uuid.UUID(document_id)

            document = Document.objects.create(
                id=document_id,
                model=model,
                title=options['title'],
                content=content,
                metadata={'source_file': options['content_file']}
            )

            # Add to vector store
            vector_service = TokenAwareVectorService(model)
            success = vector_service.add_document_to_vector_store(
                document=document,
                target_chunk_tokens=options['chunk_tokens'],
                overlap_tokens=options['overlap_tokens']
            )

            if success:
                self.stdout.write(
                    self.style.SUCCESS(
                        f'Successfully added document "{options["title"]}" with ID: {document.id}'
                    )
                )
            else:
                self.stdout.write(
                    self.style.ERROR(f'Failed to add document to vector store')
                )

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error adding document: {str(e)}')
            )