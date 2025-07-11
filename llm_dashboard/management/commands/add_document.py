from django.core.management.base import BaseCommand
from llm_dashboard.models import LLMModel, Document
from llm_dashboard.services.token_aware_vector_service import TokenAwareVectorService
import uuid

class Command(BaseCommand):
    help = 'Add a document to the vector store'

    def add_arguments(self, parser):
        parser.add_argument('--model-id', type=int, required=True, help='LLM Model ID')
        parser.add_argument('--title', type=str, required=True, help='Document title')
        parser.add_argument('--content-file', type=str, required=True, help='Path to content file')
        parser.add_argument('--document-id', type=str, help='Custom document ID (UUID)')
        parser.add_argument('--chunk-tokens', type=int, default=300, help='Target tokens per chunk')
        parser.add_argument('--overlap-tokens', type=int, default=50, help='Overlap tokens between chunks')

    def handle(self, *args, **options):
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