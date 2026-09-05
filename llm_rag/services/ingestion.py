# ingestion.py

"""
Document ingestion pipeline for the Carer Support RAG application.

PIPELINE
--------
Azure Data Lake Storage Gen2
        ↓
Download source documents
        ↓
Extract text
        ↓
LangChain text splitting
        ↓
Generate embeddings
        ↓
Upload chunks to Azure AI Search

IMPORTANT
---------
Run this script when documents are added or changed.
It is NOT called for every user question.
"""

import asyncio
import hashlib
import io
import os
from typing import Iterable, List

from dotenv import load_dotenv
from openai import AsyncOpenAI

from azure.identity.aio import (
    DefaultAzureCredential,
    get_bearer_token_provider,
)
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.storage.filedatalake.aio import DataLakeServiceClient

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader



# HELPER FUNCTIONS


def create_chunk_id(
    file_name: str,
    chunk_number: int,
) -> str:
    """
    Create a deterministic ID for one document chunk.

    The same file path + chunk number always creates the same ID.
    This prevents every ingestion run from creating a new random
    UUID for the same chunk.
    """
    raw_id = (
        f"{file_name}-{chunk_number}"
    )

    return hashlib.sha256(
        raw_id.encode("utf-8")
    ).hexdigest()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract searchable text from a PDF file."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages: List[str] = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)

    return "\n".join(pages)


def extract_document_text(
    file_name: str,
    file_bytes: bytes,
) -> str:
    """
    Extract text from a supported source file.

    Supported in this first version:
    - .pdf
    - .txt
    """
    extension = os.path.splitext(file_name)[1].lower()

    if extension == ".txt":
        return file_bytes.decode("utf-8", errors="ignore")

    if extension == ".pdf":
        return extract_text_from_pdf(file_bytes)

    print(f"Skipping unsupported file: {file_name}")
    return ""


def split_text_into_chunks(text: str) -> List[str]:
    """
    Split a large document into overlapping RAG chunks.

    chunk_size=1000:
        Keeps each unit small enough for focused retrieval.

    chunk_overlap=200:
        Preserves some context when information crosses a chunk boundary.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_text(text)


def chunk_list(items: List[str], batch_size: int) -> Iterable[List[str]]:
    """Yield lists in batches to avoid sending every chunk in one API call."""
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def escape_odata_string(value: str) -> str:
    """
    Escape apostrophes for an Azure AI Search OData string filter.

    Example:
        carer's-guide.pdf
    becomes:
        carer''s-guide.pdf
    """
    return value.replace("'", "''")


# AZURE AI SEARCH INDEX


async def create_search_index(
    index_client: SearchIndexClient,
    index_name: str,
    embedding_dimensions: int,
) -> None:
    """
    Create or update the Azure AI Search index used by rag.py.

    The field names here MUST match the fields queried in rag.py.
    """
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="carer-hnsw",
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="carer-vector-profile",
                algorithm_configuration_name="carer-hnsw",
            )
        ],
    )

    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SearchableField(
            name="title",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SearchableField(
            name="source",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
        ),
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(
                SearchFieldDataType.Single
            ),
            searchable=True,
            vector_search_dimensions=embedding_dimensions,
            vector_search_profile_name="carer-vector-profile",
        ),
    ]

    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
    )

    await index_client.create_or_update_index(index)
    print(f'Azure AI Search index "{index_name}" is ready.')


# EMBEDDINGS


async def create_embeddings(
    openai_client: AsyncOpenAI,
    embedding_deployment: str,
    texts: List[str],
    embedding_dimensions: int,
) -> List[List[float]]:
    """
    Create embeddings for a batch of chunks.

    Using batches is much more efficient than making one network
    request for every individual chunk.
    """
    response = await openai_client.embeddings.create(
        model=embedding_deployment,
        input=texts,
        dimensions=embedding_dimensions,
    )

    # The API returns embeddings in the same order as the input list.
    return [item.embedding for item in response.data]


# ============================================================
# ADLS GEN2
# ============================================================

async def download_file(
    file_system_client,
    file_path: str,
) -> bytes:
    """Download one file from Azure Data Lake Storage Gen2."""
    file_client = file_system_client.get_file_client(file_path)
    download = await file_client.download_file()
    return await download.readall()


# ============================================================
# INDEX MAINTENANCE
# ============================================================

async def delete_existing_chunks_for_source(
    search_client: SearchClient,
    source: str,
) -> None:
    """
    Delete old indexed chunks belonging to a source file before re-indexing it.

    Deterministic IDs prevent duplicate IDs, but deletion also handles the
    case where a document becomes shorter. Without this step, old high-numbered
    chunks from the previous version could remain in the index.
    """
    safe_source = escape_odata_string(source)

    results = await search_client.search(
        search_text="*",
        filter=f"source eq '{safe_source}'",
        select=["id"],
        top=1000,
    )

    documents_to_delete = []

    async for result in results:
        document_id = result.get("id")
        if document_id:
            documents_to_delete.append({"id": document_id})

    if documents_to_delete:
        delete_results = await search_client.delete_documents(
            documents=documents_to_delete
        )

        failed = [
            result for result in delete_results
            if not result.succeeded
        ]

        if failed:
            raise RuntimeError(
                f"Failed to delete {len(failed)} existing chunks "
                f"for source: {source}"
            )


# ============================================================
# PROCESS ONE DOCUMENT
# ============================================================

async def process_document(
    file_name: str,
    file_bytes: bytes,
    openai_client: AsyncOpenAI,
    search_client: SearchClient,
    embedding_deployment: str,
    embedding_dimensions: int,
    embedding_batch_size: int = 32,
) -> None:
    """
    Extract → chunk → embed → index one source document.
    """
    print(f"\nProcessing: {file_name}")

    document_text = extract_document_text(
        file_name=file_name,
        file_bytes=file_bytes,
    )

    if not document_text.strip():
        print(f"No usable text found in {file_name}")
        return

    chunks = split_text_into_chunks(document_text)

    if not chunks:
        print(f"No chunks were created for {file_name}")
        return

    print(f"Created {len(chunks)} chunks.")

    # Remove the previous indexed version of this source first.
    await delete_existing_chunks_for_source(
        search_client=search_client,
        source=file_name,
    )

    search_documents = []
    chunk_offset = 0

    # Generate embeddings in batches.
    for batch in chunk_list(chunks, embedding_batch_size):
        embeddings = await create_embeddings(
            openai_client=openai_client,
            embedding_deployment=embedding_deployment,
            texts=batch,
            embedding_dimensions=embedding_dimensions,
        )

        for local_index, (chunk_text, embedding) in enumerate(
            zip(batch, embeddings),
            start=1,
        ):
            chunk_number = chunk_offset + local_index

            search_documents.append(
                {
                    "id": create_chunk_id(
                        file_name=file_name,
                        chunk_number=chunk_number,
                    ),
                    "title": os.path.basename(file_name),
                    "source": file_name,
                    "content": chunk_text,
                    "contentVector": embedding,
                }
            )

        chunk_offset += len(batch)

    # Upload in smaller batches to avoid oversized indexing requests.
    upload_batch_size = 500
    total_uploaded = 0

    for start in range(0, len(search_documents), upload_batch_size):
        upload_batch = search_documents[
            start:start + upload_batch_size
        ]

        upload_results = await search_client.upload_documents(
            documents=upload_batch
        )

        failed = [
            result for result in upload_results
            if not result.succeeded
        ]

        if failed:
            raise RuntimeError(
                f"Failed to upload {len(failed)} chunks for {file_name}"
            )

        total_uploaded += len(upload_batch)

    print(
        f"Uploaded {total_uploaded}/{len(search_documents)} "
        f"chunks for {file_name}."
    )


# MAIN INGESTION WORKFLOW


async def ingest_documents() -> None:
    """
    Read all supported files from ADLS Gen2 and populate Azure AI Search.
    """
    load_dotenv()

    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    embedding_deployment = os.getenv("EMBEDDING_DEPLOYMENT")
    azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    azure_search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
    storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    file_system_name = os.getenv("AZURE_STORAGE_FILE_SYSTEM")

    try:
        embedding_dimensions = int(
            os.getenv("EMBEDDING_DIMENSIONS", "1536")
        )
    except ValueError as exc:
        raise ValueError(
            "EMBEDDING_DIMENSIONS must be an integer."
        ) from exc

    required_values = {
        "AZURE_OPENAI_ENDPOINT": azure_openai_endpoint,
        "EMBEDDING_DEPLOYMENT": embedding_deployment,
        "AZURE_SEARCH_ENDPOINT": azure_search_endpoint,
        "AZURE_SEARCH_INDEX_NAME": azure_search_index_name,
        "AZURE_STORAGE_ACCOUNT_NAME": storage_account_name,
        "AZURE_STORAGE_FILE_SYSTEM": file_system_name,
    }

    for variable_name, value in required_values.items():
        if not value:
            raise ValueError(
                f"{variable_name} is not configured."
            )

    credential = None
    openai_client = None
    search_client = None
    index_client = None
    datalake_service_client = None

    try:
        # One asynchronous credential is shared by Azure SDK clients.
        credential = DefaultAzureCredential()

        # IMPORTANT:
        # This code uses an Azure OpenAI v1 endpoint, so the Entra token
        # is scoped to Azure Cognitive Services.
        token_provider = get_bearer_token_provider(
            credential,
            "https://cognitiveservices.azure.com/.default",
        )

        openai_client = AsyncOpenAI(
            base_url=azure_openai_endpoint,
            api_key=token_provider,
        )

        index_client = SearchIndexClient(
            endpoint=azure_search_endpoint,
            credential=credential,
        )

        await create_search_index(
            index_client=index_client,
            index_name=azure_search_index_name,
            embedding_dimensions=embedding_dimensions,
        )

        search_client = SearchClient(
            endpoint=azure_search_endpoint,
            index_name=azure_search_index_name,
            credential=credential,
        )

        storage_url = (
            f"https://{storage_account_name}.dfs.core.windows.net"
        )

        datalake_service_client = DataLakeServiceClient(
            account_url=storage_url,
            credential=credential,
        )

        file_system_client = (
            datalake_service_client.get_file_system_client(
                file_system=file_system_name
            )
        )

        print("\nReading documents from Azure Data Lake Storage Gen2...")

        # get_paths() includes both files and directories.
        paths = file_system_client.get_paths(recursive=True)

        async for path in paths:
            if path.is_directory:
                continue

            file_name = path.name

            if not file_name.lower().endswith((".pdf", ".txt")):
                print(f"Skipping unsupported file: {file_name}")
                continue

            file_bytes = await download_file(
                file_system_client=file_system_client,
                file_path=file_name,
            )

            await process_document(
                file_name=file_name,
                file_bytes=file_bytes,
                openai_client=openai_client,
                search_client=search_client,
                embedding_deployment=embedding_deployment,
                embedding_dimensions=embedding_dimensions,
            )

        print("\nDocument ingestion completed successfully.")

    finally:
        if search_client is not None:
            await search_client.close()

        if index_client is not None:
            await index_client.close()

        if datalake_service_client is not None:
            await datalake_service_client.close()

        if openai_client is not None:
            await openai_client.close()

        if credential is not None:
            await credential.close()


if __name__ == "__main__":
    asyncio.run(ingest_documents())
