# rag.py

"""
RAG retrieval layer for the Carer Support application.

Responsibilities:
1. Receive the user's question.
2. Create a query embedding.
3. Run hybrid keyword + vector retrieval in Azure AI Search.
4. Convert retrieved chunks to LangChain Documents.
5. Build a context string for main.py.
"""

from typing import List

from openai import AsyncOpenAI
from azure.identity.aio import DefaultAzureCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
from langchain_core.documents import Document


class RAGRetriever:
    """Retrieve grounded context from Azure AI Search."""

    def __init__(
        self,
        openai_client: AsyncOpenAI,
        search_endpoint: str,
        search_index_name: str,
        embedding_deployment: str,
        embedding_dimensions: int,
        credential: DefaultAzureCredential,
    ):
        self.openai_client = openai_client
        self.embedding_deployment = embedding_deployment
        self.embedding_dimensions = embedding_dimensions

        self.search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=search_index_name,
            credential=credential,
        )

    async def create_query_embedding(
        self,
        query: str,
    ) -> List[float]:
        """
        Convert the user's question to an embedding.

        The dimensions value MUST match ingestion.py and the Azure AI
        Search contentVector field.
        """
        response = await self.openai_client.embeddings.create(
            model=self.embedding_deployment,
            input=query,
            dimensions=self.embedding_dimensions,
        )

        return response.data[0].embedding

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Document]:
        """
        Run hybrid retrieval:
        - search_text=query performs keyword/BM25 retrieval.
        - vector_queries performs semantic vector retrieval.
        """
        query_vector = await self.create_query_embedding(query)

        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="contentVector",
        )

        results = await self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=top_k,
            select=[
                "id",
                "content",
                "source",
                "title",
            ],
        )

        documents: List[Document] = []

        async for result in results:
            content = result.get("content", "")

            if not content:
                continue

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "id": result.get("id"),
                        "source": result.get("source"),
                        "title": result.get("title"),
                        "search_score": result.get("@search.score"),
                    },
                )
            )

        return documents

    async def build_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> str:
        """Retrieve relevant chunks and combine them into LLM context."""
        documents = await self.retrieve(
            query=query,
            top_k=top_k,
        )

        if not documents:
            return ""

        context_parts = []

        for number, document in enumerate(documents, start=1):
            title = document.metadata.get(
                "title"
            ) or "Unknown document"

            source = document.metadata.get(
                "source"
            ) or "Unknown source"

            context_parts.append(
                f"""[Document {number}]

Title:
{title}

Source:
{source}

Content:
{document.page_content}"""
            )

        return "\n\n---\n\n".join(context_parts)

    async def close(self) -> None:
        """Close the asynchronous Azure AI Search client."""
        await self.search_client.close()
