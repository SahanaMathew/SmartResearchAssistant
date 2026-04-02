"""
Vector Store Manager — Pinecone operations.

Retrieval approach:
- MMR (Maximal Marginal Relevance): fetches fetch_k=20 candidates by similarity,
  then selects k=5 that are both relevant AND diverse from each other.
  lambda_mult=0.7 weights similarity slightly more than diversity.
  This prevents returning 5 nearly-identical chunks from adjacent paragraphs.
"""

import os
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec


EMBEDDING_DIMENSION = 3072  # Google gemini-embedding-001 output dimension


class VectorStoreManager:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "research-assistant")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )
        self._ensure_index()

    def _ensure_index(self):
        existing_names = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing_names:
            self.pc.create_index(
                name=self.index_name,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

    def upsert(self, chunks: list, namespace: str) -> PineconeVectorStore:
        """Embed and upsert chunks into a session-scoped namespace."""
        return PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            index_name=self.index_name,
            namespace=namespace,
        )

    def get_retriever(self, namespace: str, k: int = 5):
        """Return an MMR retriever scoped to this session's namespace."""
        store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace=namespace,
        )
        return store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": 20,
                "lambda_mult": 0.7,
            },
        )

    def delete_namespace(self, namespace: str):
        """Clean up vectors for a session (called on 'Clear Session')."""
        try:
            index = self.pc.Index(self.index_name)
            index.delete(delete_all=True, namespace=namespace)
        except Exception:
            pass
