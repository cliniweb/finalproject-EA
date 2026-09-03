"""RAG layer — chunking, embedding, in-memory vector store, retrieval, quality gate."""

from app.rag.chunking import Chunk, chunk_doctor_profile, chunk_faq_text
from app.rag.store import InMemoryVectorStore
from app.rag.retriever import Retriever, RetrievedChunk

__all__ = [
    "Chunk",
    "chunk_doctor_profile",
    "chunk_faq_text",
    "InMemoryVectorStore",
    "Retriever",
    "RetrievedChunk",
]
