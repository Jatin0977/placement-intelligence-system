from typing import List, Optional
from langchain.schema import Document
from langchain_community.vectorstores import Qdrant
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from placement_intel.config import DEFAULT_COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL


class PlacementVectorStore:
    """Manages local in-memory Qdrant Vector DB for document embedding and retrieval."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL
    ):
        self.collection_name = collection_name
        
        # Initialize Local Ollama Embeddings (nomic-embed-text)
        self.embeddings = OllamaEmbeddings(
            model=embedding_model
        )
        
        # Initialize Local In-Memory Qdrant Client (Zero Cloud Setup Needed)
        self.client = QdrantClient(location=":memory:")
        self.vector_store: Optional[Qdrant] = None
        self.indexed_files: List[str] = []

    def index_documents(self, documents: List[Document]) -> int:
        """Indexes a list of Document objects into Qdrant vector database."""
        if not documents:
            return 0

        # Track unique file names indexed
        for doc in documents:
            source_name = doc.metadata.get("source_name", "Unknown File")
            if source_name not in self.indexed_files:
                self.indexed_files.append(source_name)

        if self.vector_store is None:
            # First batch of documents: initialize Qdrant vector store wrapper
            self.vector_store = Qdrant.from_documents(
                documents=documents,
                embedding=self.embeddings,
                location=":memory:",
                collection_name=self.collection_name
            )
        else:
            # Append additional documents
            self.vector_store.add_documents(documents)

        return len(documents)

    def search(self, query: str, k: int = 4) -> List[Document]:
        """Performs similarity search against the indexed placement chunks."""
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search(query, k=k)

    def get_indexed_files(self) -> List[str]:
        """Returns list of currently indexed document file names."""
        return self.indexed_files
