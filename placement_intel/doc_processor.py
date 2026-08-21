import os
import tempfile
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from placement_intel.config import CHUNK_SIZE, CHUNK_OVERLAP


class PlacementDocProcessor:
    """Processes uploaded placement documents (PDF / TXT) into metadata-rich chunks."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_and_split_file(self, file_name: str, file_bytes: bytes) -> List[Document]:
        """Loads raw bytes from a Streamlit file upload, writes to temp file, and splits into Document chunks."""
        file_ext = os.path.splitext(file_name)[1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            if file_ext == ".pdf":
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
            elif file_ext == ".txt":
                loader = TextLoader(tmp_path, encoding="utf-8")
                docs = loader.load()
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")

            # Enhance metadata with original filename
            for doc in docs:
                doc.metadata["source_name"] = file_name
                # Remove temporary path from metadata
                doc.metadata["source"] = file_name

            chunks = self.text_splitter.split_documents(docs)
            return chunks

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def load_from_filepath(self, file_path: str) -> List[Document]:
        """Loads a file directly from a local filepath (e.g. from sample_data/)."""
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()

        if file_ext == ".pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif file_ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        for doc in docs:
            doc.metadata["source_name"] = file_name
            doc.metadata["source"] = file_name

        chunks = self.text_splitter.split_documents(docs)
        return chunks
