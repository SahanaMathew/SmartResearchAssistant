"""
Document Processor — PDF loading, chunking, and embedding preparation.

Chunking strategy:
- chunk_size=800 chars (~200 tokens): large enough to hold a complete thought,
  small enough for precise retrieval.
- chunk_overlap=150 chars (18%): preserves context across chunk boundaries so
  sentences split at the edge of a chunk aren't lost.
- RecursiveCharacterTextSplitter: tries paragraph → sentence → word boundaries
  before falling back to character splits, keeping semantic units intact.
"""

import os
import tempfile
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings


MAX_PAGES_PER_FILE = 10


class DocumentProcessor:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
        )

    def process(self, uploaded_file) -> list:
        """
        Load a PDF, validate page count, chunk it, and return annotated docs.
        Raises ValueError if the file exceeds MAX_PAGES_PER_FILE.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded_file.read())
            tmp_path = f.name

        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
        finally:
            os.unlink(tmp_path)

        if len(pages) > MAX_PAGES_PER_FILE:
            raise ValueError(
                f"'{uploaded_file.name}' has {len(pages)} pages. "
                f"Maximum allowed is {MAX_PAGES_PER_FILE}."
            )

        # Annotate each page with human-readable metadata before splitting
        for page in pages:
            page.metadata["source_file"] = uploaded_file.name
            page.metadata["page_number"] = page.metadata.get("page", 0) + 1

        chunks = self.splitter.split_documents(pages)

        # Tag each chunk with a unique ID and character count
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["char_count"] = len(chunk.page_content)
            # Clean up text: collapse excessive whitespace
            chunk.page_content = " ".join(chunk.page_content.split())

        return [c for c in chunks if len(c.page_content.strip()) > 50]

    def get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        return self.embeddings

    def embed_with_retry(self, texts: list, retries: int = 3) -> list:
        """Embed with exponential backoff for rate limit handling."""
        for attempt in range(retries):
            try:
                return self.embeddings.embed_documents(texts)
            except Exception as e:
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt
                time.sleep(wait)
