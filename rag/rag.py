from pathlib import Path
import csv
from typing import List, Dict, Optional

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


class RAG:
    """
    Retrieval-Augmented Generation (RAG) system for CSV data.
    
    This class provides functionality to:
    - Load and persist a Chroma vector database
    - Ingest CSV files into the vector database
    - Retrieve relevant documents based on semantic similarity
    - Build context strings for LLM prompts
    - Enable/disable RAG functionality globally
    """

    def __init__(self):
        """
        Initialize the RAG system with default configuration.
        
        Sets up:
        - Default persist directory for the Chroma database
        - Default collection name for storing vectors
        - Embedding model (mxbai-embed-large via Ollama)
        - Vectorstore placeholder (initialized when loading/ingesting)
        - RAG enabled flag (off by default)
        """
        self.persist_directory = Path("./chroma_db")
        self.collection_name = "default_collection"
        self.embedding_model = "mxbai-embed-large"
        self.embeddings = OllamaEmbeddings(model=self.embedding_model)
        self.vectorstore: Optional[Chroma] = None
        self.enabled = False

    def load_chroma_db(self, collection_name: Optional[str] = None, persist_directory: Optional[str] = None) -> None:
        """
        Load an existing Chroma vector database from disk.
        
        Args:
            collection_name: Name of the collection to load (uses default if None)
            persist_directory: Path to the database directory (uses default if None)
        
        Note: This connects to an existing database. If the database doesn't exist,
              Chroma will create a new empty one at the specified location.
        """
        coll = collection_name or self.collection_name
        persist_dir = str(Path(persist_directory) if persist_directory else self.persist_directory)
        self.vectorstore = Chroma(
            collection_name=coll,
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
        )

    def ingest_csv(self, csv_path: str) -> None:
        """
        Ingest a single CSV file into the vector database.
        
        Args:
            csv_path: Path to the CSV file to ingest
        
        Process:
        1. Reads the CSV file and extracts all rows
        2. Converts each row into a Document with formatted content
        3. Stores metadata (source file and row index) with each document
        4. Adds all documents to the vector database with embeddings
        5. Automatically loads the database if not already loaded
        
        Document format: Each CSV row becomes "column_name: value" on separate lines
        """
        p = Path(csv_path)
        with open(p, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows: List[Dict[str, str]] = [row for row in reader]

        docs: List[Document] = []
        for i, row in enumerate(rows):
            content_lines = [f"{col}: {row.get(col, '')}" for col in fieldnames]
            content = "\n".join(content_lines).strip()
            if not content:
                continue
            docs.append(Document(page_content=content, metadata={"source": str(p), "row_index": i}))

        if self.vectorstore is None:
            self.load_chroma_db()

        self.vectorstore.add_documents(docs)
        print(f"✅ Ingested {len(docs)} rows from '{p.name}' into collection '{self.collection_name}' at '{self.persist_directory}'")

    def retrieve(self, query: str, top_k: int = 3) -> List[Document]:
        """
        Retrieve the most relevant documents for a given query.
        
        Args:
            query: The search query string
            top_k: Number of most similar documents to return (default: 3)
        
        Returns:
            List of Document objects ranked by semantic similarity to the query
        
        Raises:
            RuntimeError: If the vector database hasn't been loaded yet
        
        Note: Uses cosine similarity between query embedding and document embeddings
        """
        if not self.vectorstore:
            raise RuntimeError("Vector DB not loaded. Call load_chroma_db() or ingest_csv() first.")
        return self.vectorstore.similarity_search(query, k=top_k)

    def build_context(self, question: str, top_k: int = 3) -> str:
        """
        Build a plain text context string from relevant documents.
        
        Args:
            question: The question to find relevant context for
            top_k: Number of documents to retrieve (default: 3)
        
        Returns:
            A formatted string containing the content of the top_k most relevant
            documents, separated by double newlines. This can be directly inserted
            into an LLM prompt to provide context for answering the question.
        """
        docs = self.retrieve(question, top_k=top_k)
        return "\n\n".join(d.page_content for d in docs)
