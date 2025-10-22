from pathlib import Path
import csv
from typing import List, Dict, Optional

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


class RAG:
    """
    Minimal RAG:
    - load_chroma_db(): load existing Chroma DB
    - ingest_csv(): ingest ONE CSV file
    - retrieve(): search
    """

    def __init__(self):
        self.persist_directory = Path("./chroma_db")
        self.collection_name = "default_collection"
        self.embedding_model = "mxbai-embed-large"

        self.embeddings = OllamaEmbeddings(model=self.embedding_model)
        self.vectorstore: Optional[Chroma] = None

    def load_chroma_db(self, collection_name: Optional[str] = None, persist_directory: Optional[str] = None) -> None:
        coll = collection_name or self.collection_name
        persist_dir = str(Path(persist_directory) if persist_directory else self.persist_directory)
        self.vectorstore = Chroma(
            collection_name=coll,
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
        )

    def ingest_csv(self, csv_path: str) -> None:
        """
        Ingest ONE CSV file into the vector DB.
        No options. No batching. No extras.
        """
        p = Path(csv_path)
        # Read CSV rows
        with open(p, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows: List[Dict[str, str]] = [row for row in reader]

        # Build documents (row-level)
        docs: List[Document] = []
        for i, row in enumerate(rows):
            # Simple "col: value" per line
            content_lines = [f"{col}: {row.get(col, '')}" for col in fieldnames]
            content = "\n".join(content_lines).strip()
            if not content:
                continue
            docs.append(Document(page_content=content, metadata={"source": str(p), "row_index": i}))

        # Ensure vectorstore is ready
        if self.vectorstore is None:
            self.load_chroma_db()

        # Add all docs in one shot
        self.vectorstore.add_documents(docs)
        print(f"✅ Ingested {len(docs)} rows from '{p.name}' into collection '{self.collection_name}' at '{self.persist_directory}'")

    def retrieve(self, query: str, top_k: int = 3) -> List[Document]:
        if not self.vectorstore:
            raise RuntimeError("Vector DB not loaded. Call load_chroma_db() or ingest_csv() first.")
        return self.vectorstore.similarity_search(query, k=top_k)