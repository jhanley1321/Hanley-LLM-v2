import csv
from typing import Dict, List, Optional, Any


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.
    
    Handles:
    - Loading reviews from CSV
    - Indexing them in a vector store
    - Querying relevant documents when prompted
    """

    def __init__(self, vector_store: Any, csv_path: str):
        """
        Initialize the pipeline.

        Args:
            vector_store: Any object implementing add_documents(docs) and query(text, n_results)
            csv_path: Path to CSV file containing reviews
        """
        self.vector_store = vector_store
        self.csv_path = csv_path

    def load_csv_documents(self) -> Dict[str, str]:
        """
        Load reviews from CSV file into a dictionary.

        Returns:
            Dict mapping document IDs -> review text
        """
        docs: Dict[str, str] = {}
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                # Supports flexible CSVs: use "review" or "text" or fallback to row str
                review: Optional[str] = row.get("review") or row.get("text") or str(row)
                if review:
                    docs[f"review_{i}"] = review
        return docs

    def build_index(self, force_rebuild: bool = False) -> None:
        """
        Build (or reuse) vector store index from CSV documents.
        
        Args:
            force_rebuild: If False (default), reuse existing collection if available.
                           If True, re-index and overwrite.
        """
        # Quick check: does the store already contain docs?
        existing: List[str] = self.vector_store.query("ping", n_results=1)
        if existing and not force_rebuild:
            print("🔄 Using existing vector store (skipping rebuild).")
            return

        docs: Dict[str, str] = self.load_csv_documents()
        print(f"📥 Indexing {len(docs)} docs from {self.csv_path} ...")
        self.vector_store.add_documents(docs)
        print("✅ Index build complete!")

    def query(self, question: str, n_results: int = 3) -> List[str]:
        """
        Query the vector store for semantically similar docs.

        Args:
            question: The user’s query string
            n_results: Number of top results to return

        Returns:
            List of retrieved document texts
        """
        return self.vector_store.query(question, n_results)