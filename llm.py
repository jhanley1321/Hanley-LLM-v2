from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from typing import Optional


class LLM:
    """Lowest layer; sends prompts to the model and returns text replies."""

    def __init__(self, model: str = "llama2", temperature: float = 0.2) -> None:
        self.model_name = model
        self.temperature = temperature
        self.client: Optional[ChatOllama] = None
        self._init_model()

    def _init_model(self) -> None:
        self.client = ChatOllama(model=self.model_name, temperature=self.temperature)

    def send(self, message: str) -> str:
        """Send a message directly to the model (no RAG context)."""
        if not self.client:
            raise RuntimeError("LLM client not initialized.")
        response = self.client.invoke([HumanMessage(content=message)])
        return response.content or ""

    def send_rag(self, message: str, context: str) -> str:
        """
        Send a message to the model with retrieved context prepended.
        Caller provides the context string.
        """
        if not self.client:
            raise RuntimeError("LLM client not initialized.")

        # Compose grounded prompt
        composed = (
            "You must answer using only the provided context. "
            "If the answer is not in the context, say: Not in context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {message}"
        )

        response = self.client.invoke([HumanMessage(content=composed)])
        return response.content or ""