from langchain_community.chat_models import ChatOllama
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
        if not self.client:
            raise RuntimeError("LLM client not initialized.")
        response = self.client.invoke([HumanMessage(content=message)])
        return response.content or ""