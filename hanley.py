from llm import LLM
from chat.chat import Chat
from chat.chat_history import ChatHistory


class Hanley:
    """
    Orchestrator.
    Builds and wires the Chat, LLM, and ChatHistory layers.
    Does not process messages itself.
    """

    def __init__(self):
        # 1. Build core components
        self.llm = LLM(model="llama2", temperature=0.2)
        self.chat = Chat()
        self.history = ChatHistory(model_name=self.llm.model_name)

        # 2. Wire components together
        self.chat.set_model(self.llm)

        # Allow the chat layer to record turns directly
        self.chat.history = self.history