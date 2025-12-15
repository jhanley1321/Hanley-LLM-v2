from llm.llm import LLM
from chat.chat import Chat, ChatOrchestrator
from chat.chat_history import ChatHistory
from rag.rag import RAG
from chat.cli_interface import CLIInterface


class Hanley:
    """
    Pure orchestrator.
    Builds and wires LLM, Chat, ChatHistory, RAG, ChatOrchestrator, and CLI interface.
    """

    def __init__(self):
        self.llm = LLM(model="llama2", temperature=0.2)
        self.chat = Chat()
        self.history = ChatHistory(model_name=self.llm.model_name)
        self.rag = RAG()

        self.chat.history = self.history
        self.llm.set_rag(self.rag)

        self.chat_orchestrator = ChatOrchestrator(self.chat, self.llm)

        self.cli = CLIInterface(self)
