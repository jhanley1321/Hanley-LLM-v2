from llm import LLM
from chat.chat import Chat

class Hanley:
    """
    Pure orchestrator.
    Builds and wires the Chat and LLM layers—nothing else.
    """

    def __init__(self):
        self.llm = LLM(model="llama2", temperature=0.2)
        self.chat = Chat()
        self.chat.set_model(self.llm)