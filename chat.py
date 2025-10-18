from typing import List, Tuple


class Chat:
    """
    Minimal chat handler.
    - Maintains a simple message history as a list of (role, content).
    - Uses Hanley.llm to generate replies.
    """

    def __init__(self) -> None:
        self.history: List[Tuple[str, str]] = []

    def send(self, hanley, message: str) -> str:
        """
        Send a message and get a reply using the LLM loaded in Hanley.
        For now, we pass only the latest user message to keep it minimal.
        """
        self.history.append(("user", message))
        reply = hanley.llm.generate(message)
        self.history.append(("assistant", reply))
        return reply