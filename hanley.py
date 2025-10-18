from llm import LLM


class Hanley:
    def __init__(self) -> None:
        # Provide a path to enable incremental JSONL logging; omit to disable.
        self.llm = LLM(model="llama2", temperature=0.2, log_path="logs/chat_history.jsonl")