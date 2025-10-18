class RAGChatHandler:
    """
    Chat handler that augments user messages with retrieved context 
    before passing to an underlying LLM.
    """

    def __init__(self, rag_pipeline, llm):
        self.rag = rag_pipeline   # RAGPipeline from rag.py
        self.llm = llm            # LLM from llm.py
        self.rag.build_index()    # ensure docs indexed

    def chat_with_context(self, messages):
        """Retrieve docs, augment prompt, and stream via LLM."""
        user_msg = messages[-1]["content"]
        docs = self.rag.query(user_msg, n_results=3)
        context = "\n".join(docs)

        augmented = (
            f"Use the following context to answer:\n{context}\n\n"
            f"Question: {user_msg}"
        )
        rag_messages = messages[:-1] + [{"role": "user", "content": augmented}]

        yield from self.llm.stream_response(rag_messages)