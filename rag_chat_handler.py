class RAGChatHandler:
    """
    Chat handler that augments user messages with retrieved context 
    before passing to an underlying LLM.
    """

    def __init__(self, rag_pipeline, llm):
        self.rag = rag_pipeline   # RAGPipeline from rag.py
        self.llm = llm            # LLM from llm.py
        self.rag.build_index()    # ensure docs indexed

    def stream_chat(self, messages):
        """
        Augment final user message with retrieved docs 
        and stream the response via LLM.
        """
        user_message = messages[-1]["content"]

        # Ask RAG for relevant docs
        retrieved_docs = self.rag.query(user_message, n_results=3)
        context = "\n".join(retrieved_docs)

        # Create augmented prompt
        augmented_prompt = (
            f"Use the following context to answer:\n{context}\n\n"
            f"Question: {user_message}"
        )

        # Replace last user message with augmented version
        rag_messages = messages[:-1] + [{"role": "user", "content": augmented_prompt}]

        # ✅ Call into existing LLM stream_chat — no new logic here
        yield from self.llm.stream_chat(rag_messages)


    def run_cli(self):
        """
        Interactive CLI loop for RAG-augmented chat.
        """
        print("🤖 RAG-Enhanced Chat (type 'exit' to quit)\n")

        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            # Add user message to history
            self.messages.append({"role": "user", "content": user_input})

            # Stream assistant response
            print("Assistant: ", end="", flush=True)
            full_response = ""
            for chunk in self.stream_chat(self.messages):
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n")

            # Store assistant message in history
            self.messages.append({"role": "assistant", "content": full_response})