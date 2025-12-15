class CLIInterface:
    """
    Simple CLI driver that interacts through Hanley.
    """

    def __init__(self, hanley):
        self.hanley = hanley

    def start(self):
        orchestrator = self.hanley.chat_orchestrator

        print("\nCLI started. Type 'exit' or 'quit' to end.\n")
        while True:
            try:
                user_text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nEnding chat.\n")
                break

            if not user_text:
                continue
            if user_text.lower() in {"exit", "quit"}:
                print("\nEnding chat.\n")
                break

            reply = orchestrator.process_message(user_text)
            print(f"Assistant: {reply}\n")
