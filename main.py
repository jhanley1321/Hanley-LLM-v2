from llm import LLM
from ui import ChatUI

if __name__ == "__main__":
    # Option 1: Run CLI
    # LLM().run_cli()
    
    # Option 2: Run Streamlit UI with LLM injected
    llm = LLM()
    ChatUI(chat_handler=llm).run()