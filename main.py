from rag.vector_store import LangChainVectorStore
from rag.rag import RAGPipeline
from llm import LLM
from rag.rag_chat_handler import RAGChatHandler
from ui import ChatUI







def main():

    vs = LangChainVectorStore(collection_name="restaurant_reviews")
    llm = LLM(model="llama2")
    rag = RAGPipeline(vector_store=vs, csv_path="data/realistic_restaurant_reviews.csv")

    # Build index (reuse or rebuild depending on flag)
    rag.build_index(force_rebuild=False)

    # Wrap LLM + RAG into chat handler
    rag_chat = RAGChatHandler(rag_pipeline=rag, llm=llm)
    # Run Streamlit UI
    ChatUI(chat_handler=rag_chat).run()


if __name__ == "__main__":
    # Normal run = reuse index if present
    
    main()

    # For rebuild, change to:
    # main(force_rebuild=True)