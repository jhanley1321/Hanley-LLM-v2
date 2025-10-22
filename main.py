from hanley import Hanley

def main():
    hanley = Hanley()          # loads everything
    # chat = hanley.chat         # get chat layer

    # print(chat.send("Say a short hello."))
    # print(chat.send("Give me a fun fact about the ocean."))
    hanley.cli.start()


def main():
    hanley = Hanley()

    # Load existing DB (optional; ingest will also init if not loaded)
    hanley.rag.load_chroma_db()

    # Ingest exactly one CSV
    hanley.rag.ingest_csv("data/realistic_restaurant_reviews.csv")
   

    # Turn RAG on
    hanley.rag.enabled = True
    
    # Now all messages will use RAG automatically
    # print(hanley.chat.send("Which review mentions gluten-free and what did they say? What was the Title. What date was it, and what rating was it given?" ))
   


    # # Start CLI
    # print("\n💬 CLI Interface ready — type 'exit' or 'quit' to stop.\n")
    hanley.cli.start()

if __name__ == "__main__":
    main()
