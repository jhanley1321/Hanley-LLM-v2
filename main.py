from hanley import Hanley

def main():
    hanley = Hanley()          # loads everything
    chat = hanley.chat         # get chat layer

    print(chat.send("Say a short hello."))
    print(chat.send("Give me a fun fact about the ocean."))

if __name__ == "__main__":
    main()