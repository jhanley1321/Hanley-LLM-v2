from hanley import Hanley 


def main():
    hanley = Hanley()
    print(hanley.llm.send_message("Say a short hello."))
    print(hanley.llm.history)


if __name__ == "__main__":
    main()
