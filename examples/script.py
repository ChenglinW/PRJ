import os

def main():
    with open("testing_command.txt", "r") as file:
        content = file.read()

    text_segments = content.split("\n\n")
    for cmd in text_segments:
        os.system(cmd)














if __name__ == '__main__':
    main()

