import bert
import parsing


def main():
    sentence = "The traditional etymology is from the Latin aperire, \"to open,\" in allusion to its being the season when trees and flowers begin to \"open\"."
    newsen = parsing.main(sentence)
    print(newsen)
    newbag = bert.main(newsen)
    print(newbag)


if __name__ == '__main__':
    main()
