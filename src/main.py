import bert
import parsing
from evaluate import load


def main():
    sentence = "This is known as the geocentric model of the Universe.."
    newsen = parsing.run(sentence)
    res = bert.run(newsen)
    print(res)
    sari = load("sari")
    sources = [sentence]
    predictions = [res[0]]
    references = [
        ["This is known as the geocentric model of the Universe."]]
    sari_score = sari.compute(sources=sources, predictions=predictions, references=references)
    print(sari_score)


if __name__ == '__main__':
    main()
