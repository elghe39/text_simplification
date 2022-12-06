import os
import parsing


def split_dataset():
    with open(os.getcwd() + '/dataset/dataset/dataset.txt', 'r') as instream:
        with open(os.getcwd() + '/dataset/dataset/input.txt', 'w') as input:
            with open(os.getcwd() + '/dataset/dataset/reference.txt', 'w') as ref:
                lines = instream.readlines()
                tests = []
                temp = []
                for index, line in enumerate(lines):
                    if line == '\n':
                        tests.append(temp)
                        temp = []
                    elif index == len(lines) - 1:
                        temp.append(line)
                        tests.append(temp)
                    else:
                        temp.append(line)
                for test in tests:
                    i = 0
                    for line in test:
                        if i == 0:
                            input.write(line)
                            i += 1
                        else:
                            ref.write(line)
                    ref.write("\n")


def parse_dataset():
    with open(os.getcwd() + '/dataset/dataset/input.txt', 'r') as instream:
        with open(os.getcwd() + '/dataset/parse_dataset.txt', 'w') as outstream:
            input = instream.readlines()
            i = 1
            for sen in input:
                parse_sens = parsing.run(sen)
                for element in parse_sens:
                    outstream.write(element + '\n')
                outstream.write('\n')
                print(i)
                i += 1


if __name__ == '__main__':
    # split_dataset()
    parse_dataset()
