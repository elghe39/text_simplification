import os

if __name__ == '__main__':
    with open(os.getcwd() + '/dataset/dataset.txt', 'r') as instream:
        with open(os.getcwd() + '/dataset/input.txt', 'w') as input:
            with open(os.getcwd() + '/dataset/reference.txt', 'w') as ref:
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
