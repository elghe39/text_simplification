import os
import bert


def main():
    with open(os.getcwd() + '/dataset/parse_dataset.txt', 'r') as instream:
        with open(os.getcwd() + '/dataset/predict.txt', 'w') as outstream:
            input = instream.readlines()
            temp = []
            for index, line in enumerate(input):
                if line == '\n':
                    predict = bert.run(temp)
                    temp = []
                    for sen in predict:
                        outstream.write(sen)
                    outstream.write('\n')
                    print('ok')
                elif index == len(input) - 1:
                    temp.append(line)
                    predict = bert.run(temp)
                    for sen in predict:
                        outstream.write(sen)
                    outstream.write('\n')
                    print('ok')
                else:
                    temp.append(line)


if __name__ == '__main__':
    main()
