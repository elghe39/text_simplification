import os
from evaluate import load
from statistics import mean


def sari():
    sari = load("sari")
    with open(os.getcwd() + '/dataset/predict.txt', 'r') as predict:
        with open(os.getcwd() + '/dataset/dataset/reference.txt', 'r') as reference:
            with open(os.getcwd() + '/dataset/dataset/input.txt', 'r') as input:
                input = input.readlines()
                reference = reference.readlines()
                predict = predict.readlines()
                size = len(input)
                score = []
                j = 0
                for i in range(size):
                    ref = []
                    temp_score = []
                    while reference[j] != '\n':
                        ref.append(reference[j])
                        j += 1
                    for pre in predict:
                        if pre != '\n':
                            sari_score = sari.compute(sources=[input[i]], predictions=[pre], references=[ref])
                            temp_score.append(sari_score)
                        else:
                            temp_score_1 = []
                            for item in temp_score:
                                temp_score_1.append(item['sari'])
                            score.append(mean(temp_score_1))
                            break
                    j += 1
                with open(os.getcwd() + '/dataset/metric.txt', 'w') as f:
                    f.write('SARI: ' + str(round(mean(score), 2)))


if __name__ == '__main__':
    sari()
