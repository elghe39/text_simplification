import math
from collections import OrderedDict

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import PorterStemmer
import re

p = PorterStemmer()


# remove non-alphanumeric characters
def removeChar(sentence):
    regex = r'[^a-zA-Z0-9\s]'
    sentence = re.sub(regex, '', sentence)
    return sentence


def handleParagraph(paragraph: str):
    sentences_list = sent_tokenize(paragraph)
    sentences_copy = [word_tokenize(removeChar(sentence)) for sentence in sentences_list]
    words_list = []
    for sentence in sentences_copy:
        new_word = [p.stem(word.lower()) for word in sentence if word not in stopwords.words('english')]
        words_list.append(new_word)
    return words_list, sentences_list


def tfidf(input: list):
    word_dic = dict()
    vector_dic = dict()
    word_set = [set(each) for each in input]
    for sentence in word_set:
        for word in sentence:
            if word not in word_dic:
                word_dic[word] = 1
            else:
                word_dic[word] += 1
    for index, sen in enumerate(word_set):
        sen_vector = dict()
        for word in sen:
            tf = input[index].count(word) / len(sen)
            idf = math.log(len(input) / word_dic[word])
            sen_vector[word] = tf * idf
        vector_dic[index] = sen_vector
    return vector_dic


# calculate Cosine Similarity score
def cosineSimilarityScore(vector_x: dict, vector_y: dict):
    res: float = 0
    product_x: float = 0
    product_y: float = 0
    for word_x in vector_x:
        if word_x in vector_y:
            res += vector_y[word_x] * vector_x[word_x]
    for word_x, score_x in vector_x.items():
        product_x += score_x ** 2
    for word_y, score_y in vector_y.items():
        product_y += score_y ** 2
    return res / (math.sqrt(product_x) * math.sqrt(product_y))


def cosineSimilarity(matrix: dict):
    res = dict()
    for i in matrix:
        vector_i = matrix[i]
        temp = dict()
        for j in matrix:
            if i != j:
                vector_j = matrix[j]
                temp[j] = cosineSimilarityScore(vector_i, vector_j)
        res[i] = temp
    return res


# use TextRank algorithm
def textRank(matrix: dict, damping=0.85, epslone=0.0001):
    print(matrix)
    prob = dict()
    for i in matrix:
        prob[i] = 1 / len(matrix)
    small_enough = False
    while not small_enough:
        new_score = 0
        count = 0
        new_prob = dict()
        for i in matrix.keys():
            sum_outside: float = 0
            for j in matrix[i].keys():
                visited = list()
                sum_inside: float = 0
                for k in matrix[j].keys():
                    if [j, k] not in visited:
                        sum_inside += matrix[j][k]
                        visited.append([j, k])
                if sum_inside != 0:
                    sum_outside += (matrix[i][j] / sum_inside) * prob[j]
                new_score = (1 - damping) + damping * sum_outside
            new_prob[i] = new_score
        for index, score in prob.items():
            if abs(score - new_prob[index]) < epslone:
                count += 1
        if count == len(matrix.keys()):
            small_enough = True
        prob = new_prob
    sort = OrderedDict(reversed(list({k: v for k, v in sorted(prob.items(), key=lambda item: item[1])}.items())))
    items = list(sort.keys())[:int(len(prob) * 0.5)]
    return items


def main(paragraph: str):
    words_list, sentences_list = handleParagraph(paragraph)
    tfidf_vectorizer = tfidf(words_list)
    cosine_value = cosineSimilarity(tfidf_vectorizer)
    sort = sorted(textRank(cosine_value))
    res = [sentences_list[index] for index in sort]
    return res
