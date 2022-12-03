import numpy as np
import re
from nltk.corpus import stopwords
from nltk import pos_tag
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import torch
from transformers import BertTokenizer, BertForMaskedLM
from wordfreq import zipf_frequency
import json
import os

stop_words_ = set(stopwords.words('english'))


def process_input(input_text, word2index, sent_max_length):
    input_text = cleaner(input_text)
    clean_text = []
    index_list = []
    input_token = []
    for i, word in enumerate(input_text.split()):
        if word in word2index:
            clean_text.append(word)
            input_token.append(word2index[word])
        else:
            index_list.append(i)
    input_padded = pad_sequences(maxlen=sent_max_length, sequences=[input_token], padding="post", value=0)
    return input_padded, index_list, len(clean_text)


# Clean data
def cleaner(word):
    # word = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', '', word, flags=re.MULTILINE)
    word = re.sub('[\W]', ' ', word)
    word = re.sub('[^a-zA-Z]', ' ', word)
    return word.lower().strip()


def complete_missing_word(pred_binary, index_list, len_list):
    list_cwi_predictions = list(pred_binary[0][:len_list])
    for i in index_list:
        list_cwi_predictions.insert(i, 0)
    return list_cwi_predictions


def get_bert_candidates(input_text, list_cwi_predictions, tokenizer, model):
    numb_predictions_displayed = 10
    list_candidates_bert = []
    for word, pred in zip(input_text.split(), list_cwi_predictions):
        if (pred and (pos_tag([word])[0][1] in ['NNS', 'NN', 'VBP', 'RB', 'VBG', 'VBD'])) or (
                zipf_frequency(word, 'en')) < 3.1:
            replace_word_mask = input_text.replace(word, '[MASK]')
            text = f'[CLS]{replace_word_mask} [SEP] {input_text} [SEP] '
            tokenized_text = tokenizer.tokenize(text)
            masked_index = [i for i, x in enumerate(tokenized_text) if x == '[MASK]'][0]
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [0] * len(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])
            with torch.no_grad():
                outputs = model(tokens_tensor, token_type_ids=segments_tensors)
                predictions = outputs[0][0][masked_index]
            predicted_ids = torch.argsort(predictions, descending=True)[:numb_predictions_displayed]
            predicted_tokens = tokenizer.convert_ids_to_tokens(list(predicted_ids))
            list_candidates_bert.append((word, predicted_tokens))
    return list_candidates_bert


def main(list_texts: list):
    model_name = 'model_CWI.h5'
    path_dir = os.getcwd() + f"/model/{model_name}"
    model_CWI = load_model(path_dir)
    with open(os.getcwd() + '/dataset/word2index.json', 'r') as instream:
        word2index = json.load(instream)
    with open(os.getcwd() + '/dataset/sent_max_length.txt', 'r') as instream:
        sent_max_length = int(instream.readline().strip())

    bert_model = 'bert-large-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    model = BertForMaskedLM.from_pretrained(bert_model)

    newlist = []
    for input_text in list_texts:
        new_text = input_text
        input_padded, index_list, len_list = process_input(input_text, word2index, sent_max_length)
        pred_cwi = model_CWI.predict(input_padded)
        pred_cwi_binary = np.argmax(pred_cwi, axis=2)
        complete_cwi_predictions = complete_missing_word(pred_cwi_binary, index_list, len_list)
        bert_candidates = get_bert_candidates(input_text, complete_cwi_predictions, tokenizer, model)
        for word_to_replace, l_candidates in bert_candidates:
            tuples_word_zipf = []
            for w in l_candidates:
                if w.isalpha():
                    tuples_word_zipf.append((w, zipf_frequency(w, 'en')))
            tuples_word_zipf = sorted(tuples_word_zipf, key=lambda x: x[1], reverse=True)
            if len(tuples_word_zipf) != 0:
                new_text = re.sub(word_to_replace, tuples_word_zipf[0][0], new_text)
        newlist.append(new_text)
    return newlist
