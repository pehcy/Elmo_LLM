import torch
import pandas as pd
import re
import unicodedata
from datasets import load_dataset
import json

CORPUS_NAME = "alpaca"

PAD_token = 0
UNK_token = 1
SOS_token = 2
EOS_token = 3

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_str(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def read_vocabs(datatable: pd.DataFrame, corpus_name="alpaca"):
    output_col = datatable['output'].to_list()
    instruction_col = datatable['instruction'].to_list()
    pairs = [[normalize_str(x), normalize_str(y)] for x, y in zip(instruction_col, output_col)]
    voc = Vocab(corpus_name)
    return voc, pairs

__MAX_LEN_KV_PAIR = 50
__MIN_COUNT_WORD = 3

def is_under_maxlen(p):
    return len(p[0].split(' ')) < __MAX_LEN_KV_PAIR and len(p[1].split(' ')) < __MAX_LEN_KV_PAIR

def filter_pairs(pairs):
    return [pair for pair in pairs if is_under_maxlen(pair)]

def trim_rare_words(voc, pairs, min_count = __MIN_COUNT_WORD):
    # trim words used under the minimum count
    voc.trim(min_count)
    # filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True

        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break
        
        if keep_input and keep_output:
            keep_pairs.append(pair)
    
    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

class Vocab:
    def __init__(self, name) -> None:
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            PAD_token: "<pad>",
            UNK_token: "<unk>",
            SOS_token: "<sos>",
            EOS_token: "<eos>"
        }
        self.num_words = 4
    
    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1
    
    def trim(self, min_count):
        if self.trimmed:
            return
        
        self.trimmed = True
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
        
        # reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            PAD_token: "<pad>",
            UNK_token: "<unk>",
            SOS_token: "<sos>",
            EOS_token: "<eos>"
        }
        self.num_words = 4

        for word in keep_words:
            self.add_word(word)


def load_prepare_data(datatable: pd.DataFrame, save_dir):
    print("Start prepraing data...")
    voc, pairs = read_vocabs(datatable, "alpaca")
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filter_pairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.add_sentence(pair[0])
        voc.add_sentence(pair[1])
    print("Counted words:", voc.num_words)

    with open('word_map.json', 'w') as p:
        json.dump(voc.word2index, p)

    return voc, pairs

def indexes_from_sentence(voc, words):
    word_map = voc.word2index
    enc_c = [word_map.get(word, UNK_token) for word in words]
    return enc_c

max_len = 50

def encode_question(voc, sentence):
    words = sentence.split(' ')[:max_len]
    enc_c = indexes_from_sentence(voc, words) + [PAD_token] * (max_len - len(words))
    return enc_c

def encode_answer(voc, sentence):
    words = sentence.split(' ')[:max_len]
    enc_c = [SOS_token] + indexes_from_sentence(voc, words) + [EOS_token] + [PAD_token] * (max_len - len(words))
    return enc_c

if __name__ == "__main__":
    alpaca_dataset = load_dataset('tatsu-lab/alpaca', split='train')
    data = alpaca_dataset.to_pandas()
    voc, pairs = load_prepare_data(data, "")
    pairs = trim_rare_words(voc, pairs, 3)

    pairs_encoded = []
    for pair in pairs:
        qus = encode_question(voc, pair[0])
        ans = encode_answer(voc, pair[1])
        pairs_encoded.append([qus, ans])

    with open('pairs_encoded.json', 'w') as p:
        json.dump(pairs_encoded, p)