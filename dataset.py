import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.utils.data
import json
import random
import itertools

PAD_token = 0
UNK_token = 1
SOS_token = 2
EOS_token = 3

__MIN_COUNT_WORD = 3

class TatsuAlpacaDataset(Dataset):
    def __init__(self, voc) -> None:
        super().__init__()
        self.pairs = json.load(open('pairs_encoding.json'))
        inp_enc, out_enc = batch_to_train_data(voc, )
    
    def __getitem__(self, index):
        return super().__getitem__(index)


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


def indexes_from_sentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def zero_padding(l, fill_value=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fill_value))


def input_encode(l, voc):
    indexes_batch = [indexes_from_sentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    pad_list = zero_padding(indexes_batch)
    pad_var = torch.LongTensor(pad_list)
    return pad_var, lengths


def output_encode(l, voc):
    indexes_batch = [indexes_from_sentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    pad_list = zero_padding(indexes_batch)
    pad_var = torch.LongTensor(pad_list)
    return pad_var, lengths


def batch_to_train_data(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])

        inp_enc, lengths = input_encode(input_batch, voc)
        out_enc, _ = output_encode(output_batch, voc)

    return inp_enc, out_enc