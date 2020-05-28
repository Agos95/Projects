# -*- coding: utf-8 -*-

import torch
import re
import numpy as np
from torch.utils.data import Dataset, DataLoader
from functools import reduce
from torchvision import transforms
from gensim.models import Word2Vec


class LoadDataset(Dataset):

    def __init__(self, filepath, crop_len, transform=None, embedding=100):

        ### Load data
        text = open("Scienza_in_cucina.txt", 'r').read()

        ### Preprocess data
        # Lower case
        text = text.lower()

        # Remove some symbols
        text = re.sub("[_ª«»ºø—'()]", " ", text)
        text = re.sub("[\[*\]]", "", text)
        # Replace punctuations
        punctuation = {
            "!" : "-ESCLAMATION-",
            ":" : "-COLUMN-",
            ";" : "-SEMICOLUMN-",
            "," : "-COMMA-",
            "." : "-DOT-"
        }
        for p, w in punctuation.items():
            text = re.sub("[{}]".format(p), " {} ".format(w), text)

        # Extract the paragraph (divided by recipe number)
        par_list = re.split('\n\n[0-9].*', text)
        # Remove double new lines
        par_list = list(map(lambda s: s.replace('\n\n', '\n'), par_list))
        # split each paragraph into a list of string
        par_list = [x.split() for x in par_list]

        # dataset is made by list with at least crop_len words
        lines = []
        for x in par_list:
            if len(x) >= crop_len:
                for n in range(0, len(x), crop_len):
                    line = x[n:n+crop_len]
                    if len(line) >= crop_len: lines.append(line)

        print('Number of sentences: ', len(lines))

        ### Word2Vect
        model = Word2Vec(sentences=lines, min_count=1, iter=100, size=embedding)
        model.save('w2v.model')
        self.vocabulary_len = len(model.wv.vocab)
        print("Vocabulary size = {}".format(self.vocabulary_len))
        word2idx = {w:i for i,w in enumerate(model.wv.index2word)}
        idx2word = {i:w for i,w in enumerate(model.wv.index2word)}
        # fix punctuation and spaces
        for p, w in punctuation.items():
            idx = word2idx[w]
            idx2word[idx] = p
        for i,w in idx2word.items():
            if w not in punctuation.keys():
                idx2word[i] = " "+w

        ### Store data
        self.lines = lines
        self.transform = transform
        self.word2idx = word2idx
        self.idx2word = idx2word

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        # Get sonnet text
        text = self.lines[idx]
        # Encode with numbers
        encoded = encode_text(self.word2idx, text)
        # Create sample
        sample = {'text': text, 'encoded': encoded}
        # Transform (if defined)
        if self.transform:
            sample = self.transform(sample)
        return sample

    def vocabulary_length(self):
        return self.vocabulary_len

def encode_text(word2idx, text):
    for c in text:
        try:
            a = word2idx[c]
        except:
            text.remove(c)
    encoded = [word2idx[c] for c in text]
    return encoded


def decode_text(idx2word, encoded):
    text = [idx2word[c] for c in encoded]
    text = reduce(lambda s1, s2: s1 + s2, text)
    return text

class ToTensor():

    def __call__(self, sample):
        # Convert one hot encoded text to pytorch tensor
        encoded_onehot = torch.LongTensor(sample['encoded'])
        return {'encoded': encoded_onehot}



if __name__ == '__main__':

    #%% Initialize dataset
    filepath = 'Scienza_in_cucina.txt'

    #%% Test RandomCrop
    """crop_len = 100
    rc = RandomCrop(crop_len)
    sample = rc(sample)"""

    #%% Test OneHotEncoder
    """vocabulary_len = len(dataset.vocabulary)
    ohe = OneHotEncoder(vocabulary_len)
    sample = ohe(sample)"""


    # Test dataloader
    crop_len = 20
    trans = ToTensor()
    dataset = LoadDataset(filepath, crop_len)

    # Test sampling
    sample = dataset[0]

    print('##############')
    print('##############')
    print('TEXT')
    print('##############')
    print(sample['text'])

    print('##############')
    print('##############')
    print('ENCODED')
    print('##############')
    print(sample['encoded'])

    # Test decode function
    encoded_text = sample['encoded']
    decoded_text = decode_text(dataset.idx2word, encoded_text)

    #%% Test ToTensor
    tt = ToTensor()
    sample = tt(sample)
    print(type(sample['encoded']))
    print(sample['encoded'].shape)

    dataset = LoadDataset(filepath, crop_len, transform=trans)
    dataloader = DataLoader(dataset, batch_size=52, shuffle=True)

    for batch_sample in dataloader:
        batch_onehot = batch_sample['encoded']
        print(batch_onehot.shape)



