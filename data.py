import json
import spacy
import random
import re
import torch
import torch.utils.data as data
import os
from glob import glob

class Dataset(data.Dataset):
    def __init__(self, data_dir, tokenizer):
        self.tokenizer = tokenizer
        self.spacy = spacy.load('en_core_web_sm')
        self.figures = glob(os.path.join(data_dir, '*', 'Figure_*.json'))

    def __len__(self):
        return len(self.figures)

    def __getitem__(self, index):
        #load caption and mention
        with open(self.figures[index]) as j:
            data = json.load(j)
        
        caption = data['caption']
        positive = data['mention']

        #choose negative mention
        paper_dir = self.figures[index].split('/Figure_')[0]
        figures = glob(os.path.join(paper_dir, 'Figure_*.json'))
        if len(figures) == 1:
            negative = 'dummy paragraph'
        else:
            random.shuffle(figures)
            if self.figures[index] == figures[0]:
                negative_json = figures[1]
            else:
                negative_json = figures[0]

            with open(negative_json) as j:
                data = json.load(j)
            negative = data['mention']
        
        # mask all numerical values not to attend figure number
        caption = re.sub(r'\d+', '0', caption)
        positive = re.sub(r'\d+', '0', positive)
        negative = re.sub(r'\d+', '0', negative)

        # positive pair
        # sample sentence from a paragraph
        doc = self.spacy(positive)
        sentences = list(doc.sents)
        num = random.randrange(len(sentences))
        mention = str(sentences[num])
        sequence_p = self.tokenizer.encode(mention, text_pair = caption)
        length_p = len(self.tokenizer.tokenize(mention))

        # prone input sequence as maximum length is 512
        if len(sequence_p) > 512:
            copy = sequence_p
            sequence_p = sequence_p[:512]
            sequence_p[-1] = copy[-1]

        sequence_p = torch.tensor(sequence_p)

        # negative pair
        # sample sentence from a paragraph
        doc = self.spacy(negative)
        sentences = list(doc.sents)
        num = random.randrange(len(sentences))
        mention = str(sentences[num])
        sequence_n = self.tokenizer.encode(mention, text_pair = caption)
        length_n = len(self.tokenizer.tokenize(mention))

        # prone input sequence as maximum length is 512
        if len(sequence_n) > 512:
            copy = sequence_n
            sequence_n = sequence_n[:512]
            sequence_n[-1] = copy[-1]

        sequence_n = torch.tensor(sequence_n)

        return sequence_p, sequence_n, length_p, length_n

def collate_fn(data):
    data.sort(key = lambda x: len(x[0]), reverse=True)

    sequence_p, sequence_n, length_p, length_n = zip(*data)

    len_seq_p = [len(seq) for seq in sequence_p]
    padded_p = torch.ones(len(sequence_p), max(len_seq_p)).long()
    mask_p = torch.zeros(len(sequence_p), max(len_seq_p)).long()
    idx_p = torch.ones(len(sequence_p), max(len_seq_p)).long()

    for i, seq in enumerate(sequence_p):
        end = len_seq_p[i]
        padded_p[i, :end] = seq[:end]
        mask_p[i, :end] = 1
    for i in range(len(length_p)):
        idx_p[i, :length_p[i]] = 0

    len_seq_n = [len(seq) for seq in sequence_n]
    padded_n = torch.ones(len(sequence_n), max(len_seq_n)).long()
    mask_n = torch.zeros(len(sequence_n), max(len_seq_n)).long()
    idx_n = torch.ones(len(sequence_n), max(len_seq_n)).long()

    for i, seq in enumerate(sequence_n):
        end = len_seq_n[i]
        padded_n[i, :end] = seq[:end]
        mask_n[i, :end] = 1
    for i in range(len(length_n)):
        idx_n[i, :length_n[i]] = 0

    return padded_p, padded_n, mask_p, mask_n, idx_p, idx_n


def load_test(input_json, tokenizer):
    nlp = spacy.load('en_core_web_sm')

    with open(input_json) as j:
        input_data = json.load(j)
    abstract = input_data['abstract']
    figures = input_data['figures']

    doc = nlp(abstract)
    sents_abstract = list(doc.sents)

    figure_sequences = []

    for figure in figures:
        print(figure)
        caption = figure['caption']
        caption = re.sub(r'\d+', '0', caption)

        sequences = []
        ids = []
        
        for sequence in sents_abstract:
            tokens = tokenizer.encode(str(sequence), text_pair = caption)

            #prone input sequence as maximum length is 512
            if len(tokens) > 512:
                copy = tokens
                tokens = tokens[:512]
                tokens[-1] = copy[-1]

            tokens = torch.tensor(tokens).long()
            len_abstract = len(tokenizer.tokenize(str(sequence)))
            token_id = torch.ones(len(tokens)).long()
            token_id[:len_abstract] = 0
            
            sequences.append(tokens)
            ids.append(token_id)

        figure['sequences'] = sequences
        figure['ids'] = ids
        figure_sequences.append(figure)

    return figure_sequences