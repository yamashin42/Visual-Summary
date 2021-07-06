import numpy as np
import argparse
import torch
import torch.nn as nn
from transformers import BertTokenizer

from model import BERT_pair
from data import load_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type = str, default = 'data/test/test_sample.json')
    parser.add_argument('--model', type = str, default = 'model/CS.pth')
    parser.add_argument('--bert', type = str, default = 'scibert_scivocab_uncased')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained(args.bert)
    net = BERT_pair(args.bert)
    net = net.to(device)
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load(args.model))
    net.eval()

    figures = load_test(args.test_data, tokenizer)
    scores = []

    for figure in figures:
        score = 0
        for i in range(len(figure['sequences'])):
            tokens = torch.stack([figure['sequences'][i]]).to(device)
            ids = torch.stack([figure['ids'][i]]).to(device)
            mask = torch.ones((tokens.size(0),tokens.size(1))).to(device)

            with torch.no_grad():
                # Smaller value of model output indicates ranked higher. For clarity, multiply by (-1)
                score_segment = net(tokens, mask, ids) * (-1)
            score_segment = score_segment.item()
            score += score_segment

        scores.append(score)

    #descending order
    rank = np.argsort(-np.array(scores))
    
    print('Rank 1')
    print('Figure {}'.format(figures[rank[0]]['number']))

    print('Rank 2')
    print('Figure {}'.format(figures[rank[1]]['number']))

    print('Rank 3')
    print('Figure {}'.format(figures[rank[2]]['number']))