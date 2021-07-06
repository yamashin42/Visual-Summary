# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel



class BERT_pair(nn.Module):
    def __init__(self, bert_pretrained, dropout = 0.2):
        super(BERT_pair, self).__init__()
        self.bert = BertModel.from_pretrained(bert_pretrained)
        
        self.dropout = nn.Dropout(p=dropout)
        self.predict = nn.Linear(768, 1)


    def forward(self, text, mask, type_ids):
        encode = self.bert(text, attention_mask = mask, token_type_ids = type_ids)[0].permute(1, 0, 2)[0]
        predict = self.predict(self.dropout(encode))      
        return predict


class TripletLoss(nn.Module):
    def __init__(self, margin = 1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, positive, negative, size_average = True):
        loss = F.relu(self.margin + positive - negative)
        
        return loss.mean() if size_average else loss.sum()