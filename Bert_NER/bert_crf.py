from turtle import forward
import torch
from transformers import BertForTokenClassification, BertModel
import torch.nn as nn
from torchcrf import CRF
import config

class Bert_CRF(nn.Module):
    def __init__(self, nb_labels):
        super(Bert_CRF, self).__init__()
        self.nb_labels = nb_labels
        self.bert = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768, self.nb_labels)
        self.crf = CRF(self.nb_labels, batch_first=True)

    # Batch operation
    def forward(self, input_ids, token_type_ids, attention_mask, labels):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        features = self.classifier(sequence_output)
        attention_mask = attention_mask.byte()
        loss =  -self.crf.forward(features, labels, reduction='mean')
        b_tag_seq = self.crf.decode(features, mask=attention_mask)
        return loss, b_tag_seq

        # loss, logits = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
        # # [batch_size, sequence_length, num_labels]
        # score, path = self.crf.decode(logits, mask=attention_mask)
        # nll = self.crf(logits, labels, mask=attention_mask)
        # return score, path, nll

class Bert(nn.Module):
    def __init__(self, nb_labels):
        super(Bert, self).__init__()
        self.nb_labels = nb_labels
        self.bert = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768, self.nb_labels)

    # Batch operation
    def forward(self, input_ids, token_type_ids, attention_mask, labels, doi):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss_fn = nn.CrossEntropyLoss()

        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.nb_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        loss = loss_fn(active_logits, active_labels)
        return loss, logits 
