import enum
from math import ceil
import numpy as np 
import config
import torch

from transformers import BertTokenizer

class EntityDataset():
    # text_list = List[tokenized text]
    # label_list = List[labels for tokenized text]
    def __init__(self, text_list, label_list, doi_list, paragraph_split_length):
        self.text_list, self.label_list, self.doi_list = self.split_paragraphs(text_list, label_list, doi_list, paragraph_split_length)
        # Since words are tokenized, we don't do basic_tokenize
        self.tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_basic_tokenize=False)

        
    # Because paragraphs are over 512 tokens, we need to prep it in correct format
    def split_paragraphs(self, text_list, label_list, doi_list, paragraph_split_length):
        new_text_list = []
        new_label_list = []
        new_doi_list = []
        for i in range(len(text_list)):
            paragraph = text_list[i]
            label = label_list[i]
            if doi_list:
                doi = doi_list[i]
            splits = ceil(len(paragraph) / paragraph_split_length) # Split length for over length paragraphs (MODEL max_len refer to MAX_LEN in config.py)
            paragraphs = np.array_split(paragraph, splits)
            labels = np.array_split(label, splits)
            if doi_list:
                dois = [doi] * splits
            new_text_list.extend(paragraphs)
            new_label_list.extend(labels)
            if doi_list:
                new_doi_list.extend(dois)
            else:
                new_doi_list.extend([i] * splits) # Making sure that dataloader doesn't give error in testing
        return new_text_list, new_label_list, new_doi_list
    
    def label2id(self, labels):
        label_types = config.NER_LABELS
        label_mapping = {label: i for i, label in enumerate(label_types)}
        label_2_id_list = []
        for label in labels:
            # Might throw error when NER_LABELS is not comprehensive
            label_2_id_list.append(label_mapping[label])
        return label_2_id_list

    def id2label(self, ids):
        label_types = config.NER_LABELS
        label_mapping = {i: label for i, label in enumerate(label_types)}
        id_2_label_list = []
        for id in ids:
            id_2_label_list.append(label_mapping[id])
        return np.array(id_2_label_list)
    

    def __len__(self):
        return len(self.text_list)
    # ALIGN BIO BETTER
    def __getitem__(self, index):
        paragraph = self.text_list[index].tolist() # Because np.array_split returns [np.array([])]
        labels = self.label_list[index].tolist()
        # label_ids = self.label2id(labels)
        input_ids = []
        ner_labels_string = []        
        for i, token in enumerate(paragraph):
            # Loop through each token in Bert tokenizer - we need to align labels
            wordpiece_token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            wordpiece_token_len = len(wordpiece_token_ids)
            input_ids.extend(wordpiece_token_ids)
            # Labels need to be aligned with wordpiece tokens
            for wordpiece_token_idx in range(wordpiece_token_len):
                if labels[i].startswith('B-'):
                    if wordpiece_token_idx == 0:
                        ner_labels_string.append(labels[i])
                    else:
                        ner_labels_string.append('I' + labels[i][1:])
                else:
                    ner_labels_string.append(labels[i])

            # Or align label with words                
            # ner_labels_string.extend([labels[i]] * wordpiece_token_len)
        
        ner_labels = self.label2id(ner_labels_string)

        input_ids = input_ids[:config.MAX_LEN-2]
        ner_labels = ner_labels[:config.MAX_LEN-2]

        input_ids = [102] + input_ids + [103]
        ner_labels = [0] + ner_labels + [0]

        mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        padding_len = config.MAX_LEN - len(input_ids)

        input_ids += ([0] * padding_len)
        mask += ([0] * padding_len)
        token_type_ids += ([0] * padding_len)
        ner_labels += ([0] * padding_len)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(ner_labels, dtype=torch.long),
            'doi': torch.tensor(self.doi_list[index])
        }