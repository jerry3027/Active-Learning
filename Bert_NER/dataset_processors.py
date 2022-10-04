import os
import json
import pandas as pd

MATSCHOLAR_LABEL_DICT = {
    'O':'O', 'B-MAT':'B-CN', 'I-MAT':'I-CN', 'B-SPL':'B-PV', 'I-SPL':'I-PV', 
    'B-DSC':'B-PV', 'I-DSC':'I-PV', 'B-PRO':'B-PN', 'I-PRO':'I-PN', 'B-APL': 'O',
    'I-APL':'O', 'B-SMT':'O', 'I-SMT':'O', 'B-CMT':'O', 'I-CMT':'O'}

PRANAV_LABEL_DICT = {
    'O':'O',
    'POLYMER':'CN',
    'POLYMER_FAMILY':'CN',
    'ORGANIC':'CN',
    'MONOMER':'CN',
    'INORGANIC':'CN',
    'MATERIAL_AMOUNT':'O',
    'PROP_NAME':'PN',
    'PROP_VALUE':'PV'
}

# IF not is_PSC, we only train_text_list and train_label_list are populated, the test_text_list and test_label_list will be []
def process_PIPELINE(dataset_path, is_PSC=False, replace_dois=None):
    data_items = os.listdir(dataset_path)
    train_text_list = []
    train_label_list = []
    test_text_list = []
    test_label_list = []
    doi_list = []
    for data_item in data_items:
        doi_list.append(data_item)
        data_item_path = os.path.join(dataset_path, data_item)
        with open(os.path.join(data_item_path, 'tokenized_paragraphs.txt'), 'r') as file:
            section = json.load(file)
        with open(os.path.join(data_item_path, 'ner_dirty.txt'), 'r') as file:
            ner_dirty = json.load(file)
        with open(os.path.join(data_item_path, 'ner_clean.txt'), 'r') as file:
            ner_clean = json.load(file)
        
        if replace_dois and data_item in replace_dois:
            ner = ner_clean
        else:
            ner = ner_dirty
        
        if is_PSC:
            test_text_list.append(section)
        else:
            train_text_list.append(section)
        label_list_t = []
        word_idx = 0
        for entity in ner:
            start, end, value, type = entity
            while word_idx < start:
                label_list_t.append('O')
                word_idx += 1
            for _ in range(start, end):
                if word_idx == start:
                    label_list_t.append('B-' + type)
                else:
                    label_list_t.append('I-' + type)
                word_idx += 1
        while word_idx < len(section):
            label_list_t.append('O')
            word_idx += 1

        if is_PSC:
            test_label_list.append(label_list_t)
        else:
            train_label_list.append(label_list_t)
    return train_text_list, test_text_list, train_label_list, test_label_list, doi_list

def process_MATSCHOLAR(dataset_path):
    text_list = []
    label_list = []

    # Statistic metrics
    paragraph_count = 0
    cn_count = 0
    pn_count = 0
    pv_count = 0

    files = os.listdir(dataset_path)
    
    for file in files:
        with open(os.path.join(dataset_path, file), 'r') as f:
            lines = f.readlines()

        current_sentence = []
        current_sentence_labels = []
        current_paragraph = []
        current_paragraph_labels = []
        for line in lines:
            line = line.strip()
            if line == '':
                if current_sentence and current_sentence[-1] == '.':
                    current_paragraph.extend(current_sentence)
                    current_paragraph_labels.extend(current_sentence_labels)
                # Add to text_list and lebel_list if current sentence is not EMPTY or a title
                elif current_sentence and current_paragraph:
                    # For statistics
                    paragraph_count += 1

                    text_list.append(current_paragraph)
                    label_list.append(current_paragraph_labels)
                    current_paragraph = []
                    current_paragraph_labels = []
                current_sentence = []
                current_sentence_labels = []
            else:
                tokens = line.split()
                word = tokens[:-1]
                label = tokens[-1]
                current_sentence.append(' '.join(word))
                # change labal to our application
                if label in MATSCHOLAR_LABEL_DICT:
                    current_sentence_labels.append(MATSCHOLAR_LABEL_DICT[label])
                    # For statistics
                    if label == 'B-MAT':
                        cn_count += 1
                    elif label == 'B-PRO':
                        pn_count += 1
                    elif label in ['B-SPL', 'B-DSC']:
                        pv_count += 1

                else:
                    current_sentence_labels.append('O')
        # Add the last paragraph in document (because we don't have title afterwards to add it to text_list and label_list)
        text_list.append(current_paragraph)
        label_list.append(current_paragraph_labels)
    print(paragraph_count, cn_count, pn_count, pv_count)
    return text_list, label_list   
    
def process_Pranav(dataset_path):
    text_list = []
    label_list = []

    with open(dataset_path, 'r') as f:
        for line in f:
            document = json.loads(line)
            tokens = document['tokens']
            spans = document['spans']

            text_list_t = []
            label_list_t = []
            span_idx = 0
            for i, token in enumerate(tokens):
                text_list_t.append(token['text'])
                # Update span_idx first
                if span_idx < len(spans) and i > spans[span_idx]['token_end']:
                    span_idx += 1
                # Append 'O' to labels if finished processing spans
                if span_idx >= len(spans):
                    label_list_t.append('O')
                    continue
                # Process label of tokens
                if i >= spans[span_idx]['token_start'] and i <= spans[span_idx]['token_end']:
                    transformed_label = PRANAV_LABEL_DICT[spans[span_idx]['label']]
                    if transformed_label == 'O':
                        label_list_t.append('O')
                    elif i == spans[span_idx]['token_start']:
                        label_list_t.append('B-' + transformed_label)
                    else:
                        label_list_t.append('I-' + transformed_label)
                else: 
                    label_list_t.append('O')
            text_list.append(text_list_t)
            label_list.append(label_list_t)
    return text_list, label_list            
    

if __name__ == '__main__':
    # text_list, label_list = process_MATSCHOLAR('./Data/Datasets/Matscholar')
    # print(len(os.listdir('./Data/Datasets/Yinghao')))
    text_list, label_list = process_Pranav('./Data/Datasets/Pranav/combined_labels.jsonl')
    pass
