import torch
import pandas as pd
from ActiveLearning.Bert_NER.dataset import EntityDataset
import config
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import KFold
import engine
import numpy as np
import dataset_processors
from ActiveLearning.Bert_NER.bert_crf import Bert_CRF, Bert
import json
from sklearn import model_selection
import random


def activeLearning(num_incorrect_predictions, avg_entropy, avg_confidence, avg_margin, doi, metric, correction_ratio):
    N = int(len(doi) * correction_ratio) # Number of paragraphs
    doi = np.array(doi)
    if metric in ['Traditional', 'all']:
        dois = doi[(-np.array(num_incorrect_predictions)).argsort()[:N]].tolist()
        with open('./ActiveLearning/Bert_predictions/traditional.json', 'w+') as f:
            json.dump(dois, f)
    if metric in ['Uncertainty', 'all']:
        dois = doi[(np.array(avg_confidence)).argsort()[:N]].tolist()
        with open('./ActiveLearning/Bert_predictions/uncertainty.json', 'w+') as f:
            json.dump(dois, f)
    if metric in ['Entropy', 'all']:
        dois = doi[(-np.array(avg_entropy)).argsort()[:N]].tolist()# Verify
        with open('./ActiveLearning/Bert_predictions/entropy.json', 'w+') as f:
            json.dump(dois, f)
    if metric in ['Margin', 'all']:
        dois = doi[(np.array(avg_margin)).argsort()[:N]].tolist()
        with open('./ActiveLearning/Bert_predictions/margin.json', 'w+') as f:
            json.dump(dois, f)
    
def build_model(device, dataset):
    model = Bert(len(config.NER_LABELS))
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    # Set optimizer
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
        
    # Set scheduler
    num_train_steps = int(len(dataset) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    return model, optimizer, scheduler
    

if __name__ == '__main__':
    dataset_path = './ActiveLearning/Data/'
    bert_output_path = './ActiveLearning/Bert_predictions'
    
    train_text_list, _, train_label_list, _, doi_list = dataset_processors.process_PIPELINE(dataset_path=dataset_path, is_PSC=False)
    train_text_list = np.array(train_text_list)
    train_label_list = np.array(train_label_list)
    doi_list = np.array(doi_list)

    doi2doi_id = {doi:i for i, doi in enumerate(doi_list)}
    doi_id2doi = {v:k for k, v in doi2doi_id.items()} 
    
    doi_id_list = np.array([doi2doi_id[key] for key in doi_list])

    
    device =torch.device('cuda:0')

    best_loss = np.inf
    # Implement Cross Validation
    kf = KFold(n_splits=5)

    num_incorrect_predictions_all = []
    avg_entropy_all = []
    avg_confidence_all = []
    avg_margin_all = []
    doi_all = []

    indices = np.array(range(len(train_text_list)))
    train_paragraphs, test_paragraphs, train_labels, test_labels, indices_train, indices_test = model_selection.train_test_split(train_text_list, train_label_list, indices, random_state=3, test_size=0.1)

    # for train_idx, validation_idx in kf.split(train_paragraphs):

    #     train_split = train_paragraphs[train_idx]
    #     test_split = train_paragraphs[validation_idx]
    #     train_label = train_labels[train_idx]
    #     test_label = train_labels[validation_idx]
    #     doi_train = doi_id_list[train_idx]
    #     doi_test = doi_id_list[validation_idx]

    #     train_dataset = EntityDataset(train_split, train_label, doi_train, paragraph_split_length=512)
    #     train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE)

    #     test_dataset = EntityDataset(test_split, test_label, doi_test, paragraph_split_length=512)
    #     test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.VALID_BATCH_SIZE)
        
    #     # Create model
    #     model, optimizer, scheduler = build_model(device, train_dataset)

    #     # Training model
    #     for epoch in range(config.EPOCHS):
    #         train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
    #         print(f"Train Loss = {train_loss}")

    #     # Make prediction on test fold
    #     num_incorrect_predictions, avg_entropy, avg_confidence, avg_margin, doi = engine.test_fn(test_data_loader, model, device)
        
    #     num_incorrect_predictions_all.extend(num_incorrect_predictions)
    #     avg_entropy_all.extend(avg_entropy)
    #     avg_confidence_all.extend(avg_confidence)
    #     avg_margin_all.extend(avg_margin)
    #     doi_all.extend(doi)

    # activeLearning(num_incorrect_predictions_all, avg_entropy_all, avg_confidence_all, avg_margin_all, doi_all, metric='all', correction_ratio=0.1)

    
    # Perform inference on original data
    train_whole_dataset = EntityDataset(train_paragraphs, train_labels, doi_list=None, paragraph_split_length=512)
    train_whole_data_loader = torch.utils.data.DataLoader(train_whole_dataset, batch_size=config.TRAIN_BATCH_SIZE)


    _, _, clean_label_list, _, _ = dataset_processors.process_PIPELINE(dataset_path=dataset_path, is_PSC=False, replace_dois=indices_test.tolist())
    clean_label_list = np.array(clean_label_list)
    test_labels = clean_label_list[indices_test]

    test_whole_dataset = EntityDataset(test_paragraphs, test_labels, doi_list=None, paragraph_split_length=512)
    test_whole_data_loader = torch.utils.data.DataLoader(test_whole_dataset, batch_size=config.VALID_BATCH_SIZE)

    model, optimizer, scheduler = build_model(device, train_whole_dataset)
    
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(train_whole_data_loader, model, optimizer, device, scheduler)
        print(f"Train Loss = {train_loss}")

    
        
    metrics = engine.actual_test(test_whole_data_loader, model, device)
    print("Baseline metrics:")
    print(metrics)
    
    # Perform inference on active learned data
    
    # Uncertainty 
    with open('./ActiveLearning/Bert_predictions/uncertainty.json', 'r') as uncertainty_file:
        uncertainty_id_list = json.load(uncertainty_file)
    
    replace_doi_list = [doi_id2doi[uncertainty_id] for uncertainty_id in uncertainty_id_list]

    uncertainty_text_list, _, uncertainty_label_list, _, doi_list = dataset_processors.process_PIPELINE(dataset_path=dataset_path, is_PSC=False, replace_dois=replace_doi_list)

    train_uncertainty_dataset = EntityDataset(uncertainty_text_list, uncertainty_label_list, doi_list=None, paragraph_split_length=512)
    train_uncertainty_data_loader = torch.utils.data.DataLoader(train_uncertainty_dataset, batch_size=config.TRAIN_BATCH_SIZE)

    model, optimizer, scheduler = build_model(device, train_uncertainty_dataset)
    
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(train_uncertainty_data_loader, model, optimizer, device, scheduler)
        print(f"Train Loss = {train_loss}")
        
    metrics = engine.actual_test(test_whole_data_loader, model, device)
    print("Uncertainty metrics:")
    print(metrics)

    # Entropy
    with open('./ActiveLearning/Bert_predictions/entropy.json', 'r') as entropy_file:
        entropy_id_list = json.load(entropy_file)
    
    replace_doi_list = [doi_id2doi[entropy_id] for entropy_id in entropy_id_list]

    entropy_text_list, _, entropy_label_list, _, doi_list = dataset_processors.process_PIPELINE(dataset_path=dataset_path, is_PSC=False, replace_dois=replace_doi_list)

    train_entropy_dataset = EntityDataset(entropy_text_list, entropy_label_list, doi_list=None, paragraph_split_length=512)
    train_entropy_data_loader = torch.utils.data.DataLoader(train_entropy_dataset, batch_size=config.TRAIN_BATCH_SIZE)

    model, optimizer, scheduler = build_model(device, train_entropy_dataset)

    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(train_entropy_data_loader, model, optimizer, device, scheduler)
        print(f"Train Loss = {train_loss}")
        
    metrics = engine.actual_test(test_whole_data_loader, model, device)
    print("Entropy metrics:")
    print(metrics)

    # Traditional
    with open('./ActiveLearning/Bert_predictions/traditional.json', 'r') as traditional_file:
        traditional_id_list = json.load(traditional_file)
    
    replace_doi_list = [doi_id2doi[traditional_id] for traditional_id in traditional_id_list]

    traditional_text_list, _, traditional_label_list, _, doi_list = dataset_processors.process_PIPELINE(dataset_path=dataset_path, is_PSC=False, replace_dois=replace_doi_list)

    train_traditional_dataset = EntityDataset(traditional_text_list, traditional_label_list, doi_list=None, paragraph_split_length=512)
    train_traditional_data_loader = torch.utils.data.DataLoader(train_traditional_dataset, batch_size=config.TRAIN_BATCH_SIZE)

    model, optimizer, scheduler = build_model(device, train_traditional_dataset)

    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(train_traditional_data_loader, model, optimizer, device, scheduler)
        print(f"Train Loss = {train_loss}")
        
    metrics = engine.actual_test(test_whole_data_loader, model, device)
    print("Traditional metrics:")
    print(metrics)

    # Margin
    with open('./ActiveLearning/Bert_predictions/margin.json', 'r') as margin_file:
        margin_id_list = json.load(margin_file)
    
    replace_doi_list = [doi_id2doi[margin_id] for margin_id in margin_id_list]

    margin_text_list, _, margin_label_list, _, doi_list = dataset_processors.process_PIPELINE(dataset_path=dataset_path, is_PSC=False, replace_dois=replace_doi_list)

    train_margin_dataset = EntityDataset(margin_text_list, margin_label_list, doi_list=None, paragraph_split_length=512)
    train_margin_data_loader = torch.utils.data.DataLoader(train_margin_dataset, batch_size=config.TRAIN_BATCH_SIZE)

    model, optimizer, scheduler = build_model(device, train_margin_dataset)

    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(train_margin_data_loader, model, optimizer, device, scheduler)
        print(f"Train Loss = {train_loss}")
        
    metrics = engine.actual_test(test_whole_data_loader, model, device)
    print("Margin metrics:")
    print(metrics)

    # Random
    replace_doi_list = random.sample(range(0,90), 8)

    random_text_list, _, random_label_list, _, doi_list = dataset_processors.process_PIPELINE(dataset_path=dataset_path, is_PSC=False, replace_dois=replace_doi_list)

    train_random_dataset = EntityDataset(random_text_list, random_label_list, doi_list=None, paragraph_split_length=512)
    train_random_data_loader = torch.utils.data.DataLoader(train_random_dataset, batch_size=config.TRAIN_BATCH_SIZE)

    model, optimizer, scheduler = build_model(device, train_random_dataset)

    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(train_random_data_loader, model, optimizer, device, scheduler)
        print(f"Train Loss = {train_loss}")
        
    metrics = engine.actual_test(test_whole_data_loader, model, device)
    print("Random metrics:")
    print(metrics)