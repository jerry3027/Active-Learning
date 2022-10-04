import torch
from tqdm import tqdm
import torch.nn.functional as F
import config
import numpy as np

from transformers import BertTokenizer


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        loss, _ = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        loss, _ = model(**data)
        final_loss += loss.item()
    return final_loss / len(data_loader)

def test_fn(data_loader, model, device):
    model.eval()
    # score_all = []
    # predicted_label_all = []
    # gold_label_all = []
    # mask_all = []
    doi_all = []
    num_incorrect_predictions_all = []
    avg_entropy_all = []
    avg_confidence_all = []
    avg_margin_all = []
    

    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
            
        _, logits = model(**data)
        
        scores = F.softmax(logits, dim=2).detach().cpu()
        logits = torch.argmax(scores, dim=2).numpy()

        
        gold_label_ids = data['labels'].to('cpu').numpy()
        mask = data['attention_mask'].to('cpu').numpy()
        dois = data['doi'].to('cpu').numpy()


        epoches = logits.shape[0]

        for i in range(epoches):            
            predicted_labels = data_loader.dataset.id2label(logits[i])
            gold_labels = data_loader.dataset.id2label(gold_label_ids[i])
            current_mask = mask[i] != 0
            # Number of incorrect predictions of paragraph
            num_incorrect_predictions = np.sum(predicted_labels[current_mask] != gold_labels[current_mask])
            # Find avg entropy of paragraph
            entropy = -torch.log(scores[i][current_mask]) * scores[i][current_mask]
            avg_entropy = torch.mean(torch.sum(entropy, dim=1))
            # Find avg confidence of paragraph
            avg_confidence = torch.mean(torch.max(scores[i], dim=1).values[current_mask])
            # Find avg least margin of paragraph
            top2_scores = torch.topk(scores[i], 2, dim=1).values[current_mask]
            avg_least_margin = torch.mean(torch.abs(torch.sub(top2_scores[0], top2_scores[1])))

            avg_entropy_all.append(avg_entropy)
            avg_confidence_all.append(avg_confidence)
            avg_margin_all.append(avg_least_margin)
            num_incorrect_predictions_all.append(num_incorrect_predictions)
            # predicted_label_all.append(predicted_labels)
            # gold_label_all.append(gold_labels)
            # mask_all.append(current_mask)
            doi_all.append(dois[i])

    return num_incorrect_predictions_all, avg_entropy_all, avg_confidence_all, avg_margin_all, doi_all

def id2label(id):
    label_types = config.NER_LABELS
    label_mapping = {i: label for i, label in enumerate(label_types)}
    return label_mapping[id]

def actual_test(data_loader, model, device):
    model.eval()
    metric_dict = {}
    for label in config.NER_LABELS:
        metric_dict[label] = {'true_positive': 0, 'false_positive': 0, 'false_negative': 0}
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
            
        _, logits = model(**data)
        
        logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
        logits = logits.detach().cpu().numpy()

        
        label_ids = data['labels'].to('cpu').numpy()
        mask = data['attention_mask'].to('cpu').numpy()
        for i, epoch in enumerate(label_ids):
            for j, label in enumerate(epoch):
                if mask[i][j]:
                    predicted_label = id2label(logits[i][j])
                    gold_label = id2label(label)
                    if predicted_label == gold_label:
                        metric_dict[gold_label]['true_positive'] += 1
                    else:
                        metric_dict[gold_label]['false_negative'] += 1
                        metric_dict[predicted_label]['false_positive'] += 1
    results = {}
    for k, v in metric_dict.items():
        results[k] = {}
        results[k]['precision'] = v['true_positive'] / (v['true_positive'] + v['false_negative'])
        results[k]['recall'] = v['true_positive'] / (v['true_positive'] + v['false_positive'])
    return results
