#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, recall_score, precision_score, roc_curve, auc


# In[ ]:


def compute_ensemble_metrics(mod_list, data_loader, criterion, num_classes, device=device, show_FNs=False, compute_roc_auc=False):
    '''
    This is the same as the compute_metrics() function used during training, but this combines the models for ensemble
    prediction and computes the metrics based on the average prediction of all counterpart models. 
    '''

    # Evaluation mode
    for model in mod_list:
        model.eval()

    # Set up empty lists
    num_correct = 0
    total_loss = 0.0
    all_targets = []
    all_probabilities = []
    all_predictions = []
    false_negatives = []

    # Move data to device
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        
        # Create ensemble probabilies array
        ensemble_probabilities = torch.zeros(y.shape[0], num_classes, device=device)
        
        # Run models over data and update ensemble probabilities
        for model in mod_list:
            y_hat = model(x)
            probabilities = F.softmax(y_hat, dim=1)
            ensemble_probabilities += probabilities
        
        # Average the probabilities, make predictions, count correct predictions, and calculate loss
        ensemble_probabilities /= len(mod_list)
        ensemble_predictions = torch.argmax(ensemble_probabilities, dim=1)
        num_correct += torch.sum(ensemble_predictions == y).item()
        loss = criterion(ensemble_probabilities, y)
        total_loss += loss.item() * x.size(0)
        
        # Optional: if tag show FNs is on, this will give the indices of false negatives. This was specific to 
        # the project, as I wanted to see what acoustic signals were still being missed to see if there were patterns
        # I could identify and address.
        if show_FNs:
            fn_indices = (ensemble_predictions == 0) & (y == 1)
            for i in range(len(fn_indices)):
                if fn_indices[i]:
                    false_negatives.append((x[i].cpu().numpy(), ensemble_probabilities[i, 1].item()))

        # Move data to CPU and convert to numpy arrays from torch tensors.
        all_targets.extend(y.cpu().numpy())
        all_predictions.extend(ensemble_predictions.cpu().numpy())
        all_probabilities.extend(ensemble_probabilities.detach().cpu().numpy())

    # Compute the various metrics
    accuracy = num_correct / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader.dataset)
    cm = confusion_matrix(all_targets, all_predictions)
    f1_scores = f1_score(all_targets, all_predictions, average=None)
    f1_class_1 = f1_scores[1]
    fbeta_scores = fbeta_score(all_targets, all_predictions, average=None, beta=2)
    fbeta_class_1 = fbeta_scores[1]
    recall = recall_score(all_targets, all_predictions, average='binary')
    precision = precision_score(all_targets, all_predictions, average='binary')

    # Create dict with results
    results = {
        'accuracy': accuracy,
        'avg_loss': avg_loss,
        'cm': cm,
        'f1_class_1': f1_class_1,
        'fbeta_class_1': fbeta_class_1,
        'recall': recall,
        'precision': precision
    }
    # Add false negatives to results if tracking.
    if show_FNs:
        results['false_negatives'] = false_negatives

    # If computing ROC and AUC, this will add false postive rate, true positive rate, and roc_auc variable to results
    if compute_roc_auc:
        all_targets_np = np.array(all_targets)
        all_probabilities_np = np.array(all_probabilities)
        
        num_classes = 2
        y_true_one_hot = np.eye(num_classes)[all_targets_np]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], all_probabilities_np[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        results['fpr'] = fpr
        results['tpr'] = tpr
        results['roc_auc'] = roc_auc
        
    return results, all_targets, all_probabilities

