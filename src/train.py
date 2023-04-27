#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, recall_score
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.nn.functional as F


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[3]:



# Define a save_checkpoint function that saves state_dicts as well as important metrics
def save_checkpoint(epoch, model, optimizer, metrics, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, checkpoint_path)

# Define a loade_checkpoint function that loads state_dicts as well as important metrics
def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    return epoch, metrics

# Define training loop
def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device = device,
    scheduler = None,
    checkpoint_path: str = None,
    resume: bool = False,
    manual_lr: float = None  

) -> None:
    
    # If resuming, grab checkpoint and metatdata from saved checkpoint, else create empty lists to house new data
    if resume and checkpoint_path is not None and os.path.exists(checkpoint_path):
        last_epoch, metrics = load_checkpoint(model, optimizer, checkpoint_path)
        train_losses, test_losses, train_accs, test_accs, train_f1s, test_f1s, train_fbetas, test_fbetas, train_recalls, test_recalls = metrics
    else:
        last_epoch = 0
        train_losses = []
        test_losses = []
        train_accs = []
        test_accs = []
        train_f1s = []
        test_f1s = []
        train_fbetas = []
        test_fbetas = []
        train_recalls = []
        test_recalls = []
        
    if manual_lr is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = manual_lr

    # Move the model to the device:
    model.to(device)
    best_recall = 0
    best_f1 = 0
    # Training loop
    for epoch in range(last_epoch, last_epoch + num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr} | Current epoch: {epoch +1}")
        # Set the model to training mode:
        model.train()
        # Loop over the training data:
        train_loss = 0
        for x, y in tqdm(train_loader):
            # Move the data to the device:
            x, y = x.to(device), y.to(device)
            # Zero the gradients:
            optimizer.zero_grad()
            # Forward pass:
            y_hat = model(x)
            # Compute the loss:
            loss = criterion(y_hat, y)
            train_loss += loss.item()
            # Backward pass:
            loss.backward()
            # Update the parameters:
            optimizer.step()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Set the model to evaluation mode:
        model.eval()
        # Compute the accuracy and loss on the test data:
        test_loss = 0
        accuracy_test = 0
        with torch.no_grad():
          train_accuracy, train_loss, train_cm, train_f1, train_fbeta, train_recall = compute_metrics(model, train_loader, criterion, device)
          test_accuracy, test_loss, test_cm, test_f1, test_fbeta, test_recall = compute_metrics(model, test_loader, criterion, device)

        test_losses.append(test_loss)
        
        train_accs.append(train_accuracy)
        test_accs.append(test_accuracy)
        
        train_f1s.append(train_f1)
        test_f1s.append(test_f1)
        
        train_fbetas.append(train_fbeta)
        test_fbetas.append(test_fbeta)
        
        train_recalls.append(train_recall)
        test_recalls.append(test_recall)
        
        
        # Print the results:
        print(f"Train accuracy: {train_accuracy:.2f} | Train loss: {train_loss:.4f} |"\
              f"Train F1 score: {train_f1:.4f}| Train fbeta score: {train_fbeta:.4f}|"\
              f"Train Recall: {train_recall:.4f}")
        print(f"Test accuracy: {test_accuracy:.2f} | Test loss: {test_loss:.4f} |"\
              f"Test F1 score: {test_f1:.4f}| Test fbeta score: {test_fbeta:.4f}|"\
              f"Test Recall: {test_recall:.4f}")
        print("Train confusion matrix:")
        print(train_cm)
        print("Test confusion matrix:")
        print(test_cm)
        # This will need to be altered depending on scheduler being used. This is set up to use ReduceLRonPlateau()
        # and to reduce learning rate when test_loss plateaus

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(test_loss)
            elif isinstance(scheduler, StepLR):
                scheduler.step()

        # This updates a checkpoint after every epoch, so there is always the latest checkpoint
        if checkpoint_path is not None:
            metrics = (train_losses, test_losses, train_accs, test_accs, train_f1s, test_f1s, train_fbetas, test_fbetas, train_recalls, test_recalls)
            save_checkpoint(epoch + 1, model, optimizer, metrics, checkpoint_path)
            
            # This creates separate checkpoints every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path_10_epochs = checkpoint_path.replace('.pth', f'_epoch_{epoch + 1}.pth')
                save_checkpoint(epoch + 1, model, optimizer, metrics, checkpoint_path_10_epochs)
            
            # This saves the checkpoint when best_recall is surpassed    
            if test_recall > best_recall:
                best_recall = test_recall
                best_recall_path = checkpoint_path.replace('.pth', '_best_recall.pth')
                save_checkpoint(epoch + 1, model, optimizer, metrics, best_recall_path)
            
            # This saves a checkpoint when best_f1 is surpassed    
            if test_f1 > best_f1:
                best_f1 = test_f1
                best_f1_path = checkpoint_path.replace('.pth', '_best_f1.pth')
                save_checkpoint(epoch + 1, model, optimizer, metrics, best_f1_path)

    return train_losses, test_losses, train_accs, test_accs, train_f1s, test_f1s, train_recalls, test_recalls


def compute_metrics(model, data_loader, criterion, device=device, show_FNs = False):
    model.eval()
    num_correct = 0
    total_loss = 0.0
    all_targets = []
    all_predictions = []
    false_negatives = []

    for x, y in data_loader:
        # Move data to device
        x, y = x.to(device), y.to(device)
        # Forward pass
        y_hat = model(x)
        # Save predictions
        predictions = torch.argmax(y_hat, dim=1)
        # Count correct predictions
        num_correct += torch.sum(predictions == y).item()
        # Save loss
        loss = criterion(y_hat, y)
        total_loss += loss.item() * x.size(0)
        # If optional flag is set to True, this will save false negative results for later visualization
        if show_FNs:
            fn_indices = (predictions == 0) & (y == 1)
            
            for i in range(len(fn_indices)):
                if fn_indices[i]:
                    softmax_scores = F.softmax(y_hat, dim=1)
                    false_negatives.append((x[i].cpu().numpy(), softmax_scores[i, 1].item()))

        # Save targets and predictions for metrics calculations
        all_targets.extend(y.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
    # Compute metrics and return them
    accuracy = num_correct / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader.dataset)
    cm = confusion_matrix(all_targets, all_predictions)
    f1_scores = f1_score(all_targets, all_predictions, average= None)
    f1_class_1 = f1_scores[1]
    fbeta_scores = fbeta_score(all_targets, all_predictions, average = None, beta = 2)
    fbeta_class_1 = fbeta_scores[1]
    recall = recall_score(all_targets, all_predictions, average='binary')
    
    if show_FNs:
        return accuracy, avg_loss, cm, f1_class_1, fbeta_class_1, recall, false_negatives
    
    else:
        return accuracy, avg_loss, cm, f1_class_1, fbeta_class_1, recall


