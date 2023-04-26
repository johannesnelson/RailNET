#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# train_utils.py

import os
import random
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def create_dataloaders(train_dataset, test_dataset, train_indices, test_indices, batch_size=16):
    train_data = Subset(train_dataset, train_indices)
    test_data = Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def perform_k_fold_cross_validation(k_folds, 
                                    train_dataset, 
                                    test_dataset, 
                                    checkpoint_folder, 
                                    create_model_func, 
                                    train_func, 
                                    num_epochs, 
                                    starting_LR,
                                    criterion, 
                                    device, 
                                    batch_size=16, 
                                    seed=None):
    if seed is not None:
        set_seed(seed)

    all_train_losses = []
    all_test_losses = []
    all_train_accs = []
    all_test_accs = []
    all_train_f1s = []
    all_test_f1s = []
    all_train_recalls = []
    all_test_recalls = []

    checkpoint_folder = checkpoint_folder
    os.makedirs(checkpoint_folder, exist_ok=True)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    for fold, (train_indices, test_indices) in enumerate(kf.split(train_dataset)):
        print(f"Fold {fold + 1}/{k_folds}")

        train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, train_indices, test_indices, batch_size)

        model = create_model_func()

        optimizer = torch.optim.Adam(model.parameters(), lr = starting_LR)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.5, patience = 3, verbose= True)

        checkpoint_path = os.path.join(checkpoint_folder, f"checkpoint_fold_{fold + 1}.pth")

        train_losses, test_losses, train_accs, test_accs, train_f1s, test_f1s, train_recalls, test_recalls = train_func(
            model,
            train_loader,
            test_loader,
            num_epochs,
            optimizer,
            criterion,
            device,
            scheduler,
            checkpoint_path,
            resume=False,
            manual_lr=None
        )

        all_train_losses.append(train_losses)
        all_test_losses.append(test_losses)
        all_train_accs.append(train_accs)
        all_test_accs.append(test_accs)
        all_train_f1s.append(train_f1s)
        all_test_f1s.append(test_f1s)
        all_train_recalls.append(train_recalls)
        all_test_recalls.append(test_recalls)

    return all_train_losses, all_test_losses, all_train_accs, all_test_accs, all_train_f1s, all_test_f1s, all_train_recalls, all_test_recalls

