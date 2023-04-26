#!/usr/bin/env python
# coding: utf-8

# In[2]:

import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, Subset


# In[3]:


class Virginia_Rail_Dataset(Dataset):

    def __init__(self,
                 root_dir,
                 positive_folder,
                 negative_folder,            
                 preprocessing_transforms,
                 augmentation_transforms,
                 augmentation_noise_transforms,
                 device):
        self.root_dir = root_dir
        self.positive_folder = positive_folder
        self.negative_folder = negative_folder
        self.positive_files = os.listdir(os.path.join(self.root_dir, self.positive_folder))
        self.negative_files = os.listdir(os.path.join(self.root_dir, self.negative_folder))
        self.labels = torch.cat([torch.ones(len(self.positive_files)), torch.zeros(len(self.negative_files))])
        self.preprocessing_transforms = preprocessing_transforms
        self.augmentation_transforms = augmentation_transforms
        self.augmentation_noise_transforms = augmentation_noise_transforms
        self.device = "cpu"
        
    def __len__(self):
        return len(self.positive_files) + len(self.negative_files)


    def __getitem__(self, index, augment = True):
        if index < len(self.positive_files):
            path = os.path.join(self.root_dir, self.positive_folder, self.positive_files[index])
            label = 1            
        else:
            path = os.path.join(self.root_dir, self.negative_folder, self.negative_files[index - len(self.positive_files)])
            label = 0

        audio, _ = torchaudio.load(path)
        audio = audio.to(self.device)
        if self.augmentation_noise_transforms is not None:
            audio = self.augmentation_noise_transforms(audio)
        audio = self.preprocessing_transforms(audio)
        if self.augmentation_transforms is not None:
            audio = self.augmentation_transforms(audio)

        return audio, label


# In[ ]:




