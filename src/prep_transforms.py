#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import torch
import torchaudio
import random
import torchvision
from typing import Optional, Tuple
from torchvision.transforms import Compose


# In[7]:


class AddRandomNoise(torch.nn.Module):
    def __init__(self, noise_dir: str, snr_range: Tuple[float, float]):
        super(AddRandomNoise, self).__init__()
        self.noise_files = [
            os.path.join(noise_dir, file) for file in os.listdir(noise_dir) if file.endswith(".wav")
        ]
        self.snr_range = snr_range

    def forward(self, waveform: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        noise_file = random.choice(self.noise_files)
        noise_waveform, _ = torchaudio.load(noise_file)
        noise_waveform = noise_waveform[: waveform.shape[0], : waveform.shape[1]]
        snr = torch.FloatTensor(1).uniform_(*self.snr_range)
        return torchaudio.functional.add_noise(waveform, noise_waveform, snr, lengths)


class AddRandomGaussianNoise(torch.nn.Module):
    def __init__(self, snr_range: Tuple[float, float]):
        super(AddRandomGaussianNoise, self).__init__()
        self.snr_range = snr_range

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(waveform)
        snr = torch.FloatTensor(1).uniform_(*self.snr_range)
        return torchaudio.functional.add_noise(waveform, noise, snr)
    


# In[ ]:


def prep_all_transforms(aug_level = "moderate"):
    preprocessing_transforms = get_preprocessing_transforms()
    if aug_level == "moderate":
        augmentation_transforms = get_moderate_augmentation_transforms()
        augmentation_noise_transforms = get_moderate_augmentation_noise_transforms()
    if aug_level == "heavy":
        augmentation_transforms = get_heavy_augmentation_transforms()
        augmentation_noise_transforms = get_heavy_augmentation_noise_transforms()
        
    return preprocessing_transforms, augmentation_transforms, augmentation_noise_transforms

def get_preprocessing_transforms():
    return Compose([
        torchaudio.transforms.MelSpectrogram(n_fft=1024, hop_length=517, n_mels=128),
        torchaudio.transforms.AmplitudeToDB(),
    ])


def get_moderate_augmentation_noise_transforms():
    return Compose([
        torchvision.transforms.RandomApply(torch.nn.ModuleList([
            AddRandomNoise(noise_dir="..//data//noise_folder", snr_range=(-10, 15)),
        ]), p=0.75),
        torchvision.transforms.RandomApply(torch.nn.ModuleList([
            AddRandomGaussianNoise(snr_range=(-10, 15)),
        ]), p=0.75),
    ])


def get_heavy_augmentation_noise_transforms():
    return Compose([
        torchvision.transforms.RandomApply(torch.nn.ModuleList([
            AddRandomNoise(noise_dir="..//data//noise_folder", snr_range=(-10, 10)),
        ]), p=1.0),
        torchvision.transforms.RandomApply(torch.nn.ModuleList([
            AddRandomGaussianNoise(snr_range=(-10, 10)),
        ]), p=1.0),
    ])



def get_moderate_augmentation_transforms():
    return Compose([
        torchvision.transforms.RandomApply(torch.nn.ModuleList([
            torchvision.transforms.RandomAffine(0, translate=(0.4, 0.0)),
        ]), p=0.5),
    ])


def get_heavy_augmentation_transforms():
    return Compose([
        torchvision.transforms.RandomApply(torch.nn.ModuleList([
            torchaudio.transforms.FrequencyMasking(30),
        ]), p=0.25),
        torchvision.transforms.RandomApply(torch.nn.ModuleList([
            torchaudio.transforms.TimeMasking(75),
        ]), p=0.25),
        torchvision.transforms.RandomApply(torch.nn.ModuleList([
            torchvision.transforms.RandomAffine(0, translate=(0.4, 0.0)),
        ]), p=0.5),
    ])


