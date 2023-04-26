#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import model_maker as mm
from train import load_checkpoint
import torch
from prep_transforms import get_preprocessing_transforms


# In[1]:


def RailNET():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    heav_18 = mm.create_modified_resnet18(output_classes = 2, input_channels = 1)
    med_34 = mm.create_modified_resnet34(output_classes = 2, input_channels = 1)
    heav_34 = mm.create_modified_resnet34(output_classes = 2, input_channels = 1)
    
    _, _ =load_checkpoint(heav_18, torch.optim.Adam(heav_18.parameters()),"..//models//ResNet18_heavy_final.pth")
    _, _ =load_checkpoint(med_34, torch.optim.Adam(med_34.parameters()),"..//models//ResNet34_medium_final.pth")
    _, _ =load_checkpoint(heav_34, torch.optim.Adam(heav_34.parameters()),"..//models//ResNet34_heavy_final.pth")
    
    RailNET = [heav_18, med_34, heav_34]
    
    for model in RailNET:
        model.to(device)
    
    return RailNET


