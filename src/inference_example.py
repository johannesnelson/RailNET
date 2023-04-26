#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ensemble_inference import EnsembleInference
import model_maker as mm
from train import load_checkpoint
from load_RailNET import RailNET
import torch
from prep_transforms import get_preprocessing_transforms


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[4]:


preprocessing_transforms = get_preprocessing_transforms()


# In[2]:


RailNET = RailNET()


# In[5]:


inference_call = EnsembleInference(RailNET, preprocessing_transforms, threshold=0.5)


# In[6]:





# In[8]:


inference_call.process_folder('..//data//sample_audio', '..//data//dample_audio//output')

