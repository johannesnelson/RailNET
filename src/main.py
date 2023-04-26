#!/usr/bin/env python
# coding: utf-8

# In[1]:


from custom_dataset import Virginia_Rail_Dataset
from train_utils import perform_k_fold_cross_validation
import prep_transforms as pt
import model_maker as mm
import torch
from visualizations import plot_random_spectrogram
from train import train


# In[2]:


preprocessing_transforms, augmentation_transforms, augmentation_noise_transforms = pt.prep_all_transforms(aug_level = 'moderate')


# In[3]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[4]:


train_dataset = Virginia_Rail_Dataset(root_dir='', 
                                      positive_folder='../data/positive_class', 
                                      negative_folder='../data/negative_class', 
                                      preprocessing_transforms=preprocessing_transforms, 
                                      augmentation_transforms=augmentation_transforms,
                                      augmentation_noise_transforms=augmentation_noise_transforms,
                                      device=device,
                                    )
test_dataset = Virginia_Rail_Dataset(root_dir='', 
                                     positive_folder='../data/positive_class', 
                                     negative_folder='../data/negative_class', 
                                     preprocessing_transforms=preprocessing_transforms, 
                                     augmentation_transforms=None,
                                     augmentation_noise_transforms=None,
                                     device=device,
                                 )



# In[5]:


k_folds = 5
num_epochs = 60
criterion = torch.nn.CrossEntropyLoss()
starting_LR = 0.001
seed = 44


# In[9]:


all_train_losses, all_test_losses, all_train_accs, all_test_accs, all_train_f1s, all_test_f1s, all_train_recalls, all_test_recalls = perform_k_fold_cross_validation(
    k_folds=k_folds,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    create_model_func=lambda: mm.create_modified_resnet18(2),
    train_func=train,
    num_epochs=num_epochs,
    criterion=criterion,
    starting_LR = starting_LR,
    checkpoint_folder = 'desired_folder_path_here',
    device=device,
    batch_size=16,
    seed=seed
)


# In[ ]:




