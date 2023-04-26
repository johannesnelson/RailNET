#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, Wide_ResNet50_2_Weights


# In[ ]:


def create_modified_resnet18(output_classes, input_channels=1):
    model = models.resnet18(weights = ResNet18_Weights.DEFAULT)

    # Change the output layer to match the desired number of output classes
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, output_classes)

    # Change the first convolution layer to match the desired number of input channels
    model.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    return model

def create_modified_resnet34(output_classes,input_channels=1):
    model = models.resnet34(weights = ResNet34_Weights.DEFAULT)

    # Change the output layer to match the desired number of output classes
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, output_classes)

    # Change the first convolution layer to match the desired number of input channels
    model.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    return model


def create_modified_widenet(output_classes,input_channels=1):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2',weights=Wide_ResNet50_2_Weights.DEFAULT)


    # Change the output layer to match the desired number of output classes
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, output_classes)

    # Change the first convolution layer to match the desired number of input channels
    model.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    return model



class ResNet18WithAttention(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18WithAttention, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)  
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=8)

        # Replace the first convolutional layer to accept grayscale input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Replace the last fully connected layer to match the number of classes
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Initial layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # Residual blocks
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)

        # Self-attention after the second residual block
        batch_size, C, H, W = x.size()
        reshaped_feature_maps = x.permute(0, 2, 3, 1).reshape(batch_size, H * W, C)
        attn_output, _ = self.attention(reshaped_feature_maps, reshaped_feature_maps, reshaped_feature_maps)
        x = attn_output.view(batch_size, H, W, C).permute(0, 3, 1, 2)

        # Remaining residual blocks and average pooling
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)

        # Fully connected layer
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)

        return x
    


class BaseCNN(nn.Module):
    """
    A simple CNN with 2 convolutional layers and 2 fully-connected layers.
    """

    def __init__(self, num_channels: int = 3):
        super(BaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 6, 5) #124x252
        self.pool = nn.MaxPool2d(2, 2) #62x126
        self.conv2 = nn.Conv2d(6, 16, 5) #58x 122
        self.fc1 = nn.Linear(16 * 29 * 61, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 29 * 61)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


