# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 16:54:49 2022

@author: DJ
"""
from torch import nn
import torch.nn.functional as F

class Classifier(nn.Module):
    
    def __init__(self, hidden_units):
        super().__init__()
        
        self.fc1 = nn.Linear(25088, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 1568)
        self.output = nn.Linear(1568, 102)

        # Dropout module with drop_out drop probability
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Flaten tensor input
        x = x.view(x.shape[0], -1)

        # Now with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))  

        # Output so no dropout here
        x = F.log_softmax(self.output(x), dim=1)

        return x