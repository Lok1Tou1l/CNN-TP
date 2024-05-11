import torch

import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout = nn.Dropout(0.5)  # Dropout layer with 50% probability
        
        self.fc_input_size = 16 * 112 * 112  # Calculate the input size to the first fully connected layer
        
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)  # Batch normalization layer
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        
        out = out.view(out.size(0), -1)
        
        out = self.dropout(out)  # Apply dropout before the first fully connected layer
        
        out = self.fc1(out)
        out = self.bn1(out)  # Apply batch normalization before activation function
        out = self.relu3(out)
        out = self.fc2(out)
        
        return out
