### Neural Network for MNIST Image Classification ###
import torch.nn as nn
import torch.nn.functional as F
import torch 
import numpy as np

class CNN(nn.Module):
    def __init__(self,device=torch.device("cuda")):
        super(CNN, self).__init__()
        device = torch.device("cuda")

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 10)
        self.device = device
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
    def forward(self, x):
        x = x.reshape(-1,1,28,28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    def loss(self,y_pred,y):
        ### y_pred: (batch_size,n_classes)
        ### y: (batch_size)
        return F.nll_loss(y_pred, y)        

    def train(self,x,y):
        for epoch in range(10):
            self.optimizer.zero_grad()
            output = self.forward(x)
            loss = self.loss(output,y)
            loss.backward()
            self.optimizer.step()
        return loss.item()
