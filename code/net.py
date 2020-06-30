import torch
import torch.nn as nn
import torch.nn.functional as F

#define neural net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.pool1 = nn.MaxPool1d(kernel_size=2, padding=1)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=5, padding=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels=10, out_channels=20, kernel_size=5, padding=3, stride=1)
        self.conv3 = nn.Conv1d(in_channels=20, out_channels=10, kernel_size=5, padding=3, stride=1)
        self.conv4 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=5, padding=3, stride=1)

        self.fc1 = nn.Linear(in_features=21, out_features=150)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(in_features=150, out_features=200)
        self.batch_norm = nn.BatchNorm1d(1)
        self.fc3 = nn.Linear(in_features=200, out_features=100)
        self.fc4 = nn.Linear(in_features=100, out_features=5)
        
        
    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool1(self.conv2(x))
        x = self.pool1(self.conv3(x))
        x = self.pool1(self.conv4(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.batch_norm(x)
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x
net = Net()
