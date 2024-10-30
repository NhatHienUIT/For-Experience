import torch
import torch.nn as nn
import torch.nn.functional as F
from models.robustifier import Robustifier

class Net(nn.Module):
    def __init__(self, num_classes=9): 
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)  
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
def classifier_path():
    return Net()

def robustifier_path(x_min, x_max, x_avg, x_std, x_epsilon_defense):
    convolutional_dnn = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(128, 3, 3, 1, 1) 
    )

    return Robustifier(x_min, x_max, x_avg, x_std, x_epsilon_defense, convolutional_dnn).cuda()

