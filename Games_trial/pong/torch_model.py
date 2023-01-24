import torch 
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CNN(torch.nn.module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(4,32,kernel_size=8,strides=4)
        self.conv2 = torch.nn.Conv2d(32,64,kernel_size=4,strides=2)
        self.conv3 = torch.nn.Conv2d(64,64,kernel_size=3,strides=1)
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten(
        )
        self.dens1 = torch.nn.Linear()

