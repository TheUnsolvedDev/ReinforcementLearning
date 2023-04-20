from agents.params import *
import numpy as np
import torch

from torchsummary import summary

device = DEVICE


class DQN_model(torch.nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN_model, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(conv_out_size, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


if __name__ == '__main__':
    dqn = DQN_model(input_shape, n_actions).to(device)
    target_dqn = DQN_model(input_shape, n_actions).to(device)
    summary(dqn, input_shape)
