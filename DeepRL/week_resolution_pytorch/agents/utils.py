from agents.params import *
from agents.models import *

from collections import deque
import random

device = DEVICE

# Define replay memory


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(
            *random.sample(self.memory, batch_size))
        return torch.FloatTensor(states).to(device), torch.LongTensor(actions).to(device), torch.FloatTensor(rewards).to(device), torch.FloatTensor(next_states).to(device), torch.FloatTensor(dones).to(device)

    def __len__(self):
        return len(self.memory)


def save_model(model, optimizer, episode, save_path):
    torch.save({
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)


def load_model(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    episode = checkpoint['episode']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return episode
