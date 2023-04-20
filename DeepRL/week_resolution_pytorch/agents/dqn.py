import torch
import gc
import os
import tqdm
from torch.utils.tensorboard import SummaryWriter

from agents.params import *
from agents.models import *
from agents.utils import *

SAVE_PATH = 'weights/dqn_breakout.pth'
LOAD_PATH = 'weights/dqn_breakout.pth'

dqn = DQN_model(input_shape, n_actions).to(device)
target_dqn = DQN_model(input_shape, n_actions).to(device)
target_dqn.load_state_dict(dqn.state_dict())
target_dqn.eval()


def select_action(state, eps):
    if random.random() < eps:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            return dqn(state).argmax().item()


def DQN(env):
    # Define optimizer and criterion
    optimizer = torch.optim.Adam(dqn.parameters(), lr=ALPHA)
    criterion = torch.nn.SmoothL1Loss()
    memory = ReplayMemory(MEMORY_CAPACITY)

    log_dir = os.path.join(
        "logs/logs_DQN", env.unwrapped.spec.id+'_events')
    writer = SummaryWriter(log_dir)

    episode_rewards = []
    state = env.reset()[0]
    done = truncated = False

    for t in tqdm.tqdm(range(1, NUM_EPISODES+1)):
        gc.collect()
        eps = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) *
                  min(t / EPSILON_DECAY, 1))
        episode_rewards.append(0)
        while not (done or truncated):
            action = select_action(torch.FloatTensor(
                state).unsqueeze(0).to(device), eps)
            next_state, reward, done, truncated, info = env.step(action)
            memory.push(state, action, reward, next_state, done)
            state = next_state
            episode_rewards[-1] += reward

            if len(memory) > BATCH_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(
                    BATCH_SIZE)
                q_values = dqn(states).gather(
                    1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_dqn(next_states).max(1)[0].detach()
                expected_q_values = rewards + GAMMA * \
                    next_q_values * (1 - dones)
                loss = criterion(q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        writer.add_scalar('episode_reward', episode_rewards[-1], t)
        writer.add_scalar('average_reward', np.mean(episode_rewards[:-50]), t)
        writer.add_scalar('epsilon', eps, t)
        writer.add_scalar('loss', loss.item(), t)

        if t % SAVE_FREQUENCY == 0:
            save_model(dqn, optimizer, t, SAVE_PATH)

        if t % TARGET_UPDATE_FREQUENCY == 0:
            target_dqn.load_state_dict(dqn.state_dict())

        if done:
            state = env.reset()[0]
            episode_rewards.append(0)

        if t % 100 == 0:
            print('Timestep: {}, Mean Episode Reward: {:.2f}'.format(
                t, np.mean(episode_rewards[:-50])))
