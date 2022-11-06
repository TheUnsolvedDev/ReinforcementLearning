from kaggle_environments import evaluate, make, utils
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from dqn_nn_model import DeepModel, DQN
from environment import ConnectX
from params import *

env = make('connectx', debug=True)
env.reset()


def Plot(data, title):
    plt.figure()
    plt.plot(data)
    plt.xlabel('Episode')
    plt.ylabel(title)
    plt.savefig(title+'.png')
    plt.close()


def play_game(env, TrainNet, TargetNet, epsilon, copy_step, Global_Step_Counter):
    turns = 0
    env.reset()
    Metric_Buffer = {key: [] for key in Metric_Titles}
    while not env.game_over():
        active = env.current_player()

        observations = env.get_state()[active].observation
        action, Q_Values = TrainNet.get_action(observations, epsilon)
        Q_Values = [val for val in Q_Values if val != Discard_Q_Value]
        Metric_Buffer['Avg_Q'].append(np.mean(Q_Values))
        Metric_Buffer['Max_Q'].append(np.max(Q_Values))
        Metric_Buffer['Min_Q'].append(np.min(Q_Values))

        prev_observations = observations
        env.step([action if i == active else None for i in [0, 1]])
        reward = env.get_state()[active].reward

        # Convert environment's [0,0.5,1] reward scheme to [-1,1]
        if env.game_over():
            if reward == 1:  # Won
                reward = 1
            elif reward == 0:  # Lost
                reward = -1
            else:  # Draw
                reward = 0
        else:
            reward = 0

        next_active = 1 if active == 0 else 0

        observations = env.get_state()[next_active].observation
        exp = {'inputs': TrainNet.preprocess(prev_observations), 'a': action, 'r': reward, 'inputs2': TrainNet.preprocess(
            observations), 'done': env.game_over()}
        TrainNet.add_experience(exp)

        turns += 1
        total_turns = Global_Step_Counter+turns
        # Train the training model by using experiences in buffer and the target model
        if total_turns % Steps_Till_Backprop == 0:
            TrainNet.train(TargetNet)
        if total_turns % copy_step == 0:
            # Update the weights of the target model when reaching enough "copy step"
            TargetNet.copy_weights(TrainNet)
    results = {key: [] for key in Metric_Titles}
    for metric_name in Metric_Titles:
        results[metric_name] = np.mean(Metric_Buffer[metric_name])
    return results, turns


def train():
    env = ConnectX()

    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Here we will store metrics for plotting after training
    Metrics = {key: [] for key in Metric_Titles}
    Metrics_Buffer = {key: [] for key in Metric_Titles}  # Downsampling buffer

    # Initialize models
    TrainNet = DQN(num_states, num_actions, hidden_units, gamma,
                   max_experiences, min_experiences, batch_size, lr)
    TargetNet = DQN(num_states, num_actions, hidden_units, gamma,
                    max_experiences, min_experiences, batch_size, lr)

    if Train:
        Global_Step_Counter = 0
        pbar = tqdm(range(episodes))
        pbar2 = tqdm()
        epsilon = epsilon_ep
        for n in pbar:
            epsilon = max(min_epsilon, epsilon * decay)
            results, steps = play_game(
                env, TrainNet, TargetNet, epsilon, copy_step, Global_Step_Counter)
            Global_Step_Counter += steps
            for metric_name in Metric_Titles:
                Metrics_Buffer[metric_name].append(results[metric_name])

            if Global_Step_Counter % N_Downsampling_Episodes == 0:
                for metric_name in Metric_Titles:  # Downsample our metrics from the buffer
                    Metrics[metric_name].append(
                        np.mean(Metrics_Buffer[metric_name]))
                    Metrics_Buffer[metric_name].clear()
                pbar.set_postfix({
                    'Steps': Global_Step_Counter,
                    'Updates': Global_Step_Counter*batch_size/Steps_Till_Backprop
                })

                pbar2.set_postfix({
                    'max_Q': Metrics['Max_Q'][-1],
                    'avg_Q': Metrics['Avg_Q'][-1],
                    'min_Q': Metrics['Min_Q'][-1],
                    'epsilon': epsilon,
                    'turns': steps
                })

        for metric_name in Metric_Titles:
            Plot(Metrics[metric_name], metric_name)

        TrainNet.save_weights('./weights.h5')
    else:
        TrainNet.load_weights('./weights.h5')


if __name__ == '__main__':
    env.run(['random', 'random'])
    env.render(mode='human')

    train()
