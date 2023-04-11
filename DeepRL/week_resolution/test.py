import os
import multiprocessing

envs = ['CartPole-v1', 'Acrobot-v1']
models = ['reinforce', 'reinforce_baseline', 'DQN', 'ActorCritic']

commands = []
for env in envs:
    for model in models:
        commands.append('python train.py -t '+model+' -env '+env)

with multiprocessing.Pool(processes=8) as pool:
    pool.map(os.system, commands)
