!/bin/bash
python3 DQN_Discrete.py --env_name "CartPole-v0"
python3 DQN_Discrete.py --env_name "CartPole-v1"
python3 DQN_Discrete.py --env_name "MountainCar-v0"
python3 DQN_Discrete.py --env_name "Acrobot-v1"
python3 DQN_Discrete.py --env_name "LunarLander-v2"

python3 DoubleDQN_Discrete.py --env_name "CartPole-v0"
python3 DoubleDQN_Discrete.py --env_name "CartPole-v1"
python3 DoubleDQN_Discrete.py --env_name "MountainCar-v0"
python3 DoubleDQN_Discrete.py --env_name "Acrobot-v1"
python3 DoubleDQN_Discrete.py --env_name "LunarLander-v2"

python3 DuelingDQN_Discrete.py --env_name "CartPole-v0"
python3 DuelingDQN_Discrete.py --env_name "CartPole-v1"
python3 DuelingDQN_Discrete.py --env_name "MountainCar-v0"
python3 DuelingDQN_Discrete.py --env_name "Acrobot-v1"
python3 DuelingDQN_Discrete.py --env_name "LunarLander-v2"

python3 DuelingDoubleDQN_Discrete.py --env_name "CartPole-v0"
python3 DuelingDoubleDQN_Discrete.py --env_name "CartPole-v1"
python3 DuelingDoubleDQN_Discrete.py --env_name "MountainCar-v0"
python3 DuelingDoubleDQN_Discrete.py --env_name "Acrobot-v1"
python3 DuelingDoubleDQN_Discrete.py --env_name "LunarLander-v2"