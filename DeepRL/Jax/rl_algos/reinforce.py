import jax
import jax.numpy as jnp
import optax
import tqdm
import numpy as np
import gymnax
import gc
import os
import haiku as hk
from torch.utils.tensorboard import SummaryWriter
import pickle


rng = jax.random.PRNGKey(0)
rng, key_reset, key_act, key_step = jax.random.split(rng, 4)
ENV_ID = "CartPole-v1"  # "Acrobot-v1"

MAX_LENGTH = 1000
NUM_ENVS = 12
NUM_EPISODES = 2000
LEARNING_RATE = 1e-3
BATCH_SIZE = 128

env, env_params = gymnax.make(ENV_ID)
vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))


class PolicyModel(hk.Module):
    def __init__(self, num_hidden=32, num_outputs=2):
        super(PolicyModel, self).__init__()
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

    def __call__(self, inputs):
        dense1 = hk.Linear(self.num_hidden)(inputs)
        act1 = jax.nn.relu(dense1)
        dense2 = hk.Linear(self.num_hidden)(act1)
        act2 = jax.nn.relu(dense2)
        outputs = hk.Linear(self.num_outputs)(act2)
        return jax.nn.softmax(outputs)


def reinforce_loss(params, obs, actions, rewards):
    probs = model.apply(params, rng, obs)
    log_probs = jnp.log(probs+1e-6)
    actions_onehot = jax.nn.one_hot(actions, num_classes=len(actions[0]))
    action_log_probs = jnp.sum(log_probs * actions_onehot, axis=-1)
    return -jnp.mean(action_log_probs * rewards)


@hk.transform
def model(x):
    net = PolicyModel()
    return net(x)


def update(params, optimizer, opt_state, batch, grad_acc_steps):
    obs, actions, rewards = batch
    loss = 0.0
    for i in range(grad_acc_steps):
        subbatch = (obs[i::grad_acc_steps],
                    actions[i::grad_acc_steps], rewards[i::grad_acc_steps])
        subloss, grad = jax.value_and_grad(reinforce_loss)(params, *subbatch)
        loss += subloss
        if i == 0:
            avg_grad = grad
        else:
            avg_grad = jax.tree_map(
                lambda g1, g2: g1 + g2, avg_grad, grad)
    avg_grad = jax.tree_map(lambda g: g / grad_acc_steps, avg_grad)
    updates, opt_state = optimizer.update(avg_grad, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss / grad_acc_steps


def discount_rewards(rewards, gamma=0.99):
    discounted = np.zeros_like(rewards)
    running_sum = 0
    for i in reversed(range(len(rewards))):
        running_sum = running_sum * gamma + rewards[i]
        discounted[i] = running_sum
    return discounted


def normalise(rewards):
    return (rewards - np.mean(rewards, axis=1, keepdims=True)) / np.std(rewards, axis=1, keepdims=True) + 1e-6


def train(env_params, num_envs):
    input_shape = (4,)
    params = model.init(rng, jnp.zeros(input_shape))
    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = optimizer.init(params)

    log_dir = os.path.join(
        ENV_ID+"_logs_policy_gradient")
    writer = SummaryWriter(log_dir)

    averages = []
    for episode in tqdm.tqdm(range(NUM_EPISODES)):
        vmap_keys = jax.random.split(jax.random.PRNGKey(episode), num_envs)
        obs, state = vmap_reset(vmap_keys, env_params)
        dones = False
        states, actions, rewards = [], [], []
        for step in range(MAX_LENGTH):
            if dones:
                break
            action = jnp.argmax(model.apply(params, rng, obs), axis=1)
            n_obs, n_state, reward, dones, _ = vmap_step(
                vmap_keys, state, action, env_params)
            dones = jnp.all(dones)
            reward = jnp.array(reward, dtype=jnp.float32).reshape((-1, 1))

            states.append(obs)
            actions.append(action)
            rewards.append(reward)

            obs = n_obs
            state = n_state

        gc.collect()
        result = np.array(rewards).reshape((-1, NUM_ENVS))
        rewards = (np.array(rewards))
        discounted_rewards = (discount_rewards(rewards))
        rewards = rewards.reshape((-1, NUM_ENVS))

        print(rewards)
        input()
        print(np.sum(result, axis=0))
        input()

        policy_loss = 0
        grad_acc_steps = 1
        for batch_start in range(0, len(states), BATCH_SIZE):
            batch_end = batch_start + BATCH_SIZE
            batch_states = jnp.array(
                states[batch_start:batch_end]).reshape(-1, 4)
            batch_actions = jnp.array(
                actions[batch_start:batch_end]).reshape(-1, 1)
            batch_rewards = jnp.array(
                discounted_rewards[batch_start:batch_end]).reshape(-1, 1)

            params, opt_state, loss = update(
                params, optimizer, opt_state, (batch_states, batch_actions, batch_rewards), grad_acc_steps)
            policy_loss += loss

        averages.append(np.mean(result, axis=1))
        for i in range(NUM_ENVS):
            writer.add_scalar('1/episode_reward_'+str(i),
                              np.sum(result, axis=0)[i], episode)
            writer.add_scalar('2/average_reward_'+str(i), np.mean(
                averages[:-50])/NUM_ENVS, episode)
        writer.add_scalar('loss', float(policy_loss), episode)

        if episode % 50 == 0:
            print('Total Reward:', sum(rewards.reshape(-1,)/NUM_ENVS),
                  'Loss:', float(policy_loss))
            print(params)
            with open('weights/weights_policy_gradient_'+ENV_ID+"_"+str(episode)+".pkl", "wb") as f:
                pickle.dump(params, f)

# # Load the saved parameters from a file
# with open("my_variable.pkl", "rb") as f:
#     my_variable = pickle.load(f)

# # Print the loaded variable
# print(my_variable)


if __name__ == '__main__':
    train(env_params, num_envs=NUM_ENVS)
