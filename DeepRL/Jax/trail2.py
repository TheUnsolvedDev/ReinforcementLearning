import numpy as np
from jax import numpy as jnp, jit
import jax
from jax.nn import relu, log_softmax
from jax.random import PRNGKey
import gymnasium as gym
import haiku as hk
import optax

rng = jax.random.PRNGKey(0)

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


@hk.transform
def model(x):
    net = PolicyModel()
    return net(x)


def sample_categorical(key, logits, axis=-1):
    return jnp.argmax(random.gumbel(key, logits.shape, logits.dtype) + logits, axis=axis)


def main(batch_size=256, env_name="CartPole-v1"):
    env = gym.make(env_name)
    params = model.init(rng, jnp.zeros(env.observation_space.n))
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)

    @jit
    def loss(observations, actions, rewards_to_go):
        logprobs = log_softmax(model.apply(observations))
        action_logprobs = logprobs[jnp.arange(logprobs.shape[0]), actions]
        return -jnp.mean(action_logprobs * rewards_to_go, axis=0)


    shaped_loss = loss.shaped(jnp.zeros((1,) + env.observation_space.shape),
                              jnp.array([0]), jnp.array([0]))

    @jit
    def sample_action(state, key, observation):
        loss_params = opt.get_parameters(state)
        logits = policy.apply_from({shaped_loss: loss_params}, observation)
        return sample_categorical(key, logits)

    rng_init, rng = jax.random.split(rng)
    returns, observations, actions, rewards_to_go = [], [], [], []

    for i in range(250):
        while len(observations) < batch_size:
            observation = env.reset()
            episode_done = False
            rewards = []

            while not episode_done:
                rng_step, rng = jax.random.split(rng)
                action = sample_action(state, rng_step, observation)
                observations.append(observation)
                actions.append(action)

                observation, reward, episode_done, info = env.step(int(action))
                rewards.append(reward)

            returns.append(np.sum(rewards))
            rewards_to_go += list(np.flip(np.cumsum(np.flip(rewards))))

        print(f'Batch {i}, recent mean return: {np.mean(returns[-100:]):.1f}')

        state = opt.update(loss.apply, state, jnp.array(observations[:batch_size]),
                           jnp.array(actions[:batch_size]), jnp.array(
                               rewards_to_go[:batch_size]),
                           jit=True)

        observations = observations[batch_size:]
        actions = actions[batch_size:]
        rewards_to_go = rewards_to_go[batch_size:]


if __name__ == '__main__':
    main()
