import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
from jax.experimental import optimizers
import gymnax

rng = jax.random.PRNGKey(0)
rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

env, env_params = gymnax.make("Pendulum-v1")
vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
num_envs = 8
vmap_keys = jax.random.split(rng, num_envs)

obs, state = vmap_reset(vmap_keys, env_params)
n_obs, n_state, reward, done, _ = vmap_step(
    vmap_keys, state, jnp.zeros(num_envs), env_params)


def loss_fn(params, obs, act, reward):
    log_probs = policy_fn(params, obs)
    return -jnp.mean(log_probs * reward)


@jit
def policy_fn(obs_dim, act_dim):
    key = jax.random.PRNGKey(0)
    params = {
        'W': jax.random.normal(key, (act_dim, obs_dim)),
        'b': jnp.zeros(act_dim)
    }

    @jit
    def policy(params, obs, return_log_prob=False):
        logits = jnp.dot(params['W'], obs) + params['b']
        action_probs = jax.nn.softmax(logits)
        action = jax.random.categorical(jax.random.PRNGKey(0), logits)
        log_prob = jnp.log(action_probs[action])
        if return_log_prob:
            return action, log_prob
        else:
            return action

    return params, policy


@jit
def update(params, opt_state, obs, act, reward, opt_update, get_params):
    grads = grad(loss_fn)(params, obs, act, reward)
    updates, new_opt_state = opt_update(0, grads, opt_state)
    new_params = get_params(updates)
    return new_params, new_opt_state


def reinforce(env_fn, policy_fn, num_episodes=1000, learning_rate=0.01, gamma=0.99):
    opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    params = policy_fn.init(env.observation_space, env.action_space)

    episode_rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        # Collect episode
        obs_list = []
        act_list = []
        reward_list = []
        while not done:
            act, log_prob = policy_fn(params, obs, return_log_prob=True)
            obs_list.append(obs)
            act_list.append(act)
            reward, obs, done, info = env.step(act)
            reward_list.append(reward)
            episode_reward += reward

        # Update policy
        obs_arr = jnp.array(obs_list)
        act_arr = jnp.array(act_list)
        reward_arr = jnp.array(reward_list)
        params, opt_state = update(
            params, opt_state, obs_arr, act_arr, reward_arr, opt_update, get_params)

        # Print progress
        episode_rewards.append(episode_reward)
        if i % 10 == 0:
            print(
                f"Episode {i}: Avg. Reward = {jnp.mean(episode_rewards[-10:]):.2f}")

    return params
