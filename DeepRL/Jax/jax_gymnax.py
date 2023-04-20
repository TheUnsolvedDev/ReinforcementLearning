

import jax
import jax.numpy as jnp
import gymnax

rng = jax.random.PRNGKey(0)
rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

# Instantiate the environment & its settings.
env, env_params = gymnax.make("Pendulum-v1")

vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

num_envs = 8
vmap_keys = jax.random.split(rng, num_envs)

obs, state = vmap_reset(vmap_keys, env_params)
n_obs, n_state, reward, done, _ = vmap_step(
    vmap_keys, state, jnp.zeros(num_envs), env_params)
print(n_obs)