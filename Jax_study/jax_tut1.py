import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time

from jax import grad, jit, vmap, pmap

from jax import lax
from jax import make_jaxpr 
from jax import random
from jax import device_put 

if __name__ == '__main__':
	# jax and numpy similarity
	x_np = np.linspace(0,10,1000)
	y_np = 2*np.sin(x_np)*np.cos(x_np)

	plt.plot(x_np, y_np)
	plt.show()

	x_jnp = jnp.linspace(0,10,1000)
	y_jnp = 2*jnp.sin(x_jnp)*np.cos(x_jnp)

	plt.plot(x_jnp, y_jnp)
	plt.show()

	#jax array immutable
	size = 10
	index = 0
	value = 23

	x = np.arange(size)
	print(x)
	x[index] = value
	print(x)

	x = jnp.arange(size)
	print(x)
	y = x.at[index].set(value)
	print(y)

	#random number geneartion
	seed = 0
	key = random.PRNGKey(seed)
	x = random.normal(key,(10,))
	print(type(x),x)

	#agnostic accelerator
	size = 10000

	x_jnp = random.normal(key,(size,size),dtype = jnp.float32)
	x_np = np.random.normal((size,size)).astype(np.float32)

	start = time.time()
	jnp.dot(x_jnp,x_jnp.T).block_until_ready()
	end = time.time()
	print(end - start)

	start = time.time()
	np.dot(x_np,x_np.T)
	end = time.time()
	print(end - start)

	start = time.time()
	jnp.dot(x_np,x_np.T).block_until_ready()
	end = time.time()
	print(end - start)
	
	