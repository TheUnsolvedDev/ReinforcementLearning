import numpy as np

a = np.arange(24).reshape(2, 12, 1)
print(a)
print(np.mean(a,axis = 1))