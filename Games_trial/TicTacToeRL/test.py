import numpy as np

from agents import load_model
from params import *

Q = load_model("O"+MODEL_FILE)
for state,values in Q.items():
    state = np.array(state).reshape((3,3))
    values = np.array(values).reshape((3,3))
    
    print(state)
    print(values)
    input()