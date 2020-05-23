import numpy as np
import matplotlib.pyplot as plt

# an elementary cellular automata,
# following Wolfram's paper
# "Cellular Automata as Simle Self Organising Systems"
# except the rule is changed at each step to the state
# of the previous step

size = 8
n = 100

# initialize state

state = np.zeros(size)
all_states = np.zeros((n, size))

rule = state

# update state according to rule

# separate state 
