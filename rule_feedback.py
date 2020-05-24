import numpy as np
import matplotlib.pyplot as plt

# an elementary cellular automata,
# following Wolfram's paper
# "Cellular Automata as Simple Self-Organizing Systems"
# except the rule is changed at each step to the state
# of the previous step

def permutations(n):
    # gives all pemutations of n bits
    if n != int(n):
        raise(ValueError, "permutations encountered a non-int number of bits")
    if n < 1:
        raise(ValueError, "permutations encountered a negative number of bits")
    if n == 1:
        ps = np.zeros((2,1))
        ps[1] = 1
        return ps
    else:
        subarrs = np.vstack((permutations(n-1),permutations(n-1)))
        ps = np.zeros((subarrs.shape[0],1))
        ps[ps.size//2:] = 1
        return np.hstack((ps, subarrs))

def apply_rule_to_group(group, pattern, rule):
    return rule[np.where(np.prod(pattern==group,axis=1))]

apply_rule = np.vectorize(apply_rule_to_group, excluded=["pattern","rule"], signature="(n)->()")

size = 8
group_size = int(np.log2(size))
if group_size != np.log2(size):
    raise(ValueError,"size must be a power of 2")
n = 100

# flipped to match Wolfram's paper
pattern = np.flip(permutations(group_size), axis=0)

# initialize state

state = np.zeros(size)
all_states = np.zeros((n, size))

state[1] = 1

# update state according to rule

# separate state into groups of three (periodic boundaries)
# [periodic bounds a problem? too small for structure?]

repeated_states = np.hstack((state[state.size-1:],state,state[:1]))

# TODO: optimize if possible
groups = np.array([repeated_states[i:i+group_size] for i in range(state.size)])

# apply rule to each group, according to pattern
next_state = apply_rule(groups, pattern=pattern, rule=state).T


