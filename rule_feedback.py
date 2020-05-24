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
        raise ValueError("permutations encountered a non-int number of bits")
    if n < 1:
        raise ValueError("permutations encountered a negative number of bits")
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

size = 2**3
group_size = int(np.log2(size))
if group_size != np.log2(size):
    raise ValueError("size must be a power of 2")
n = 100

# flipped to match Wolfram's paper
pattern = np.flip(permutations(group_size), axis=0)

# initialize state
initial_state = np.array([0,0,0,1,1,1,1,0])

state = initial_state
all_states = np.zeros((n, size))

rule = state

# update state according to rule

# possible to apply rules all in one go?
# does the impossibility imply something (complexity?)
for i in range(n):
    
    # save the current state
    all_states[i] = state

    # separate state into groups of three (periodic boundaries)
    # [periodic bounds a problem? too small for structure?]
    # TODO: optimize if possible
    repeated_states = np.hstack((state[state.size-group_size//2:],state,state[:group_size//2]))
    groups = np.array([repeated_states[i:i+group_size] for i in range(state.size)])
    
    # apply rule to each group, according to pattern
    # TODO: get rid of that transpose
    state = apply_rule(groups, pattern=pattern, rule=rule).T

    rule = state

# classic cellular automaton for comparison
all_states_classic = np.zeros((n,size))
state_classic = initial_state
rule_classic = np.copy(state_classic)

for i in range(n):

    all_states_classic[i] = state_classic

    repeated_states = np.hstack((state_classic[state_classic.size-group_size//2:],state_classic,state_classic[:group_size//2]))
    groups = np.array([repeated_states[i:i+group_size] for i in range(state_classic.size)])

    state_classic = apply_rule(groups, pattern=pattern, rule=rule_classic).T



plt.subplot(1,2,1)
plt.title("Rule depends on state")
plt.imshow(all_states)
plt.subplot(1,2,2)
plt.title("Classic cellular automaton")
plt.imshow(all_states_classic)
plt.show()
