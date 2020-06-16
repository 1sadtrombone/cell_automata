import numpy as np
import matplotlib.pyplot as plt

# an elementary cellular automata,
# following Wolfram's paper
# "Cellular Automata as Simple Self-Organizing Systems"
# except the rule is changed at each step to the state
# of the previous step

def combinations(n, k):
    # gives all combinations of n integers from 0 to k
    if n != int(n):
        raise ValueError("combinations encountered a non-int number of integers")
    if n < 1:
        raise ValueError("combinations encountered a negative number of integers")
    if k != int(k):
        raise ValueError("combinations encountered a non-int parameter k")
    if n == 1:
        ps = np.arange(k).reshape((-1,1))
        return ps
    else:
        subarrs = np.tile(combinations(n-1, k), (k,1))
        ps = np.arange(k)
        ps = np.repeat(ps,subarrs.shape[0]//k).reshape((-1,1))

        return np.hstack((ps, subarrs))

def apply_rule_to_group(group, pattern, rule):
    if pattern.shape[0] != rule.shape[0]:
        raise ValueError("apply rule encountered length of pattern != length of rule")
    return rule[np.where(np.prod(pattern==group,axis=1))]

apply_rule = np.vectorize(apply_rule_to_group, excluded=["pattern","rule"], signature="(n)->()")

number_of_types = 4
group_size = 2
size = number_of_types**group_size

n = 50

# flipped to match Wolfram's paper
pattern = np.flip(combinations(group_size, number_of_types), axis=0)

# initialize state
seed_size = 10
initial_state = np.hstack((np.zeros((size-seed_size)//2), np.random.randint(number_of_types, size=seed_size), np.zeros((size-seed_size)//2)))
#initial_state = np.random.randint(number_of_types, size=size)
#initial_state = np.array([0,2,0,2,0,2,1,0,0])
initial_state[-1] = 0 # to ensure "legality"

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


cmap = 'tab10'

plt.subplot(1,2,1)
plt.title("Rule is State")
plt.imshow(all_states, cmap=cmap, aspect='auto')
plt.subplot(1,2,2)
plt.title("Constant Rule")
plt.imshow(all_states_classic, cmap=cmap, aspect='auto')
plt.show()
