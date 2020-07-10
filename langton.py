import numpy as np
import matplotlib.pyplot as plt

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

def run_cell_automaton(init_state, rule, number_of_types, n):
    # rule and init state -> cell automaton run for n steps

    # infer group size
    group_size = int(np.round(np.log(rule.size) / np.log(number_of_types)))
    
    pattern = np.flip(combinations(group_size, number_of_types), axis=0)

    state = init_state

    all_states = np.zeros((n,init_state.size))

    for i in range(n):

        all_states[i] = state

        repeated_states = np.hstack((state[state.size-group_size//2:],state,state[:group_size//2]))
        groups = np.array([repeated_states[i:i+group_size] for i in range(state.size)])
        
        state = apply_rule(groups, pattern=pattern, rule=rule).T

    return all_states

def random_rule(rng, number_of_types, lamb=None):
    # generate random rule with neighbourhoods of size 2*rng+1, and number_of_types cell types
    # if lamb specified, make a rule with that lambda

    if lamb is None:
        rule = np.random.randint(number_of_types, size=number_of_types**(2*rng+1))

    else:
        # create rule where none map to dead
        rule = np.random.randint(1,number_of_types, size=number_of_types**(2*rng+1))
        dead_transitions = np.random.choice([1,0],rule.size,p=[1-lamb,lamb])
        rule[np.where(dead_transitions)] = 0
        
    return rule

def perturb_rule(rule, number_of_types, more_chaotic=False):
    # make a rule less (or more) chaotic by changing one outcome
    # assumes 0 is the quiescent (dead) state
    
    if more_chaotic:
        valid_inds = np.where(rule == 0)[0]
        ind = np.random.choice(valid_inds)
        rule[ind] = np.random.randint(number_of_types) + 1 #set to non-quiescent state
        
    else:
        valid_inds = np.where(rule != 0)[0]
        ind = np.random.choice(valid_inds)
        rule[ind] = 0

    return rule

def get_lambda(rule):
    # calculate a rule's lambda value, as specified by Langton

    return np.sum(rule != 0) / rule.size
        

def entropy(timeseries, number_of_types):
    # time series of single cell -> entropy
    # assumes types go 0, 1 ... k

    entropy = 0
    for i in range(number_of_types):
        p = np.sum(timeseries == i) / timeseries.size
        if p > 0:
            entropy -= p * np.log2(p)

    return entropy

vect_entropy = np.vectorize(entropy, excluded=["number_of_types"], signature="(n)->()")
    
def mutual_info(timeseries_1, timeseries_2, number_of_types):
    # two time series -> mutual information

    if timeseries_1.size != timeseries_2.size:
        raise ValueError('mutual information encountered timeseries of different sizes')

    mutual = 0
    for i in range(number_of_types):
        for j in range(number_of_types):
            p = np.sum((timeseries_1 == i) * (timeseries_2 == j)) / timeseries_1.size
            if p > 0:
                mutual += p * np.log2(p)
    mutual += entropy(timeseries_1, number_of_types) + entropy(timeseries_2, number_of_types)
    return mutual

vect_mutual_info = np.vectorize(mutual_info, excluded=["number_of_types"], signature="(n),(m)->()")

def avg_entropy(all_states, number_of_types):
    
    entropies = vect_entropy(all_states.T, number_of_types=number_of_types)

    return np.mean(entropies)

def avg_mutual_info(all_states, number_of_types):

    infos = vect_mutual_info(all_states.T, np.roll(all_states,1).T, number_of_types=number_of_types)

    return np.mean(infos)


def show_cells(all_states, cmap='tab10'):

    plt.imshow(all_states, cmap=cmap)
    plt.show()

if __name__=='__main__':

    plot_dir = 'plots'
    name = 'final_run'
    plot_prefix = f'{plot_dir}/{name}'

    N = 30 # number of runs
    n = 500 # time steps per run
    size = 128 # system size

    k = 4
    rng = 2
    max_lamb = 1 - 1/k

    init = np.random.randint(k, size=size)
    
    entropies = np.zeros(N)
    mutual_infos = np.zeros(N)
    lambs = np.zeros(N)
    
    for i in range(N):

        lamb = np.random.uniform(0,max_lamb)
        rule = random_rule(rng, k, lamb=lamb)

        cells = run_cell_automaton(init, rule, k, n)

        entropies[i] = avg_entropy(cells, k)
        mutual_infos[i] = avg_mutual_info(cells, k)
        lambs[i] = lamb
        
        if mutual_infos[i] > 0.5:
            print(i, "high news")
            plt.imshow(cells, cmap='tab10')
            plt.title(f'$\lambda$:{lamb} \nentropy:{entropies[i]} \nmutual info:{mutual_infos[i]}')
            plt.tight_layout()
            plt.savefig(f'{plot_prefix}_highMI_example_{i}')

        if entropies[i] < 0.2 and entropies[i] > 0.1:
            print(i, "low H")
            plt.imshow(cells, cmap='tab10')
            plt.title(f'$\lambda$:{lamb} \nentropy:{entropies[i]} \nmutual info:{mutual_infos[i]}')
            plt.tight_layout()
            plt.savefig(f'{plot_prefix}_lowH_example_{i}')

        if entropies[i] > 1:
            print(i, "high H")
            plt.imshow(cells, cmap='tab10')
            plt.title(f'$\lambda$:{lamb} \nentropy:{entropies[i]} \nmutual info:{mutual_infos[i]}')
            plt.tight_layout()
            plt.savefig(f'{plot_prefix}_highH_example_{i}')


            

    plt.plot(lambs, entropies, 'k.')
    plt.savefig(f'{plot_prefix}_avg_entropy')
    plt.clf()
    
    plt.plot(lambs, mutual_infos, 'k.')
    plt.savefig(f'{plot_prefix}_avg_mutual_info')
    plt.clf()

    plt.plot(entropies/np.max(entropies), mutual_infos, 'k.')
    plt.savefig(f'{plot_prefix}_the_plot')
    plt.xlabel('Average Entropy')
    plt.ylabel('Average News')
    plt.clf()

    plt.plot(entropies, mutual_infos, 'k.')
    plt.xlabel('Average Entropy')
    plt.ylabel('Average News')
    plt.savefig(f'{plot_prefix}_the_plot_unnormed')
    plt.clf()
    
    
    '''
    k = 3
    #rule = np.zeros(3**k)
    rule = random_rule(2,k,0.6)
    
    init = np.random.randint(k,size=128)
    test_cells = run_cell_automaton(init, rule, k, 100)

    show_cells(test_cells)
    '''