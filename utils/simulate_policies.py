
### File to simulate different policies and compute the expected cost and number of communication rounds

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import components.client_oracle as client_oracle
import tqdm as tqdm
oracle = client_oracle.Oracle()
oracle_states = oracle.get_oracle_states()



### Simulating all same actions for all rounds
def greedy_policy(L,O,N_learners):
    return np.zeros((N_Learners,L,O),dtype=int)
def get_action_from(policy_parameters,learner_state):
    policy_parameters = policy_parameters.flatten()
    lower_thresholds = policy_parameters[0::3].flatten()
    upper_thresholds = policy_parameters[1::3].flatten()
    probabilities = np.sin(policy_parameters[2::3])**2
    for i in range(len(lower_thresholds)):
        if learner_state < lower_thresholds[i]:
            return i
        elif learner_state > lower_thresholds[i] and learner_state <= upper_thresholds[i]:
            return np.random.choice([i,i+1],p=[probabilities[i],1-probabilities[i]])
    return len(lower_thresholds)
def get_current_learner(learner_states,oracle_states):
    current_learner_min = 100
    current_learner = 0
    for i in range(len(learner_states)):
        if ((oracle_states[i]+1)*(learner_states[i]+1)<=0):
            print("Error",oracle_states,learner_states)
        if (1/((oracle_states[i]+1)*(learner_states[i]+1)))<current_learner_min:
            current_learner_min = (1/((oracle_states[i]+1)*(learner_states[i]+1)))
            current_learner = i
    return current_learner

policy_parameters_load = np.load("parameters/policy_parameters_spane_trained.npy").mean(axis=0)[-1,:,:,:]
### Greedy Scheduling, Random Action
spane_action = lambda l,o,n: get_action_from(policy_parameters_load[n,o],l)
greedy_action_0 = lambda l,o,n: 0
greedy_action_1 = lambda l,o,n: 1
greedy_action_2 = lambda l,o,n: 2
random_action = lambda l,o,n: np.random.choice([-1,0,1,2])

N_rounds = 80
N_Learners = 5

N_mc = 10
policies = [spane_action,greedy_action_0,greedy_action_1,greedy_action_2,random_action]

N_communication_rounds = np.zeros((len(policies),N_mc))
learning_cost = np.zeros((len(policies),N_mc))

learner_states_store = np.ones((N_mc,len(policies),N_rounds,N_Learners))*12
fig,axs = plt.subplots(len(policies),1,figsize = (10,40))
action_count = np.zeros((len(policies),4))
run_simulation = True
if run_simulation:
    for i in tqdm.tqdm(range(N_mc)):
        for p in range(len(policies)):
            learner_states = learner_states_store[i,p,:,:].copy()
            for round in np.arange(N_rounds-1):
                learner_states[round,learner_states[round]<=0] = 0
                if learner_states[round].sum()<=0:
                    print(learner_states[round])
                    learner_states[round:,:] = 0
                    break
                oracle_states = oracle.get_oracle_states()
                learner_states[round+1] = learner_states[round]

                current_learner = get_current_learner(learner_states[round],oracle_states)
                action = policies[p](learner_states[round,current_learner],oracle_states[current_learner],current_learner)
                if learner_states[round,current_learner] <= 0:
                    learner_states_store[i,p,round+1:,current_learner] = 0
                    continue
                    
                action_count[p,action] += 1

                oracle.update_client_selection_matrix(action)
                round_success = oracle.get_oracle_success(current_learner)
                learner_states[round+1,current_learner] -= round_success
            learner_states_store[i,p,:,:] = learner_states.copy()
    np.save("parameters/learner_states.npy",learner_states_store)
    np.save("parameters/action_count.npy",action_count)
            # print(learner_states_store[i,p,-1,:])

learner_states_store = np.load("parameters/learner_states.npy")
action_count = np.load("parameters/action_count.npy")
print(action_count)
for p in range(len(policies)):
    axs[p].plot(np.arange(N_rounds),learner_states_store[:,p,:,:].mean(axis=0))
    axs[p].set_title("Action = "+str(p))
plt.savefig("plots/learner_states.png")



### Random Scheduling




