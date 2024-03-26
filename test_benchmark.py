import numpy as np
import matplotlib.pyplot as plt
from utils.funcs import get_action_from, get_current_learner, retrieve_cost, get_current_learner_random
from components.client_oracle import Oracle
from components.net import CNN
import tqdm as tqdm
#### Test spane algorithm against greedy and random scheduling
#### Two key benchmarks: number of queries posed by each learner and the average number of classes from learner preference in the round that the learner learns in

### Policies 
# policy_parameters_load = np.load("parameters/policy_parameters_spane_trained.npy").mean(axis=0)[-1,:,:,:]
## Synthetic policy parameters
N = 5
O = 5
U = 3
policy_parameters_load = np.zeros((N,O,3*(U-1)))
policy_parameters_load[:,:,0::3] = np.array([[15,20],[10,15],[5,10],[0,5],[0,0]])
policy_parameters_load[:,:,1::3] = np.array([[20,25],[15,20],[10,15],[5,10],[0,5]])
policy_parameters_load[:,:,2::3] =  np.array([[1],[1],[1],[1],[1]])
print(policy_parameters_load)
### Greedy Scheduling, Random Action
spane_action = lambda l,o,n: get_action_from(policy_parameters_load[n,o],l)
greedy_action_0 = lambda l,o,n: 0
greedy_action_1 = lambda l,o,n: 1
greedy_action_2 = lambda l,o,n: 2
random_action = lambda l,o,n: np.random.choice([0,1,2])

N_rounds = 2000
N_Learners = 5
N_classes = 10

N_mc = 100
policies = [spane_action,greedy_action_0,greedy_action_1,greedy_action_2,random_action,random_action]
scheduling = [get_current_learner, get_current_learner_random, get_current_learner_random, get_current_learner_random, get_current_learner_random, get_current_learner]
N_communication_rounds = np.zeros((len(policies),N_mc))

learner_states_store = np.ones((N_mc,len(policies),N_rounds,N_Learners))*200
class_counts = np.zeros((len(policies),N_mc,N_rounds,N_Learners,2))
round_participation = np.zeros((len(policies),N_mc,N_rounds,N_Learners))
action_rounds = np.zeros((len(policies),N_mc,N_rounds,N_Learners)) 
num_rounds = np.zeros((len(policies),N_mc,N_Learners))

PERFORM_EXP = 1

if PERFORM_EXP:
    for mc_round in tqdm.tqdm(range(N_mc)):
        oracle = Oracle()
        oracle.initialize_clients(CNN)
        for policy in range(len(policies)):
            learner_done = np.zeros((N_Learners))
            for round in np.arange(N_rounds-1):
                if learner_done.sum() == N_Learners:
                    break
                current_round_learner_states = learner_states_store[mc_round,policy,round,:].copy()
                oracle_states = oracle.get_oracle_states("class")
                
                current_learner = scheduling[policy](current_round_learner_states,oracle_states)
                if learner_done[current_learner] == 1:
                    continue
                num_rounds[policy,mc_round,current_learner] += 1

                action = policies[policy](current_round_learner_states[current_learner],oracle_states[current_learner],current_learner)
                action_rounds[policy,mc_round,round,current_learner] = action
                oracle.update_client_selection_matrix(action,oracle.learner_class_preference[current_learner])
                round_success = oracle.get_oracle_success(current_learner,"class")
                if policy != 4 and policy != 5:
                    current_round_learner_states[current_learner] -= round_success
                else:
                    current_round_learner_states[current_learner] -= 1
                if current_round_learner_states[current_learner] == 0:
                    learner_done[current_learner] = 1
                current_round_learner_states[current_round_learner_states<0] = 0
                learner_states_store[mc_round,policy,round+1,:] = current_round_learner_states.copy()
                class_counts[policy,mc_round,round+1,current_learner,:] = oracle.class_dist[oracle.learner_class_preference[current_learner]]
                round_participation[policy,mc_round,round+1,current_learner] = 1
    np.save("parameters/learner_states.npy",learner_states_store)
    np.save("parameters/class_counts.npy",class_counts)
    np.save("parameters/round_participation.npy",round_participation)
    np.save("parameters/action_rounds.npy",action_rounds)

action_rounds = np.load("parameters/action_rounds.npy")
learner_states_store = np.load("parameters/learner_states.npy")
class_counts = np.load("parameters/class_counts.npy")
round_participation = np.load("parameters/round_participation.npy")


print(class_counts.shape)
## For each mc round, for each learner take average of only the rounds that the learner participated in
class_counts_mean = np.zeros((len(policies),N_mc,N_Learners,2))
action_rounds_mean = np.zeros((len(policies),N_mc,N_Learners))
learner_states_mean = np.zeros((len(policies),N_mc,N_Learners))
for policy in range(len(policies)):
    for mc_round in range(N_mc):
        for learner in range(N_Learners):
            action_rounds_mean_learner = action_rounds[policy,mc_round,round_participation[policy,mc_round,:,learner]==1,learner].mean(0)
            action_rounds_mean[policy,mc_round,learner] = action_rounds_mean_learner if np.sum(round_participation[policy,mc_round,:,learner]==1)>0 else np.array([0])
            class_counts_mean_learner = class_counts[policy,mc_round,round_participation[policy,mc_round,:,learner]==1,learner,:].mean(0)
            class_counts_mean[policy,mc_round,learner,:] = class_counts_mean_learner if np.sum(round_participation[policy,mc_round,:,learner]==1)>0 else np.array([0,0])
            learner_states_mean_learner = learner_states_store[mc_round,policy,round_participation[policy,mc_round,:,learner]==1,learner]
            learner_states_mean[policy,mc_round,learner] = learner_states_mean_learner[-1] if np.sum(round_participation[policy,:,learner]==1)>0 else np.array([0])
print(learner_states_mean.mean(1))
print(num_rounds.mean(1))
### highlight nan indices
class_counts_mean = np.nan_to_num(class_counts_mean, copy=False, nan=0)
print(class_counts_mean.shape)
print(class_counts_mean.mean(1))
print(action_rounds_mean.mean(1))
print((class_counts_mean.mean(1)[0]-class_counts_mean.mean(1)[-1]).mean())
