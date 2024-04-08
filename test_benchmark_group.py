import numpy as np
import matplotlib.pyplot as plt
from utils.funcs import get_action_from, get_current_learner, retrieve_cost, get_current_learner_random
from components.group_oracle import Oracle
from components.net import CNN
import tqdm as tqdm
#### Test spane algorithm against greedy and random scheduling
#### Two key benchmarks: number of queries posed by each learner and the average number of classes from learner preference in the round that the learner learns in

### Policies 
## Synthetic policy parameters
N = 5
O = 5
U = 4
M= 50
policy_parameters_load = np.zeros((N,O,3*(U-1)))
policy_parameters_load[:,:,0::3] = np.array([[M,M,M],[M,M,M],[M-30,M-25,M-20],[M-40,M-40,M-40],[0,0,0]])
policy_parameters_load[:,:,1::3] = np.array([[M,M,M],[M,M,M],[M-20,M-15,M-10],[M-40,M-40,M-40],[0,0,0]])
policy_parameters_load[:,:,2::3] =  np.array([[1],[1],[1],[1],[1]])
print(policy_parameters_load)
### Greedy Scheduling, Random Action

### Greedy Scheduling, Random Action
spane_action = lambda l,o,n: get_action_from(policy_parameters_load[n,o],l)
greedy_action_3 = lambda l,o,n: 3
random_action = lambda l,o,n: np.random.choice(U)
N_rounds = 2000
N_Learners = 5
N_classes = 10

N_mc = 20
policies = [spane_action,spane_action,greedy_action_3,greedy_action_3,random_action,random_action]
scheduling = [get_current_learner_random, get_current_learner, get_current_learner_random, get_current_learner, get_current_learner_random, get_current_learner]

N_communication_rounds = np.zeros((len(policies),N_mc))

learner_states_store = np.ones((N_mc,len(policies),N_rounds,N_Learners))*M  ### Stores the learner states for each mc round
round_participation = np.zeros((len(policies),N_mc,N_rounds,N_Learners)) ### Stores the round participation for each mc round
action_rounds = np.zeros((len(policies),N_mc,N_rounds,N_Learners)) ### Stores the actions for each mc round
oracle_states_store = np.zeros((N_mc,len(policies),N_rounds,N_Learners))

round_success_store = np.zeros((O,U,2))
incentive_used = np.zeros((len(policies),N_mc,N_rounds,N_Learners))
PERFORM_EXP = 1

if PERFORM_EXP:
    for mc_round in tqdm.tqdm(range(N_mc)):
        oracle = Oracle()
        oracle.initialize_clients(CNN)
        for policy in range(len(policies)):
            learner_done = np.zeros((N_Learners))
            for round_ in tqdm.tqdm(np.arange(N_rounds-1)):
                oracle_states = oracle.get_oracle_states("group")
                oracle_states_store[mc_round,policy,round_,:] = oracle_states.copy().flatten()
                if learner_done.sum() < N_Learners:
                    
                    current_round_learner_states = learner_states_store[mc_round,policy,round_,:].copy()
                    
                    current_learner = scheduling[policy](current_round_learner_states,oracle_states)
                    if learner_done[current_learner] == 1:
                        continue

                    action = policies[policy](current_round_learner_states[current_learner],oracle_states[current_learner],current_learner)
                    action_rounds[policy,mc_round,round_,current_learner] = action
                    if action != 0:
                        action -= 1
                        incentive_used[policy,mc_round,round_,current_learner] += action + 1
                        oracle.update_client_selection_matrix(action,oracle.learner_group_preference[current_learner])
                        round_success = oracle.get_oracle_success(current_learner,"group")

                        round_success_store[oracle_states[current_learner],action+1,0] += round_success
                        round_success_store[oracle_states[current_learner],action+1,1] += 1

                        if policy != 4 and policy != 5:
                            current_round_learner_states[current_learner] -= round_success
                        else:
                            current_round_learner_states[current_learner] -= 1
                        if current_round_learner_states[current_learner] == 0:
                            learner_done[current_learner] = 1
                oracle.update_client_participation_matrix()
                current_round_learner_states[current_round_learner_states<0] = 0
                learner_states_store[mc_round,policy,round_+1,:] = current_round_learner_states.copy()
                round_participation[policy,mc_round,round_+1,current_learner] = 1
    np.save("parameters/learner_states.npy",learner_states_store)
    np.save("parameters/round_participation.npy",round_participation)
    np.save("parameters/action_rounds.npy",action_rounds)
    np.save("parameters/oracle_states.npy",oracle_states_store)
    np.save("parameters/round_success.npy",round_success_store)
    np.save("parameters/incentive_used.npy",incentive_used)


action_rounds = np.load("parameters/action_rounds.npy")
learner_states_store = np.load("parameters/learner_states.npy")
round_participation = np.load("parameters/round_participation.npy")
oracle_states_store = np.load("parameters/oracle_states.npy")
round_success_store = np.load("parameters/round_success.npy")
incentive_used = np.load("parameters/incentive_used.npy")

action_rounds_mean = np.zeros((len(policies),N_mc,N_Learners))
learner_states_mean = np.zeros((len(policies),N_mc,N_Learners))
for policy in range(len(policies)):
    for mc_round in range(N_mc):
        for learner in range(N_Learners):
            action_rounds_mean_learner = action_rounds[policy,mc_round,round_participation[policy,mc_round,:,learner]==1,learner].mean(0)
            action_rounds_mean[policy,mc_round,learner] = action_rounds_mean_learner if np.sum(round_participation[policy,mc_round,:,learner]==1)>0 else np.array([0])
            learner_states_mean_learner = learner_states_store[mc_round,policy,round_participation[policy,mc_round,:,learner]==1,learner]
            learner_states_mean[policy,mc_round,learner] = learner_states_mean_learner[-1] if np.sum(round_participation[policy,:,learner]==1)>0 else np.array([0])
print(learner_states_mean.mean(1))

print(action_rounds_mean.mean(1))
