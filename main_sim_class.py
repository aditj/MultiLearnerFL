import numpy as np
import matplotlib.pyplot as plt
from utils.funcs import get_action_from, get_current_learner, retrieve_cost, get_current_learner_random
from components.client_oracle import Oracle
from components.net import CNN
import tqdm as tqdm
from components.learner import Learner

N = 5
O = 5
U = 4
M = 90
policy_parameters_load = np.zeros((N,O,3*(U-1)))
policy_parameters_load[:,:,0::3] = np.array([[M,M,M],[M-10,M-10,M-10],[M-60,M-60,M-60],[M-80,M-80,M-80],[0,0,0]])
policy_parameters_load[:,:,1::3] = np.array([[M,M,M],[M-10,M-10,M-10],[M-60,M-60,M-60],[M-80,M-80,M-80],[0,0,0]])
policy_parameters_load[:,:,2::3] =  np.array([[1],[1],[1],[1],[1]])

### Greedy Scheduling, Random Action
spane_action = lambda l,o,n: get_action_from(policy_parameters_load[n,o],l)
greedy_action_3 = lambda l,o,n: 3
random_action = lambda l,o,n: np.random.choice(U)
N_rounds = 2000
N_Learners = 5
N_classes = 10

N_mc = 1
policies = [spane_action,spane_action,greedy_action_3,greedy_action_3,random_action,random_action]
scheduling = [get_current_learner_random, get_current_learner, get_current_learner_random, get_current_learner, get_current_learner_random, get_current_learner]

N_communication_rounds = np.zeros((len(policies),N_mc))

learner_states_store = np.ones((N_mc,len(policies),N_rounds,N_Learners))*M  ### Stores the learner states for each mc round
round_participation = np.zeros((len(policies),N_mc,N_rounds,N_Learners)) ### Stores the round participation for each mc round
action_rounds = np.zeros((len(policies),N_mc,N_rounds,N_Learners)) ### Stores the actions for each mc round
oracle_states_store = np.zeros((N_mc,len(policies),N_rounds,N_Learners))

round_success_store = np.zeros((O,U,2))
incentive_used = np.zeros((len(policies),N_mc,N_rounds,N_Learners))
evaluation_metric = np.zeros((len(policies),N_mc,N_rounds,N_Learners))


oracle = Oracle()
learners = [Learner(oracle.learner_class_preference[i],CNN) for i in range(N_Learners)]


PERFORM_EXP =  0

if PERFORM_EXP:
    for mc_round in tqdm.tqdm(range(N_mc)):
        oracle = Oracle()
        oracle.initialize_clients(CNN)
        for policy in range(len(policies)):
            for learner in learners:
                ### reset weights
                learner.random_init_weights()
            learner_done = np.zeros((N_Learners))
            for round_ in np.arange(N_rounds-1):
                oracle_states = oracle.get_oracle_states("class")
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
                        oracle.update_client_selection_matrix(action,oracle.learner_class_preference[current_learner])
                        round_success = oracle.get_oracle_success(current_learner,"class")
                        
                        round_success_store[oracle_states[current_learner],action+1,0] += round_success
                        round_success_store[oracle_states[current_learner],action+1,1] += 1

                        if policy == 4 or policy == 5:
                            round_success = 1
                        current_round_learner_states[current_learner] -= round_success
                        if round_success == 1:
                            weights = learners[current_learner].get_weights()
                            oracle.train(weights=weights)
                            learners[current_learner].set_weights(oracle.aggregate_weights())
                            evaluation_metric[policy,mc_round,round_,current_learner] = learners[current_learner].evaluate()
                            print("Round: ",round_," Policy: ",policy," Learner: ",current_learner," Evaluation: ",evaluation_metric[policy,mc_round,round_,current_learner],"Oracle State:", oracle_states[current_learner])
                        if current_round_learner_states[current_learner] == 0:
                            learner_done[current_learner] = 1
                oracle.update_client_partition_matrix()
                current_round_learner_states[current_round_learner_states<0] = 0
                learner_states_store[mc_round,policy,round_+1,:] = current_round_learner_states.copy()
                round_participation[policy,mc_round,round_+1,current_learner] = 1
    np.save("parameters/learner_states.npy",learner_states_store)
    np.save("parameters/round_participation.npy",round_participation)
    np.save("parameters/action_rounds.npy",action_rounds)
    np.save("parameters/oracle_states.npy",oracle_states_store)
    np.save("parameters/round_success.npy",round_success_store)
    np.save("parameters/incentive_used.npy",incentive_used)
    np.save("parameters/evaluation_metric.npy",evaluation_metric)

action_rounds = np.load("parameters/action_rounds.npy")
learner_states_store = np.load("parameters/learner_states.npy")
round_participation = np.load("parameters/round_participation.npy")
oracle_states_store = np.load("parameters/oracle_states.npy")
round_success_store = np.load("parameters/round_success.npy")
incentive_used = np.load("parameters/incentive_used.npy")
evaluation_metric = np.load("parameters/evaluation_metric.npy")

evaluation_metric_mean = np.zeros((len(policies),N_mc,N_Learners,M))
for policy in range(len(policies)):
    for mc_round in range(N_mc):
        for learner in range(N_Learners):
            evaluation_metric_rounds = evaluation_metric[policy,mc_round,round_participation[policy,mc_round,:,learner]==1,learner]

            for m in range(min(len(evaluation_metric_rounds),M-1)):
                evaluation_metric_mean[policy,mc_round,learner,m] = evaluation_metric_rounds[m]

print(evaluation_metric_mean[:,:,:,:].max(axis=2))
print(evaluation_metric[2,:,:,0])