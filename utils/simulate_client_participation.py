import numpy as np
import matplotlib.pyplot as plt
from components.client_oracle import Oracle
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


N_rounds = 80 ## Number of rounds

N_Learners = 5 ### Number of Learners

U = [0,1,2] ### Actions
### Simulating all same actions for all rounds
N_mc = 10
N_total = 20
learner_states_store = np.ones((N_mc,len(U),N_rounds,N_Learners))*N_total
fig,axs = plt.subplots(len(U),1)
for i in range(N_mc):
    for u in U:
        learner_states = learner_states_store[i,u,:,:].copy()
        oracle = Oracle(N_learners=N_Learners)
        oracle_states_groups = oracle.get_oracle_states()   
        for round in np.arange(N_rounds-1):
            learner_states[round,learner_states[round]<=0] = 0
            if learner_states[round].sum()<=0:
                print(learner_states[round])
                learner_states[round:,:] = 0
                break
            oracle_states = oracle.get_oracle_states()
            learner_states[round+1] = learner_states[round]
            current_learner = get_current_learner(learner_states[round],oracle_states)
            action = u
            oracle.update_client_selection_matrix(action)
            round_success_groups = oracle.get_oracle_success(current_learner)
            learner_states[round+1,current_learner] -= round_success_groups
        learner_states_store[i,u,:,:] = learner_states.copy()
        print(learner_states_store[i,u,-1,:])

for u in U:
    axs[u].plot(np.arange(N_rounds),learner_states_store[:,u,:,:].mean(axis=0))
    axs[u].set_title("Action = "+str(u))
plt.savefig("plots/learner_states_sim.png")
### Benchmark SPANE parameters v/s all same actions (greedy) v/s random actions



