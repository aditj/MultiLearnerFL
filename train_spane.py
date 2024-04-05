## File to train SPANE parameters on simulated data and oracle access to the MDP
import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm
from utils.funcs import get_action_from, get_current_learner, retrieve_cost
from components.net import CNN
 
from components.client_oracle import Oracle

delta = np.array([0.15,0.15,0.15,0.15,0.15])
M = np.array([1,1,1,1,1])
L = 30
N = 5
U = 3
O = 5

# Learning Cost
C_L = np.zeros((N,O,U))
C_L_ind = np.vstack([np.linspace(1,0.5,O),np.linspace(2,0.5,O),np.linspace(3,0.5,O)]).T
C_L = np.vstack([C_L_ind.reshape(1,O,U)]*N)
assert C_L.shape == (N,O,U) ### 

# Queue Cost
C_Q_L = np.zeros((L,U))
C_Q_L = np.vstack([np.linspace(0,2,L),np.linspace(0,1,L),np.linspace(0,0,L)]).T
assert C_Q_L.shape == (L,U)
C_Q_O = np.zeros((N,O))
C_Q_O = np.vstack([np.linspace(0,1,O)]*N,)
assert C_Q_O.shape == (N,O)

constraints = np.array([0.5]*N)

learner_states = np.ones((N),dtype=int)*(15)
value_vector = np.ones((N,O,O,L,L))
old_value_vector = np.zeros((N,O,O,L,L))
policy = np.zeros((N,O,L))

step_lagrange = 0.01
scale_parameter = 2000

oracle = Oracle()
oracle.initialize_clients(CNN)
### Set numpy print precision to 2
np.set_printoptions(precision=1)
# print(oracle.return_success_probability_classes(100000,[[0,1],[2,3],[4,5],[6,7],[8,9]]))
# P = oracle.return_oracle_probability_classes(100000)
# for i in range(5):
#     print(P[i],"Oracle "+str(i))
#     print(P[i].cumsum(axis=1),"Oracle "+str(i))

n_mc = 1
n_iter = 200000
learner_count = np.zeros((N))
COST_TYPE = 'localized_sum'
policy_parameters_store = np.zeros((n_mc,n_iter,N,O,3*(U-1)))
DO_SPANE = 1
if DO_SPANE:
    for mc in tqdm.tqdm(range(n_mc)):
        step_parameter = 0.8

        np.random.seed(mc)
        policy_parameters = np.ones((N,O,3*(U-1)))*10
        policy_parameters[:,:,2::3] = np.pi/8
        lagrange_parameters = np.ones((N))
        
        learner_states = np.ones((N),dtype=int)*10

        oracle_states = oracle.get_oracle_states("class")
        C_Q_L_zeros = np.zeros((L,U))
        C_Q_O_zeros = np.zeros((N,O))
        C_L_zeros = np.zeros((N,O,U))
        delta_value = [0.5,0.5,np.pi/8]*(U-1)
        delta_value = np.vstack([delta_value]*O)
        for i in tqdm.tqdm(range(n_iter)):
            oracle_states = oracle.get_oracle_states("class")
            current_learner = get_current_learner(learner_states,oracle_states)
            current_learner_state = learner_states[current_learner]
            current_oracle_state = oracle_states[current_learner]
            delta_parameters = np.random.choice([0,1],size=(O,3*(U-1)))

            policy_parameters_current = policy_parameters[current_learner,:,:]
            policy_parameters_plus = policy_parameters_current + delta_parameters*delta_value
            policy_parameters_minus = policy_parameters_current - delta_parameters*delta_value
                
            current_action = get_action_from(policy_parameters_plus[current_oracle_state],current_learner_state)
            learning_cost_incurred_plus = retrieve_cost(current_learner,current_learner,current_learner_state,current_oracle_state,current_action,C_L,C_Q_L_zeros,C_Q_O_zeros,lagrange_parameters[current_learner],N=2,cost_type=COST_TYPE)
            queue_cost_incurred_plus = retrieve_cost(current_learner,current_learner,current_learner_state,current_oracle_state,current_action,C_L_zeros,C_Q_L,C_Q_O,lagrange_parameters[current_learner],N=2,cost_type=COST_TYPE)
            current_action = get_action_from(policy_parameters_minus[current_oracle_state],current_learner_state)
            learning_cost_incurred_minus = retrieve_cost(current_learner,current_learner,current_learner_state,current_oracle_state,current_action,C_L,C_Q_L_zeros,C_Q_O_zeros,lagrange_parameters[current_learner],N=2,cost_type=COST_TYPE)
            queue_cost_incurred_minus = retrieve_cost(current_learner,current_learner,current_learner_state,current_oracle_state,current_action,C_L_zeros,C_Q_L,C_Q_O,lagrange_parameters[current_learner],N=2,cost_type=COST_TYPE)
            current_action = get_action_from(policy_parameters_current[current_oracle_state],current_learner_state)
            queuecost_current = retrieve_cost(current_learner,current_learner,current_learner_state,current_oracle_state,current_action,C_L_zeros,C_Q_L,C_Q_O,lagrange_parameters[current_learner],N=2,cost_type=COST_TYPE)
            learning_cost_gradient = ((learning_cost_incurred_plus - learning_cost_incurred_minus)/(2*delta_value))*delta_parameters
            queue_cost_gradient = ((queue_cost_incurred_plus - queue_cost_incurred_minus)/(2*delta_value))*delta_parameters
            policy_parameters[current_learner] = policy_parameters_current - step_parameter*(learning_cost_gradient + queue_cost_gradient*max([0,lagrange_parameters[current_learner]+(queuecost_current - constraints[current_learner])]))
            lagrange_parameters[current_learner] = max([1 - (step_parameter/scale_parameter)*lagrange_parameters[current_learner],lagrange_parameters[current_learner] + step_parameter*(queuecost_current - constraints[current_learner])])
            oracle.update_client_selection_matrix(current_action,oracle.learner_class_preference[current_learner])
            round_success_or_not = oracle.get_oracle_success(current_learner,"class")
            learner_states[current_learner] -= round_success_or_not
            learner_states[learner_states<0] = 1
            arrivals = [np.random.choice([0,M[learner_index]], p =[1-delta[learner_index],delta[learner_index]]) for learner_index in range(N)]
            learner_states += np.array(arrivals)
            learner_states[learner_states>L-1] = L-1
            
            learner_count[current_learner] += 1

            lagrange_parameters = np.minimum(lagrange_parameters,20)
            lagrange_parameters = np.maximum(lagrange_parameters,-20)
            policy_parameters[:,:,0::3] = np.minimum(policy_parameters[:,:,0::3],50)
            policy_parameters[:,:,0::3] = np.maximum(policy_parameters[:,:,0::3],-30)
            policy_parameters[:,:,1::3] = np.minimum(policy_parameters[:,:,1::3],50)
            policy_parameters[:,:,1::3] = np.maximum(policy_parameters[:,:,2::3],-30)


            policy_parameters_store[mc,i] = policy_parameters

            step_parameter = step_parameter*0.999995
            #if not round_success_or_not:
            # print("Round Success",current_learner,current_oracle_state,round_success_or_not)
    np.save('parameters/policy_parameters_spane_trained.npy',policy_parameters_store)
policy_parameters_store = np.load('parameters/policy_parameters_spane_trained.npy').mean(axis=0)
fig,ax = plt.subplots(N,O,figsize=(20,20))
## clean plot
for learner in np.arange(N):
    for oracle in np.arange(O):
        ax[learner,oracle].plot(np.arange(n_iter),policy_parameters_store[:,learner,oracle,0],label="Lower Threshold Action 0->p")
        ax[learner,oracle].plot(np.arange(n_iter),policy_parameters_store[:,learner,oracle,3],label="Lower Threshold Action 1->p")
ax[0,0].legend()

plt.xlabel("Oracle State")
plt.ylabel("Learner")


plt.savefig('plots/policy_parameters_spane.png')
plt.close()