import numpy as np
import matplotlib.pyplot as plt

import tqdm as tqdm
P_O = np.array([
[0.7, 0.2, 0.05, 0.03,  0.02],
[0.6, 0.2, 0.1, 0.03,  0.07],
[0.1, 0.1, 0.2, 0.2,  0.4],
[0.1,0.1,0.1,0.2,0.5],
[0.02,0.02,0.06,0.2,0.7 ]
]) 
# fs = np.array([[0.1,0.3,0.5,0.7,0.4],[0.1,0.3,0.5,0.7,0.5],[0.1,0.3,0.5,0.7,0.6],[0.1,0.3,0.5,0.7,0.7],[0.1,0.3,0.5,0.7,1]])
fs = np.array([[0,0.1,0.2],[0.2,0.3,0.4],[0.4,0.5,0.6],[0.6,0.7,0.8],[0.8,0.9,1]]*5).reshape(5,5,3)

delta = np.array([0.12,0.14,0.16,0.18,0.2])
M = np.array([1,1,1,1,1])
L = 20
N = 5
U = 3
O = 5

# Learning Cost
C_L = np.zeros((N,O,U))
C_L_ind = np.vstack([np.linspace(0.5,0,O),np.linspace(0.75,0,O),np.linspace(1,0,O)]).T
C_L = np.vstack([C_L_ind.reshape(1,O,U)]*N)
assert C_L.shape == (N,O,U) ### 

# Queue Cost
C_Q_L = np.zeros((L,U))
C_Q_L = np.vstack([np.linspace(0,1,L),np.linspace(0,0.5,L),np.linspace(0,0,L)]).T
assert C_Q_L.shape == (L,U)
C_Q_O = np.zeros((N,O))
C_Q_O = np.vstack([np.linspace(0,1,O)]*N,)
assert C_Q_O.shape == (N,O)

constraints = np.array([0.5]*N)

def retrieve_cost(learner_index,current_learner,current_learner_state,current_oracle_state,current_action,C_L,C_Q_L,C_Q_O,lagrange,N=2,cost_type='zero_sum'):
    queue_cost = C_Q_L[current_learner_state,current_action]*C_Q_O[current_learner,current_oracle_state]
    learning_cost = C_L[current_learner,current_oracle_state,current_action]
    
    cost = learning_cost + queue_cost*lagrange
    if learner_index == current_learner:
        return cost
    elif cost_type == 'zero_sum':
        return -cost/(N-1)
    else:
        return 0


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


def calculate_average_cost(policy,P_O,O,L,T,N,fs,delta,Ms,discount_factor,C_Q_L,C_Q_O):
    
    oracle_states = np.array(np.random.choice(np.arange(O),size=(N)))
    learner_states = np.ones(N)*(L-1)
    costs = np.zeros(N)
    discount_factors = np.ones((N,T))*discount_factor

    for t in range(T):
        current_learner = get_current_learner(learner_states,oracle_states)
        current_oracle_state = int(oracle_states[current_learner])
        current_learner_state = int(learner_states[current_learner])
        action_learner = int(policy[current_learner][current_oracle_state][current_learner_state])
        #action_learner = np.random.choice(np.arange(U),p = action_prob_learner)
        arrivals = [np.random.choice([0,Ms[learner_index]], p =[1-delta[learner_index],delta[learner_index]]) for learner_index in range(N)]
        learner_states += np.array(arrivals)
        learner_states[learner_states>L-1] = L-1
        learner_states[current_learner] -= np.random.choice([0,1],p = [1-fs[current_learner][current_oracle_state][action_learner],fs[current_learner][current_oracle_state][action_learner]])
        learner_states[learner_states<0] = 0
        C_L = np.zeros((N,O,U))
        costs[:] += discount_factors[:,t]*np.array([retrieve_cost(learner_index,current_learner,current_learner_state,current_oracle_state,action_learner,C_L,C_Q_L,C_Q_O,lagrange=1) for learner_index in range(N)])
        discount_factors *= discount_factor
        # append to list end 

        oracle_states = [np.random.choice(np.arange(O),p=P_O[oracle_states[i]]) for i in range(N) ]
    return costs
def calculate_average_cost_running(policy,P_O,O,L,T,N,fs,delta,Ms,discount_factor,C_L):
    oracle_states = np.array(np.random.choice(np.arange(O),size=(N)))
    learner_states = np.ones(N)*(L-1)
    costs = np.zeros(N)
    discount_factors = np.ones((N,T))*discount_factor

    for t in range(T):
        current_learner = get_current_learner(learner_states,oracle_states)
        current_oracle_state = int(oracle_states[current_learner])
        current_learner_state = int(learner_states[current_learner])
        action_learner = int(policy[current_learner][current_oracle_state][current_learner_state])
        #action_learner = np.random.choice(np.arange(U),p = action_prob_learner)
        arrivals = [np.random.choice([0,Ms[learner_index]], p =[1-delta[learner_index],delta[learner_index]]) for learner_index in range(N)]
        learner_states += np.array(arrivals)
        learner_states[learner_states>L-1] = L-1
        learner_states[current_learner] -= action_learner*np.random.choice([0,1],p = [1-fs[current_learner][current_oracle_state],fs[current_learner][current_oracle_state]])
        C_Q_L = np.zeros((L,U))
        C_Q_O = np.zeros((N,O))
       
        costs[:] += discount_factors[:,t]*np.array([retrieve_cost(learner_index,current_learner,current_learner_state,current_oracle_state,action_learner,C_L,C_Q_L,C_Q_O,lagrange=0) for learner_index in range(N)])
        discount_factors *= discount_factor
        # append to list end 

        oracle_states = [np.random.choice(np.arange(O),p=P_O[oracle_states[i]]) for i in range(N) ]

    return costs


learner_states = np.ones((N),dtype=int)*(L-1)
value_vector = np.ones((N,O,O,L,L))
old_value_vector = np.zeros((N,O,O,L,L))
policy = np.zeros((N,O,L))

discount_factor = 0.9
step_lagrange = 0.5

def get_action_from(policy_parameters,learner_state):
    lower_thresholds = policy_parameters[0::3]
    upper_thresholds = policy_parameters[1::3]
    probabilities = np.sin(policy_parameters[2::3])**2
    for i in range(len(lower_thresholds)):
        if learner_state < lower_thresholds[i]:
            return i
        elif learner_state > lower_thresholds[i] and learner_state <= upper_thresholds[i]:
            return np.random.choice([i,i+1],p=[probabilities[i],1-probabilities[i]])
    return len(lower_thresholds)
    
n_mc = 1
n_iter = 100000
COST_TYPE = 'localized_sum'
policy_parameters_store = np.zeros((n_mc,n_iter,N,O,3*(U-1)))
DO_SPANE = True
if DO_SPANE:
    for mc in tqdm.tqdm(range(n_mc)):
        np.random.seed(mc)
        policy_parameters = np.ones((N,O,3*(U-1)))*10
        policy_parameters[:,:,2::3] = np.pi/8
        lagrange_parameters = np.ones((N))
        scale_parameter = 2000
        step_parameter = 0.5
        learner_states = np.ones((N),dtype=int)*(L-1)
        oracle_states = np.array([1,4,3,2,0])
        C_Q_L_zeros = np.zeros((L,U))
        C_Q_O_zeros = np.zeros((N,O))
        C_L_zeros = np.zeros((N,O,U))
        delta_value = [1,1,np.pi/16]*(U-1)
        delta_value = np.vstack([delta_value]*O)
        for i in tqdm.tqdm(range(n_iter)):
            current_learner = get_current_learner(learner_states,oracle_states)
            current_learner_state = learner_states[current_learner]
            current_oracle_state = oracle_states[current_learner]
            delta_parameters = np.random.choice([0,1],size=(O,3*(U-1)))

            policy_parameters_current = policy_parameters[current_learner]
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
            policy_parameters[current_learner] = policy_parameters_current - step_parameter*(learning_cost_gradient + queue_cost_gradient*np.max([0,lagrange_parameters[current_learner]+(queuecost_current - constraints[current_learner])]))
            lagrange_parameters[current_learner] = np.max([1 - (step_parameter/scale_parameter)*lagrange_parameters[current_learner],lagrange_parameters[current_learner] + step_parameter*(queuecost_current - constraints[current_learner])])

            

            learner_states[current_learner] -= np.random.choice([0,1],p = [1-fs[current_learner][current_oracle_state][current_action],fs[current_learner][current_oracle_state][current_action]])
            learner_states[learner_states<0] = 0
            arrivals = [np.random.choice([0,M[learner_index]], p =[1-delta[learner_index],delta[learner_index]]) for learner_index in range(N)]

            learner_states += np.array(arrivals)
            learner_states[learner_states>L-1] = L-1
            
            oracle_states = [np.random.choice(np.arange(O),p=P_O[oracle_states[i]]) for i in range(N) ]
            # print(fs[current_learner][current_oracle_state])
            # print(learning_cost_gradient)
            # print(current_action,learner_states,oracle_states,current_learner)
            # print(current_learner_state,current_oracle_state,current_action*np.random.choice([0,1],p = [1-fs[current_learner][current_oracle_state],fs[current_learner][current_oracle_state]]))
            policy_parameters_store[mc,i] = policy_parameters

    np.save('parameters/policy_parameters.npy',policy_parameters_store)
policy_parameters_store = np.load('parameters/policy_parameters.npy')

fig,axs = plt.subplots(5,5,figsize=(10,10))


# Plot iterations of the policy parameters
for o in range(O):
    for n in range(N):
        axs[n,o].plot(policy_parameters_store.mean(axis=0)[:,n,o,0])
# add overall x and y labels
fig.text(0.5, 0.04, 'Oracle States', ha='center')
fig.text(0.04, 0.5, 'Learners', va='center', rotation='vertical')
plt.savefig('plots/policy_parameters_np.png')

# Plot policies from the policy parameters

def get_policy_from_parameters(policy_parameters,L):
    policy = np.zeros(L)
    lower_thresholds = policy_parameters[0::3]
    upper_thresholds = policy_parameters[1::3]
    probabilities = np.sin(policy_parameters[2::3])**2
    for i in range(len(lower_thresholds)):
        policy[:int(lower_thresholds[i])] = 0
        policy[int(lower_thresholds[i]):int(upper_thresholds[i])] = probabilities[i]
    policy[int(upper_thresholds[i]):] = len(lower_thresholds)
    return policy

fig,axs = plt.subplots(5,5,figsize=(10,10))
for o in range(O):
    for n in range(N):
        axs[n,o].plot(np.arange(L),get_policy_from_parameters(policy_parameters_store.mean(axis=0)[-1,n,o],L))
# add overall x and y labels
fig.text(0.5, 0.04, 'Oracle States', ha='center')
fig.text(0.04, 0.5, 'Learners', va='center', rotation='vertical')
plt.savefig('plots/policy_np.png')


