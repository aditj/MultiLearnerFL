import numpy as np
import matplotlib.pyplot as plt

import tqdm as tqdm
P_O = np.array([
[0.8, 0.1, 0.05, 0.03,  0.02],
[0.6, 0.2, 0.1, 0.03,  0.07],
[0.5, 0.2, 0.2, 0,  0.1],
[0.4,0.2,0.2,0.1,0.1],
[0.02,0.02,0.1,0.18,0.68 ]
]) 
fs = np.array([[0,0.2,0.4,0.6,0.9],[0,0.2,0.4,0.6,0.9]])
delta = np.array([0.2,0.2])
M = np.array([4,4])
L = 20
N = 2
U = 2
O = 5

# Learning Cost
C_L = np.zeros((N,O,U))
C_L_ind = np.vstack([np.zeros(O),np.linspace(1,0,O)]).T
C_L = np.vstack([C_L_ind.reshape(1,O,U)]*N)
assert C_L.shape == (N,O,U) ### 

# Queue Cost
C_Q_L = np.zeros((L,U))
C_Q_L = np.vstack([np.linspace(0,1,L),np.zeros((L))]).T
assert C_Q_L.shape == (L,U)
C_Q_O = np.zeros((N,O))
C_Q_O = np.vstack([np.linspace(0,1,O)]*N,)
assert C_Q_O.shape == (N,O)

constraints = np.array([0.5,0.5])

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
def calculate_learner_state_probabilities(current_learner_state,current_action,current_oracle_state,fs,M,delta,current_learner_index,L):
    
    learner_state_possible = [max(0,current_learner_state-1),current_learner_state,min(current_learner_state+M[current_learner_index]-1,L-1),min(current_learner_state+M[current_learner_index],L-1)]
    decreasing_prob = fs[current_learner_index,current_oracle_state]*current_action*(1-delta[current_learner_index])
    same_prob = ((1 - fs[current_learner_index,current_oracle_state])*current_action + (1-current_action))*(1-delta[current_learner_index])
    increasingbyoneless_prob = fs[current_learner_index,current_oracle_state]*(1-current_action)*(delta[current_learner_index])
    increasingbyone_prob = ((1-fs[current_learner_index,current_oracle_state])*(1-current_action) + (current_action))*(delta[current_learner_index])
    learner_state_probabilities = [ decreasing_prob,same_prob,increasingbyoneless_prob,increasingbyone_prob]

    return learner_state_possible,learner_state_probabilities
def calculate_average_value(value_vector,learner_index,current_learner,current_oracle_states,current_learner_states,current_action,P_O,fs,M,delta):
    expected_value = 0
    current_learner_state = current_learner_states[current_learner]
    current_oracle_state = current_oracle_states[current_learner]
    for oracle_state_1 in range(O): # PLayer 1
        for oracle_state_2 in range(O): # Player 2
            P_O_prod = P_O[current_oracle_states[0],oracle_state_1]*P_O[current_oracle_states[1],oracle_state_2]
            learner_state_possible,learner_state_probabilities = calculate_learner_state_probabilities(current_learner_state,current_action,current_oracle_state,fs,M,delta,current_learner,L)
            for learner_state,learner_state_prob in zip(learner_state_possible,learner_state_probabilities):
                learner_state_copy = current_learner_states.copy()
                learner_state_copy[current_learner] = learner_state
                expected_value += P_O_prod*learner_state_prob*value_vector[learner_index,oracle_state_1,oracle_state_2,learner_state_copy[0],learner_state_copy[1]]
    return expected_value

def get_current_learner(learner_states,oracle_states):
    current_learner = 0 
    if ((oracle_states[1]+1)*(learner_states[1]+1)<=0) or ((oracle_states[0]+1)*(learner_states[0]+1)<=0):
        print("Error",oracle_states,learner_states)
    if (1/((oracle_states[1]+1)*(learner_states[1]+1))>1/((oracle_states[0]+1)*(learner_states[0]+1))):
        current_learner = 1
    return current_learner

def generate_policy_from_vector(value_vector,learner_index,N,O,L,U9):
    policy = np.zeros((N,O,L,U))
    for oracle_state_1 in range(O):
        for oracle_state_2 in range(O):
            for learner_state_1 in range(L):
                for learner_state_2 in range(L):
                    min_value = 100
                    min_action = 0
                    for action in range(U):
                        if value_vector[learner_index,oracle_state_1,oracle_state_2,learner_state_1,learner_state_2,action] < min_value:
                            min_value = value_vector[learner_index,oracle_state_1,oracle_state_2,learner_state_1,learner_state_2][action]
                            min_action = action
                    policy[learner_index,oracle_state_1,oracle_state_2,learner_state_1,learner_state_2][min_action] = 1
    return policy
def calculate_average_cost(policy,P_O,O,L,T,N,fs,delta,Ms,discount_factor,C_Q_L,C_Q_O):
    
    oracle_states = np.array([0,O-1])
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
        learner_states[learner_states<0] = 0
        C_L = np.zeros((N,O,U))
        costs[:] += discount_factors[:,t]*np.array([retrieve_cost(learner_index,current_learner,current_learner_state,current_oracle_state,action_learner,C_L,C_Q_L,C_Q_O,lagrange=1) for learner_index in range(N)])
        discount_factors *= discount_factor
        # append to list end 

        oracle_state_1 = np.random.choice(np.arange(O),p=P_O[oracle_states[0]])
        oracle_state_2 = np.random.choice(np.arange(O),p=P_O[oracle_states[1]])
        oracle_states = [oracle_state_1,oracle_state_2] 
    return costs

def calculate_average_cost_running(policy,P_O,O,L,T,N,fs,delta,Ms,discount_factor,C_L):
    oracle_states = np.array([0,O-1])
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

        oracle_state_1 = np.random.choice(np.arange(O),p=P_O[oracle_states[0]])
        oracle_state_2 = np.random.choice(np.arange(O),p=P_O[oracle_states[1]])
        oracle_states = [oracle_state_1,oracle_state_2]

    return costs


learner_states = np.ones((N),dtype=int)*(L-1)
value_vector = np.ones((N,O,O,L,L))
old_value_vector = np.zeros((N,O,O,L,L))
policy = np.zeros((N,O,L))

discount_factor = 0.9
step_lagrange = 0.5
COST_TYPE = 'localized_sum'
I = 20
lagrange = np.ones(N)*0.5
DO_VI = True
if DO_VI:
    for j in tqdm.tqdm(range(I)):
        old_value_vector = np.zeros((N,O,O,L,L))
        while ((value_vector - old_value_vector)**2).sum()>1:
            old_value_vector = value_vector.copy()

            for oracle_state_1 in range(O):
                for oracle_state_2 in range(O):
                    oracle_states = np.array([oracle_state_1,oracle_state_2])
                    for learner_state_1 in range(L):
                        for learner_state_2 in range(L):
                            learner_states = np.array([learner_state_1,learner_state_2])
                            current_learner = get_current_learner([learner_state_1,learner_state_2],[oracle_state_1,oracle_state_2])
                            current_learner_state = learner_states[current_learner]
                            current_oracle_state = oracle_states[current_learner]
                            min_value = 100
                            min_action = 0
                            for action in range(U):
                                cost_incurred = retrieve_cost(current_learner,current_learner,current_learner_state,current_oracle_state,action,C_L,C_Q_L,C_Q_O,lagrange[current_learner],N=2,cost_type=COST_TYPE)
                            # print(cost_incurred)
                                expected_value = calculate_average_value(value_vector,current_learner,current_learner,oracle_states,learner_states,action,P_O,fs,M,delta)
                                if cost_incurred + discount_factor*expected_value < min_value:
                                    min_value = cost_incurred + discount_factor*expected_value
                                    min_action = action
                            value_vector[current_learner,oracle_state_1,oracle_state_2,learner_state_1,learner_state_2] = min_value
                            policy[current_learner,current_oracle_state,current_learner_state] = min_action
                            for learner_index in range(N):
                                if learner_index==current_learner:
                                    continue
                                else:
                                    cost_incurred = retrieve_cost(learner_index,current_learner,current_learner_state,current_oracle_state,min_action,C_L,C_Q_L,C_Q_O,lagrange[learner_index],N=2,cost_type=COST_TYPE)
                                    expected_value = calculate_average_value(value_vector,learner_index,current_learner,oracle_states,learner_states,min_action,P_O,fs,M,delta)
                                    value_vector[learner_index,oracle_state_1,oracle_state_2,learner_state_1,learner_state_2] =  cost_incurred + discount_factor*expected_value
            print(((value_vector - old_value_vector)**2).sum())
    # policy = generate_policy_from_vector(value_vector,0,N,O,L,U)
        print("Running Avg Cost: ",calculate_average_cost_running(policy,P_O,O,L,1000,N,fs,delta,M,discount_factor,C_L))
        costs = calculate_average_cost(policy,P_O,O,L,1000,N,fs,delta,M,discount_factor,C_Q_L,C_Q_O)
        lagrange = lagrange - step_lagrange*(costs - constraints)
        print(costs-constraints)
    np.save('policy.npy',policy)
policy = np.load('policy.npy')
print("Running Learning Cost: ",calculate_average_cost_running(policy,P_O,O,L,1000,N,fs,delta,M,discount_factor,C_L))
print("Queue Cost: ",calculate_average_cost(policy,P_O,O,L,1000,N,fs,delta,M,discount_factor,C_Q_L,C_Q_O))
## Plot policy 
fig,axs = plt.subplots(2,5)

for o in range(O):
    axs[0,o].plot(policy[0,o,:])
    axs[1,o].plot(policy[1,o,:])
plt.savefig('policy.png')


### Tests to be done to verify the policy is correct 
