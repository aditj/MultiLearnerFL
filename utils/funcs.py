import numpy as np
import matplotlib.pyplot as plt
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

def get_current_learner_random(learner_states,oracle_states):
    if np.sum(learner_states>0) == 0:
        return np.random.choice(np.arange(len(learner_states)))
    else:
        return np.random.choice(np.where(learner_states>0)[0])

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

def sigmoid_function(x,shift=0,tau=0.1):
    return 1/(1+np.exp((-x+shift)/tau))

def get_action_from(policy_parameters,learner_state):
    policy_parameters = policy_parameters.flatten()
    n_thresholds = 2*len(policy_parameters)//3
    sorted_thresholds = np.sort(policy_parameters[0:n_thresholds])
    probabilities = np.sin(policy_parameters[n_thresholds::])**2
    action = 0
    for i in range(sorted_thresholds.shape[0]):
        if i % 2 == 0:
            action+=sigmoid_function(learner_state,shift=sorted_thresholds[i])*(1-probabilities[i//2])
        else:    
            action+=sigmoid_function(learner_state,shift=sorted_thresholds[i])*probabilities[i//2]
        
    action_index = np.floor(action).astype(int)
    action_prob = action - action_index
    ### check if action_prob is nan
    if action_prob == None or np.isnan(action_prob):
        action_prob = 0
        action_index = 0
        print("Exception",action,policy_parameters)
        # raise Exception
        
    action_returned = np.random.choice([action_index,action_index+1],p=[1-action_prob,action_prob])
    return action_returned

    

def test_get_action_from(policy_parameters,learner_state_range):
    actions = np.zeros((len(learner_state_range)))
    for i in range(len(learner_state_range)):
        actions[i] = get_action_from(policy_parameters,learner_state_range[i])
    plt.plot(learner_state_range,actions)
    plt.savefig("./spane_simulate/testsigmoid.png")

test_get_action_from(np.array([4,20,10,21,0.5,0.3]),np.arange(0,40,0.1))