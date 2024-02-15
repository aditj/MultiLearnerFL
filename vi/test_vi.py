import numpy as np
import itertools
def createP(O,L,E,P_O,fs,M,delta):
    P = np.zeros((A,O*L*E,O*L*E))
    for a in range(A):
        for o in range(O):
            for l in range(L):
                for e in range(E):
                    if l == 0: # if the learner state is 0 then the learner can only go to 1
                        p_success = 0
                    else:
                        p_success = fs[o,a]
                    for o_prime in range(O):
                        p_o_o_prime = P_O[o,o_prime] # probability of transition to o_prime from o
                        for l_prime in range(L):
                            l_transition_success = l_prime == l + e*M - 1
                            l_transition_failure = l_prime == l + e*M 
                            if l>=L-M: 
                                l_transition_success = l_prime == min(l+M-1,L-2)
                                l_transition_failure = l_prime == min(l+M,L-1)
                            for e_prime in range(E):
                                p_e_prime = e_prime*delta + (1-e_prime)*(1-delta)
                                if l >= L-M:
                                    p_e_prime = 1 - e_prime
                                P[a,o*L*E+l*E+e,o_prime*L*E+l_prime*E+e_prime] = p_o_o_prime*(l_transition_success*p_success + l_transition_failure*(1-p_success))*p_e_prime 
    return P
# Write transition kernel and cost function 

U = 2 # Number of actions
O = 3 # Number of states
E = 2 # Number of arrival states
N = 2 # Number of learners
M = np.array([4]*N) # Number of arrivals
delta = np.array([0.2]*N) # Arrival Probabilities
L = 10 # Number of queue states


P_O = np.array([
[0.8, 0.1, 0.05, 0.03,  0.02],
[0.6, 0.2, 0.1, 0.03,  0.07],
[0.5, 0.2, 0.2, 0,  0.1],
[0.4,0.2,0.2,0.1,0.1],
[0.02,0.02,0.1,0.18,0.68 ]
])  # Transition probabilities for oracle states
P_Os = np.array([P_O]*N)

fs = np.array([[0,0.3,0.6],
               [0,0.4,0.7],
               [0,0.5,0.9],
               [0,0.6,0.9],
               [0,0.6,1]]) # Success probabilities for oracle states

def generate_cost_matrix(N,U,O,i,oracle_bad_cost = 0.9,oracle_good_cost = 0.1):
    cost_matrix = np.ones((N*U,O))
    cost_row = np.linspace(oracle_bad_cost,oracle_good_cost,O)   
    for n in range(N):
        if n==i:
            cost_matrix[n*U+1,:] = cost_row
        else:
            cost_matrix[n*U,:] = - np.ones((1,O))/(N-1)
            cost_matrix[n*U+1,:] = - cost_row/(N-1)
    return cost_matrix

oracle_costs_limits = [
    [0.1,0.9],
    [0.2,0.8],
    [0.3,0.7],
    [0.8,0.2],
    [0.1,0.9]
]

cost_matrix_learners = np.zeros((N,N*U,O))
for i in range(N):
    cost_matrix_learners[i,:,:] = generate_cost_matrix(N,U,O,i,oracle_costs_limits[i][0],oracle_costs_limits[i][1])

## Zero sum condition check
for i in range(N):
    for o in range(O):
        
        print(i,o,(cost_matrix_learners[:,i*U+1,o]))
        assert np.sum(cost_matrix_learners[:,i*U,o]) == 0
        assert np.sum(cost_matrix_learners[:,i*U,o]) == 0

### Value Iteration
I = 100 
num_states = (O*L)**N
states = [np.arange(O*L) for n in range(N)]
v = np.ones((N,)+(O*L,)*N)
pi = np.ones((N,)+(O*L,)*N)
discount_factor = 0.9  

def calculate_average_value_function(current_state_vector,current_value,current_learner,P_0s,fs,M,delta,a):
    return
# State = \sum
for i in range(I-1):
    for element in itertools.product(*states):
        min_time = 100
        current_learner = 0
        j = 0
        for player_state in element:
            o = player_state//L
            l = player_state%L
            if min_time > 1/(o*l):
                min_time = 1/(o*l)
                current_learner = j
            j+=1
        current_state_vector = [(element[k]//L,element[k]%L) for k in range(N)]
        pi[np.concatenate([current_learner],np.flatten(current_state_vector))] = U
        min_value  = 0 
        current_value = v[current_learner,np.concatenate([current_learner],np.flatten(current_state_vector))]
        for a in range(U):
            actions = np.zeros((N*U))    
            cost_incurred = cost_matrix_learners[current_learner,current_learner*U+a,current_state_vector[current_learner][0]]
            expected_value = calculate_average_value_function(current_state_vector,current_value,current_learner,a,P_Os,fs,M,delta)
            value_ascribed = cost_incurred + discount_factor*expected_value
            if value_ascribed < min_value:
                min_value = value_ascribed
                pi[np.concatenate([current_learner],np.flatten(current_state_vector))] = a
        v[np.concatenate([current_learner],np.flatten(current_state_vector))] = min_value

        for k in range(N):
            if k == current_learner:
                continue
            cost_incurred = cost_matrix_learners[k,current_learner*U+a,current_state_vector[k][0]]
            expected_value = calculate_average_value_function(current_state_vector,current_value,current_learner,a,P_Os,fs,M,delta)
            v[np.concatenate([j],np.flatten(current_state_vector))] = cost_incurred + discount_factor*expected_value
        

