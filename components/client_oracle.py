import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm
import torch
import pandas as pd
from components.client import Client
class Oracle():
    def __init__(self,
            U=[0,1,2],
                n_clients = 20,
                n_classes = 10,
                n_group_attributes = 2,
                n_categories_per_group = [2,5],
                N_max = 60,
                max_classes = 2,
                p_stay = [[0.7,0.3],[0.3,0.7]],
                N_learners = 5,
                client_dirichlet_alpha = [[0.6,0.4],[0.42,0.2,0.16,0.14,0.08]],
                client_dirichlet_alpha_class = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                learner_class_preference = np.array([[0,1],[2,3],[4,5],[6,7],[8,9]]), ### Learner class preference,
                state_thresholds = np.array([0,0.2,0.25,0.30,0.35]), ### Thresholds on class distribution to determine oracle state for each learner
                state_thresholds_groups = [0,0.2,0.4,0.6,0.8],
                success_thresholds = 0.6,
                success_thresholds_groups = 0.75,
                learner_group_preference =  [0,3,1,4,5],
                N_mins = [0,0.5,1], ### Minimum number of samples per class per client
                dist_groups = [[[0.9,0.1],[0.7,0.1,0.1,0.1]],
                                [[0.8,0.2],[0.5,0.1,0.2,0.2]],
                                [[0.7,0.3],[0.45,0.19,0.18,0.18]]
                ],
                dist_classes = np.array([[0.1,0.1],
                              [0.2,0.2],
                              [0.3,0.3]]),
                client_dataset_path = "data/client_dataset/"
                
                 ):
        self.U = U
        self.n_clients = n_clients
        self.n_classes = n_classes
        self.n_group_attributes = n_group_attributes
        self.n_categories_per_group = n_categories_per_group
        self.N_max = N_max
        self.max_classes = max_classes
        self.p_stay = p_stay
        self.N_learners = N_learners
        self.classes = np.arange(n_classes)
        self.client_dirichlet_alpha_class = client_dirichlet_alpha_class
        self.n_datapoints = 2800
        self.initialize_client_dataset()
        self.client_dirichlet_alpha = client_dirichlet_alpha
        self.learner_class_preference = learner_class_preference
        self.state_thresholds = state_thresholds
        self.state_thresholds_groups = state_thresholds_groups
        self.success_thresholds = success_thresholds
        self.success_thresholds_groups = success_thresholds_groups
        self.learner_group_preference = learner_group_preference
        self.N_mins = N_mins*self.N_max
        self.dist_groups = dist_groups
        self.client_selection_matrix = np.random.choice([0,1],(self.n_clients,1))
        self.client_dataset_selection_matrix = np.zeros((self.n_clients,self.n_classes))
        self.client_dataset_selection_matrix_groups = np.zeros((self.n_clients,sum(self.n_categories_per_group)))
        self.dist_classes = dist_classes
        self.oracle_states_classes = np.zeros((self.N_learners,1)) ### Learner states for each round

        self.oracle_states_groups = np.zeros((self.N_learners,1)) ### Learner states for each round
        self.client_dataset_path = client_dataset_path
        self.create_client_datasets_files("mnist")

        ### Learner parameters

    def initialize_client_dataset(self):
        
        self.client_dataset = np.random.choice(np.arange(self.N_max), size = (self.n_clients,self.n_classes))
        self.client_dataset = np.zeros_like(self.client_dataset)
        for client in range(self.n_clients):
            p_client = np.random.dirichlet(self.client_dirichlet_alpha_class)
            p_client = [0.33,0.33,0.34,0,0,0,0,0,0,0]
            np.random.shuffle(p_client)
            
            clients_class_indices = np.random.choice(np.arange(self.n_classes), size = self.n_datapoints,p = p_client)
            for class_index in clients_class_indices:
                self.client_dataset[client,class_index] += 1
        self.client_dataset_size = self.client_dataset.sum(axis=1) # total number of samples
        if self.client_dataset_size.min() == 0:
            self.initialize_client_dataset()
            return
        self.client_class_coefficient = self.client_dataset/self.client_dataset_size.reshape(self.n_clients,1) # class coefficient for each client
    def get_class_dist(self):
        return self.client_dataset.sum(axis=0)

    
    def update_client_selection_matrix(self,u,learner_class_preference):
        
        if u==-1:
            return
        #### Sample datapoints for each client 
        self.client_dataset_selection_matrix = np.zeros((self.n_clients,self.n_classes))
        #### Compute probability of each class for each client
        p_class = np.zeros((self.n_classes,))
        learner_class_preference = np.array(learner_class_preference)
        
        p_class[learner_class_preference] = 1# self.dist_classes[u].reshape(-1,1)
        ### Set other values to 1 - 2*self.dist_classes[u]
        # other_classes = [i for i in range(self.n_classes) if i not in learner_class_preference]
        # p_class[other_classes] = (1 - sum(self.dist_classes[u]))/(self.n_classes-len(self.dist_classes[u]))
        # p_class = p_class.flatten()
        # participating_clients = np.where(self.client_selection_matrix==1)[0]
        # class_coefficient = self.client_class_coefficient[participating_clients].mean(axis=0)
        # class_coefficient = class_coefficient/np.sum(class_coefficient)
        # p_class = p_class*class_coefficient + 1e-6
        p_class = p_class/np.sum(p_class)
        # print("p_class: ",p_class)
        for client in range(self.n_clients):
            if self.client_selection_matrix[client]:
                sample_factor = [10,3,1.1][u]
                dataset_size = np.random.randint(self.n_datapoints//sample_factor,self.n_datapoints) ### Dataset size for each participating client
                try:
                    class_size_coeff = p_class#np.random.dirichlet(p_class)
                except:
                    print("Error",p_class)
                    import pdb;pdb.set_trace()
                    class_size_coeff = np.ones_like(p_class)/len(p_class)
                self.client_dataset_selection_matrix[client] = (class_size_coeff*dataset_size).astype(int)
                self.clients[client].sample_training_data(self.client_dataset_selection_matrix[client])

        self.class_dist = self.client_dataset_selection_matrix.sum(axis=0)
        self.class_dist = self.class_dist/np.sum(self.class_dist)

    def update_client_partition_matrix(self):
        for client in range(self.n_clients):
            self.client_selection_matrix[client] = np.random.choice([0,1],p = self.p_stay[int(self.client_selection_matrix[client])])
        if self.client_selection_matrix.sum() == 0:
            random_client = np.random.randint(0,self.n_clients)
            self.client_selection_matrix[random_client] = 1
        
    def get_oracle_states(self,type_state="group"):
        participating_clients = np.where(self.client_selection_matrix==1)[0]

        class_coefficient = self.client_class_coefficient[participating_clients].mean(axis=0)
        class_coefficient = class_coefficient/np.sum(class_coefficient)
        self.class_coefficient = class_coefficient

        for learner in range(self.N_learners):
            learner_class_coefficient = class_coefficient[self.learner_class_preference[learner][0]]+class_coefficient[self.learner_class_preference[learner][1]]
            self.oracle_states_classes[learner] = np.where(self.state_thresholds<=learner_class_coefficient)[0][-1]
        if type_state == "class":
            return self.oracle_states_classes.astype(int)
        
        

    def get_oracle_success(self,learner,type_state="group"):
        round_success_coefficient = self.class_dist[self.learner_class_preference[learner][0]]+self.class_dist[self.learner_class_preference[learner][1]]
        if type_state == "class":
            return self.success_thresholds <= round_success_coefficient
        
    

    #### Compute probability transition matrix for each learner
    #### Test for Oracle States
    def return_oracle_probability_test(self,N_times,type_state="group"):
        P = np.zeros((self.N_learners,len(self.state_thresholds),len(self.state_thresholds)))

        for i in range(N_times):
            old_oracle_states = self.get_oracle_states(type_state).flatten()
            if type_state == "group":
                self.update_client_selection_matrix_by_group(-1)
            else:
                self.update_client_selection_matrix(-1,[0,1])
            oracle_states = self.get_oracle_states(type_state).flatten()

            for learner in range(self.N_learners):
                P[learner,old_oracle_states[learner],oracle_states[learner]] += 1
        P = P/P.sum(axis=2,keepdims=True)
        return P
    
    ### Probabilities for Different Actions for each learner
    ### Test for Failure Probabilities 
    def return_success_probability_test(self,N_times,learner_preference,type_state="group"):
        fs = np.zeros((self.N_learners,len(self.U),len(self.state_thresholds)))
        oracle_counts = np.zeros((self.N_learners,len(self.U),len(self.state_thresholds)))
        for u in self.U: 
            for i in tqdm.tqdm(range(N_times)):
                oracle_states = self.get_oracle_states(type_state).flatten()
                for learner in range(self.N_learners):
                    if type_state == "group":
                        self.update_client_selection_matrix_by_group(u,learner_preference[learner])
                    else:
                        self.update_client_selection_matrix(u,learner_preference[learner])
                    oracle_counts[learner,u,oracle_states[learner]] += 1
                    if self.get_oracle_success(learner,type_state):
                        fs[learner,u,oracle_states[learner]] += 1
        fs = fs/oracle_counts
        return fs
    

    def create_client_datasets_files(self,dataset_name):
        df = pd.read_csv("data/"+dataset_name+"_train.csv")
        self.dataset_name = dataset_name
        for client in range(self.n_clients):
            df_client = pd.DataFrame()
            for class_idx in range(self.n_classes):
                df_class = df[df['label']==class_idx]
                df_class = df_class.sample(n=self.client_dataset[client,class_idx])
                df_client = pd.concat([df_client,df_class])
            df_client.to_csv(self.client_dataset_path+dataset_name+"_client_"+str(client)+".csv",index=False)

    def initialize_clients(self,neural_network):
        self.clients = []
        for client_id in range(self.n_clients):
            self.clients.append(Client(client_id,self.client_dataset_path+self.dataset_name+"_client_"+str(client_id)+".csv",neural_network))
    
    def train(self,weights):
        losses = []

        for client in tqdm.tqdm(np.nonzero(self.client_selection_matrix)[0]):
            self.clients[client].set_weights(weights)
            losses.append(self.clients[client].train())
        return np.mean(losses)

    def aggregate_weights(self):
         
        weights = self.clients[0].get_weights()
        ### zero out the weights
        for key in weights.keys():
            weights[key] = weights[key]*0
        for client in np.nonzero(self.client_selection_matrix.flatten()==1)[0]:
            for key in weights.keys():
                weights[key] += self.clients[int(client)].get_weights()[key]
        
        for key in weights.keys():
            weights[key] = weights[key]/np.sum(self.client_selection_matrix)
        return weights
