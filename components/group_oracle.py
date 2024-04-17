import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm
import torch
import pandas as pd
from components.client import Client
class Oracle():
    def __init__(self,
            U=[0,1,2],
                n_clients = 80,
                n_classes = 10,
                n_group_attributes = 2,
                n_categories_per_group = [2,5],
                N_max = 60,
                p_stay = [[0.7,0.3],[0.3,0.7]],
                N_learners = 5,
                client_dirichlet_alpha = [[0.6,0.4],[0.42,0.2,0.16,0.14,0.08]],
                state_thresholds_groups = [0,0.05,0.1,0.2,0.3],
                success_thresholds_groups = 0.75,
                learner_group_preference =  [[0,1],[0,3],[1,0],[1,1],[0,2]],
                N_mins = [0,0.5,1], ### Minimum number of samples per class per client
                dist_groups = [[[0.5,0.3],[0.7,0.1,0.1,0.1,0.1]],
                                [[0.6,0.4],[0.8,0.2,0.2,0.2,0.2]],
                                [[0.7,0.7],[0.9,0.5,0.5,0.5,0.5]]
                ],
                client_dataset_path = "data/client_dataset/",
                dataset_name = "utkface"
                
                 ):
        self.U = U
        self.n_clients = n_clients
        self.n_classes = n_classes
        self.n_group_attributes = n_group_attributes
        self.n_categories_per_group = n_categories_per_group
        self.N_max = N_max
        self.p_stay = p_stay
        self.N_learners = N_learners
        self.client_dataset_path = client_dataset_path
        self.dataset_name = dataset_name
        self.initialize_client_dataset()
        self.client_dirichlet_alpha = client_dirichlet_alpha
        self.state_thresholds_groups = state_thresholds_groups
        self.success_thresholds_groups = success_thresholds_groups
        self.learner_group_preference = learner_group_preference
        self.N_mins = N_mins*self.N_max
        self.dist_groups = dist_groups
        self.client_selection_matrix = np.random.choice([0,1],(self.n_clients,1))
        self.client_dataset_selection_matrix_groups = np.zeros((self.n_clients,sum(self.n_categories_per_group)))
        
        self.oracle_states_groups = np.zeros((self.N_learners,1)) ### Learner states for each round
        
        ### Learner parameters

    def initialize_client_dataset(self):
        data = pd.read_csv("data/utkface.csv")

        self.n_data_per_client = 200
        self.client_dataset = np.zeros((self.n_clients,self.n_data_per_client,sum(self.n_categories_per_group)))

        self.n_genders = 2
        gender_dist = data['gender'].value_counts()
        gender_dist = gender_dist/gender_dist.sum()
        self.n_races = data['race'].unique().shape[0]
        race_dist = data['race'].value_counts()
        race_dist = race_dist/race_dist.sum()

        for client in range(self.n_clients):
            client_dataset = pd.DataFrame(columns=data.columns)
            p_client_gender = np.random.dirichlet(gender_dist)
            p_client_race = np.random.dirichlet(race_dist)
            for data_index in range(self.n_data_per_client):
                sample_race = np.random.choice(np.arange(self.n_races),size=1,p=p_client_race)[0]
                sample_gender = np.random.choice(np.arange(self.n_genders),size=1,p=p_client_gender)[0]
                
                try:
                    sample_idx = data[(data['race']==sample_race)&(data['gender']==sample_gender)].sample(n=1).index

                    client_dataset = pd.concat([client_dataset,data.loc[sample_idx]])

                    data = data.drop(sample_idx)
                    self.client_dataset[client,data_index,sample_gender]=1
                    self.client_dataset[client,data_index,self.n_genders+sample_race]=1
                except ValueError:
                    print(data.shape)

                    print("No data left",sample_gender,sample_race)
                    continue
            client_dataset.to_csv(self.client_dataset_path+self.dataset_name+"_client_"+str(client)+".csv")
        self.client_gender_coefficient = self.client_dataset[:,:,:2].sum(axis=1)/self.n_data_per_client
        self.client_race_coefficient = self.client_dataset[:,:,2:].sum(axis=1)/self.n_data_per_client
        
    def update_client_selection_matrix(self,u,learner_group_preference):
        
        if u==-1:
            return
        #### Sample datapoints for each client 
        self.client_dataset_selection_matrix = np.zeros((self.n_clients,self.n_data_per_client,sum(self.n_categories_per_group)))
        self.group_dist = np.zeros((sum(self.n_categories_per_group),1))
        #### Compute probability of each class for each client
        p_race = np.zeros((self.n_races))
        learner_gender_preference = learner_group_preference[0]
        learner_race_preference = learner_group_preference[1]
        p_gender = np.zeros((self.n_genders))
        p_gender[learner_gender_preference] = self.dist_groups[u][0][learner_gender_preference]
        p_gender[1-learner_gender_preference] = 1 - p_gender[learner_gender_preference]
        
        p_race[learner_race_preference] = self.dist_groups[u][1][learner_race_preference]
        other_races = [i for i in range(self.n_races) if i != learner_race_preference]
        p_race[other_races] = (1 - sum(p_race))/(self.n_races-1)
        p_race = p_race.flatten()
        participating_clients = np.where(self.client_selection_matrix==1)[0]
        gender_coefficient = self.client_gender_coefficient[participating_clients].mean(axis=0)
        gender_coefficient = gender_coefficient/np.sum(gender_coefficient)
        p_gender = p_gender*gender_coefficient + 1e-6
        p_gender = p_gender/np.sum(p_gender)

        race_coefficient = self.client_race_coefficient[participating_clients].mean(axis=0)
        race_coefficient = race_coefficient/np.sum(race_coefficient)
        race_coefficient = p_race*race_coefficient + 1e-6
        p_race = p_race/np.sum(p_race)


        for client in range(self.n_clients):
            if self.client_selection_matrix[client]:
                dataset_size = np.random.randint(1,self.n_data_per_client) ### Dataset size for each participating client
                race_size_coeff = np.random.dirichlet(p_race)
                gender_size_coeff = np.random.dirichlet(p_gender)              
                race_samples = np.random.choice(np.arange(self.n_races),size=dataset_size,p = race_size_coeff)
                gender_samples = np.random.choice(np.arange(self.n_genders),size=dataset_size,p = gender_size_coeff)
                self.client_dataset_selection_matrix[client,:,gender_samples] = 1
                self.client_dataset_selection_matrix[client,:,self.n_genders + race_samples] = 1
                self.clients[client].sample_training_data_group(self.client_dataset_selection_matrix[client],self.n_genders,self.n_races)
                self.group_dist += self.clients[client].get_ohe_training_data(self.n_genders,self.n_races).sum(axis=0).reshape(-1,1)
       
        self.group_dist[0:self.n_categories_per_group[0]] = self.group_dist[0:self.n_categories_per_group[0]]/np.max(self.group_dist[0:self.n_categories_per_group[0]])
        self.group_dist[self.n_categories_per_group[0]:] = self.group_dist[self.n_categories_per_group[0]:]/np.max(self.group_dist[self.n_categories_per_group[0]:])
    def update_client_participation_matrix(self):
        for client in range(self.n_clients):
            self.client_selection_matrix[client] = np.random.choice([0,1],p = self.p_stay[int(self.client_selection_matrix[client])])
        if self.client_selection_matrix.sum() == 0:
            random_client = np.random.randint(0,self.n_clients)
            self.client_selection_matrix[random_client] = 1
        
    def get_oracle_states(self,type_state="group"):
        participating_clients = np.where(self.client_selection_matrix==1)[0]

        gender_coefficient = self.client_gender_coefficient[participating_clients].sum(axis=0)
        gender_coefficient = gender_coefficient/np.sum(gender_coefficient)

        race_coefficient = self.client_race_coefficient[participating_clients].sum(axis=0)
        race_coefficient = race_coefficient/np.sum(race_coefficient)



        for learner in range(self.N_learners):
            learner_gender_coefficient = gender_coefficient[self.learner_group_preference[learner][0]]
            learner_race_coefficient = race_coefficient[self.learner_group_preference[learner][1]]
            learner_group_coefficient = learner_gender_coefficient*learner_race_coefficient
            self.oracle_states_groups[learner] = np.where(self.state_thresholds_groups<=learner_group_coefficient)[0][-1]

        return self.oracle_states_groups.astype(int)
    

        

    def get_oracle_success(self,learner,type_state="group"):
        round_success_coefficients_groups = self.group_dist[self.learner_group_preference[learner][0]] + self.group_dist[self.n_genders + self.learner_group_preference[learner][1]]
        return self.success_thresholds_groups <= round_success_coefficients_groups[0]

    
    def plot_group_dist_for_rounds(self,N_rounds,action = 2):
            
            group_counts = np.zeros((N_rounds,sum(self.n_categories_per_group)))
            for i in range(N_rounds):        
                self.update_client_selection_matrix_by_group(action)
                for group_index in range(sum(self.n_categories_per_group)):
                    group_counts[i,group_index] = self.client_dataset_selection_matrix_groups[:,group_index].sum()
    
            learner_group_count = np.zeros((self.N_learners,N_rounds))
            plt.figure()
            for learner in range(self.N_learners):
                learner_group_count[learner] = group_counts[:,self.learner_group_preference[learner]]
                plt.plot(np.arange(N_rounds),learner_group_count[learner],label = "Learner "+str(learner))
            plt.legend()
            plt.savefig("plots/group_dist.png")
            plt.close()

    #### Compute probability transition matrix for each learner
    #### Test for Oracle States
    def return_oracle_probability_test(self,N_times,type_state="group"):
        P = np.zeros((self.N_learners,len(self.state_thresholds),len(self.state_thresholds)))

        for i in range(N_times):
            old_oracle_states = self.get_oracle_states(type_state).flatten()
            self.update_client_selection_matrix_by_group(-1)
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
                    self.update_client_selection_matrix(u,learner_preference[learner])
                    oracle_counts[learner,u,oracle_states[learner]] += 1
                    if self.get_oracle_success(learner,type_state):
                        fs[learner,u,oracle_states[learner]] += 1
        fs = fs/oracle_counts
        return fs
    

    def initialize_clients(self,neural_network):
        self.clients = []
        for client_id in range(self.n_clients):
            self.clients.append(Client(client_id,self.client_dataset_path+self.dataset_name+"_client_"+str(client_id)+".csv",neural_network))
    

