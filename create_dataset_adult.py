
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import torch as torch
import tqdm as tqdm
from components import client_oracle 
import os 
import pickle as pkl


### Load UTKFace Dataset
dataset_path = "data/archive/UTKFace/"
files = os.listdir(dataset_path)

### the first three characters separated by _ of the file name are the age,gender and race of the person
data = []
for file_name in files:
    if "Zone.Identifier" in file_name:
        continue
    data.append(file_name.split("_")[:4])
data = pd.DataFrame(data,columns=["age","gender","race","name"])
data = data[data['race'].isin([str(x) for x in [0,1,2,3,4]])].reset_index(drop=True)

race_columns = ['race_' + str(x) for x in [0,1,2,3,4]]

#### create binary labels for one hot encoding of race
def ohe(x,n):
    x_ohe = np.zeros(n)
    x_ohe[x] = 1
    return x_ohe
data['race_ohe'] = data['race'].apply(lambda x: ohe(int(x),n=5))
data['gender'] = data['gender'].apply(lambda x: 1 if x=='1' else 0)
data[race_columns] = data['race_ohe'].to_list()
data.drop(columns=['race_ohe','race'],inplace=True)

gender_dist_factor = 0.6
race_dist_factor = [0.4,0.2,0.2,0.1,0.1]

n_clients = 100
client_files = [{"files":[],"racedist":[0 for i in range(5)],"genderdist":[0,0]} for i in range(n_clients)]
n_samples  = 200

CREATE_CLIENT_DATASETS = 0
if CREATE_CLIENT_DATASETS:
    for client in tqdm.tqdm(range(n_clients)):
        ### Filter data using client_dataset 
        race_dist = np.random.dirichlet(race_dist_factor,size=1).flatten()
        gender_dist = np.random.dirichlet([gender_dist_factor,1-gender_dist_factor],size=1).flatten()
        for race in range(5):
            for gender in range(2):
                try: 
                    client_datapoints = data[(data['race_'+str(race)]==1) & (data['gender']==gender)].sample(n=int(n_samples*race_dist[race]*gender_dist[gender]))
                    data = data.drop(client_datapoints.index)
                    client_files[client]['files'].extend(client_datapoints['name'].to_list())
                    client_files[client]['racedist'][race] += len(client_datapoints['name'].to_list())
                    client_files[client]['genderdist'][gender] += len(client_datapoints['name'].to_list())
                except:
                    print(f"Length of data: {len(client_datapoints)}")
                    pass

    print("Created Client Datasets")
    with open("data/client_datasets.pkl","wb") as f:
        pkl.dump(client_files,f)

with open("data/client_datasets.pkl","rb") as f:
    client_files = pkl.load(f)

N_MC = 100
N_rounds = 100
learner_preference = [([4],[0,1]),([1,2],[1]),([0],[1]),([3],[0,1]),([0],[0])]
N = 5
O = 3
P_O = np.zeros((N_MC,N,O,O))

def compute_oracle_state(learner_preference,participating_race_dict,participating_gender_dict):
    preference_thresholds = [0,0.05,0.1,0.15,0.2]
    preference_thresholds = [0,0.1,0.2]
    race_prop = 0
    gender_prop = 0
    for race in learner_preference[0]:
        race_prop+= participating_race_dict[race]/np.sum(participating_race_dict)
    for gender in learner_preference[1]:
        gender_prop+= participating_gender_dict[gender]/np.sum(participating_gender_dict)
        
    total_prop = race_prop*gender_prop
    # print(total_prop)
    return np.digitize(total_prop,preference_thresholds)-1
race_dist_rounds = np.zeros((N_MC,N_rounds,5))
gender_dist_rounds = np.zeros((N_MC,N_rounds,2))
RUN_MC =0
if RUN_MC:
    for mc in tqdm.tqdm(range(N_MC)):
        client_participation = np.random.binomial(1,0.2,size=(n_clients))
        participation_probability =[[0.7,0.3],[0.3,0.7]]


        oracle_states = np.zeros((N,1))   

        for round in range(N_rounds):
            
            
            participating_clients = np.where(client_participation==1)[0]
            participating_race_dict = np.zeros(5)
            participating_gender_dict = np.zeros(2)
            for client in participating_clients:
                participating_race_dict += np.array(client_files[client]['racedist'])
                participating_gender_dict += np.array(client_files[client]['genderdist']) 
            race_dist_rounds[mc,round] = participating_race_dict
            gender_dist_rounds[mc,round] = participating_gender_dict
            for learner in range(N):
                
                new_oracle_state = compute_oracle_state(learner_preference[learner],participating_race_dict,participating_gender_dict)
                if round > 0:
                    P_O[mc,learner,int(oracle_states[learner]),new_oracle_state] += 1

                #print(oracle_state,new_oracle_state)
            for learner in range(N):
                oracle_states[learner] = compute_oracle_state(learner_preference[learner],participating_race_dict,participating_gender_dict)
            for client in range(n_clients):
                client_participation[client] = np.random.choice([0,1],p=participation_probability[client_participation[client]])
            print(sum(client_participation))
    np.save("data/parameters/race_dist_round.npy",race_dist_rounds)
    np.save("data/parameters/gender_dist_round.npy",gender_dist_rounds)
    np.save("data/parameters/P_O.npy",P_O)
P_O = np.load("data/parameters/P_O.npy")
race_dist_rounds = np.load("data/parameters/race_dist_round.npy")
gender_dist_rounds = np.load("data/parameters/gender_dist_round.npy")
print(race_dist_rounds.mean(axis=(0)))
print(gender_dist_rounds.mean(axis=(0)))

P_O = P_O.mean(axis=0)
# P_O = P_O/P_O.sum(axis=2,keepdims=True)
### numpy print options
np.set_printoptions(precision=2)
print(P_O)