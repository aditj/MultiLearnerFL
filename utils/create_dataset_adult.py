
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import torch as torch
import tqdm as tqdm
import os 

# oracle = client_oracle.Oracle()


### Load UTKFace Dataset

### the first three characters separated by _ of the file name are the age,gender and race of the person
if not os.path.exists("data/utkface.csv"):
    data = []
    dataset_path = "data/archive/UTKFace/"
    files = os.listdir(dataset_path)
    for file_name in files:

        if "jpg" in file_name.split("_")[2]:
            continue
        data.append(file_name.split("_")[:4])
    data = pd.DataFrame(data,columns=["age","gender","race","label"])
    print(data.describe())
    data.to_csv("data/utkface.csv",index=False)

data = pd.read_csv("data/utkface.csv")
print(data.describe())
n_clients = 80
n_data_per_client = 100
n_genders = 2
gender_dist = data['gender'].value_counts()
gender_dist = gender_dist/gender_dist.sum()
n_race = data['race'].unique().shape[0]
race_dist = data['race'].value_counts()
race_dist = race_dist/race_dist.sum()
gender_counts = np.zeros(n_genders)
race_counts = np.zeros(n_race)
clients = []
for client in tqdm.tqdm(range(n_clients)):
    client_race = np.random.choice(np.arange(n_race),size=1,p= race_dist.values)[0]
    client_gender = np.random.choice(np.arange(n_genders),size=1,p=gender_dist.values)[0]
    gender_counts[client_gender] += 1
    race_counts[client_race] += 1
    sample_indices = data[(data['race']==client_race)&(data['gender']==client_gender)].sample(n=n_data_per_client).index
    client_data = data.loc[sample_indices]
    data = data.drop(sample_indices)
    # client_data.to_csv("data/client_dataset/"+str(client)+".csv",index=False)
    clients.append([client_race,client_gender])
print(race_counts,gender_counts)
data = pd.read_csv("data/utkface.csv")

clients = np.zeros((n_clients,n_data_per_client,n_genders+n_race))
gender_counts = np.zeros((n_clients,n_genders))
race_counts = np.zeros((n_clients,n_race))
for client in tqdm.tqdm(range(n_clients)):
    client_race_dist = np.random.dirichlet(race_dist.values)
    client_gender_dist = np.random.dirichlet(gender_dist.values)
    client_data = []
    for data_index in range(n_data_per_client):
        sample_race = np.random.choice(np.arange(n_race),size=1,p=client_race_dist)[0]
        sample_gender = np.random.choice(np.arange(n_genders),size=1,p=client_gender_dist)[0]
        
        try:
            sample_idx = data[(data['race']==sample_race)&(data['gender']==sample_gender)].sample(n=1).index
            data = data.drop(sample_idx)
            gender_counts[client,sample_gender] += 1
            race_counts[client,sample_race] += 1
        
            clients[client,data_index,sample_gender]=1
            clients[client,data_index,n_genders+sample_race]=1
        except ValueError:
            print(data.shape)
            print(data[(data['race']==client_race)],data[(data['gender']==client_gender)])

            print("No data left",sample_gender,sample_race)
            continue
        

    # client_data.to_csv("data/client_dataset/"+str(client)+".csv",index=False)
print(race_counts,gender_counts)

print(clients)
np.save("parameters/clients.npy",clients)
