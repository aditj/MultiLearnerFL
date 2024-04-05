
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import torch as torch
import tqdm as tqdm
import os 

# oracle = client_oracle.Oracle()


### Load UTKFace Dataset
dataset_path = "data/archive/UTKFace/"
files = os.listdir(dataset_path)
### the first three characters separated by _ of the file name are the age,gender and race of the person
data = []
for file_name in files:

    if "jpg" in file_name.split("_")[2]:
        continue
    data.append(file_name.split("_")[:4])
data = pd.DataFrame(data,columns=["age","gender","race","label"])
print(data.describe())
data.to_csv("data/utkface.csv",index=False)

data = pd.read_csv("data/utkface.csv")
n_clients = 20
n_data_per_client = 1000
n_genders = 2
gender_dist = data['gender'].value_counts()
gender_dist = gender_dist/gender_dist.sum()
n_race = data['race'].unique().shape[0]
race_dist = data['race'].value_counts()
race_dist = race_dist/race_dist.sum()
for client in tqdm.tqdm(range(n_clients)):
    client_race = np.random.choice(np.arange(n_race),size=1,p= race_dist.values)
    client_gender = np.random.choice(np.arange(n_genders),size=1,p=gender_dist.values)
    sample_indices = data[()]

exit()
n_clients = oracle.n_clients
n_classes = oracle.n_classes
for client in tqdm.tqdm(range(n_clients)):
    ### Filter data using client_dataset 
    client_classes = oracle.client_dataset[client]
    client_dataset = []
    for class_index in np.arange(n_classes):
        client_dataset.append(data[data['label']==class_index].sample(n=client_classes[class_index]))
    
    client_dataset = pd.concat(client_dataset)
    client_dataset = client_dataset.reset_index(drop=True)
    client_dataset.to_csv("data/client_dataset/"+str(client)+".csv",index=False)
print("Created Client Datasets")