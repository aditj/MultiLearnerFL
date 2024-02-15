
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import torch as torch
import tqdm as tqdm
from components import client_oracle 

oracle = client_oracle.Oracle()


### Load MNIST Dataset
data = pd.read_csv("data/mnist_train.csv")

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