import numpy as np
import pandas as pd
import torch
from components.net import CNN
#### Client 
##### Initialize client with dataset
##### Define NN Model for client
##### Mechanism to sample data from client dataset
##### Train NN with respect to sampled data
##### Validate NN with respect to standard validation dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Client():
    def __init__(self,
                 client_id, 
                 client_dataset_path="data/client_datasets/",
                    neural_network=CNN,
                    n_classes = 10
    ):
        self.client_id = client_id
        self.client_dataset_path = client_dataset_path
        self.neural_network = neural_network().to(DEVICE)
        self.df = pd.read_csv(self.client_dataset_path)
        self.n_classes = n_classes
        self.training_data = pd.DataFrame()
        
    def set_weights(self,weights):
        self.neural_network.load_state_dict(weights)
    
    def get_weights(self):
        return self.neural_network.state_dict()
    def sample_training_data_group(self,dataset_selection_row,n_genders,n_races):
        self.training_data = pd.DataFrame()
        for gender in range(n_genders):
            for race in range(n_races):
                n_samples = dataset_selection_row[:,gender + n_races*race].sum()
                if n_samples> len(self.df)
        for sample in range(self.n_data)
    def sample_training_data(self,dataset_selection_row):

        self.training_data = pd.DataFrame()

        for class_idx in range(self.n_classes):
            if dataset_selection_row[class_idx] != 0 and self.df[self.df['label']==class_idx].size != 0:
                # print("class_idx: ",class_idx,", dataset_selection_row[class_idx]: ",dataset_selection_row[class_idx],", self.df[self.df['label']==class_idx].size: ",len(self.df[self.df['label']==class_idx]))
                if dataset_selection_row[class_idx] > len(self.df[self.df['label']==class_idx].index):
                    sampled_training_data = self.df[self.df['label']==class_idx].index
                else:
                    sampled_training_data = np.random.choice(self.df[self.df['label']==class_idx].index, size=int(dataset_selection_row[class_idx]), replace=False)
                self.training_data = pd.concat([self.training_data,self.df.iloc[sampled_training_data,:]])
        self.training_data = self.training_data.reset_index(drop=True)


    
    def train(self):
        if self.training_data.shape[0] == 0:
            print("No training data available for client ",self.client_id)
            return 0
        x = torch.from_numpy(self.training_data.iloc[:,1:].values).float().to(self.neural_network.device)
        y = torch.from_numpy(self.training_data.iloc[:,0].values).long().to(self.neural_network.device)
        loss = self.neural_network.train(x,y)
        return loss



               