
#### Learner class ####
#### Learning Task: MNIST Image Classification
#### Initiliaze Neural Network, Validation Dataset (Preference of Learner's Class) 
#### Methods
#### 1. Get Weights of Neural Network
#### 2. Evaluate on Validation Dataset
#### 3. Update Weights of Neural Network
import pandas as pd
import torch
class Learner():
    def __init__(
            self,
            class_preference,
            neural_network,
            create_validation_dataset=True,
            dataset_path = "./data/mnist_train.csv",
            validation_data_per_class = 100
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.class_preference = class_preference
        self.neural_network = neural_network().to(device)
        self.dataset_path = dataset_path
        self.validation_data_per_class = validation_data_per_class

        if create_validation_dataset:
            self.init_validation_dataset()
        
    def init_validation_dataset(self):
        df = pd.read_csv(self.dataset_path)
        self.validation_dataset = []
        for class_index in self.class_preference:
            self.validation_dataset.append(df[df['label']==class_index].sample(n=self.validation_data_per_class))
        self.validation_dataset = pd.concat(self.validation_dataset)

    def get_weights(self):
        return self.neural_network.state_dict()
    
    def evaluate(self):
        x = self.validation_dataset.iloc[:,1:].values
        y = self.validation_dataset.iloc[:,0].values
        x = torch.from_numpy(x).float().to(self.neural_network.device)
        y = torch.from_numpy(y).long().to(self.neural_network.device)
        output = self.neural_network.forward(x)
        _, predicted = torch.max(output.data, 1)
        total = y.size(0)
        correct = (predicted == y).sum().item()
        return correct/total
    def set_weights(self,weights):
        self.neural_network.load_state_dict(weights)