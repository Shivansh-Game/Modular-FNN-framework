import torch
import json
from torch.utils.data import Dataset

class JSONDataset(Dataset):
    def __init__(self, file_path, features_name, label_name, loss_type):
        with open(file_path, 'r') as f:
            data = json.load(f)

        X_list = [item[features_name] for item in data]
        y_list = [item[label_name] for item in data]
        
        # turn em into tensors
        self.X = torch.tensor(X_list, dtype=torch.float32)
        
        if loss_type == "cross_entropy":
            # it expects long data type values
            self.y = torch.tensor(y_list, dtype=torch.long)
        elif loss_type == "mse" or loss_type == "bce":
            # These expect floats
            self.y = torch.tensor(y_list, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Supported: 'mse', 'cross_entropy', 'bce'")

        # for 1d labels
        if self.y.dim() == 1:
            self.y = self.y.unsqueeze(1)
        
        
        self.n_inputs = self.X.shape[1]
        if loss_type == "cross_entropy":
             self.n_outputs = len(torch.unique(self.y))
        else:
             self.n_outputs = self.y.shape[1]
            
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
