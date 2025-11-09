import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataset import JSONDataset
from torch.utils.data import DataLoader

# simple vectors as inputs, outputs logits
class SkeletalModel(nn.Module):
    # hidden_layers is a list of sizes with a length of the number of hidden layers, i.e [128, 64, 32]
    def __init__(self, n_inputs, n_outputs, hidden_layers):
        
        super(SkeletalModel, self).__init__()
        
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(n_inputs, hidden_layers[0]))
        
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            

        self.output_layer = nn.Linear(hidden_layers[-1], n_outputs)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x)) # passes the output of the last layer into the next until the last hiddedn layer
            
        # output layer
        x = self.output_layer(x)
        
        # return logits
        return x

        
# loads trained model

def load_model(model_path):
    print(f"Loading model from {model_path}...")
    try:
        # load model data
        data = torch.load(model_path)
        
        # get model details
        n_ip = data['n_inputs']
        x_op = data['n_outputs']
        h_config = data['hidden_layers']
        
        # remake model
        model = SkeletalModel(
            n_inputs=n_ip, 
            n_outputs=x_op, 
            hidden_layers=h_config
        )

        # load weights
        model.load_state_dict(data['model_state'])
        
        print("Model loaded successfully.")
        model.eval()
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()
        
        
def json_trainer(h_config, model_path, file_path, batch_size, epochs, LR, features_name, label_name, loss_type):
    
    dataset = JSONDataset(file_path, features_name, label_name, loss_type=loss_type)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    n_ip = dataset.n_inputs
    x_op = dataset.n_outputs
    
    print(f"Data loaded: Found {len(dataset)} samples.")
    print(f"Auto-detected n_inputs: {n_ip}")
    print(f"Auto-detected n_outputs: {x_op}")
    print(f"Using loss function: {loss_type}")
    
    model = SkeletalModel(
        n_inputs=n_ip, 
        n_outputs=x_op, 
        hidden_layers=h_config
    )
    
    if loss_type == "mse":
        criterion = nn.MSELoss()
    elif loss_type == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif loss_type == "bce":
        # 0 or 1 classifications
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_X, batch_y in data_loader:
            
            if loss_type == "cross_entropy":
                # Cross_entropy expects (N) as the shape not (N,1)
                batch_y = batch_y.squeeze(1)
                
            # forward pass
            outputs = model(batch_X)
            # loss calc
            loss = criterion(outputs, batch_y)
            
            # backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # update total loss
            total_loss += loss.item()
            
        # gives loss every 5 epochs
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(data_loader)
            print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')
            
            
    print("--- Training Complete ---")
    
    data_to_save = {
        'model_state': model.state_dict(),
        'n_inputs': n_ip,
        'n_outputs': x_op,
        'hidden_layers': h_config,
        'loss_type': loss_type
    }
    
    try:
        torch.save(data_to_save, model_path)
        print(f"Model and config saved successfully to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")