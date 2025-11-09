import torch
from skeletalmodel import json_trainer, load_model
from dataset import JSONDataset

json_trainer(
    h_config=[8],
    model_path="model.pth",
    file_path="train.json",
    batch_size=4,
    epochs=1000,
    LR=0.001,
    features_name="features",
    label_name="label",
    loss_type="mse"
)
model = load_model("model.pth")


try:
    test_data = JSONDataset("test.json", "features", "label", "mse")
        
except Exception as e:
    print(f"Error loading or processing test file: {e}")
    
# run model
with torch.no_grad():
    predictions = model(test_data.X)
    

print("\nModel Predictions vs. Actual Labels:")
print("------------------------------------------")
for i in range(len(predictions)):
    pred = predictions[i]
    
    label = test_data.y[i]
    
    input_features = test_data.X[i] 
    
    print(f"Sample {i + 1}:")
    
    print(f"  Input:     {input_features.tolist()}")
    
    print(f"  Prediction: [{pred[0]:.2f}, {pred[1]:.2f}]")
    print(f"  Actual:     [{label[0]:.2f}, {label[1]:.2f}]\n")
