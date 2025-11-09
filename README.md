# Modular FNN Framework

Simple and reusable framework to train FNNs for numerical json data

## Features
  * **The Model** The `SkeletalModel` class builds an FFN with any number of hidden layers based on the `h_config` list (e.g., `[128, 64, 32]`).
  * **Shape Detection:** The `json_trainer` looks at the `JSONDataset` to determine the `n_inputs` and `n_outputs` for the model.
  * **Multi-Loss Support:** Easily switch between loss criterion by changing the `loss_type` parameter. Supported types:
      * `"mse"`: For regression tasks (predicting numbers).
      * `"cross_entropy"`: For multi-class classification (predicting categories).
      * `"bce"`: For binary classification (predicting 0 or 1).
  * **Save/Load:** Saves both the model weights (`model_state`) and its architecture configuration (`n_inputs`, `n_outputs`, `hidden_layers`, `loss_type`) into a single `.pth` file for safe and easy loading.

## Requirements
  * Python 3
  * PyTorch

## 📁 File Structure

```
.
├── main.py             # script to run training and testing
├── skeletalmodel.py    # Core logic: SkeletalModel, json_trainer, load_model
├── dataset.py          # JSONDataset class
├── train.json          # training data
└── test.json           # testing data
```

-----

## How to Use

### 1. Format Your JSON Data

The `JSONDataset` class expects a JSON file containing a single list of data objects.

**For Regression (`loss_type="mse"`): (Note it's the same for BCE but the label should be either [0.0] or [0.1])**

  * `features`: A list of numbers.
  * `label`: A list of numbers to predict.

*`train.json` (Example)*

```json
[
  {
    "features": [10.0, 10.0, 5.0, 0.0, 0.0, 0.0],
    "label": [15.0, 10.0]
  },
  {
    "features": [20.0, 20.0, 0.0, 10.0, 0.0, 0.0],
    "label": [20.0, 30.0]
  }
]
```


**For Classification (`loss_type="cross_entropy"`):**

  * `features`: A list of numbers.
  * `label`: A single integer representing the class (e.g., 0, 1, 2).

*`classify_data.json` (Example)*

```json
[
  {
    "features": [0.1, 0.8, 0.2],
    "label": 0
  },
  {
    "features": [0.9, 0.1, 0.1],
    "label": 1
  }
]
```

### 2\. Train a New Model

Use the `json_trainer` function in your main script.

```python
import torch
from skeletalmodel import json_trainer, load_model
from dataset import JSONDataset

# --- Define Your Training ---
HIDDEN_CONFIG = [128, 64, 32] # 3 hidden layers (for the training data I used I obviously lowered this but you can add as many hidden layers of any size you want)
MODEL_PATH = "my_model.pth"
TRAIN_FILE = "train.json"

# --- Run the Trainer ---
json_trainer(
    h_config=HIDDEN_CONFIG,
    model_path=MODEL_PATH,
    file_path=TRAIN_FILE,
    batch_size=16,
    epochs=100,
    LR=0.001,
    features_name="features",
    label_name="label",
    loss_type="mse" # Change this to "cross_entropy" or "bce" for other tasks
)
```

### 3\. Load and Use Your Model

Use the `load_model` function to get your trained model for inference.

```python
# --- Load the model ---
# You don't need to know the hidden config,
# it's loaded from the file.
model, loss_type = load_model("my_model.pth") # returns a tuple of model, loss_type
```
