import torch
import numpy as np


def action_to_array(action_dict, seq_len):
    action_array = np.zeros((seq_len, 5))
    action_names = ["forward", "jump", "attack"]
    
    action_array[:, :2] = action_dict["camera"]
    
    for i, action_name in enumerate(action_names):
        action_array[:, i + 2] = action_dict[action_name]
    
    return action_array


def load_model(model_path, device="cpu"):
    model = torch.load(model_path, map_location=torch.device(device))
    model.eval()
    
    model.name = model_path.split("/")[-1]
    
    return model
    