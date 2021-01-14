import torch
import numpy as np


def rgb_to_grey(img):
    """
    Convert image from rgb to grey scale.

    Parameters
    ----------
    img: np.ndarray, shape [batch, H, W, 3]
        Single image or batch of images.

    Returns
    -------
    np.ndarray, shape [batch, H, W, 1]
        Grey scaled image or batch of images. 
    
    """
    rgb_to_grey_vec = np.array([0.2989, 0.5870, 0.1140]).reshape(-1, 1)
    
    return img @ rgb_to_grey_vec


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
    
    # for old versions of models
    if not hasattr(model, "name"):
        model.name = model_path.split("/")[-1]
    
    return model
    