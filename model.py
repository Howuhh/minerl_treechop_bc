import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNetRGB(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 4, 2)
        
        self.l1 = nn.Linear(64 * 2 * 2, 128)
        self.l2 = nn.Linear(128, 5)  
        
    def _process_input(self, X):
        if len(X.shape) == 3:
            X = X.unsqueeze(0)
            assert len(X.shape) == 4, "input shape should be [batch, H, W, C] or [H, W, C]"

        X = X.permute(0, 3, 1, 2)  # from BHWC to BCHW
        X = X / 255  # normalize images
            
        return X

    def forward(self, X):
        X = self._process_input(X)
        
        out = F.relu(self.conv1(X))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        
        out = out.reshape(out.shape[0], -1)
        out = F.relu(self.l1(out))
        out = self.l2(out)
        
        out[:, 2:] = torch.sigmoid(out[:, 2:])
        
        return out
    
    def predict(self, X):
        out = self.forward(X)[0]  # prediction only for first frame
    
        torch.clamp_(out[:2], -180, 180)  # camera move degrees 
        
        probs = out[2:].detach().numpy()
        actions_sampled = np.random.binomial(1, probs)

        action_names = ["forward", "jump", "attack"]
        
        actions = {
            "camera": out[:2].detach().numpy(),
            "sneak": 0,
            "sprint": 0,
            "back": 0,
            "left": 0,
            "right": 0,
        }
        
        for name, action in zip(action_names, actions_sampled):
            actions[name] = action

        return actions
