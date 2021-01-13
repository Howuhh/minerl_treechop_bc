import torch
import torch.nn as nn
import numpy as np

from model import ConvNetRGB
from wrappers import TreeChopDataset
from utils import action_to_array, load_model


def train_treechop(experiment_name, data_path, save_path, load_path=None, stack_frames=1, seq_len=64, epochs=10, lr=0.0001):
    data = TreeChopDataset(data_dir=data_path, stack=stack_frames)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Training on: ", device)
    
    if load_path is None:
        # TODO: Не обязательно 3 канала, может 1 если чб, надо поправить
        model = ConvNetRGB(in_channels=3*stack_frames).to(device)
    else:
        model = load_model(load_path, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    camera_loss, action_loss = nn.MSELoss(), nn.BCELoss()

    errors = []
    for epoch in range(epochs):
        epoch_errors = []
        for i, (state, action) in enumerate(data.seq_iter(seq_len=seq_len)):
            model.zero_grad()
        
            state = torch.tensor(state).float().to(device)
            true_actions = torch.tensor(action_to_array(action, seq_len)).float().to(device)
                    
            pred_actions = model(state)
            
            loss = (camera_loss(pred_actions[:, :2], true_actions[:, :2]) + 
                    action_loss(pred_actions[:, 2:], true_actions[:, 2:]))

            epoch_errors.append(loss.cpu().detach().numpy().flatten())

            loss.backward()
            optimizer.step()
            
        torch.save(model, save_path + experiment_name)
        errors.append(np.mean(epoch_errors))
        print(f"Epoch {epoch} -- Mean Loss {errors[epoch]}")

    np.save(save_path + experiment_name + '_log.npy', np.array(errors))
    
if __name__ == "__main__":
    train_treechop("test_model", "data", "model/", seq_len=200, stack_frames=2, lr=1e-3)