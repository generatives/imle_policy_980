import torch
from torch.utils.data import Dataset 
import zarr 
import numpy as np
import os

class PolicyDataset(Dataset):
    def __init__(self, dataset_path, pred_horizon=1, obs_horizon=1, action_horizon=1, dataset_percentage=1.0):
        self.states = zarr.open(os.path.join(dataset_path, "state"), mode="r")
        self.actions = zarr.open(os.path.join(dataset_path, "action"), mode="r")
        
        if isinstance(dataset_percentage, tuple):
            start_frac, end_frac = dataset_percentage
            N = self.states.shape[0]
            start_idx = int(N * start_frac)
            end_idx = int(N * end_frac)
            self.states = self.states[start_idx:end_idx]
            self.actions = self.actions[start_idx:end_idx]
        else:
            N = int(self.states.shape[0] * dataset_percentage)
            self.states = self.states[:N]
            self.actions = self.actions[:N]
        
        self.stats = {
            "state_mean": np.mean(self.states, axis=0),
            "state_std": np.std(self.states, axis=0),
            "action_mean": np.mean(self.actions, axis=0),
            "action_std": np.std(self.actions, axis=0),
        }
        
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        
    def __len__(self):
        return self.states.shape[0] - self.obs_horizon - self.pred_horizon 
    
    def __getitem__(self, idx):
        obs = self.states[idx : idx + self.obs_horizon]
        action = self.actions[idx + self.obs_horizon : idx + self.obs_horizon + self.pred_horizon]
        
        
        return {
            "obs": torch.tensor(obs, dtype=torch.float32),
            "action": torch.tensor(action, dtype=torch.float32),
        }