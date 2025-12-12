import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class GameplayDataset(Dataset):
    """
    PyTorch Dataset for gameplay data.
    Preloads all data into memory for fast training.
    Data is preprocessed during loading (transpose) for efficiency.
    """
    
    def __init__(self, data_dir: str, max_files: int = None, transform=None):
        data_path = Path(data_dir)
        npz_files = sorted(data_path.glob("*.npz"))
        
        if max_files:
            npz_files = npz_files[:max_files]
        
        print(f"Loading {len(npz_files)} files into memory...")
        
        all_obs = []
        all_act = []
        
        for npz_file in npz_files:
            data = np.load(npz_file)
            obs = data['obs']
            # Preprocess: transpose from (H, W, C) to (C, H, W) during loading
            if obs.shape[-1] == 3:  # If channels last
                obs = obs.transpose(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            all_obs.append(obs)
            all_act.append(data['act'])
            data.close()
        
        self.observations = np.concatenate(all_obs, axis=0)
        self.actions = np.concatenate(all_act, axis=0)
        
        print(f"Loaded {len(self.observations):,} frames")
        print(f"Observation shape: {self.observations.shape}")
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        obs = self.observations[idx]
        act = self.actions[idx]
        
        # Data already preprocessed during loading, just convert to tensor
        return {
            'observation': torch.from_numpy(obs).float(),
            'action': torch.from_numpy(act).float()
        }


class SequenceGameplayDataset(Dataset):
    """Sequential dataset for RNN training."""
    
    def __init__(self, data_dir: str, sequence_length: int = 32, 
                 max_files: int = None, transform=None):
        self.transform = transform
        self.sequence_length = sequence_length
        
        data_path = Path(data_dir)
        npz_files = sorted(data_path.glob("*.npz"))
        
        if max_files:
            npz_files = npz_files[:max_files]
        
        all_obs = []
        all_act = []
        
        for npz_file in npz_files:
            data = np.load(npz_file)
            all_obs.append(data['obs'])
            all_act.append(data['act'])
            data.close()
        
        self.observations = np.concatenate(all_obs, axis=0)
        self.actions = np.concatenate(all_act, axis=0)
        
        self.num_sequences = len(self.observations) - sequence_length + 1
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        obs_seq = self.observations[idx:idx + self.sequence_length]
        act_seq = self.actions[idx:idx + self.sequence_length]
        
        if self.transform:
            obs_seq = np.stack([self.transform(obs) for obs in obs_seq])
        
        return {
            'observation': torch.from_numpy(obs_seq).float(),
            'action': torch.from_numpy(act_seq).float()
        }
