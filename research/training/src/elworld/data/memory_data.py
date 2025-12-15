import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json

from elworld.model.vision import VisionModel


class MemoryDataset(Dataset):
    """
    Dataset for memory model training.
    Loads VQ-VAE checkpoint, encodes all frames to visual tokens,
    and creates sequences for transformer training.
    """
    
    def __init__(
        self, 
        data_dir: str,
        checkpoint_path: str,
        max_files: int = None,
        sequence_length: int = 2,
        device='cuda'
    ):
        """
        Args:
            data_dir: Directory containing .npz gameplay files
            checkpoint_path: Path to trained VQ-VAE checkpoint (e.g., 'checkpoints/best_model')
            max_files: Maximum number of files to load
            sequence_length: Number of consecutive frames per sequence (default 2 for frame_t → frame_t+1)
            device: Device for encoding frames
        """
        self.sequence_length = sequence_length
        self.device = device
        
        # Load VQ-VAE model
        print(f"\n[1/3] Loading VQ-VAE from {checkpoint_path}...")
        self.vqvae = self._load_vqvae(checkpoint_path, device)
        self.vqvae.eval()
        
        # Load gameplay data
        print(f"\n[2/3] Loading gameplay data from {data_dir}...")
        data_path = Path(data_dir)
        npz_files = sorted(data_path.glob("*.npz"))
        
        if max_files:
            npz_files = npz_files[:max_files]
        
        all_obs = []
        all_act = []
        
        for npz_file in npz_files:
            data = np.load(npz_file)
            obs = data['obs']
            # Preprocess: (H, W, C) -> (C, H, W)
            if obs.shape[-1] == 3:
                obs = obs.transpose(0, 3, 1, 2)
            all_obs.append(obs)
            all_act.append(data['act'])
            data.close()
        
        observations = np.concatenate(all_obs, axis=0)
        actions = np.concatenate(all_act, axis=0)
        
        print(f"Loaded {len(observations):,} frames")
        print(f"Observation shape: {observations.shape}")
        print(f"Action shape: {actions.shape}")
        
        # Encode all frames to visual tokens
        print(f"\n[3/3] Encoding frames to visual tokens...")
        self.visual_tokens = self._encode_frames(observations)
        self.actions = torch.from_numpy(actions).float()
        
        # Free VQ-VAE model from GPU to save memory
        print(f"\n[4/4] Freeing VQ-VAE model from GPU...")
        del self.vqvae
        if device == 'cuda':
            torch.cuda.empty_cache()
            print(f"✓ GPU memory freed")
        
        # Create sequences
        # Need sequence_length frames for input + 1 more for target
        # So max idx where idx+sequence_length is still valid
        self.num_sequences = len(self.visual_tokens) - sequence_length
        
        print(f"\n✓ Dataset ready:")
        print(f"  Total sequences: {self.num_sequences:,}")
        print(f"  Visual tokens shape: {self.visual_tokens.shape}")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Tokens per frame: {self.visual_tokens.shape[1]}")
        print(f"  Memory: Visual tokens stored in CPU RAM, will be moved to GPU during training")
    
    def _load_vqvae(self, checkpoint_path, device):
        """Load trained VQ-VAE model."""
        checkpoint_path = Path(checkpoint_path)
        
        # Load config
        config_path = checkpoint_path / "config.json"
        with open(config_path, 'r') as f:
            vision_config = json.load(f)
        
        # Create model
        model = VisionModel(
            num_hidden=vision_config['num_hidden'],
            res_layer=vision_config['res_layer'],
            res_hidden=vision_config['res_hidden'],
            input_channels=vision_config['input_channels'],
            num_embedding=vision_config['num_embedding'],
            embedding_dim=vision_config['latent_dim'],
            commitment_cost=vision_config['commitment_cost']
        ).to(device)
        
        # Load weights
        model_path = checkpoint_path / "model.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        print(f"✓ VQ-VAE loaded")
        return model
    
    def _encode_frames(self, observations):
        """Encode all frames to visual tokens using VQ-VAE."""
        batch_size = 256  # Process in batches to avoid OOM
        num_frames = len(observations)
        all_tokens = []
        
        with torch.no_grad():
            for i in range(0, num_frames, batch_size):
                batch = observations[i:i+batch_size]
                batch_tensor = torch.from_numpy(batch).float().to(self.device)
                
                # Normalize
                if batch_tensor.max() > 1.0:
                    batch_tensor = batch_tensor / 255.0
                
                # Encode
                output = self.vqvae(batch_tensor)
                tokens = output['encoding_indices']  # [B, H, W]
                
                # Flatten spatial dimensions: [B, H, W] -> [B, H*W]
                tokens_flat = tokens.view(tokens.size(0), -1)
                all_tokens.append(tokens_flat.cpu())
                
                if (i // batch_size) % 10 == 0:
                    print(f"  Encoded {i}/{num_frames} frames...")
        
        # Concatenate all batches
        visual_tokens = torch.cat(all_tokens, dim=0)  # [N, 768]
        print(f"✓ Encoded {num_frames} frames to {visual_tokens.shape}")
        
        return visual_tokens
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        """
        Returns a sequence for training.
        
        Returns:
            dict with:
                - 'input_tokens': [seq_len, 768] - Input visual tokens
                - 'actions': [seq_len, action_dim] - Actions taken
                - 'target_tokens': [seq_len, 768] - Target next-frame tokens (shifted by 1)
        """
        # Get sequence of frames
        input_tokens = self.visual_tokens[idx:idx + self.sequence_length]
        actions = self.actions[idx:idx + self.sequence_length]
        
        # Target is next frame (shifted by 1)
        # For each frame, predict the next frame's tokens
        target_tokens = self.visual_tokens[idx+1:idx + self.sequence_length + 1]
        
        return {
            'input_tokens': input_tokens,      # [seq_len, 768]
            'actions': actions,                # [seq_len, action_dim]
            'target_tokens': target_tokens     # [seq_len, 768]
        }
