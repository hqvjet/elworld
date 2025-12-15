import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_general_config(config: Dict) -> Dict:
    """Get general configuration."""
    return config.get('general_config', {})


def get_vision_config(config: Dict) -> Dict:
    """Get vision (VAE) configuration."""
    return config.get('vision_config', {})


def get_memory_config(config: Dict) -> Dict:
    """Get memory (Transformer/MinGPT) configuration."""
    return config.get('memory_config', {})


def load_gameplay_data(data_dir: str, max_files: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all gameplay data from directory.
    
    Args:
        data_dir: Directory containing .npz files
        max_files: Maximum number of files to load (None = all)
        
    Returns:
        Tuple of (observations, actions)
        - observations: shape (total_frames, H, W, C)
        - actions: shape (total_frames, action_dim)
    """
    data_path = Path(data_dir)
    npz_files = sorted(data_path.glob("*.npz"))
    
    if max_files:
        npz_files = npz_files[:max_files]
    
    print(f"Loading {len(npz_files)} files from {data_dir}")
    
    all_obs = []
    all_act = []
    
    for npz_file in npz_files:
        print(f"  Loading {npz_file.name}...")
        data = np.load(npz_file)
        all_obs.append(data['obs'])
        all_act.append(data['act'])
    
    # Concatenate all data
    observations = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_act, axis=0)
    
    print(f"Total frames loaded: {len(observations)}")
    print(f"Observation shape: {observations.shape}")
    print(f"Action shape: {actions.shape}")
    
    return observations, actions


def load_gameplay_data_lazy(data_dir: str, max_files: int = None) -> List[Tuple[str, int]]:
    """
    Get list of gameplay data files without loading into memory.
    Returns list of (filepath, num_frames) tuples.
    
    Args:
        data_dir: Directory containing .npz files
        max_files: Maximum number of files to include (None = all)
        
    Returns:
        List of (filepath, num_frames) tuples
    """
    data_path = Path(data_dir)
    npz_files = sorted(data_path.glob("*.npz"))
    
    if max_files:
        npz_files = npz_files[:max_files]
    
    file_info = []
    total_frames = 0
    
    for npz_file in npz_files:
        data = np.load(npz_file)
        num_frames = len(data['obs'])
        file_info.append((str(npz_file), num_frames))
        total_frames += num_frames
        data.close()
    
    print(f"Found {len(file_info)} files with {total_frames} total frames")
    return file_info
