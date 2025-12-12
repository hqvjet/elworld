import sys
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent.parent))

from elworld.data.gameplay_data import GameplayDataset, SequenceGameplayDataset


def setup_vision_data(data_path, vision_config, general_config) -> DataLoader:
    """
    Setup data for vision model (VQ-VAE) training.
    Vision model needs individual frames for reconstruction.
    
    Args:
        data_path: Path to data directory
        vision_config: Vision configuration dict
        general_config: General configuration dict
        
    Returns:
        DataLoader for vision training
    """
    print(f"\n{'='*60}")
    print("Setting up Vision Dataset")
    print(f"{'='*60}")
    
    vision_dataset = GameplayDataset(
        data_dir=data_path,
        max_files=general_config.get('total_play'),
        transform=None  # No transform needed, preprocessing done during load
    )
    
    # Optimized DataLoader settings for maximum throughput
    vision_dataloader = DataLoader(
        vision_dataset,
        batch_size=vision_config['batch_size'],
        shuffle=True,
        num_workers=4,  # Use 4 worker processes for parallel data loading
        pin_memory=True,  # Faster GPU transfer
        prefetch_factor=2,  # Prefetch 2 batches per worker
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    print(f"Vision dataloader ready:")
    print(f"  Total samples: {len(vision_dataset)}")
    print(f"  Total batches: {len(vision_dataloader)}")
    print(f"  Batch size: {vision_config['batch_size']}")
    print(f"  Num workers: 4 (parallel data loading)")
    print(f"  Prefetch factor: 2")
    print(f"  Persistent workers: True")
    
    return vision_dataloader


def setup_memory_data(data_path, memory_config, vision_config, general_config, sequence_length=32):
    """
    Setup data for memory model (MDN-GRU) training.
    Memory model needs sequences of (latent_states, actions) pairs.
    
    Args:
        data_path: Path to data directory
        memory_config: Memory configuration dict
        vision_config: Vision config (for latent encoding)
        general_config: General configuration dict
        sequence_length: Length of sequences for RNN
        
    Returns:
        DataLoader for memory training
    """
    print(f"\n{'='*60}")
    print("Setting up Memory Dataset")
    print(f"{'='*60}")
    print("[TODO] Memory model needs latent features from trained vision model")
    print("This requires:")
    print("  1. Load trained VQ-VAE checkpoint")
    print("  2. Encode all frames to latent space")
    print("  3. Create sequences of (latent, action) pairs")
    print("  4. Train MDN-GRU to predict next latent given (latent, action)")
    
    # TODO: Implement after vision model is trained
    # memory_dataset = SequenceGameplayDataset(
    #     data_dir=data_path,
    #     sequence_length=sequence_length,
    #     max_files=general_config.get('total_play'),
    #     transform=memory_transform
    # )
    
    # memory_dataloader = DataLoader(
    #     memory_dataset,
    #     batch_size=memory_config.get('batch_size', 32),
    #     shuffle=True,
    #     num_workers=4,
    #     pin_memory=True
    # )
    
    return None


def setup_control_data(data_path, control_config, general_config):
    """
    Setup data for control model training.
    Control model learns to predict actions from observations.
    
    Args:
        data_path: Path to data directory
        control_config: Control configuration dict
        general_config: General configuration dict
        
    Returns:
        DataLoader for control training
    """
    print(f"\n{'='*60}")
    print("Setting up Control Dataset")
    print(f"{'='*60}")
    print("[TODO] Control model implementation")
    print("This requires:")
    print("  1. Observations (raw or latent)")
    print("  2. Corresponding actions")
    print("  3. Train controller to predict action given observation")
    
    # TODO: Implement control dataset
    # control_dataset = GameplayDataset(
    #     data_dir=data_path,
    #     max_files=general_config.get('total_play'),
    #     transform=control_transform
    # )
    
    # control_dataloader = DataLoader(
    #     control_dataset,
    #     batch_size=control_config.get('batch_size', 32),
    #     shuffle=True,
    #     num_workers=4,
    #     pin_memory=True
    # )
    
    return None


def setup_data(mode, data_path, vision_config=None, memory_config=None, 
               control_config=None, general_config=None):
    """
    Main function to setup data based on training mode.
    
    Args:
        mode: Training mode ("vision", "memory", "control")
        data_path: Path to data directory
        vision_config: Vision configuration dict
        memory_config: Memory configuration dict
        control_config: Control configuration dict
        general_config: General configuration dict
        
    Returns:
        DataLoader(s) for the specified mode
    """
    if mode == "vision":
        return setup_vision_data(data_path, vision_config, general_config)
    
    elif mode == "memory":
        return setup_memory_data(data_path, memory_config, vision_config, general_config)
    
    elif mode == "control":
        return setup_control_data(data_path, control_config, general_config)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
