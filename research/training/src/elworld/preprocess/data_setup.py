import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent.parent))

from elworld.data.gameplay_data import GameplayDataset, SequenceGameplayDataset
from elworld.data.memory_data import MemoryDataset


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


def setup_memory_data(data_path, memory_config, vision_config, general_config):
    """
    Setup data for memory model (Transformer/MinGPT) training.
    Memory model needs sequences of visual tokens (encoding_indices) and actions.
    
    Args:
        data_path: Path to data directory
        memory_config: Memory configuration dict with:
            - block_size: Context length for transformer (e.g., 1537)
            - vocab_size: Codebook size from VQ-VAE (512)
            - grid_latent_dim: [H, W] of latent space (e.g., [24, 32])
            - batch_size: Training batch size
        vision_config: Vision config (for latent encoding)
        general_config: General configuration dict
        
    Returns:
        DataLoader for memory training
    """
    print(f"\n{'='*60}")
    print("Setting up Memory Dataset (Transformer)")
    print(f"{'='*60}")
    print("[TODO] Memory model needs visual tokens from trained vision model")
    print("This requires:")
    print("  1. Load trained VQ-VAE checkpoint")
    print("  2. Encode all frames to visual tokens (encoding_indices)")
    print("  3. Flatten tokens: [B, 24, 32] -> [B, 768] visual token sequence")
    print("  4. Create context windows of length block_size (1537 tokens)")
    print("  5. Train Transformer to predict next token given previous tokens + action")
    print(f"\nMemory config:")
    print(f"  Block size: {memory_config.get('block_size', 'N/A')}")
    print(f"  Vocab size: {memory_config.get('vocab_size', 'N/A')}")
    print(f"  Grid latent: {memory_config.get('grid_latent_dim', 'N/A')}")
    print(f"  Num layers: {memory_config.get('num_layers', 'N/A')}")
    print(f"  Num heads: {memory_config.get('num_heads', 'N/A')}")
    
    # Create memory dataset with encoded visual tokens
    checkpoint_path = "checkpoints/vision/best_model"  # Use best VQ-VAE model
    
    memory_dataset = MemoryDataset(
        data_dir=data_path,
        checkpoint_path=checkpoint_path,
        max_files=general_config.get('total_play'),
        sequence_length=2,  # Frame_t -> Frame_t+1 prediction
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    memory_dataloader = DataLoader(
        memory_dataset,
        batch_size=memory_config.get('batch_size', 64),
        shuffle=True,
        num_workers=0,  # Single process - safe and stable
        pin_memory=True
    )
    
    print(f"\nMemory dataloader ready:")
    print(f"  Total sequences: {len(memory_dataset)}")
    print(f"  Total batches: {len(memory_dataloader)}")
    print(f"  Batch size: {memory_config.get('batch_size', 64)}")
    
    return memory_dataloader


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
