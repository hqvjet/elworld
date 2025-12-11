import torch

from elworld.train.trainer import Trainer

if __name__ == "__main__":
    # ============= CONFIGURE HERE =============
    mode = "vision"  # "vision", "memory", "control"
    config_path = "config.yaml"
    device = 'cuda' # 'cuda', 'cpu', or None for auto-detect
    # ==========================================
    
    # Create and run trainer
    trainer = Trainer(
        config_path=config_path,
        mode=mode,
        device=device
    )
    
    trainer.train()