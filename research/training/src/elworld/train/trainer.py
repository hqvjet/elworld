import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import load_config, get_general_config, get_vision_config, get_memory_config
from elworld.train.pipeline.vision_trainer import VisionTrainer
from elworld.preprocess.data_setup import setup_data


class Trainer:
    """
    Main trainer that orchestrates training of vision, memory, and control models.
    
    Modes:
        - "vision": Train only the vision model (VQ-VAE)
        - "memory": Train only the memory model (MDN-GRU)
        - "control": Train only the control model
    """
    
    def __init__(self, config_path="config.yaml", mode="vision", device=None):
        """
        Args:
            config_path: Path to config.yaml file
            mode: Training mode ("vision", "memory", "control")
            device: Device to use for training (None = auto-detect)
        """
        self.config_path = config_path
        self.mode = mode.lower()
        
        self.config = load_config(config_path)
        self.general_config = get_general_config(self.config)
        self.vision_config = get_vision_config(self.config)
        self.memory_config = get_memory_config(self.config)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        valid_modes = ["vision", "memory", "control"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode '{self.mode}'. Must be one of {valid_modes}")
        
        # Initialize trainers
        self.vision_trainer = None
        self.memory_trainer = None
        self.control_trainer = None
        
        # Data loaders
        self.vision_dataloader = None
        self.memory_dataloader = None
        self.control_dataloader = None
        
        print(f"Initialized Trainer in '{self.mode}' mode")
    
    def train_vision(self):
        """Train vision model (VQ-VAE)."""
        print(f"\n{'='*60}")
        print("Training Vision Model (VQ-VAE)")
        print(f"{'='*60}")
        
        # Setup data
        data_path = self.general_config.get('data_path', '../recorded')
        self.vision_dataloader = setup_data(
            mode="vision",
            data_path=data_path,
            vision_config=self.vision_config,
            general_config=self.general_config
        )
        
        # Initialize vision trainer
        self.vision_trainer = VisionTrainer(
            vision_config=self.vision_config,
            device=self.device
        )
        
        print(f"\nModel architecture:")
        print(f"  Input size: {self.vision_config['input_size']}")
        print(f"  Input channels: {self.vision_config['input_channels']}")
        print(f"  Latent dim: {self.vision_config['latent_dim']}")
        print(f"  Num embeddings: {self.vision_config['num_embedding']}")
        print(f"  Num hidden: {self.vision_config['num_hidden']}")
        print(f"  Residual layers: {self.vision_config['res_layer']}")
        print(f"  Commitment cost: {self.vision_config['commitment_cost']}")
        
        print(f"\nTraining parameters:")
        print(f"  Batch size: {self.vision_config['batch_size']}")
        print(f"  Learning rate: {self.vision_config['learning_rate']}")
        print(f"  Epochs: {self.vision_config['num_epochs']}")
        
        # Train (auto-saves checkpoint when done)
        self.vision_trainer.train(self.vision_dataloader)
        
        print(f"\nVision model training completed! ✓")
    
    def train_memory(self):
        """Train memory model (MDN-GRU)."""
        print(f"\n{'='*60}")
        print("Training Memory Model (MDN-GRU)")
        print(f"{'='*60}")
        
        # Setup data
        data_path = self.general_config.get('data_path', '../recorded')
        self.memory_dataloader = setup_data(
            mode="memory",
            data_path=data_path,
            memory_config=self.memory_config,
            vision_config=self.vision_config,
            general_config=self.general_config
        )
        
        if self.memory_dataloader is None:
            print("Skipping memory training (not implemented yet)...")
            return
        
        print("[TODO] Memory trainer implementation")
    
    def train_control(self):
        """Train control model."""
        print(f"\n{'='*60}")
        print("Training Control Model")
        print(f"{'='*60}")
        
        # Setup data
        data_path = self.general_config.get('data_path', '../recorded')
        self.control_dataloader = setup_data(
            mode="control",
            data_path=data_path,
            control_config=None,  # TODO: Add control_config to config.yaml
            general_config=self.general_config
        )
        
        if self.control_dataloader is None:
            print("Skipping control training (not implemented yet)...")
            return
        
        print("[TODO] Control trainer implementation")
    
    def train(self):
        """Execute training based on mode."""
        print(f"\n{'='*60}")
        print(f"Starting Training - Mode: {self.mode.upper()}")
        print(f"{'='*60}")
        
        if self.mode == "vision":
            self.train_vision()
        elif self.mode == "memory":
            self.train_memory()
        elif self.mode == "control":
            self.train_control()
        
        print(f"\n{'='*60}")
        print(f"Training completed! ✓")
        print(f"{'='*60}")



