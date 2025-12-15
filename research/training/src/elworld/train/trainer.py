import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import load_config, get_general_config, get_vision_config, get_memory_config
from elworld.train.pipeline.vision_trainer import VisionTrainer
from elworld.train.pipeline.memory_trainer import MemoryTrainer
from elworld.preprocess.data_setup import setup_data
from elworld.utils.real_vs_vision import extract_video


class Trainer:
    """
    Main trainer that orchestrates training of vision, memory, and control models.
    
    Modes:
        - "vision": Train only the vision model (VQ-VAE)
        - "memory": Train only the memory model (Transformer/MinGPT)
        - "control": Train only the control model
        - "extract_video": Create comparison video (actual vs vision model)
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
        
        valid_modes = ["vision", "memory", "control", "extract_video"]
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
            device=self.device,
            checkpoint_dir='checkpoints/vision'
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
        """Train memory model (Transformer/MinGPT)."""
        print(f"\n{'='*60}")
        print("Training Memory Model (Transformer/MinGPT)")
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
        
        # Initialize memory trainer
        self.memory_trainer = MemoryTrainer(
            memory_config=self.memory_config,
            device=self.device,
            checkpoint_dir='checkpoints/memory'
        )
        
        print(f"\nModel architecture:")
        print(f"  Vocab size: {self.memory_config['vocab_size']}")
        print(f"  Block size: {self.memory_config['block_size']}")
        print(f"  Embedding dim: {self.memory_config['n_emb']}")
        print(f"  Num layers: {self.memory_config['num_layers']}")
        print(f"  Num heads: {self.memory_config['num_heads']}")
        print(f"  Action dim: {self.memory_config['action_dim']}")
        
        print(f"\nTraining parameters:")
        print(f"  Batch size: {self.memory_config['batch_size']}")
        print(f"  Learning rate: {self.memory_config['learning_rate']}")
        print(f"  Epochs: {self.memory_config['num_epochs']}")
        
        # Train
        self.memory_trainer.train(self.memory_dataloader)
        
        print(f"\nMemory model training completed! ✓")
    
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
    
    def extract_vision_video(self, data_file=None, output_file=None, max_frames=None):
        """Create comparison video: actual vs vision model reconstruction."""
        print(f"\n{'='*60}")
        print("Extracting Vision Model Comparison Video")
        print(f"{'='*60}")
        
        # Default values
        data_path = self.general_config.get('data_path', '../recorded')
        if data_file is None:
            # Use first recorded file by default
            data_file = f"{data_path}/elsword_gameplay_01.npz"
        elif not data_file.startswith('/'):
            # Relative path
            data_file = f"{data_path}/{data_file}"
        
        if output_file is None:
            output_file = "vision_comparison.mp4"
        
        checkpoint_path = "checkpoints/vision/best_model"
        
        print(f"\nConfiguration:")
        print(f"  Data file: {data_file}")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Output: {output_file}")
        print(f"  Max frames: {max_frames if max_frames else 'All'}")
        print(f"  Device: {self.device}")
        
        # Call extract_video function
        extract_video(
            data_path=data_file,
            checkpoint_path=checkpoint_path,
            output_path=output_file,
            max_frames=max_frames,
            fps=self.general_config.get('frame_rate', 20),
            device=str(self.device)
        )
        
        print(f"\n✓ Video extraction completed!")
    
    def train(self):
        """Execute training or evaluation based on mode."""
        print(f"\n{'='*60}")
        print(f"Starting - Mode: {self.mode.upper()}")
        print(f"{'='*60}")
        
        if self.mode == "vision":
            self.train_vision()
        elif self.mode == "memory":
            self.train_memory()
        elif self.mode == "control":
            self.train_control()
        elif self.mode == "extract_video":
            # Extract video with default settings
            # Can be customized by calling extract_vision_video() directly
            self.extract_vision_video(
                data_file=None,  # Use first file
                output_file="vision_comparison.mp4",
                max_frames=500  # Process first 500 frames
            )
        
        print(f"\n{'='*60}")
        print(f"Task completed! ✓")
        print(f"{'='*60}")



