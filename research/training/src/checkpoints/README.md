# Checkpoint Directory Structure

This directory contains saved model checkpoints organized by model type.

## Structure

```
checkpoints/
├── vision/
│   ├── best_model/                    # Best VQ-VAE model (lowest loss)
│   │   ├── model.pth                  # Model weights
│   │   ├── optimizer.pth              # Optimizer state
│   │   ├── scheduler.pth              # LR scheduler state
│   │   ├── config.json                # Model configuration
│   │   ├── training_info.json         # Training metadata
│   │   └── README.md                  # Checkpoint info
│   └── vision_model_checkpoint_N/     # Epoch N checkpoint
│       └── ...
│
├── memory/
│   ├── best_model/                    # Best MinGPT model (lowest loss)
│   │   └── ...
│   └── memory_model_checkpoint_N/     # Epoch N checkpoint
│       └── ...
│
└── control/  (future)
    └── ...
```

## Model Types

### Vision (VQ-VAE)
- **Purpose**: Compress images to discrete visual tokens
- **Input**: RGB images [B, 3, 192, 256]
- **Output**: Visual tokens [B, 24, 32] (indices in codebook)
- **Checkpoint naming**: `vision_model_checkpoint_N/`

### Memory (Transformer/MinGPT)
- **Purpose**: Predict next frame tokens given current frame + action
- **Input**: Visual tokens [B, 768] + actions [B, 22]
- **Output**: Next frame token predictions [B, 768, 512]
- **Checkpoint naming**: `memory_model_checkpoint_N/`

### Control (Future)
- **Purpose**: Predict actions from observations
- **Checkpoint naming**: `control_model_checkpoint_N/`

## Migration

If you have old checkpoints in the root `checkpoints/` directory, run:

```bash
python migrate_checkpoints.py
```

This will move:
- `checkpoints/best_model/` → `checkpoints/vision/best_model/`
- `checkpoints/vision_model_checkpoint_N/` → `checkpoints/vision/vision_model_checkpoint_N/`
- Any memory/control checkpoints to their respective directories

## Usage

### Loading Vision Model
```python
from elworld.model.vision import VisionModel

# Load from best model
checkpoint_path = "checkpoints/vision/best_model"
model = VisionModel.from_checkpoint(checkpoint_path)
```

### Loading Memory Model
```python
from elworld.model.memory import MemoryModel

# Load from specific checkpoint
checkpoint_path = "checkpoints/memory/memory_model_checkpoint_50"
model = MemoryModel.from_checkpoint(checkpoint_path)
```

## Checkpoint Files

Each checkpoint folder contains:
- **model.pth**: PyTorch state dict with model weights
- **optimizer.pth**: Optimizer state for resuming training
- **scheduler.pth**: Learning rate scheduler state
- **config.json**: Model configuration (architecture params)
- **training_info.json**: Training metrics (epoch, loss, lr, time)
- **README.md**: Human-readable checkpoint information

## Best Model Selection

The `best_model/` folder is automatically updated during training when:
- A new epoch achieves lower validation loss
- Training completes and the best checkpoint is identified

Use `best_model/` for inference and deployment.
