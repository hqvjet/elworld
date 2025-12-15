# Vision Model Checkpoint - Epoch 123

## Training Metrics

- **Epoch:** 123
- **Total Loss:** 0.011695
- **Reconstruction Loss:** 0.006705
- **VQ Loss:** 0.004990
- **Learning Rate:** 0.000300
- **Epoch Time:** 209.45s

**🏆 This is the best model so far!**

## Files

- `model.pth` - Model weights
- `optimizer.pth` - Optimizer state
- `scheduler.pth` - LR scheduler state
- `training_info.json` - Complete training metadata
- `config.json` - Model configuration

## Usage

```python
# Load model
model = VisionModel(**config)
model.load_state_dict(torch.load('model.pth'))
```
