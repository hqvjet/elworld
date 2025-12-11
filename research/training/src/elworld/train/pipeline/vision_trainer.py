import torch
import tqdm
import time
from pathlib import Path

from elworld.model.vision import VisionModel

class VisionTrainer:
    def __init__(self, vision_config, device='cuda', checkpoint_dir='checkpoints', save_every=10):
        """
        Args:
            vision_config: Dictionary from config.yaml's vision_config section
            device: Device to train on ('cuda' or 'cpu')
            checkpoint_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
        """
        self.vision_config = vision_config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        
        self.model = VisionModel(
            num_hidden=vision_config['num_hidden'],
            res_layer=vision_config['res_layer'],
            res_hidden=vision_config['res_hidden'],
            input_channels=vision_config['input_channels'],
            num_embedding=vision_config['num_embedding'],
            embedding_dim=vision_config['latent_dim'],
            commitment_cost=vision_config['commitment_cost']
        ).to(device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=vision_config['learning_rate']
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=20
        )
        
        self.criterion = torch.nn.MSELoss()
        
        self.num_epochs = vision_config['num_epochs']
        self.batch_size = vision_config['batch_size']
        
        # Track best model
        self.best_loss = float('inf')
        self.best_epoch = 0
        
        print(f"\n{'='*60}")
        print("VisionTrainer Initialized")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Learning Rate: {vision_config['learning_rate']}")
        print(f"Total Epochs: {self.num_epochs}")
        print(f"{'='*60}\n")

    def train(self, dataloader):
        self.model.train()
        
        print(f"\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}")
        print(f"Dataset size: {len(dataloader.dataset):,} samples")
        print(f"Batches per epoch: {len(dataloader)}")
        print(f"Samples per batch: {self.batch_size}")
        print(f"Total iterations: {self.num_epochs * len(dataloader):,}")
        print(f"Checkpoint every: {self.save_every} epochs")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_vq_loss = 0.0
            
            print(f"\n{'─'*60}")
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"{'─'*60}")
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                mem_before = torch.cuda.memory_allocated() / 1024**2
                print(f"GPU Memory (start): {mem_before:.2f} MB")
            
            batch_times = []
            
            for batch_idx, batch in enumerate(tqdm.tqdm(dataloader, desc=f"Training")):
                batch_start = time.time()
                
                if isinstance(batch, dict):
                    inputs = batch['observation'].to(self.device, non_blocking=True)
                else:
                    inputs = batch.to(self.device, non_blocking=True)
                
                if inputs.max() > 1.0:
                    inputs = inputs / 255.0
                
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(inputs)
                
                recon_loss = self.criterion(outputs['x_recon'], inputs)
                vq_loss = outputs['vq_loss']
                loss = recon_loss + vq_loss
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_vq_loss += vq_loss.item()
                
                batch_times.append(time.time() - batch_start)
            
            avg_loss = epoch_loss / len(dataloader)
            avg_recon = epoch_recon_loss / len(dataloader)
            avg_vq = epoch_vq_loss / len(dataloader)
            
            epoch_time = time.time() - epoch_start_time
            avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\n{'─'*60}")
            print(f"Epoch {epoch+1}/{self.num_epochs} Summary")
            print(f"{'─'*60}")
            print(f"Loss:        {avg_loss:.6f}")
            print(f"  Recon:     {avg_recon:.6f}")
            print(f"  VQ:        {avg_vq:.6f}")
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"Time:        {epoch_time:.2f}s (avg batch: {avg_batch_time*1000:.2f}ms)")
            
            if torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated() / 1024**2
                mem_peak = torch.cuda.max_memory_allocated() / 1024**2
                print(f"GPU Memory:  {mem_after:.2f} MB (peak: {mem_peak:.2f} MB)")
            
            # Update learning rate scheduler
            self.scheduler.step(avg_loss)
            
            # Save best model
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_epoch = epoch + 1
                best_path = self.checkpoint_dir / "vision_model_best.pth"
                self.save_checkpoint(str(best_path), epoch + 1, avg_loss, is_best=True)
                print(f"✓ New best model saved! Loss: {avg_loss:.6f}")
            
            # Save checkpoint every N epochs
            if (epoch + 1) % self.save_every == 0:
                checkpoint_path = self.checkpoint_dir / f"vision_model_epoch_{epoch+1}.pth"
                self.save_checkpoint(str(checkpoint_path), epoch + 1, avg_loss)
                print(f"✓ Checkpoint saved: {checkpoint_path.name}")
            
            print(f"{'─'*60}")
        
        # Save final checkpoint
        final_path = self.checkpoint_dir / "vision_model_final.pth"
        self.save_checkpoint(str(final_path), self.num_epochs, avg_loss)
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"  Best loss: {self.best_loss:.4f} (epoch {self.best_epoch})")
        print(f"  Final checkpoint: {final_path}")
        print(f"  Best checkpoint: {self.checkpoint_dir / 'vision_model_best.pth'}")
        print(f"{'='*60}")
    
    def save_checkpoint(self, path, epoch=None, loss=None, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.vision_config,
            'epoch': epoch,
            'loss': loss,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'best_loss' in checkpoint:
            self.best_loss = checkpoint['best_loss']
            self.best_epoch = checkpoint['best_epoch']
        
        epoch = checkpoint.get('epoch', 'unknown')
        loss = checkpoint.get('loss', 'unknown')
        print(f"Checkpoint loaded from {path}")
        print(f"  Epoch: {epoch}, Loss: {loss}")