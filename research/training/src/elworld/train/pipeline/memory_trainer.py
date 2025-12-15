import torch
import tqdm
import time
import json
from pathlib import Path
from datetime import datetime

from elworld.model.memory import MemoryModel


class MemoryTrainer:
    def __init__(self, memory_config, device='cuda', checkpoint_dir='checkpoints/memory'):
        """
        Args:
            memory_config: Dictionary from config.yaml's memory_config section
            device: Device to train on ('cuda' or 'cpu')
            checkpoint_dir: Directory to save checkpoints (default: checkpoints/memory)
        """
        self.memory_config = memory_config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create best_model directory
        self.best_model_dir = self.checkpoint_dir / "best_memory_model"
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = MemoryModel(
            vocab_size=memory_config['vocab_size'],
            block_size=memory_config['block_size'],
            n_emb=memory_config['n_emb'],
            num_layers=memory_config['num_layers'],
            num_heads=memory_config['num_heads'],
            dropout=memory_config['dropout'],
            action_dim=memory_config['action_dim']
        ).to(device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=memory_config['learning_rate']
        )
        
        # GradScaler for AMP
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        self.num_epochs = memory_config['num_epochs']
        self.batch_size = memory_config['batch_size']
        
        # Track best model
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.start_epoch = 0
        
        # Try to resume from existing checkpoint
        self._load_existing_checkpoint()
        
        print(f"\n{'='*60}")
        print("MemoryTrainer Initialized")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        print(f"AMP (Mixed Precision): {'Enabled' if self.use_amp else 'Disabled'}")
        print(f"Model Parameters: {self.model.get_num_params():,}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Learning Rate: {memory_config['learning_rate']}")
        print(f"Total Epochs: {self.num_epochs}")
        print(f"Block Size: {memory_config['block_size']}")
        print(f"Vocab Size: {memory_config['vocab_size']}")
        print(f"{'='*60}\n")

    def _load_existing_checkpoint(self):
        """Find and load the latest checkpoint if exists."""
        # Load best model info
        if self.best_model_dir.exists():
            metadata_path = self.best_model_dir / "training_info.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.best_loss = metadata.get('best_loss', float('inf'))
                self.best_epoch = metadata.get('best_epoch', 0)
        
        # Find latest checkpoint
        latest_checkpoint = self._find_latest_checkpoint()
        if latest_checkpoint:
            print(f"\n{'='*60}")
            print(f"Found existing checkpoint: {latest_checkpoint.name}")
            print(f"{'='*60}")
            self.load_checkpoint(latest_checkpoint)
            
            metadata_path = latest_checkpoint / "training_info.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.start_epoch = metadata.get('epoch', 0)
            print(f"{'='*60}\n")
    
    def _find_latest_checkpoint(self):
        """Find the latest checkpoint folder."""
        checkpoint_folders = list(self.checkpoint_dir.glob("memory_model_checkpoint_*"))
        if not checkpoint_folders:
            return None
        
        epochs = []
        for folder in checkpoint_folders:
            try:
                epoch_num = int(folder.name.split('_')[-1])
                epochs.append((epoch_num, folder))
            except ValueError:
                continue
        
        if not epochs:
            return None
        
        latest = max(epochs, key=lambda x: x[0])
        return latest[1]
    
    def train(self, dataloader):
        self.model.train()
        
        print(f"\n{'='*60}")
        print("Starting Memory Model Training")
        print(f"{'='*60}")
        print(f"Dataset size: {len(dataloader.dataset):,} sequences")
        print(f"Batches per epoch: {len(dataloader)}")
        print(f"Samples per batch: {self.batch_size}")
        print(f"Total iterations: {self.num_epochs * len(dataloader):,}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        if self.start_epoch > 0:
            print(f"Resuming from epoch: {self.start_epoch}")
            print(f"Best loss so far: {self.best_loss:.6f} (epoch {self.best_epoch})")
        print(f"{'='*60}\n")
        
        for epoch in range(self.start_epoch, self.num_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            
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
                
                # Get batch data
                input_tokens = batch['input_tokens'].to(self.device, non_blocking=True)  # [B, seq_len, 768]
                actions = batch['actions'].to(self.device, non_blocking=True)  # [B, seq_len, action_dim]
                target_tokens = batch['target_tokens'].to(self.device, non_blocking=True)  # [B, seq_len, 768]
                
                # Process each frame in sequence independently
                # Flatten batch and sequence: [B, seq_len, 768] -> [B*seq_len, 768]
                B, seq_len, token_len = input_tokens.shape
                input_tokens_flat = input_tokens.view(B * seq_len, token_len)  # [B*seq_len, 768]
                target_tokens_flat = target_tokens.view(B * seq_len, token_len)  # [B*seq_len, 768]
                
                # Actions: one per frame, expand later in model
                # [B, seq_len, action_dim] -> [B*seq_len, action_dim]
                actions_flat = actions.view(B * seq_len, -1)  # [B*seq_len, action_dim]
                
                self.optimizer.zero_grad(set_to_none=True)
                
                # Use AMP for faster training
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        output = self.model(
                            input_tokens_flat,
                            actions=actions_flat,
                            targets=target_tokens_flat
                        )
                        loss = output['loss']
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    output = self.model(
                        input_tokens_flat,
                        actions=actions_flat,
                        targets=target_tokens_flat
                    )
                    loss = output['loss']
                    
                    loss.backward()
                    self.optimizer.step()
                
                epoch_loss += loss.item()
                batch_times.append(time.time() - batch_start)
            
            avg_loss = epoch_loss / len(dataloader)
            epoch_time = time.time() - epoch_start_time
            avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\n{'─'*60}")
            print(f"Epoch {epoch+1}/{self.num_epochs} Summary")
            print(f"{'─'*60}")
            print(f"Loss:        {avg_loss:.6f}")
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"Time:        {epoch_time:.2f}s (avg batch: {avg_batch_time*1000:.2f}ms)")
            
            if torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated() / 1024**2
                mem_peak = torch.cuda.max_memory_allocated() / 1024**2
                print(f"GPU Memory:  {mem_after:.2f} MB (peak: {mem_peak:.2f} MB)")
            
            # Update learning rate scheduler
            self.scheduler.step(avg_loss)
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Save checkpoint
            checkpoint_dir = self.checkpoint_dir / f"memory_model_checkpoint_{epoch+1}"
            self.save_checkpoint_folder(checkpoint_dir, epoch + 1, avg_loss, current_lr, epoch_time)
            print(f"✓ Checkpoint saved: {checkpoint_dir.name}/")
            
            # Save best model
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_epoch = epoch + 1
                self.save_checkpoint_folder(self.best_model_dir, epoch + 1, avg_loss, current_lr, epoch_time, is_best=True)
                print(f"✓ New best model saved! Loss: {avg_loss:.6f}")
            
            print(f"{'─'*60}")
        
        print(f"\n{'='*60}")
        print(f"Memory Model Training completed!")
        print(f"  Best loss: {self.best_loss:.6f} (epoch {self.best_epoch})")
        print(f"  Best model: {self.best_model_dir}/")
        print(f"  Total checkpoints: {self.num_epochs}")
        print(f"{'='*60}")
    
    def save_checkpoint_folder(self, folder_path, epoch, loss, lr, epoch_time, is_best=False):
        """Save model checkpoint in folder structure."""
        folder_path = Path(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        torch.save(self.model.state_dict(), folder_path / "model.pth")
        torch.save(self.optimizer.state_dict(), folder_path / "optimizer.pth")
        torch.save(self.scheduler.state_dict(), folder_path / "scheduler.pth")
        
        # Save metadata
        metadata = {
            'epoch': epoch,
            'loss': loss,
            'learning_rate': lr,
            'epoch_time': epoch_time,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'is_best': is_best,
            'timestamp': datetime.now().isoformat(),
            'config': self.memory_config
        }
        
        with open(folder_path / "training_info.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        with open(folder_path / "config.json", 'w') as f:
            json.dump(self.memory_config, f, indent=2)
        
        # Create README
        with open(folder_path / "README.md", 'w') as f:
            f.write(f"# Memory Model Checkpoint - Epoch {epoch}\n\n")
            f.write(f"## Training Metrics\n\n")
            f.write(f"- **Epoch:** {epoch}\n")
            f.write(f"- **Loss:** {loss:.6f}\n")
            f.write(f"- **Learning Rate:** {lr:.6f}\n")
            f.write(f"- **Epoch Time:** {epoch_time:.2f}s\n")
            if is_best:
                f.write(f"\n**🏆 This is the best model so far!**\n")
    
    def load_checkpoint(self, folder_path):
        """Load model checkpoint from folder."""
        folder_path = Path(folder_path)
        
        self.model.load_state_dict(torch.load(folder_path / "model.pth", map_location=self.device))
        
        if (folder_path / "optimizer.pth").exists():
            self.optimizer.load_state_dict(torch.load(folder_path / "optimizer.pth", map_location=self.device))
        
        if (folder_path / "scheduler.pth").exists():
            self.scheduler.load_state_dict(torch.load(folder_path / "scheduler.pth", map_location=self.device))
        
        if (folder_path / "training_info.json").exists():
            with open(folder_path / "training_info.json", 'r') as f:
                metadata = json.load(f)
            
            self.best_loss = metadata.get('best_loss', float('inf'))
            self.best_epoch = metadata.get('best_epoch', 0)
            
            print(f"Checkpoint loaded from {folder_path}")
            print(f"  Epoch: {metadata.get('epoch', 'unknown')}")
            print(f"  Loss: {metadata.get('loss', 'unknown')}")
            print(f"  Best Loss: {self.best_loss:.6f} (epoch {self.best_epoch})")
