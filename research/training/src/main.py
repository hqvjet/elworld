import torch

from elworld.train.trainer import Trainer

if __name__ == "__main__":
    # ============= CONFIGURE HERE =============
    mode = "extract_video"  # "vision", "memory", "control", "extract_video"
    config_path = "config.yaml"
    device = 'cuda' # 'cuda', 'cpu', or None for auto-detect
    
    # Video extraction settings (only for extract_video mode)
    video_config = {
        'data_file': None,  # None = use first file (elsword_gameplay_01.npz)
        'output_file': "vision_comparison.mp4",
        'max_frames': None  # None = process all frames
    }
    # ==========================================
    
    # Create and run trainer
    trainer = Trainer(
        config_path=config_path,
        mode=mode,
        device=device
    )
    
    # Run based on mode
    if mode == "extract_video":
        trainer.extract_vision_video(
            data_file=video_config['data_file'],
            output_file=video_config['output_file'],
            max_frames=video_config['max_frames']
        )
    else:
        trainer.train()