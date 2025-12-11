# 🎮 Elworld: AI-Powered Elsword Automation

> **Tired of endless grinding day and night?** Let AI master Elsword for you while you rest.

<div align="center">

> [!WARNING]  
> **🚧 UNDER ACTIVE DEVELOPMENT 🚧**  
> This project is still in early development stages. Vision model training is complete, but Memory and Control models are still in progress.  
> **Not ready for production use yet!** Star and watch for updates.

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.9-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Status](https://img.shields.io/badge/Status-In%20Development-orange.svg)

*An autonomous gameplay system powered by World Models and Reinforcement Learning*

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Contributing](#-contributing)

</div>

---

## 📌 TL;DR

Elworld is an AI system that learns to play Elsword autonomously by watching and learning from human gameplay. Instead of spending countless hours grinding dungeons manually, Elworld trains neural networks to play like a human player - making decisions, executing combos, and navigating dungeons automatically.

**What you get:**
- 🤖 AI that learns your playstyle from recorded gameplay
- 🎯 Autonomous dungeon clearing without macros or simple scripts  
- 🧠 Deep learning models that understand game states and make strategic decisions
- 🚀 GPU-accelerated training for fast learning

---

## 🎯 For Everyone: What Does This Do?

### The Problem
Elsword is an amazing game, but the grinding is brutal. Farming materials, leveling characters, and running the same dungeons hundreds of times is exhausting. Traditional "bots" get detected easily because they follow rigid patterns.

### The Solution
Elworld uses cutting-edge AI technology to learn how to play Elsword by watching **you** play. Think of it as teaching a really smart student:

1. **You play normally** - The system records your gameplay (video + keyboard inputs)
2. **AI learns from you** - Neural networks study your patterns, strategies, and decision-making
3. **AI plays for you** - Once trained, the AI can play autonomously, mimicking human behavior

### Why This Approach?
Unlike simple macros or scripts that repeat the same actions, Elworld:
- **Adapts to different situations** - Understands game state and makes decisions
- **Plays like a human** - Natural timing, varied patterns, strategic thinking
- **Handles complexity** - Can navigate dynamic environments and unexpected scenarios

---

## 🔬 For Developers: Technical Overview

Elworld implements a **World Model + Reinforcement Learning** architecture, inspired by cutting-edge research in model-based RL and game AI.

### Why World Models?

Pure RL approaches (like DQN, PPO) struggle with Elsword because:
- **High-dimensional visual input** (800×600 RGB frames)
- **Complex action space** (22 discrete actions + combos)
- **Sparse rewards** in dungeon environments
- **Sample inefficiency** - millions of frames needed

World Models solve this by learning a compressed representation of the game:

```
Raw Game Frame (800×600×3) → Vision Model → Latent State (64-dim)
                                    ↓
                            World Model (MDN-RNN)
                                    ↓
                            Controller (Policy)
```

### Architecture Components

#### 1. **Vision Model (VQ-VAE)**
- **Purpose:** Compress visual observations into discrete latent codes
- **Architecture:** Vector-Quantized Variational AutoEncoder
- **Input:** 256×192×3 RGB frames
- **Output:** 64-dimensional latent representations
- **Why VQ-VAE?** Discrete codes enable more stable world model training

```python
Encoder → Quantization (512 codes) → Decoder
  ↓                                      ↓
256×192×3                          256×192×3 (reconstructed)
```

#### 2. **Memory Model (MDN-RNN)**
- **Purpose:** Learn temporal dynamics and predict future states
- **Architecture:** Mixture Density Network + GRU
- **Input:** Current latent state + action
- **Output:** Distribution of next latent state
- **Key Feature:** Handles stochasticity in game environment

#### 3. **Control Model (Policy)**
- **Purpose:** Decide which action to take given current state
- **Training:** Reinforcement Learning in latent space
- **Advantage:** Train in imagination, not real game (sample efficient!)

### Why This Architecture for Elsword?

1. **Visual Complexity:** Elsword has rich graphics, particle effects, and dynamic environments. VQ-VAE compresses this into manageable representations.

2. **Temporal Dependencies:** Combo execution, skill cooldowns, and enemy patterns require understanding sequences. MDN-RNN captures these temporal relationships.

3. **Sample Efficiency:** Training directly in-game requires millions of frames. World Models let us train in latent space, requiring 100× less data.

4. **Interpretability:** We can visualize what the AI "imagines" will happen, debug behavior, and understand decision-making.

### Tech Stack

- **Framework:** PyTorch 2.0+ with CUDA acceleration
- **Computer Vision:** OpenCV, MSS for screen capture
- **Data Pipeline:** NumPy, custom lazy-loading datasets
- **Training:** Mixed precision, gradient accumulation, LR scheduling
- **Models:** Custom implementations of VQ-VAE, MDN-RNN architectures

---

## 🚀 Installation

### Prerequisites

- **Python 3.12+**
- **CUDA-capable GPU** (RTX 3050 or better recommended)
- **CUDA Toolkit 12.9+**
- **8GB+ RAM**
- **Elsword game installed**

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/elworld.git
cd elworld

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### GPU Setup (Important!)

Ensure CUDA is properly configured:
```bash
nvidia-smi  # Should show your GPU
nvcc --version  # Should show CUDA 12.9+
```

---

## 📖 Usage

### Phase 1: Data Collection

Record your gameplay to create training data:

```bash
# Configure capture region in research/data_collection/record_gameplay.py
# Then run:
./research/scripts/collect_data.sh
```

**Controls during recording:**
- `F10` - Start recording gameplay
- `F9` - Stop current recording
- ESC - Emergency exit

**Tips:**
- Play naturally, use various strategies
- Record 5+ complete dungeon runs
- Aim for 3+ minutes per gameplay session

### Phase 2: Train Vision Model

Train the VQ-VAE to compress visual observations:

```bash
./research/scripts/train_vision.sh
```

**Training config:** `research/training/src/config.yaml`
```yaml
vision_config:
  batch_size: 256
  learning_rate: 0.0003
  num_epochs: 500
  latent_dim: 64
```

**Expected training time:** ~4-8 hours on RTX 3050

**Monitor training:**
- Checkpoints saved to `research/training/src/checkpoints/`
- Best model saved as `vision_model_best.pth`
- Logs show reconstruction quality

### Phase 3: Train Memory Model (Coming Soon)

```bash
./research/scripts/train_memory.sh
```

### Phase 4: Train Controller (Coming Soon)

```bash
./research/scripts/train_control.sh
```

### Phase 5: Deploy & Play (Coming Soon)

```bash
# Run the trained AI
python product/main.py
```

---

## 📁 Project Structure

```
elworld/
├── research/
│   ├── data_collection/       # Gameplay recording tools
│   │   ├── record_gameplay.py
│   │   ├── reshape_data.py
│   │   └── ...
│   ├── training/              # Training pipeline
│   │   ├── src/
│   │   │   ├── config.yaml    # Training configuration
│   │   │   ├── main.py        # Training entry point
│   │   │   ├── utils.py       # Config & data utilities
│   │   │   └── elworld/
│   │   │       ├── model/     # Model architectures
│   │   │       ├── train/     # Training logic
│   │   │       ├── data/      # Dataset classes
│   │   │       └── preprocess/
│   │   └── recorded/          # Gameplay data storage
│   └── scripts/               # Convenience scripts
│       ├── collect_data.sh
│       ├── train_vision.sh
│       ├── train_memory.sh
│       └── train_control.sh
└── product/                   # Production deployment (WIP)
    ├── main.py
    ├── model_architectures/
    └── pipeline/
```

---

## 🔧 Configuration

### Training Configuration

Edit `research/training/src/config.yaml`:

```yaml
general_config:
  total_play: 5          # Number of gameplay recordings
  frame_rate: 20         # FPS for recording
  data_path: "../recorded/"

vision_config:
  batch_size: 256        # Adjust based on GPU memory
  learning_rate: 0.0003  # AdamW learning rate
  num_epochs: 500        # Training iterations
  latent_dim: 64         # Compression dimension
  num_embedding: 512     # VQ codebook size
```

### Recording Configuration

Edit `research/data_collection/record_gameplay.py`:

```python
MANUAL_REGION = {
    "top": 210,      # Y coordinate
    "left": 560,     # X coordinate  
    "width": 800,    # Capture width
    "height": 600    # Capture height
}

GAMEPLAY_DURATION = 180  # 3 minutes per recording
FPS = 20                 # 20 frames per second
```

---

## 🤝 Contributing

Contributions are welcome! This project is still in active development.

### Areas We Need Help With:
- 🧠 **RL expertise** - Implementing stable controller training
- 🎮 **Game mechanics** - Understanding Elsword-specific behaviors
- ⚡ **Optimization** - Faster inference, lower latency
- 📊 **Visualization** - Better training monitoring
- 🧪 **Testing** - Multi-character, multi-dungeon validation

---

## 🎓 Research & References

This project builds upon recent advances in model-based RL:

- **World Models** - Ha & Schmidhuber (2018)
- **VQ-VAE** - van den Oord et al. (2017)
- **Dreamer** - Hafner et al. (2020)
- **MuZero** - Schrittwieser et al. (2020)

Key insights applied to Elsword:
- Discrete latent representations for stability
- Temporal modeling with recurrent networks
- Learning in imagination for sample efficiency

---

## ⚠️ Disclaimer

**Important Legal & Ethical Notes:**

1. **Game Terms of Service:** Using automation tools may violate Elsword's Terms of Service. Use at your own risk.

2. **Educational Purpose:** This project is primarily for learning and research in:
   - Deep Reinforcement Learning
   - Computer Vision
   - Game AI Development

3. **Fair Play:** Consider the impact on other players. This tool is designed for single-player content only.

4. **No Warranty:** This software is provided "as is" without warranty of any kind.

5. **Responsibility:** Users are solely responsible for how they use this tool.

---

## 📞 Contact

**Author:** Hoang Quoc Viet (Raphael Hoang) (hqvjet)
- GitHub: [@hqvjet](https://github.com/hqvjet)
- Email: viethq.1906@gmail.com

---

## 💖 Support This Project

If Elworld saves you time or you find it valuable, consider supporting development:

### Ways to Support:
- ⭐ **Star this repository** - It helps others discover the project
- 🐛 **Report bugs** - Help us improve stability
- 💡 **Suggest features** - Share your ideas
- 🔀 **Contribute code** - PRs are welcome!

### Donate:

<div align="center">

**Scan QR Code to Support via PayPal/Banking**

<img src="assets/donate.svg" alt="Donate QR Code" width="300">

*Every contribution helps keep this project alive and improving!*

</div>

Your support helps maintain and improve Elworld. Every contribution is appreciated! 🙏

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with ❤️ for the Elsword community**

*Because your time is valuable, and grinding isn't.*

[⬆ Back to top](#-elworld-ai-powered-elsword-automation)

</div>
