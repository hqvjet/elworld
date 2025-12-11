"""
Script to analyze action data from .npz files
Display key combos used and their frequencies
"""

import numpy as np
import sys

KEYS_TO_LOG = [
    'f8', 'up', 'down', 'left', 'right', 'z', 'x',
    '1', '3',
    'q', 'w', 'e', 'r', 't',
    'a', 's', 'd', 'c', 'f',
    'enter', 'ctrl', 'esc'
]

def action_to_string(action_vec):
    """Convert action vector to combo string"""
    pressed_keys = [KEYS_TO_LOG[i] for i, val in enumerate(action_vec) if val == 1]
    return '+'.join(pressed_keys) if pressed_keys else 'none'

def analyze_actions(npz_file):
    """Analyze npz file and display action statistics"""
    
    print("=" * 60)
    print(f"📊 ACTION DATA ANALYSIS: {npz_file}")
    print("=" * 60)
    
    try:
        data = np.load(npz_file)
        
        if 'act' not in data:
            print("❌ 'act' key not found in file!")
            return
        
        actions = data['act']
        num_frames = len(actions)
        
        print(f"\n📈 General info:")
        print(f"   Total frames: {num_frames}")
        print(f"   Action shape: {actions.shape}")
        
        # Count combos
        action_strings = [action_to_string(a) for a in actions]
        unique_combos, counts = np.unique(action_strings, return_counts=True)
        
        # Sort by frequency
        sorted_indices = np.argsort(-counts)
        
        print(f"\n🎮 Top 20 Key Combos (total {len(unique_combos)} combos):")
        print(f"{'Rank':<6}{'Combo':<30}{'Count':<10}{'Percent':<10}")
        print("-" * 60)
        
        for rank, idx in enumerate(sorted_indices[:20], 1):
            combo = unique_combos[idx]
            count = counts[idx]
            percent = (count / num_frames) * 100
            print(f"{rank:<6}{combo:<30}{count:<10}{percent:>6.2f}%")
        
        # Per-key statistics
        print(f"\n⌨️  Individual key frequency:")
        for i, key in enumerate(KEYS_TO_LOG):
            if i < actions.shape[1]:
                key_count = np.sum(actions[:, i])
                key_percent = (key_count / num_frames) * 100
                print(f"   {key:<8}: {key_count:>6} frames ({key_percent:>5.1f}%)")
        
        # Analyze combo complexity
        combo_complexity = np.sum(actions, axis=1)  # Number of keys pressed simultaneously
        avg_complexity = np.mean(combo_complexity)
        max_complexity = np.max(combo_complexity)
        
        print(f"\n🔢 Combo complexity:")
        print(f"   Average: {avg_complexity:.2f} keys/frame")
        print(f"   Max: {max_complexity} keys simultaneously")
        
        # Complexity distribution
        print(f"\n📊 Distribution of simultaneous key presses:")
        for num_keys in range(int(max_complexity) + 1):
            count = np.sum(combo_complexity == num_keys)
            percent = (count / num_frames) * 100
            bar = "█" * int(percent / 2)
            print(f"   {num_keys} keys: {count:>6} ({percent:>5.1f}%) {bar}")
        
        print("\n" + "=" * 60)
        
    except FileNotFoundError:
        print(f"❌ File not found: {npz_file}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        # Default: analyze gameplay_01 file
        filename = "elsword_gameplay_01.npz"
    
    analyze_actions(filename)
