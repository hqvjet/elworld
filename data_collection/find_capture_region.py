"""
Helper script to find exact capture region for Elsword game
Run this script to determine correct coordinates for MANUAL_REGION
"""

import mss
import cv2
import numpy as np
import time

print("=" * 60)
print("🔍 CAPTURE REGION FINDER TOOL")
print("=" * 60)

# Display all monitors
with mss.mss() as sct:
    print("\n📺 Monitor list:")
    for i, monitor in enumerate(sct.monitors):
        if i == 0:
            print(f"   [{i}] ALL SCREENS: {monitor}")
        else:
            print(f"   [{i}] Monitor {i}: {monitor}")
    
    print("\n🎯 Capturing main screen in 3 seconds...")
    time.sleep(3)
    
    # Capture primary screen (monitor 1)
    monitor = sct.monitors[1]
    screenshot = sct.grab(monitor)
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    # Save image for inspection
    cv2.imwrite("screen_capture_test.png", img)
    print(f"✅ Saved image 'screen_capture_test.png' ({monitor['width']}x{monitor['height']})")
    print("\n📝 Instructions:")
    print("   1. Open file 'screen_capture_test.png'")
    print("   2. Use paint/photoshop to identify game region coordinates")
    print("   3. Get top-left corner coordinates (left, top) and dimensions (width, height)")
    print("   4. Update MANUAL_REGION in record_gameplay.py")
    print("\n💡 Example:")
    print("   MANUAL_REGION = {")
    print("       'top': 50,      # Y coordinate of game's top-left corner")
    print("       'left': 100,    # X coordinate of game's top-left corner")
    print("       'width': 1600,  # Game window width")
    print("       'height': 900   # Game window height")
    print("   }")
    
    print("\n" + "=" * 60)

# Display all windows (if pygetwindow available)
try:
    import pygetwindow as gw
    print("\n🪟 List of all open windows:")
    windows = gw.getAllWindows()
    for i, w in enumerate(windows[:20]):  # Show first 20 windows only
        if w.title.strip():  # Skip windows without title
            print(f"   [{i}] '{w.title}'")
            print(f"       Pos({w.left}, {w.top}) Size({w.width}x{w.height})")
    
    # Find Elsword windows
    print("\n🎮 Looking for Elsword windows:")
    elsword_windows = gw.getWindowsWithTitle("Elsword")
    if elsword_windows:
        for i, w in enumerate(elsword_windows):
            print(f"   [{i}] {w.title}")
            print(f"       Top: {w.top}, Left: {w.left}")
            print(f"       Width: {w.width}, Height: {w.height}")
    else:
        print("   ❌ Elsword windows not found")
        print("   💡 Make sure the game is running and try searching with different name")
        
except ImportError:
    print("\n⚠️ pygetwindow not available - skipping this section")

print("\n" + "=" * 60)
