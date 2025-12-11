import numpy as np
import cv2

# Input npz file name
INPUT_FILE = "elsword_gameplay_01.npz"
# Output video file name
OUTPUT_FILE = "debug_replay.mp4"
# Video frame rate (FPS)
FPS = 20

def npz_to_video():
    try:
        # 1. Load data
        print(f"Reading file {INPUT_FILE}...")
        data = np.load(INPUT_FILE)
        
        # Check keys in file (usually 'obs', 'act'...)
        print("Keys found:", list(data.keys()))
        
        if 'obs' not in data:
            print("Error: 'obs' key containing images not found.")
            return

        images = data['obs']
        num_frames, height, width, channels = images.shape
        print(f"Data OK! Total: {num_frames} frames. Size: {width}x{height}")

        # 2. Configure Video Writer
        # 'mp4v' for .mp4, try 'XVID' for .avi if error occurs
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, (width, height))

        print("Rendering video...")
        
        for i in range(num_frames):
            frame = images[i]
            
            # IMPORTANT: OpenCV uses BGR color space, while World Model typically saves RGB
            # Must convert back to prevent blue tint in video
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            out.write(frame_bgr)
            
            if i % 100 == 0:
                print(f"Processed {i}/{num_frames} frames...")

        out.release()
        print(f"\n✅ Done! Open '{OUTPUT_FILE}' to view.")

    except FileNotFoundError:
        print(f"❌ Error: File '{INPUT_FILE}' not found. Check the filename.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    npz_to_video()