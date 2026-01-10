import cv2
import subprocess
import pandas as pd
from pathlib import Path
import time
from collections import deque

# OpenFace paths - automatically find relative to this script
SCRIPT_DIR = Path(__file__).parent
OPENFACE_DIR = SCRIPT_DIR.parent.parent / "OpenFace_2.2.0_win_x64"
FEATURE_EXTRACTION = OPENFACE_DIR / "FeatureExtraction.exe"

print("=" * 60)
print("🎭 Simple Live Discomfort Detector")
print("=" * 60)
print(f"OpenFace: {'✓ Found' if FEATURE_EXTRACTION.exists() else '✗ Not Found'}")

if not FEATURE_EXTRACTION.exists():
    print(f"❌ OpenFace not found at {FEATURE_EXTRACTION}")
    exit(1)

print("\nStarting live detection...")
print("Press 'q' in the video window to quit\n")

# Setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam")
    exit(1)

output_dir = Path("test_output")
output_dir.mkdir(exist_ok=True)

fps = 15
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

clip_count = 0

while True:
    clip_count += 1
    test_video = output_dir / f"test_{clip_count}.avi"
    
    # Record clip
    out = cv2.VideoWriter(str(test_video), fourcc, fps, (width, height))
    
    print(f"Recording clip {clip_count}...")
    for i in range(45):  # 3 seconds at 15 fps
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow('Live Detection - Press q to quit', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                out.release()
                cap.release()
                cv2.destroyAllWindows()
                print("\n✨ Stopped by user")
                exit(0)
    
    out.release()
    
    print(f"Analyzing clip {clip_count}...")

    print(f"Analyzing clip {clip_count}...")
    
    # Run OpenFace
    cmd = [
        str(FEATURE_EXTRACTION),
        "-f", str(test_video),
        "-out_dir", str(output_dir),
        "-q"  # Quiet mode
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check for CSV
    csv_file = output_dir / f"{test_video.stem}.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        print(f"✓ Clip {clip_count}: {len(df)} frames analyzed")
        
        if len(df) > 0:
            # Check for landmarks
            landmark_cols = [col for col in df.columns if col.strip().startswith('x_') or col.strip().startswith('y_')]
            au_cols = [col for col in df.columns if 'AU' in col]
            print(f"  Landmarks: {len(landmark_cols)//2} points, Action Units: {len(au_cols)}")
        
        # Cleanup
        csv_file.unlink()
        test_video.unlink()
    else:
        print(f"✗ No CSV generated for clip {clip_count}")
    
    print()

print("\n" + "=" * 60)
