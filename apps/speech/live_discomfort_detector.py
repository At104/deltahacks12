import cv2
import subprocess
import pandas as pd
from pathlib import Path
import time

# OpenFace paths
SCRIPT_DIR = Path(__file__).parent
OPENFACE_DIR = SCRIPT_DIR.parent.parent / "OpenFace_2.2.0_win_x64"
FEATURE_EXTRACTION = OPENFACE_DIR / "FeatureExtraction.exe"

class DiscomfortAnalyzer:
    """Analyze facial discomfort from video using OpenFace"""
    
    def __init__(self):
        self.output_dir = Path("live_output")
        self.output_dir.mkdir(exist_ok=True)
        
    def detect_discomfort_from_data(self, row):
        """Detect discomfort indicators from facial action units"""
        confidence = row.get(' confidence', 0)
        if confidence < 0.5:  # Lowered from 0.7
            return None
        
        score = 0
        indicators = []
        
        # Pain/tension indicators (lowered thresholds)
        au04 = row.get(' AU04_r', 0)  # Brow Lowerer
        if au04 > 1.0:  # Lowered from 2
            score += au04 * 2.0
            indicators.append(f"Brow:{au04:.1f}")
        
        au06 = row.get(' AU06_r', 0)  # Cheek Raiser
        au07 = row.get(' AU07_r', 0)  # Lid Tightener
        if au06 > 1.0 or au07 > 1.0:  # Lowered from 2
            score += (au06 + au07) * 1.5
            indicators.append(f"Eye:{au06:.1f},{au07:.1f}")
        
        au09 = row.get(' AU09_r', 0)  # Nose Wrinkler
        if au09 > 1.0:  # Lowered from 1.5
            score += au09 * 2.0
            indicators.append(f"Nose:{au09:.1f}")
        
        au10 = row.get(' AU10_r', 0)  # Upper Lip Raiser
        if au10 > 1.0:  # Lowered from 1.5
            score += au10 * 2.0
            indicators.append(f"UpperLip:{au10:.1f}")
        
        # Tension (lowered thresholds)
        au14 = row.get(' AU14_r', 0)
        au20 = row.get(' AU20_r', 0)
        au23 = row.get(' AU23_r', 0)
        if any(au > 1.0 for au in [au14, au20, au23]):  # Lowered from 1.5
            score += sum([au for au in [au14, au20, au23] if au > 1.0]) * 1.5
            indicators.append(f"Lip:{au14:.1f},{au20:.1f},{au23:.1f}")
        
        au17 = row.get(' AU17_r', 0)  # Chin Raiser
        if au17 > 1.0:  # Lowered from 2
            score += au17 * 1.5
            indicators.append(f"Chin:{au17:.1f}")
        
        # Gaze aversion
        gaze_x = abs(row.get(' gaze_angle_x', 0))
        gaze_y = abs(row.get(' gaze_angle_y', 0))
        if gaze_x > 0.3 or gaze_y > 0.3:  # Lowered from 0.5
            score += 2
            indicators.append(f"Gaze:{gaze_x:.1f},{gaze_y:.1f}")
        
        # Head down
        head_pitch = row.get(' pose_Rx', 0)
        if head_pitch > 0.2:  # Lowered from 0.3
            score += 2
            indicators.append(f"Head:{head_pitch:.1f}")
        
        # Reduce if smiling
        au12 = row.get(' AU12_r', 0)
        if au12 > 1.5:  # Lowered from 2
            score -= au12 * 0.5
            indicators.append(f"Smile:{au12:.1f}")
        
        if indicators:
            print(f"[Frame] {', '.join(indicators)} -> Score: {score:.2f}")
        
        return {'score': max(0, score), 'confidence': confidence}
    
    def get_discomfort_level(self, score):
        """Convert score to level and color"""
        if score < 2:
            return "Comfortable 😊", (0, 255, 0)
        elif score < 5:
            return "Mild Discomfort 😐", (0, 200, 200)
        elif score < 8:
            return "Moderate Discomfort 😰", (0, 165, 255)
        else:
            return "High Discomfort 😣", (0, 0, 255)
    
    def get_relaxation_level(self, discomfort_score):
        """Convert discomfort score to relaxation level (inverse)"""
        # Normalize discomfort to 0-10 scale, then invert to get relaxation
        # Max discomfort ~15, so normalize and invert
        normalized = min(discomfort_score / 15.0, 1.0)
        relaxation = (1.0 - normalized) * 10.0
        
        if relaxation >= 8:
            return relaxation, "Very Relaxed 😌"
        elif relaxation >= 6:
            return relaxation, "Relaxed 🙂"
        elif relaxation >= 4:
            return relaxation, "Somewhat Tense 😐"
        else:
            return relaxation, "Tense 😰"
    
    
    def analyze_video(self, video_path):
        """Analyze video file and return average discomfort score"""
        cmd = [
            str(FEATURE_EXTRACTION),
            "-f", str(video_path),
            "-out_dir", str(self.output_dir),
            "-wild",
            "-q"
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, timeout=60)
            
            csv_file = self.output_dir / f"{video_path.stem}.csv"
            
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                scores = []
                
                for idx in range(len(df)):
                    analysis_result = self.detect_discomfort_from_data(df.iloc[idx])
                    if analysis_result:
                        scores.append(analysis_result['score'])
                
                csv_file.unlink()
                
                if scores:
                    return sum(scores) / len(scores)
            
            return None
                
        except Exception:
            return None

def record_and_analyze(duration=None, output_file="recording.avi", show_preview=True):
    """Record video and analyze discomfort level.
    
    Args:
        duration: Recording duration in seconds (None = until 'q' pressed)
        output_file: Path to save the video file
        show_preview: Whether to show live preview window
    
    Returns:
        float: Average discomfort score (0-10+), or None if analysis failed
    """
    analyzer = DiscomfortAnalyzer()
    output_path = analyzer.output_dir / output_file
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 15
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    start_time = time.time()
    
    print(f"\n▶ Recording{'...' if duration else ' (press q to stop)...'}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        elapsed = time.time() - start_time
        
        if duration and elapsed >= duration:
            break
        
        out.write(frame)
        
        if show_preview:
            display_frame = frame.copy()
            text = f"Recording: {int(elapsed)}s"
            cv2.putText(display_frame, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Recording', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    out.release()
    if show_preview:
        cv2.destroyAllWindows()
    
    recording_time = time.time() - start_time
    print(f"✓ Recorded {recording_time:.1f}s")
    print("Analyzing...")
    
    avg_discomfort = analyzer.analyze_video(output_path)
    
    if avg_discomfort is not None:
        discomfort_level, _ = analyzer.get_discomfort_level(avg_discomfort)
        relaxation_score, relaxation_level = analyzer.get_relaxation_level(avg_discomfort)
        
        print(f"\nDiscomfort Score: {avg_discomfort:.2f} - {discomfort_level}")
        print(f"Relaxation Score: {relaxation_score:.2f}/10 - {relaxation_level}")
    else:
        print("⚠️ Analysis failed")
    
    return avg_discomfort
    

def interactive_mode():
    """Run interactive recording mode with prompts"""
    print("\n🎭 Live Discomfort Detector")
    
    if not FEATURE_EXTRACTION.exists():
        print("❌ OpenFace not found")
        return
    
    analyzer = DiscomfortAnalyzer()
    
    # Check for old files
    old_files = list(analyzer.output_dir.glob("*"))
    old_files = [f for f in old_files if f.is_file() or f.is_dir()]
    if old_files:
        response = input(f"Found {len(old_files)} old file(s) in output. Clear all? (y/n): ").strip().lower()
        if response == 'y':
            import shutil
            for f in old_files:
                try:
                    if f.is_file():
                        f.unlink()
                    elif f.is_dir():
                        shutil.rmtree(f)
                except Exception:
                    pass
            print("✓ Cleared")
    
    print("\nPress 'q' during recording to stop")
    input("Press Enter to start...")
    
    result = record_and_analyze(duration=None, output_file="recording.avi")
    
    if result:
        print(f"\n{'='*40}")
        print("Recording saved to live_output/recording.avi")
        print(f"{'='*40}")

def main():
    interactive_mode()

if __name__ == "__main__":
    main()
