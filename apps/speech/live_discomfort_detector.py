import cv2
import subprocess
import pandas as pd
from pathlib import Path
import time
from collections import deque
import threading

# OpenFace paths - automatically find relative to this script
SCRIPT_DIR = Path(__file__).parent
OPENFACE_DIR = SCRIPT_DIR.parent.parent / "OpenFace_2.2.0_win_x64"
FEATURE_EXTRACTION = OPENFACE_DIR / "FeatureExtraction.exe"

class LiveDiscomfortDetector:
    """Real-time facial discomfort detection using OpenFace"""
    
    def __init__(self):
        self.discomfort_history = deque(maxlen=30)
        self.output_dir = Path("live_output")
        self.output_dir.mkdir(exist_ok=True)
        self.analysis_count = 0
        
        # Current analysis results
        self.current_score = 0
        self.current_level = "Starting..."
        self.current_color = (255, 255, 255)
        self.current_landmarks = {}
        self.is_analyzing = False
        
    def detect_discomfort_from_data(self, row):
        """Detect discomfort indicators from facial action units"""
        confidence = row.get(' confidence', 0)
        if confidence < 0.7:
            return None
        
        score = 0
        indicators = []
        landmarks = {}
        
        # Extract facial landmarks
        for i in range(68):
            x_col = f' x_{i}'
            y_col = f' y_{i}'
            if x_col in row and y_col in row:
                landmarks[i] = (int(row[x_col]), int(row[y_col]))
        
        # Pain/tension indicators
        au04 = row.get(' AU04_r', 0)  # Brow Lowerer
        if au04 > 2:
            score += au04 * 1.5
            indicators.append("Tension")
        
        au06 = row.get(' AU06_r', 0)  # Cheek Raiser
        au07 = row.get(' AU07_r', 0)  # Lid Tightener
        if au06 > 2 and au07 > 2:
            score += (au06 + au07)
            indicators.append("Pain")
        
        au09 = row.get(' AU09_r', 0)  # Nose Wrinkler
        if au09 > 1.5:
            score += au09 * 1.5
            indicators.append("Discomfort")
        
        au10 = row.get(' AU10_r', 0)  # Upper Lip Raiser
        if au10 > 1.5:
            score += au10 * 1.5
            indicators.append("Disgust")
        
        # Tension
        au14 = row.get(' AU14_r', 0)
        au20 = row.get(' AU20_r', 0)
        au23 = row.get(' AU23_r', 0)
        if any(au > 1.5 for au in [au14, au20, au23]):
            score += sum([au for au in [au14, au20, au23] if au > 1.5])
            indicators.append("Lip Tension")
        
        au17 = row.get(' AU17_r', 0)  # Chin Raiser
        if au17 > 2:
            score += au17
            indicators.append("Sadness")
        
        # Gaze aversion
        gaze_x = abs(row.get(' gaze_angle_x', 0))
        gaze_y = abs(row.get(' gaze_angle_y', 0))
        if gaze_x > 0.5 or gaze_y > 0.5:
            score += 2
            indicators.append("Looking Away")
        
        # Head down
        head_pitch = row.get(' pose_Rx', 0)
        if head_pitch > 0.3:
            score += 1.5
            indicators.append("Head Down")
        
        # Reduce if smiling
        au12 = row.get(' AU12_r', 0)
        if au12 > 2:
            score -= au12 * 0.5
        
        return {
            'score': max(0, score),
            'indicators': indicators,
            'confidence': confidence,
            'landmarks': landmarks
        }
    
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
    
    def analyze_video_async(self, video_path):
        """Analyze video in background thread"""
        self.is_analyzing = True
        
        # Run OpenFace
        cmd = [
            str(FEATURE_EXTRACTION),
            "-f", str(video_path),
            "-out_dir", str(self.output_dir),
            "-q"  # Quiet mode
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            
            # Read results
            csv_file = self.output_dir / f"{video_path.stem}.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                
                if len(df) > 0:
                    # Use middle frame for better stability
                    mid_idx = len(df) // 2
                    analysis_result = self.detect_discomfort_from_data(df.iloc[mid_idx])
                    
                    if analysis_result:
                        self.current_score = analysis_result['score']
                        self.current_level, self.current_color = self.get_discomfort_level(analysis_result['score'])
                        self.current_landmarks = analysis_result['landmarks']
                        self.discomfort_history.append(analysis_result['score'])
                        self.analysis_count += 1
                
                # Cleanup
                csv_file.unlink()
            
            # Cleanup video
            if video_path.exists():
                video_path.unlink()
                
        except Exception as e:
            print(f"Analysis error: {e}")
        
        self.is_analyzing = False
    
    def draw_landmark_lines(self, frame, landmarks, start, end, color, thickness=2):
        """Helper to draw connected landmark lines"""
        for i in range(start, end):
            if i in landmarks:
                # Draw point
                cv2.circle(frame, landmarks[i], 2, color, -1)
                # Draw line to next point
                if i+1 in landmarks:
                    pt1 = landmarks[i]
                    pt2 = landmarks[i+1]
                    cv2.line(frame, pt1, pt2, color, thickness)
    
    def run_live_detection(self):
        """Run live discomfort detection"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open webcam")
            return
        
        print("\n🎭 Live Discomfort Detector Started")
        print("Press 'q' to quit\n")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 15
        
        frame_buffer = []
        frames_per_clip = 20  # ~1.3 second clips for stable processing
        frame_count = 0
        clip_count = 0
        
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Use frame directly (no flip)
            display_frame = frame.copy()
            
            # Collect frames
            frame_buffer.append(frame.copy())
            frame_count += 1
            
            # Process clip when buffer is full and not currently analyzing
            if len(frame_buffer) >= frames_per_clip and not self.is_analyzing:
                clip_count += 1
                video_path = self.output_dir / f"clip_{clip_count}.avi"
                
                # Save clip
                out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
                for f in frame_buffer:
                    out.write(f)
                out.release()
                
                # Analyze in background
                thread = threading.Thread(target=self.analyze_video_async, args=(video_path,))
                thread.daemon = True
                thread.start()
                
                frame_buffer = []
            
            # Draw facial landmarks
            if self.current_landmarks:
                self.draw_landmark_lines(display_frame, self.current_landmarks, 0, 16, (100, 255, 100))
                self.draw_landmark_lines(display_frame, self.current_landmarks, 17, 21, (255, 255, 0))
                self.draw_landmark_lines(display_frame, self.current_landmarks, 22, 26, (255, 255, 0))
                self.draw_landmark_lines(display_frame, self.current_landmarks, 36, 41, (0, 255, 255))
                self.draw_landmark_lines(display_frame, self.current_landmarks, 42, 47, (0, 255, 255))
                self.draw_landmark_lines(display_frame, self.current_landmarks, 27, 35, (255, 100, 255))
                self.draw_landmark_lines(display_frame, self.current_landmarks, 48, 59, (255, 100, 100))
                self.draw_landmark_lines(display_frame, self.current_landmarks, 60, 67, (255, 100, 100))
            
            # Calculate average
            avg_score = sum(self.discomfort_history) / len(self.discomfort_history) if self.discomfort_history else 0
            
            # Draw UI overlay
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (10, height-200), (630, height-10), (0, 0, 0), -1)
            display_frame = cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0)
            
            y_pos = height - 180
            cv2.putText(display_frame, "LIVE DISCOMFORT DETECTION", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            y_pos += 35
            status_text = f"Status: {self.current_level}"
            if self.is_analyzing:
                status_text += " (Analyzing...)"
            cv2.putText(display_frame, status_text, (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.current_color, 2)
            
            y_pos += 30
            cv2.putText(display_frame, f"Score: {self.current_score:.1f} | Avg: {avg_score:.1f}", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Draw score bar
            y_pos += 35
            bar_length = int((min(self.current_score, 10) / 10) * 400)
            cv2.rectangle(display_frame, (20, y_pos), (420, y_pos+20), (50, 50, 50), -1)
            cv2.rectangle(display_frame, (20, y_pos), (20 + bar_length, y_pos+20), self.current_color, -1)
            cv2.rectangle(display_frame, (20, y_pos), (420, y_pos+20), (255, 255, 255), 2)
            
            cv2.putText(display_frame, "Press 'q' to quit", (470, y_pos+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('Live Discomfort Detector', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n✨ Session Summary:")
        if self.discomfort_history:
            avg = sum(self.discomfort_history) / len(self.discomfort_history)
            max_score = max(self.discomfort_history)
            print(f"   Clips Analyzed: {self.analysis_count}")
            print(f"   Average Discomfort: {avg:.2f}")
            print(f"   Peak Discomfort: {max_score:.2f}")
            level, _ = self.get_discomfort_level(avg)
            print(f"   Overall: {level}")

def main():
    print("=" * 60)
    print("🎭 Real-Time Facial Discomfort Detector")
    print("=" * 60)
    
    if not FEATURE_EXTRACTION.exists():
        print(f"❌ OpenFace not found at {FEATURE_EXTRACTION}")
        return
    
    print("✓ OpenFace found")
    print("\nThis will analyze your facial expressions in real-time")
    print("and detect signs of discomfort, tension, or pain.")
    print()
    
    input("Press Enter to start...")
    
    detector = LiveDiscomfortDetector()
    detector.run_live_detection()

if __name__ == "__main__":
    main()
