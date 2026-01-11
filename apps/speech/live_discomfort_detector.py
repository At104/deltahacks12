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
        self.discomfort_history = deque(maxlen=100)
        self.all_scores = []
        self.output_dir = Path("live_output")
        self.output_dir.mkdir(exist_ok=True)
        self.analysis_count = 0
        
        # Current analysis results
        self.current_score = 0
        self.current_level = "Starting..."
        self.current_color = (255, 255, 255)
        self.current_landmarks = {}
        self.is_analyzing = False
    
    def clear_clips(self):
        """Delete all video clips from output directory"""
        if self.output_dir.exists():
            for file in self.output_dir.glob("*.avi"):
                try:
                    file.unlink()
                except Exception:
                    pass
            for file in self.output_dir.glob("*.csv"):
                try:
                    file.unlink()
                except Exception:
                    pass
        
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
        
        cmd = [
            str(FEATURE_EXTRACTION),
            "-f", str(video_path),
            "-out_dir", str(self.output_dir),
            "-wild",
            "-q"
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, timeout=10)
            
            csv_file = self.output_dir / f"{video_path.stem}.csv"
            
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                
                if len(df) > 0:
                    mid_idx = len(df) // 2
                    analysis_result = self.detect_discomfort_from_data(df.iloc[mid_idx])
                    
                    if analysis_result:
                        self.current_score = analysis_result['score']
                        self.current_level, self.current_color = self.get_discomfort_level(analysis_result['score'])
                        self.current_landmarks = analysis_result['landmarks']
                        self.discomfort_history.append(analysis_result['score'])
                        self.all_scores.append(analysis_result['score'])
                        self.analysis_count += 1
                
                csv_file.unlink()
            
            if video_path.exists():
                video_path.unlink()
                
        except Exception:
            pass
        
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
        
        print("\n▶ Recording...\n")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 15
        
        frame_buffer = []
        frames_per_clip = 20  # ~1.3 second clips for stable processing
        frame_count = 0
        clip_count = 0
        start_time = time.time()  # Track recording start time
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4V codec
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Track elapsed time
            elapsed_time = time.time() - start_time
            
            # Use frame directly (no flip)
            display_frame = frame.copy()
            
            # Collect frames
            frame_buffer.append(frame.copy())
            frame_count += 1
            
            # Process clip when buffer is full and not currently analyzing
            if len(frame_buffer) >= frames_per_clip and not self.is_analyzing:
                clip_count += 1
                video_path = self.output_dir / f"clip_{clip_count}.avi"
                
                out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
                for f in frame_buffer:
                    out.write(f)
                out.release()
                
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
            cv2.rectangle(overlay, (10, height-230), (630, height-10), (0, 0, 0), -1)
            display_frame = cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0)
            
            y_pos = height - 210
            cv2.putText(display_frame, "LIVE DISCOMFORT DETECTION", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Elapsed time
            y_pos += 30
            time_text = f"Recording Time: {int(elapsed_time)}s"
            cv2.putText(display_frame, time_text, (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)
            
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
            
            cv2.putText(display_frame, "Press 'q' to stop", (470, y_pos+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('Live Discomfort Detector', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        time.sleep(2)  # Brief wait for any pending analyses
        
        print("\n" + "="*60)
        print("✨ RECORDING COMPLETE - FINAL SUMMARY")
        print("="*60)
        
        if self.all_scores:
            total_recording_time = time.time() - start_time
            avg_discomfort = sum(self.all_scores) / len(self.all_scores)
            max_discomfort = max(self.all_scores)
            
            print(f"\nDuration: {total_recording_time:.1f}s | Clips: {self.analysis_count}")
            print(f"\nAverage Discomfort: {avg_discomfort:.2f}")
            print(f"Peak Discomfort: {max_discomfort:.2f}")
            
            avg_level, _ = self.get_discomfort_level(avg_discomfort)
            print(f"\nOverall: {avg_level}")
            
            # Distribution
            comfortable = sum(1 for s in self.all_scores if s < 2)
            mild = sum(1 for s in self.all_scores if 2 <= s < 5)
            moderate = sum(1 for s in self.all_scores if 5 <= s < 8)
            high = sum(1 for s in self.all_scores if s >= 8)
            total = len(self.all_scores)
            
            print(f"\nComfortable: {comfortable/total*100:.0f}% | Mild: {mild/total*100:.0f}% | Moderate: {moderate/total*100:.0f}% | High: {high/total*100:.0f}%")
        else:
            print("\n⚠️ No data collected.")
        
        print("\n" + "="*60)

def main():
    print("\n🎭 Live Discomfort Detector")
    
    if not FEATURE_EXTRACTION.exists():
        print(f"❌ OpenFace not found")
        return
    
    detector = LiveDiscomfortDetector()
    
    # Check for old clips
    old_clips = list(detector.output_dir.glob("*.avi"))
    if old_clips:
        response = input(f"Found {len(old_clips)} old clip(s). Clear them? (y/n): ").strip().lower()
        if response == 'y':
            detector.clear_clips()
            print("✓ Clips cleared")
    
    print("Press 'q' to stop and see results\n")
    input("Press Enter to start...")
    
    detector.run_live_detection()

if __name__ == "__main__":
    main()
