# Speech Transcription Service

This Python application provides live speech-to-text transcription with automatic translation to English using Google's Gemini API.

## Setup

1. Install dependencies:
   ```bash
    brew install portaudio
    pip install -r apps/speech/requirements.txt
    brew install flac

   ```

2. Set up your Gemini API key:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```

3. If rate limit, switch to any model below:
   ```
    models/gemini-2.5-flash-lite
    models/gemini-2.5-pro
    models/gemini-3-flash-preview
    models/gemini-2.5-flash
   ```

## Usage

Run the live transcription:
```bash
python speech_transcription.py
```

The application will:
- Listen for speech through your microphone
- Transcribe the speech to text
- Translate non-English text to English using Gemini API
- Display both the original transcription and translation

Press Ctrl+C to stop the transcription.

## Requirements

- Python 3.7+
- Microphone access
- Internet connection for API calls
- Gemini API key

---

## Facial Discomfort Detector

Real-time facial analysis system that detects signs of discomfort, tension, or pain using OpenFace.

### Setup

1. **Download OpenFace** (Windows x64):
   - Download from [OpenFace Releases](https://github.com/TadasBaltrusaitis/OpenFace/releases)
   - Extract to `dlh12/OpenFace_2.2.0_win_x64/`

2. **Download OpenFace Models**:
   ```bash
   cd OpenFace_2.2.0_win_x64
   powershell -ExecutionPolicy Bypass -File .\download_models.ps1
   ```

3. **Install Python Dependencies**:
   ```bash
   pip install opencv-python pandas
   ```

### Usage

**Live Discomfort Detector** (with facial landmarks and scoring):
```bash
python live_discomfort_detector.py
```

**Simple Live Detector** (continuous testing):
```bash
python simple_live_detector.py
```

### Features

- **Real-time Analysis**: Processes video clips continuously (~1.3 seconds per clip)
- **Facial Landmarks**: Displays 68-point facial landmark tracking
- **Discomfort Scoring**: Analyzes Action Units (AU) to detect:
  - Brow tension (AU04)
  - Pain expressions (AU06, AU07)
  - Nose wrinkle/discomfort (AU09)
  - Disgust (AU10)
  - Lip tension (AU14, AU20, AU23)
  - Sadness (AU17)
  - Gaze aversion
  - Head down posture
- **Live Feedback**: Color-coded status (Green: Comfortable, Yellow: Mild, Orange: Moderate, Red: High discomfort)
- **Session Summary**: Average and peak discomfort scores

### Controls

- Press `q` to quit the detector
- Press `Enter` to start analysis (live_discomfort_detector)

### Output

Results are saved in `live_output/` and `test_output/` directories (automatically cleaned up during analysis).