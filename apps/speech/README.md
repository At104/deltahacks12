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