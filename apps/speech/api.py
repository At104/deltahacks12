import os
import io
import wave
import base64
import json
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
import ollama
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Local models - no API keys needed!
WHISPER_MODEL_SIZE = "base"  # fast and accurate
OLLAMA_MODEL = "gemma2:2b"  # lightweight model

# Patient data storage directory
PATIENTS_DIR = Path(__file__).parent.parent.parent / "saved_patients"
PATIENTS_DIR.mkdir(exist_ok=True)

print("🔄 Loading local Whisper model...")
stt_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
print("✅ Whisper model loaded!")
print(f"📁 Patient data will be saved to: {PATIENTS_DIR.absolute()}")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok", 
        "mode": "local",
        "whisper_model": WHISPER_MODEL_SIZE,
        "ollama_model": OLLAMA_MODEL
    })

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    
    try:
        # Save audio to BytesIO
        audio_data = io.BytesIO(audio_file.read())
        
        # Transcribe locally with Whisper (detects language automatically)
        print("📝 Transcribing with Whisper...")
        segments, info = stt_model.transcribe(audio_data, beam_size=5)
        original_text = " ".join([s.text for s in segments]).strip()
        detected_language = info.language
        
        if not original_text:
            return jsonify({"error": "No speech detected"}), 400
        
        print(f"🎤 Heard: {original_text}")
        print(f"🌍 Detected language: {detected_language}")
        
        # Only translate if NOT English
        if detected_language.lower() == 'en':
            print("✅ Already in English, skipping translation")
            translation = original_text
        else:
            # Translate locally with Ollama
            print(f"🌐 Translating from {detected_language} to English with Ollama...")
            response = ollama.chat(model=OLLAMA_MODEL, messages=[
                {'role': 'system', 'content': 'Translate the user text to English. Return ONLY the translation, nothing else.'},
                {'role': 'user', 'content': original_text}
            ])
            
            translation = response['message']['content'].strip()
            print(f"✅ Translation: {translation}")
        
        return jsonify({
            "original": original_text,
            "translation": translation,
            "language": detected_language,
            "success": True
        })
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/save-patient', methods=['POST'])
def save_patient():
    try:
        data = request.json
        patient_name = data.get('name', 'Unknown')
        symptoms = data.get('symptoms', '')
        photo_base64 = data.get('photo', '')
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        # Sanitize patient name for folder
        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in patient_name)
        safe_name = safe_name.strip().replace(' ', '_')
        
        # Create unique folder: PatientName_YYYYMMDD_HHMMSS
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{safe_name}_{timestamp_str}"
        patient_folder = PATIENTS_DIR / folder_name
        patient_folder.mkdir(exist_ok=True)
        
        # Save transcript
        transcript_file = patient_folder / "symptoms.txt"
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(f"Patient: {patient_name}\n")
            f.write(f"Date: {timestamp}\n")
            f.write(f"\nSymptoms:\n{symptoms}\n")
        
        # Save photo if provided
        if photo_base64:
            # Remove data URL prefix if present
            if ',' in photo_base64:
                photo_base64 = photo_base64.split(',')[1]
            
            photo_file = patient_folder / "photo.jpg"
            photo_bytes = base64.b64decode(photo_base64)
            with open(photo_file, 'wb') as f:
                f.write(photo_bytes)
        
        # Save metadata
        metadata = {
            "patient_name": patient_name,
            "timestamp": timestamp,
            "symptoms": symptoms,
            "has_photo": bool(photo_base64)
        }
        metadata_file = patient_folder / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved patient data to: {patient_folder}")
        
        return jsonify({
            "success": True,
            "folder_path": str(patient_folder.absolute()),
            "folder_name": folder_name
        })
    
    except Exception as e:
        print(f"❌ Error saving patient data: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print(f"\n🚀 Local transcription API running on http://localhost:5001")
    print(f"   Using Whisper ({WHISPER_MODEL_SIZE}) + Ollama ({OLLAMA_MODEL})")
    print(f"   No API keys needed - 100% local processing!")
    print(f"   Patient data: {PATIENTS_DIR.absolute()}\n")
    app.run(host='0.0.0.0', port=5001, debug=True)
