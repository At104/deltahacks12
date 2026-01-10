import os
import wave
import io
import pyaudio
import threading
import datetime
import ollama
from faster_whisper import WhisperModel

# --- CONFIGURATION ---
# Models: "tiny", "base", "small", "medium", "large-v3"
# "base" is fast and accurate enough for most tasks.
WHISPER_MODEL_SIZE = "base" 
OLLAMA_MODEL = "gemma3:4b"

# Global array to store history
session_data = []

class LocalRecorder:
    def __init__(self):
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000 # Whisper prefers 16kHz
        self.p = pyaudio.PyAudio()
        self.frames = []
        self.recording = False

    def start(self):
        self.recording = True
        self.frames = []
        self.thread = threading.Thread(target=self._record)
        self.thread.start()
        print("\n🔴 Local Recording... (Press Enter to stop)")

    def _record(self):
        stream = self.p.open(format=self.format, channels=self.channels,
                             rate=self.rate, input=True, frames_per_buffer=self.chunk)
        while self.recording:
            self.frames.append(stream.read(self.chunk, exception_on_overflow=False))
        stream.stop_stream()
        stream.close()

    def stop(self):
        self.recording = False
        self.thread.join()
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))
        audio_buffer.seek(0)
        return audio_buffer

def main():
    print("🔄 Loading local Whisper model (this may take a moment)...")
    # 'cpu' is safe for everyone; use 'cuda' if you have an NVIDIA GPU
    stt_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    recorder = LocalRecorder()

    print(f"--- Local AI Manager (Whisper + {OLLAMA_MODEL}) ---")

    try:
        while True:
            cmd = input("\n[Enter] to Record, [S] to show array, [Q] to quit: ").lower()
            
            if cmd == '':
                recorder.start()
                input() # Wait for stop
                audio_file = recorder.stop()
                
                print("📝 Transcribing locally...")
                segments, _ = stt_model.transcribe(audio_file, beam_size=5)
                original_text = " ".join([s.text for s in segments]).strip()
                
                if not original_text:
                    print("⚠️ No speech detected.")
                    continue

                print(f"🎤 Heard: {original_text}")
                print("🌐 Translating via Ollama...")
                
                # Local LLM translation call
                response = ollama.chat(model=OLLAMA_MODEL, messages=[
                    {'role': 'system', 'content': 'Translate the user text to English. Return ONLY the translation.'},
                    {'role': 'user', 'content': original_text}
                ])
                
                translation = response['message']['content'].strip()
                
                entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "original": original_text,
                    "translation": translation
                }
                session_data.append(entry)
                
                print(f"✅ Result: {translation}")

            elif cmd == 's':
                print("\n--- Local session_data ---")
                for item in session_data: print(item)
            
            elif cmd == 'q':
                break
    finally:
        print(f"\nDone. Session captured {len(session_data)} entries.")

if __name__ == "__main__":
    main()