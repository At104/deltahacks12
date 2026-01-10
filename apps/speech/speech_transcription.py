import os
import wave
import io
import pyaudio
import threading
import google.generativeai as genai

# Configuration
API_KEY = os.getenv('GEMINI_API_KEY')
MODEL_NAME = 'models/gemini-2.5-flash-lite'

class AudioRecorder:
    def __init__(self):
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.p = pyaudio.PyAudio()
        self.frames = []
        self.recording = False

    def start_recording(self):
        self.recording = True
        self.frames = []
        self.thread = threading.Thread(target=self._record)
        self.thread.start()
        print("🔴 Recording... (Press Enter to stop)")

    def _record(self):
        stream = self.p.open(format=self.format, channels=self.channels,
                             rate=self.rate, input=True, frames_per_buffer=self.chunk)
        while self.recording:
            data = stream.read(self.chunk, exception_on_overflow=False)
            self.frames.append(data)
        stream.stop_stream()
        stream.close()

    def stop_recording(self):
        self.recording = False
        self.thread.join()
        
        # Convert to WAV in memory
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))
        audio_buffer.seek(0)
        return audio_buffer

def translate_speech():
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    recorder = AudioRecorder()

    while True:
        input("\n🎤 Press Enter to START recording...")
        recorder.start_recording()
        
        input() # Wait for second Enter
        audio_file = recorder.stop_recording()
        print("⏳ Processing with Gemini...")

        # Multimodal Prompt
        prompt = "Transcribe this audio. If it is not English, translate it to English. Output format: [Original Text] -> [English Translation]"
        
        try:
            response = model.generate_content([
                prompt,
                {"mime_type": "audio/wav", "data": audio_file.read()}
            ])
            print(f"\n✨ Result:\n{response.text}\n")
        except Exception as e:
            print(f"❌ Error: {e}")

        if input("Continue? (y/n): ").lower() != 'y':
            break
        
if __name__ == "__main__":
    if not API_KEY:
        print("Please set your GEMINI_API_KEY environment variable.")
    else:
        try:
            translate_speech()
        finally:
            # This ensures the terminal returns to normal even if gRPC grumbles
            print("\nShutting down gracefully...")
            os._exit(0)