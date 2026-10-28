import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
import numpy as np
import io
import requests
import soundfile as sf
from collections import deque
from groq import Groq
import sounddevice as sd
import os
from dotenv import load_dotenv


# 1. Сначала загружаем переменные из файла .env
# force_reload=True заставляет библиотеку перечитать файл, даже если он был в кэше
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'), override=True)

# 2. Только ПОСЛЕ этого получаем значения
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 3. Проверка (если ключей нет, программа сразу скажет об этом и остановится)
if not DEEPGRAM_API_KEY or not GROQ_API_KEY:
    print("ОШИБКА: Ключи не загружены!")
    print(f"DEEPGRAM: {DEEPGRAM_API_KEY}")
    print(f"GROQ: {GROQ_API_KEY}")
    print("Проверь файл .env в папке D:\\")
    sys.exit(1)

# 4. Инициализация клиента
from groq import Groq
client_groq = Groq(api_key=GROQ_API_KEY, timeout=30.0)

# ===== SETTINGS =====
SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.015
SILENCE_DURATION = 1.5
MIN_RECORD_DURATION = 0.8
MAX_RECORD_DURATION = 45.0
PRE_RECORD_BUFFER_SEC = 1.0


class AudioAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Audio Assistant")
        self.root.geometry("900x800")
        self.root.resizable(True, True)
        
        self.is_listening = False
        self.worker_thread = None
        self.selected_llm_model = None
        
        self.create_widgets()
        self.setup_groq_model()
        self.populate_microphones()
        self.log("App ready. Choose microphone and click Start.")
    
    def create_widgets(self):
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text="AI Audio Assistant", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=5)
        
        self.listen_btn = ttk.Button(top_frame, text="Start Listening", 
                                     command=self.toggle_listening)
        self.listen_btn.pack(side=tk.RIGHT, padx=10)
        
        self.clear_btn = ttk.Button(top_frame, text="Clear", command=self.clear_fields)
        self.clear_btn.pack(side=tk.RIGHT, padx=5)
        
        self.status_label = ttk.Label(top_frame, text="Idle", foreground="gray")
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        device_frame = ttk.Frame(self.root, padding="10")
        device_frame.pack(fill=tk.X)
        
        ttk.Label(device_frame, text="Microphone:").pack(side=tk.LEFT, padx=5)
        self.mic_combo = ttk.Combobox(device_frame, width=50, state="readonly")
        self.mic_combo.pack(side=tk.LEFT, padx=10)
        
        level_frame = ttk.Frame(self.root, padding="5")
        level_frame.pack(fill=tk.X)
        ttk.Label(level_frame, text="Level:").pack(side=tk.LEFT)
        
        self.level_bar = ttk.Progressbar(level_frame, length=400, mode='determinate')
        self.level_bar.pack(side=tk.LEFT, padx=10)
        
        self.level_label = ttk.Label(level_frame, text="0.000", width=8)
        self.level_label.pack(side=tk.LEFT)
        
        text_frame = ttk.LabelFrame(self.root, text="Recognized text", padding="10")
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.transcript_text = scrolledtext.ScrolledText(text_frame, height=6, 
                                                          font=("Consolas", 11))
        self.transcript_text.pack(fill=tk.BOTH, expand=True)
        
        ai_frame = ttk.LabelFrame(self.root, text="AI response", padding="10")
        ai_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.ai_response_text = scrolledtext.ScrolledText(ai_frame, height=6, 
                                                           font=("Consolas", 11))
        self.ai_response_text.pack(fill=tk.BOTH, expand=True)
        
        log_frame = ttk.LabelFrame(self.root, text="Event log", padding="10")
        log_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=4, 
                                                   font=("Consolas", 9), state=tk.DISABLED)
        self.log_text.pack(fill=tk.X)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def populate_microphones(self):
        devices = sd.query_devices()
        mic_devices = []
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                mic_devices.append(f"{i}: {dev['name']}")
        
        self.mic_combo['values'] = mic_devices
        if mic_devices:
            for i, dev in enumerate(mic_devices):
                if dev.startswith("3:"):
                    self.mic_combo.current(i)
                    break
            if self.mic_combo.get() == "":
                self.mic_combo.current(0)
    
    def log(self, message):
        try:
            self.log_text.config(state=tk.NORMAL)
            timestamp = time.strftime("%H:%M:%S")
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        except:
            pass
    
    def update_status(self, status, color="gray"):
        try:
            self.status_label.config(text=status, foreground=color)
        except:
            pass
    
    def update_level(self, volume):
        try:
            level_percent = min(volume / 0.1 * 100, 100)
            self.level_bar['value'] = level_percent
            self.level_label.config(text=f"{volume:.3f}")
        except:
            pass
    
    def clear_fields(self):
        self.transcript_text.delete(1.0, tk.END)
        self.ai_response_text.delete(1.0, tk.END)
        self.log("Fields cleared.")
    
    def setup_groq_model(self):
        try:
            models = client_groq.models.list()
            candidates = [
                m.id for m in models.data
                if any(x in m.id.lower() for x in ['llama3', 'llama-3', 'mixtral', 'gemma', 'qwen'])
                and 'guard' not in m.id.lower()
                and 'whisper' not in m.id.lower()
            ]
            
            if candidates:
                self.selected_llm_model = candidates[0]
                self.log(f"Groq model: {self.selected_llm_model}")
            else:
                self.log("WARNING: No suitable LLM model found")
        except Exception as e:
            self.log(f"Groq error: {e}")
    
    def toggle_listening(self):
        if self.is_listening:
            self.is_listening = False
            self.listen_btn.config(text="Start Listening")
            self.update_status("Stopped", "gray")
            self.log("Listening stopped.")
        else:
            self.is_listening = True
            self.listen_btn.config(text="Stop Listening")
            self.update_status("Listening...", "green")
            self.log("Started listening.")
            
            self.worker_thread = threading.Thread(target=self.listen_microphone, daemon=True)
            self.worker_thread.start()
    
    def listen_microphone(self):
        selected_mic = self.mic_combo.get()
        device_index = int(selected_mic.split(":")[0].strip()) if selected_mic else 3
        
        self.log(f"Using microphone device #{device_index}")
        
        stream = sd.InputStream(
            device=device_index,
            samplerate=SAMPLE_RATE, channels=1, dtype=np.float32,
            blocksize=int(SAMPLE_RATE * 0.1), latency='low'
        )
        stream.start()
        
        try:
            while self.is_listening:
                pre_buffer = deque(maxlen=int(PRE_RECORD_BUFFER_SEC * 10))
                audio_chunks = []
                is_recording = False
                silence_start = None
                start_time = None
                
                while self.is_listening:
                    data, _ = stream.read(int(SAMPLE_RATE * 0.1))
                    data_float = data.astype(np.float32)
                    volume = float(np.max(np.abs(data_float)))
                    current_time = time.time()
                    
                    self.root.after(0, self.update_level, volume)
                    
                    if volume > VAD_THRESHOLD and not is_recording:
                        is_recording = True
                        start_time = current_time
                        silence_start = None
                        audio_chunks.extend(list(pre_buffer))
                        self.root.after(0, self.update_status, "Recording...", "blue")
                        self.root.after(0, self.log, "Speech detected.")
                    
                    if is_recording:
                        audio_chunks.append(data)
                        
                        if volume < VAD_THRESHOLD:
                            if silence_start is None:
                                silence_start = current_time
                            elif current_time - silence_start > SILENCE_DURATION:
                                self.root.after(0, self.log, "Silence detected, processing...")
                                break
                        else:
                            silence_start = None
                        
                        if current_time - start_time > MAX_RECORD_DURATION:
                            self.root.after(0, self.log, "Max duration reached.")
                            break
                    
                    if not is_recording:
                        pre_buffer.append(data)
                
                if audio_chunks and self.is_listening:
                    audio_data = np.concatenate(audio_chunks, axis=0)
                    self.root.after(0, self.process_audio, audio_data)
                    time.sleep(0.2)
        finally:
            stream.stop()
            stream.close()
    
    def process_audio(self, audio_data):
        self.update_status("Transcribing...", "orange")
        
        audio_data = audio_data - np.mean(audio_data)
        peak = float(np.max(np.abs(audio_data)))
        if peak > 0:
            audio_data = audio_data * (0.7 / peak)
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        if float(np.max(np.abs(audio_data))) < 0.3:
            gain_factor = 10 ** (12 / 20)
            audio_data = np.clip(audio_data * gain_factor, -1.0, 1.0)
        
        text = self.transcribe_deepgram(audio_data)
        
        if text:
            self.transcript_text.insert(tk.END, f"{text}\n\n")
            self.transcript_text.see(tk.END)
            self.log(f"Recognized: {text[:60]}")
            
            self.update_status("Analyzing...", "purple")
            response = self.analyze_with_llm(text)
            
            if response:
                self.ai_response_text.insert(tk.END, f"{response}\n\n")
                self.ai_response_text.see(tk.END)
                self.log("AI response received.")
        
        self.update_status("Listening...", "green")
    
    def transcribe_deepgram(self, audio_data):
        try:
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_int16, SAMPLE_RATE, format='WAV', subtype='PCM_16')
            wav_buffer.seek(0)
            audio_bytes = wav_buffer.read()
            
            url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&language=ru"
            headers = {
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "audio/wav"
            }
            
            response = requests.post(url, headers=headers, data=audio_bytes, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                text = result['results']['channels'][0]['alternatives'][0]['transcript']
                return text.strip()
            else:
                self.log(f"Deepgram error {response.status_code}")
                return None
        
        except Exception as e:
            self.log(f"Deepgram error: {e}")
            return None
    
    def analyze_with_llm(self, text):
        if not self.selected_llm_model:
            self.log("No LLM model selected.")
            return None
        try:
            completion = client_groq.chat.completions.create(
                model=self.selected_llm_model,
                messages=[
                    {"role": "system", "content": "Answer briefly in Russian (2-3 sentences)."},
                    {"role": "user", "content": text}
                ],
                temperature=0.5,
                max_tokens=300
            )
            return completion.choices[0].message.content
        except Exception as e:
            self.log(f"LLM error: {e}")
            return None
    
    def on_closing(self):
        self.is_listening = False
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAssistantApp(root)
    root.mainloop()
