import os
import queue
import json
import threading
import pyaudio
import numpy as np
import time

try:
    from vosk import Model, KaldiRecognizer, SetLogLevel
    HAS_VOSK = True
except ImportError:
    HAS_VOSK = False

class VoiceCommandSystem:
    def __init__(self, model_path="model", target_words=None, wake_word=None, device_index=None):
        self.running = False
        self.command_queue = queue.Queue()
        self.thread = None
        self.stream = None
        self.p = None
        self.rec = None
        self.enabled = False
        self.wake_word = wake_word.lower() if wake_word else None
        self.waiting_for_command = False
        self.last_wake_time = 0
        self.timeout = 2.0 # Maximum seconds to wait for a command after the wake word
        self.sample_rate = 44100 # Standard sample rate for audio output
        
        
        if not HAS_VOSK:
            print("ERROR: 'vosk' library not found. Please run 'pip install vosk'")
            return

        # Check if the model exists
        if not os.path.exists(model_path):
            print(f"\n[VOICE] WARNING: Vosk model not found in folder '{model_path}'.")
            print("[VOICE] Run 'bash scripts/download_vosk_model.sh' or download 'vosk-model-small-en-us-0.15' from https://alphacephei.com/vosk/models")
            print("[VOICE] and extract it into a folder named 'model' in the project root.\n")
            self.enabled = False
            return

        try:
            # Silence Vosk logs to keep the console clean
            SetLogLevel(-1)
            
            print(f"[VOICE] Loading Vosk model from '{model_path}'...")
            self.model = Model(model_path)
            
            if target_words:
                # Build JSON grammar to restrict recognition, adding [unk] for unknown sounds
                words_list = [w.lower() for w in target_words]
                if self.wake_word:
                    words_list.append(self.wake_word)
                grammar = json.dumps(list(set(words_list + ["[unk]"])))
                self.rec = KaldiRecognizer(self.model, 16000, grammar)
            else:
                self.rec = KaldiRecognizer(self.model, 16000)
            # PyAudio setup for microphone input
            self.p = pyaudio.PyAudio()
            
            # --- DEVICE SELECTION LOGIC ---
            # Check if the preferred device_index is valid and has input channels
            actual_device_index = None
            if device_index is not None:
                try:
                    device_info = self.p.get_device_info_by_index(device_index)
                    if device_info.get('maxInputChannels', 0) > 0:
                        actual_device_index = device_index
                        print(f"[VOICE] Using preferred input device: {device_info.get('name')}")
                except Exception:
                    print(f"[VOICE] Preferred device index {device_index} not found. Falling back to system default.")

            # Open input stream (input device index can be specified if needed)
            self.stream = self.p.open(format=pyaudio.paInt16, 
                                      channels=1, 
                                      rate=16000, 
                                      input=True,
                                      input_device_index=actual_device_index,
                                      frames_per_buffer=800)
            self.stream.start_stream()
            
            self.running = True
            self.enabled = True
            
            # Start the listening thread
            self.thread = threading.Thread(target=self._loop)
            self.thread.daemon = True
            self.thread.start()
            
            # Audio setup for beeps
            self.audio_p = pyaudio.PyAudio()
            self.audio_stream = self.audio_p.open(format=pyaudio.paFloat32,
                                                  channels=1,
                                                  rate=self.sample_rate,
                                                  output=True)
            
            # Pre-calculate the beep for the wake word
            self.wake_beep_freq = 800 # Hz
            self.wake_beep_duration = 0.1 # seconds
            self.wake_beep_wave = self._generate_beep_wave(self.wake_beep_freq, self.wake_beep_duration, 0.35)

            if self.wake_word:
                print(f"[VOICE] Voice System: READY (Say '{self.wake_word}' to activate)")
            else:
                print("[VOICE] Voice System: READY (Continuous listening active)")
            
        except Exception as e:
            print(f"[VOICE] Initialization error: {e}")
            self.enabled = False
            self.running = False

    def _generate_beep_wave(self, frequency, duration, volume=0.5):
        """
        Generates a sine wave for a beep sound.
        Args:
            frequency (float): Frequency of the beep in Hz.
            duration (float): Duration of the beep in seconds.
            volume (float): Volume of the beep (0.0 to 1.0).
        Returns:
            bytes: Audio data as bytes.
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        wave = volume * np.sin(2 * np.pi * frequency * t)
        return wave.astype(np.float32).tobytes()

    def _loop(self):
        while self.running:
            try:
                # Read a smaller chunk for lower latency
                data = self.stream.read(800, exception_on_overflow=False)

                # Feed audio to the recognizer
                is_final = self.rec.AcceptWaveform(data)

                # First, check for a timeout if we're waiting for a command
                if self.waiting_for_command and (time.time() - self.last_wake_time > self.timeout):
                    self.waiting_for_command = False
                    print(" >> [VOICE] Timeout waiting for command. Say wake word to reactivate.")
                    self.rec.Reset()

                if is_final:
                    result = json.loads(self.rec.Result())
                    text = result.get("text", "")
                    
                    # If we were waiting for a command, this final result IS the command
                    if text and self.waiting_for_command:
                        print(f" >> [VOICE] Command received: '{text}'")
                        self.command_queue.put(text)
                        self.waiting_for_command = False # Reset state after getting command
                    
                    # Handle continuous listening mode (if no wake word is set)
                    elif text and not self.wake_word:
                        self.command_queue.put(text)
                else:
                    # --- KEY CHANGE FOR RESPONSIVENESS ---
                    # On partial results, check if the wake word has been said.
                    # This allows us to react instantly without waiting for the user to stop talking.
                    if self.wake_word and not self.waiting_for_command:
                        partial_result = json.loads(self.rec.PartialResult())
                        if self.wake_word in partial_result.get("partial", ""):
                            print(f" >> [VOICE] Wake Word '{self.wake_word}' detected! Awaiting command...")
                            if self.audio_stream:
                                self.audio_stream.write(self.wake_beep_wave)
                            self.waiting_for_command = True
                            self.last_wake_time = time.time()
                            
                            # CRITICAL: Reset the recognizer to clear "jarvis" from the buffer.
                            # This ensures the next final result will only be the command itself.
                            self.rec.Reset()
            
            except (IOError, Exception):
                # This can happen if there's an issue with the audio device.
                # It's better to let it pass silently in the loop to avoid crashing.
                pass

    def get_latest_command(self):
        """Returns the latest voice command (or None) by clearing the queue."""
        last_cmd = None
        while not self.command_queue.empty():
            last_cmd = self.command_queue.get()
        return last_cmd

    def close(self):
        # Clean up resources
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.audio_p:
            self.audio_p.terminate()
