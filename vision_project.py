import cv2
import numpy as np
import time
import importlib
import threading
import torch

from iphone_lidar import IPhoneLiDARStream, MockLiDARStream
from voice_control import VoiceCommandSystem

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    print("ERROR: 'ultralytics' library not found. Please run 'pip install ultralytics' to install it.")
    exit()

# --- CLASSE AUDIO FEEDBACK (Theremin Style) ---
class AudioFeedback:
    def __init__(self):
        self.active = False # Default OFF
        self.distance = 5.0
        self.running = True
        self.thread = None
        self.p = None
        
        try:
            import pyaudio
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(format=pyaudio.paFloat32,
                                      channels=1,
                                      rate=44100,
                                      output=True)
            self.thread = threading.Thread(target=self._loop)
            self.thread.daemon = True # This makes sure the audio thread cleans up nicely when the main program exits.
            self.thread.start()
            print(" >> Audio Feedback System: READY (Press 'O' to activate)")
            
            # Playing a little test sound at startup to confirm everything's working.
            self.play_startup_sound()
            
            # --- OPTIMIZATION: Pre-calculating the target change beep to keep the main loop super smooth ---
            sample_rate = 44100
            t = np.linspace(0, 0.1, int(sample_rate * 0.1), False) # 100ms duration
            self.target_change_wave = (0.6 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32).tobytes()
            
        except ImportError:
            print("\n !!! CRITICAL AUDIO ERROR !!!")
            print(" The 'pyaudio' library is not installed correctly.")
            print(" You might need to run: brew install portaudio && pip install pyaudio in your terminal to fix this.\n") # Providing clear instructions for fixing the issue.

    def play_startup_sound(self):
        if not self.stream: return
        # A quick double beep for testing (440Hz -> 880Hz)
        sample_rate = 44100
        t = np.linspace(0, 0.15, int(sample_rate * 0.15), False)
        
        # First tone
        tone1 = 0.5 * np.sin(2 * np.pi * 440 * t)
        self.stream.write(tone1.astype(np.float32).tobytes())
        time.sleep(0.05)
        
        # Second tone
        tone2 = 0.5 * np.sin(2 * np.pi * 880 * t)
        self.stream.write(tone2.astype(np.float32).tobytes())
        print(" >> Startup Sound Played.")
    
    def _loop(self):
        sample_rate = 44100
        beep_freq = 600.0 # Hz (Slightly lowered the frequency to make it a bit more audible)
        beep_duration = 0.08 # 80ms beep
        
        # Pre-calculating the beep's waveform (a simple sine wave).
        t = np.linspace(0, beep_duration, int(sample_rate * beep_duration), False)
        beep_wave = 0.8 * np.sin(2 * np.pi * beep_freq * t) # Bumped up the volume a bit (from 0.2 to 0.8).
        
        # Adding a small fade in/out to prevent those annoying 'clicks' at the start/end of the sound.
        fade_len = 200
        beep_wave[:fade_len] *= np.linspace(0, 1, fade_len)
        beep_wave[-fade_len:] *= np.linspace(1, 0, fade_len)
        beep_bytes = beep_wave.astype(np.float32).tobytes()

        while self.running: # Keep beeping as long as we're running
            if self.active and self.distance <= 2.0:
                # Audio Range: From 0.2m (fast beeps) to 2.0m (slow beeps).
                d = max(0.2, min(2.0, self.distance))
                
                # Mapping the repetition interval:
                # 0.2m means a 0.05s pause (almost continuous), 2.0m means a 1.00s pause (slow beep).
                norm = (d - 0.2) / 1.8 # Normalize distance to a 0-1 range
                wait_time = 0.05 + (norm * 0.95) # Map to a wait time (faster when closer)
                
                if self.stream:
                    self.stream.write(beep_bytes)
                
                time.sleep(float(wait_time))
            else:
                time.sleep(0.1)

    def update_distance(self, dist):
        self.distance = dist

    def play_target_change_sound(self):
        if not self.stream or not hasattr(self, 'target_change_wave'): return
        # --- OPTIMIZATION: Playing this sound in a separate thread so it doesn't block the main loop ---
        def play():
            self.stream.write(self.target_change_wave)
            
        threading.Thread(target=play, daemon=True).start()

    def play_voice_target_change_sound(self):
        if not self.stream: return
        sample_rate = 44100
        t = np.linspace(0, 0.1, int(sample_rate * 0.1), False) # 100ms duration
        # A different sound, maybe a lower frequency
        wave = (0.6 * np.sin(2 * np.pi * 500 * t)).astype(np.float32).tobytes()
        
        def play():
            self.stream.write(wave)
            
        threading.Thread(target=play, daemon=True).start()

    def close(self):
        self.running = False
        if self.thread: self.thread.join(timeout=0.5)
        if self.stream: 
            self.stream.stop_stream()
            self.stream.close()
        if self.p: self.p.terminate()

# --- DETECTION THREAD (YOLO) ---
# This way, the main thread won't get stuck waiting for the AI inference to complete.
class DetectionThread(threading.Thread):
    def __init__(self, model_path, device, conf_threshold):
        super().__init__()
        self.daemon = True
        self.model_path = model_path
        self.model = None # Loaded inside the thread
        self.device = device
        self.conf_threshold = conf_threshold
        
        self.running = False # flag to control when this thread should be active.
        self.lock = threading.Lock()
        
        # This is where the main thread will drop off frames for us to process.
        self.frame_to_process = None
        self.new_frame_event = threading.Event() # We use this to signal when a new frame is ready for processing.
        
        # This is where the main thread will pick up our latest detection results.
        self.latest_results = None

    def run(self):
        self.running = True
        # --- Model loading INSIDE the thread ---
        # This is crucial for stability with MPS/CUDA: it prevents the model object
        # from being shared across threads, which can cause race conditions and crashes.
        self.model = YOLO(self.model_path)
        print("[DetectionThread] Started. AI inference is now running in the background.")
        
        while self.running:
            self.new_frame_event.wait() # Just waiting for a new frame to process
            if not self.running:
                break

            # Grabbing a copy of the frame to avoid any messy trace conditions while processing.
            with self.lock:
                frame = self.frame_to_process
                self.new_frame_event.clear() # Reset the event

            if frame is not None and self.model is not None:
                # --- THE HEAVY LIFTING ---
                # This is where the actual YOLO inference happens, all in the background.
                # We're using imgsz=480 for a good balance of speed and accuracy for the detection model.
                yolo_results = self.model(frame, imgsz=480, verbose=False, conf=self.conf_threshold, device=self.device)
                
                # Storing the results in a thread-safe way for the main thread to pick up later.
                with self.lock:
                    self.latest_results = yolo_results

    def update_frame(self, frame):
        # If the detection thread is ready, we'll give it a new frame to work on
        if not self.new_frame_event.is_set():
            with self.lock:
                self.frame_to_process = frame
            self.new_frame_event.set()

    def get_latest_results(self):
        with self.lock:
            return self.latest_results

    def stop(self):
        self.running = False
        self.new_frame_event.set() # Unblocking the waiting thread so it can exit its loop.

def main():
    print("=== VISION PROJECT ===")
    
    # 0. DEVICE SELECTION (GPU/CPU)
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
        print(" >> GPU ACCELERATION: ACTIVE (Apple Metal/MPS)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print(" >> GPU ACCELERATION: ACTIVE (NVIDIA CUDA)") # This should be fast.
    else:
        print(" >> GPU ACCELERATION: NOT AVAILABLE (Using CPU)") # Might be a bit slower.

    # 1. AI MODEL LOADING (YOLOv11)
    print("Loading YOLO11 (Deep Learning) model...")
    # It'll automatically download 'yolo11n-seg.pt' (the Nano Segmentation model) the very first time you run it.
    model_path = "yolo11n-seg.pt"
    model = YOLO(model_path) # using the Nano model here for a good balance of speed and accuracy, perfect for real-time.

    # 2. TARGET CONFIGURATION (Selection Menu)
    SELECTABLE_TARGETS = [
        (41, "Cup"),        # Default
        (39, "Bottle"),
        (67, "iPhone"),
        (63, "Laptop"),
        (56, "Chair"),
        (65, "Remote"),
        (73, "Book"),
        (0, "Person"),
        (42, "Fork"),
        (44, "Spoon"),
    ]
    current_target_idx = 0 # 'Cup' is the default target.

    # 3. WEBCAM SETUP
    use_lidar = False
    lidar_stream = IPhoneLiDARStream(port=7100)
    
    # Attempting connection (remember to change the host to your iPhone's IP if you're on Wi-Fi, otherwise localhost for USB).
    if lidar_stream.connect(host='127.0.0.1'): 
        use_lidar = True
        cap = None # using the iPhone, no need for standard webcam.
    else:
        print("Falling back to the Standard Webcam (with simulated depth).")
        cap = cv2.VideoCapture(0)
        lidar_stream = MockLiDARStream(cap) # Using a mock stream to simulate depth for testing purposes.


    # Detection Parameters - Setting up how confident we need to be about detections.
    # With YOLO, we don't really need a separate scale_factor for detection; we'll just feed it the cropped image directly.
    conf_threshold = 0.6 
    
    detection_interval = 1
    frame_counter = 0
    
    active_class_id = None
    
    # Phosphene processing configuration
    # GLOBAL CONSTANT: This defines the number of discrete grayscale levels used throughout the entire system.
    PHOSPHENE_INTENSITY_LEVELS = 16 
    
    current_phosphene_version = "v4" # starting with version 4, best/more realistic one
    # Initial loading of phosphene renderer module.
    phosphene_module = importlib.import_module(f"phosphene_processing_{current_phosphene_version}")
    PhospheneRenderer = getattr(phosphene_module, 'PhospheneRenderer')

    # Phosphene Parameters - Setting up visual simulation.
    # Initializing the realistic renderer (we only need to do this once!).
    # A dropout_rate of 0.1 means 10% of electrodes are 'dead', which adds realism.
    spv_renderer = PhospheneRenderer(electrode_grid=(32, 32), 
                                     lattice_structure=(1, 0.5, 1),
                                     dropout_rate=0.1, 
                                     intensity_levels=PHOSPHENE_INTENSITY_LEVELS)
    # Visualization Parameters - All the toggles and settings for what can be seen on screen, and how.
    pulse_frequency = 2.0 #Hz
    enable_lidar_validation = True # The LiDAR filter is always on by default.
    show_all_classes = False       # Toggle this to show all 80 COCO classes (Key 'A').
    detection_enabled = True       # The main switch for object detection (Key 'D').
    enable_lidar_dimming = True    # Background dimming is always active.

    pure_depth_mode = False        # Toggle for "Depth Only" mode (Key 'P').
    show_only_spv = False          # Toggle to display only SPV (Simulated Phosphene Vision) (Key 'V').
    pure_rgb_mode = False          # Toggle for "Pure RGB" mode (Key 'G').
    enable_audio = False           # Toggle for Audio Feedback (Key 'O').
    substitute_vision = False        # Toggle to substitute original vision with SPV input (Key 'S').

    # Variables for temporal smoothing (this really helps reduce flickering in the output).
    prev_spv_input = None
    smoothing_factor = 0.1 # 0.0 means no smoothing at all, 0.9 gives a long trail (a ghosting effect).

    # Variables for Semantic Auto-Range - This helps ensure focus adapts intelligently to the scene.
    # Initial focus distance.
    smooth_focus_dist = 2.0  # Initial focus distance.
    range_threshold_factor = 0.25 # Increased for greater stability (requires a 25% change in distance to update focus).
    focus_roi_percentage = 0.25   # The size of the central focus region as a percentage of screen dimension.
    print("Starting Loop... Press 'q' to exit.")

    # Detection cache for skipped frames
    cached_box = None
    cached_polygon = None
    cached_class_id = None
    cached_detections = []
    cached_dist = None

    # Audio Initialization - Getting sound system all set up.
    audio_sys = AudioFeedback()

    # Voice Control Initialization - Setting up voice assistant, "Jarvis"!
    # --- VOCABULARY CONFIGURATION ---
    # pulling out all the target names from list (like "Cup", "Bottle").
    vocab = [t[1] for t in SELECTABLE_TARGETS] 
    # System commands available via voice
    vocab.extend(["stop", "start", "audio"])
    # device_index=None uses the system default microphone; set an index to pick a specific input device
    voice_sys = VoiceCommandSystem(model_path="model", target_words=vocab, wake_word="jarvis", device_index=None)

    # --- YOLO THREADING ---
    detection_thread = DetectionThread(model_path, device, conf_threshold)
    detection_thread.start()

    # Variables for the synchronized visual pulse
    pulse_phase = 0.0
    last_pulse_time = time.time()

    while True:
        frame_counter += 1
        
        # Calculating delta time for smooth animations and consistent pulsing effects.
        current_time = time.time()
        dt = current_time - last_pulse_time
        last_pulse_time = current_time

        # --- VOICE CONTROL --- - Listening intently for your commands
        if voice_sys.enabled:
            voice_cmd = voice_sys.get_latest_command()
            if voice_cmd:
                print(f" >> Voice Command Detected: '{voice_cmd}'") # Just confirming what heard.
                # Checking if any of the spoken words match one of our predefined targets.
                for idx, (_, name) in enumerate(SELECTABLE_TARGETS):
                    if name.lower() in voice_cmd.lower():
                        current_target_idx = idx
                        print(f" >> TARGET CHANGED: {name}")
                        audio_sys.play_voice_target_change_sound()
                        break
                
                # Handling some extra system commands.
                if "stop" in voice_cmd:
                    detection_enabled = False
                    print(" >> Detection: OFF (Voice)")
                elif "start" in voice_cmd:
                    detection_enabled = True
                    print(" >> Detection: ON (Voice)")
                elif "audio" in voice_cmd:
                    enable_audio = not enable_audio
                    audio_sys.active = enable_audio
                    print(f" >> Audio: {'ON' if enable_audio else 'OFF'} (Voice)")

                    
        # Hybrid Acquisition (RGB + Depth) - Getting  visual and depth data from the stream.
        if use_lidar:
            ret, raw_frame, depth_map, conf_map = lidar_stream.read()
            if ret and raw_frame is None:
                continue # waiting for the buffer to fill up during initial milliseconds.
        else:
            ret, raw_frame, depth_map, conf_map = lidar_stream.read()
            
        if not ret or raw_frame is None: break

        # --- OPTIMIZATION: DOWN-SCALING ---
        # This step significantly reduces the load on the CPU/GPU while still maintaining
        # good enough quality for both YOLO detection and the Phosphene rendering.
        # reduced the target width from 960.
        # This maintains sufficient quality for the overall pipeline.
        target_width = 640
        h_raw, w_raw = raw_frame.shape[:2]
        if w_raw > target_width:
            scale = target_width / w_raw
            new_h = int(h_raw * scale)
            raw_frame = cv2.resize(raw_frame, (target_width, new_h), interpolation=cv2.INTER_LINEAR)

        # --- SENSOR DETECTION (FaceID vs LiDAR) ---
        # FaceID (TrueDepth) is device type 0, which is good for macro/close-up distances (< 1m).
        # LiDAR (World) is device type 1, generally better for environmental sensing (> 1m).
        is_face_id = False
        if hasattr(lidar_stream, 'get_device_type') and lidar_stream.get_device_type() == 0:
            is_face_id = True
            
        # --- DEPTH ALIGNMENT --- - Making sure depth data lines up with the video.
        h, w = raw_frame.shape[:2]
        if depth_map is not None and (depth_map.shape[0] != h or depth_map.shape[1] != w):
            # Resizing the depth map to precisely match the RGB video dimensions for proper overlay.
            depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
        
        if conf_map is not None and (conf_map.shape[0] != h or conf_map.shape[1] != w):
            conf_map = cv2.resize(conf_map, (w, h), interpolation=cv2.INTER_NEAREST)

        # --- A. SQUARE CROP (1:1) ---
        min_dim = min(h, w)
        r_offset = (h - min_dim) // 2
        c_offset = (w - min_dim) // 2
        
        square_frame = raw_frame[r_offset:r_offset+min_dim, c_offset:c_offset+min_dim]
        frame_gray = cv2.cvtColor(square_frame, cv2.COLOR_BGR2GRAY)
        
        # Also cropping the Depth Map if it's available, to match the square frame.
        square_depth = None
        square_conf = None
        if depth_map is not None:
            square_depth = depth_map[r_offset:r_offset+min_dim, c_offset:c_offset+min_dim]
        if conf_map is not None:
            square_conf = conf_map[r_offset:r_offset+min_dim, c_offset:c_offset+min_dim]

        # --- 90° ROTATION (LANDSCAPE MODE) --- - Orienting tour view correctly.
        # Rotating the image to account for horizontal phone use.
        square_frame = cv2.rotate(square_frame, cv2.ROTATE_90_CLOCKWISE)
        frame_gray = cv2.rotate(frame_gray, cv2.ROTATE_90_CLOCKWISE)
        if square_depth is not None:
            square_depth = cv2.rotate(square_depth, cv2.ROTATE_90_CLOCKWISE)
        if square_conf is not None:
            square_conf = cv2.rotate(square_conf, cv2.ROTATE_90_CLOCKWISE)


        # --- B. OBJECT DETECTION (YOLOv11 in Background) ---
        # 1. Hand the current frame to the detection thread (non-blocking)
        if detection_enabled and (frame_counter % detection_interval == 0):
            detection_thread.update_frame(square_frame)

        # 2. Fetch the latest results from the thread (non-blocking)
        yolo_results = detection_thread.get_latest_results()
        # 3. Process the results (this part is fast and can remain in the main loop)
        detected_box = None # Initializing detected_box to None for this frame.
        detected_polygon = None
        is_detected_now = False
        detected_class_id = None
        all_valid_detections = []
        target_dist_detected = None

        if detection_enabled and yolo_results is not None:
            best_conf = 0
            target_id = SELECTABLE_TARGETS[current_target_idx][0]
            
            for r in yolo_results: # Looping through all the detections found by YOLO.
                boxes = r.boxes # Bounding boxes.
                masks = r.masks # Segmentation masks.
                
                for i, box in enumerate(boxes):
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    # either show everything detected, or just the selected target.
                    if show_all_classes or cls == target_id: 
                        x1, y1, x2, y2 = box.xyxy[0].tolist() # Extracting the bounding box coordinates (top-left, bottom-right).
                        w_box = x2 - x1
                        h_box = y2 - y1
                        
                        is_valid_3d = True # assume it's valid until proven otherwise.
                        dist = None # Initializing distance
                        
                        if is_valid_3d:
                            all_valid_detections.append((x1, y1, w_box, h_box, cls))
                            if cls == target_id and conf > best_conf:
                                best_conf = conf
                                
                                # --- LiDAR Distance Calculation (Moved here for optimization) ---
                                if enable_lidar_validation and square_depth is not None: # If LiDAR is active and we have depth data
                                    if masks is not None: # If we have a segmentation mask, that's the most accurate way to get depth
                                        poly = masks.xy[i].astype(np.int32)
                                        mask_obj = np.zeros(square_depth.shape, dtype=np.uint8)
                                        cv2.fillPoly(mask_obj, [poly], 1)
                                        
                                        valid_depths = square_depth[(mask_obj > 0) & (square_depth > 0)] # Only considering valid depth pixels that are part of the object.
                                        if valid_depths.size > 0:
                                            dist = np.median(valid_depths)

                                    if dist is None: # Fallback to bounding box center
                                        cx, cy = int(x1 + w_box/2), int(y1 + h_box/2)
                                        sample_w = max(1, int(w_box * 0.2))
                                        sample_h = max(1, int(h_box * 0.2))
                                        
                                        y_s = max(0, cy - sample_h//2)
                                        y_e = min(square_depth.shape[0], cy + sample_h//2)
                                        x_s = max(0, cx - sample_w//2)
                                        x_e = min(square_depth.shape[1], cx + sample_w//2)
                                        
                                        patch = square_depth[y_s:y_e, x_s:x_e]
                                        valid_pixels = patch[patch > 0]
                                        if valid_pixels.size > 0:
                                            dist = np.median(valid_pixels)

                                detected_box = [x1, y1, w_box, h_box]
                                is_detected_now = True
                                detected_class_id = cls
                                if dist is not None:
                                    target_dist_detected = dist # Storing that precise distance
                                
                                if masks is not None: 
                                    detected_polygon = masks.xy[i]
            
            # Updating cache with the latest valid results for the target, if any were found.
            if is_detected_now:
                cached_box = detected_box
                cached_polygon = detected_polygon
                cached_class_id = detected_class_id
                cached_dist = target_dist_detected
            
            # Updating the cache for all detections (this is particularly useful for the "show all classes" mode).
            cached_detections = all_valid_detections

        # If detection is active but the target was not found in this cycle,
        # detected_box, detected_polygon, etc. stay None,
        # so the target highlight disappears immediately.

        # --- C. RAW DETECTION (NO FILTER) ---
        final_box = detected_box if is_detected_now else None # This is the bounding box to actually use for rendering the target.
        final_polygon = detected_polygon if is_detected_now else None # And the polygon, if it's available for the target.
        
        if is_detected_now: 
            active_class_id = detected_class_id 
        else:
            active_class_id = None # Resetting if the target is no longer detected.

        # --- D. PROSTHETIC VISION SIMULATION --- - Crafting the phosphene view from processed data.
        # STRATEGY: Brightness is directly related to Depth, simulating how proximity might be perceived.
        # The entire scene is rendered based on the depth map, emphasizing closer objects.
        # The closer an object is, the brighter it appears in the phosphene view, providing a sense of depth.

        # base input for the phosphene renderer (a fallback if LiDAR isn't active, using grayscale RGB).
        frame_gray_norm = frame_gray.astype(float) / 255.0
        scene_map = frame_gray_norm * 0.2 # Default dimming, making the background a little less dark to highlight foreground.
        depth_perceptual = None # placeholder but needs to be defined.
        current_audio_dist = 5.0 # Defaulting to silence for audio feedback if no target is found.

        if pure_rgb_mode:
            # Pure RGB Mode: Direct brightness from the grayscale RGB image (completely ignoring LiDAR depth data).
            scene_map = frame_gray_norm

        elif enable_lidar_dimming and square_depth is not None:
            # --- 1. PRE-PROCESSING (Pulizia Dati) ---
            # Clip at 10m so the distant background is not lost
            d_proc = square_depth.copy()
            d_proc = np.nan_to_num(d_proc, nan=10.0, posinf=10.0, neginf=0.0)
            d_proc = np.clip(d_proc, 0, 10.0)

            # --- 2. DYNAMIC FOCUS CALCULATION (CENTRAL ROI) ---
            fh, fw = d_proc.shape # Getting the frame height and width for ROI calculation.
            roi_size = int(min_dim * focus_roi_percentage)
            roi = d_proc[fh//2 - roi_size//2 : fh//2 + roi_size//2, 
                         fw//2 - roi_size//2 : fw//2 + roi_size//2]
            
            # use the median of valid pixels within this central focus area to determine the current focus distance.
            valid_roi = roi[roi > 0.1]
            if valid_roi.size > 0:
                current_focus = np.median(valid_roi) # This gives the median distance in ROI.
                
                # Proportional threshold update: This helps prevent tiny, jittery changes in focus.
                # If the distance changes by more than 20% update smooth focus.
                if abs(current_focus - smooth_focus_dist) > (smooth_focus_dist * range_threshold_factor): 
                    # Using an Exponential Moving Average (EMA) for a smooth transition; giving more weight to past values for stability.
                    smooth_focus_dist = 0.9 * smooth_focus_dist + 0.1 * current_focus

            # --- 3. NEW DIMMING LOGIC BASED ON FOCUS --- - Making the scene adapt dynamically to the focus distance.
            # The "focus" essentially determines how steep the brightness gradient is across the scene.
            # To keep brightness consistent at the focus point, we're using a fixed multiplier.
            # A distant focus results in a gentle gradient, while a close focus makes it steeper, emphasizing nearby objects.
            focus_falloff_range = max(4.0, smooth_focus_dist * 2.0) # We want at least 4.0m to ensure the background remains somewhat visible.
            min_brightness = 0.10 # A minimum of 10% brightness for the background.

            # Slightly dim near objects when the focus is far away
            # Max brightness at 0m decreases slightly as the focus distance increases
            # clip(a, a_min, a_max)
            max_brightness_at_zero_dist = np.clip(1.0 - (smooth_focus_dist / 10.0) * 0.08, 0.7, 1.0)

            # Compute brightness from distance and dynamic range
            # Brightness ranges from max_brightness_at_zero_dist (for objects up close) to min_brightness (for objects far away), creating a depth effect.
            proximity = (1.0 - np.clip(d_proc / focus_falloff_range, 0, 1)) * \
                        (max_brightness_at_zero_dist - min_brightness) + min_brightness
            if pure_depth_mode:
                scene_map = proximity
            else:
                
                # --- A. STRUCTURAL EDGES (Geometry) --- - Finding the outlines of objects using depth data.
                if is_face_id:
                    # --- FACE ID FIX: REMOVE "BLACK HALO" ---
                    
                    # 1. CRITICAL FIX: DEPTH EROSION
                    # The sensor fails at object borders and reads "infinity" (black).
                    # Eroding the distance map expands the dark regions (near, low values)
                    # over the bright regions (far, high values).
                    # This slightly "inflates" the face and covers the black error halo.
                    kernel_fix = np.ones((5,5), np.uint8)
                    d_proc_fixed = cv2.erode(d_proc, kernel_fix, iterations=1)
                    
                    # 2. Normalization
                    d_norm_face = (d_proc_fixed / 1.0)
                    d_uint8 = (np.clip(d_norm_face, 0, 1) * 255).astype(np.uint8)

                    # 3. BILATERAL FILTER (smooth skin)
                    # Keeps edges sharp while removing surface noise
                    d_smooth = cv2.bilateralFilter(d_uint8, 7, 75, 75)

                    # 4. Balanced Canny (high sensitivity)
                    # Now that the black hole is gone, we can be precise.
                    depth_edges = cv2.Canny(d_smooth, 30, 80)

                    # 5. Edge thickening
                    depth_edges = cv2.dilate(depth_edges, np.ones((3,3), np.uint8), iterations=1)
                    depth_edges_norm = depth_edges.astype(float) / 255.0

                    # 6. "HIGH CONTRAST" VISUALIZATION
                    # Recompute proximity from the corrected map (no black holes)
                    proximity = 1.0 - (d_proc_fixed / 1.0)
                    proximity = np.clip(proximity, 0, 1)
                    
                    # Mid-gray body (0.5) + saturated white edges (1.0)
                    # 'maximum' ensures the white edge always wins over the gray
                    scene_map = np.maximum(proximity * 0.5, depth_edges_norm)
                    
                    # Hack: prevent the LiDAR logic below from overwriting everything
                    depth_perceptual = proximity 
                    
                else:
                    # --- ASYMMETRIC DEPTH OF FIELD (Physical Noise + Focus Logic) ---
                    
                    # 1. Balanced denoising to preserve structural integrity.
                    d_clean = cv2.medianBlur(d_proc.astype(np.float32), 3)
                    
                    # 2. Morphological Gradient for robust edge detection.
                    kernel = np.ones((3,3), np.uint8)
                    grad_mag = cv2.morphologyEx(d_clean, cv2.MORPH_GRADIENT, kernel)
                    
                    # 3. Physical Noise Floor with Focus-Dependent Gain.
                    # Combines distance-based noise modeling with focal plane stabilization.
                    thresh_phys = (0.01 + 0.02 * d_clean)# * (1 + smooth_focus_dist * 0.1)

                    # 4. Asymmetric DoF Penalty (Suppresses background beyond focal plane).
                    bg_penalty = np.maximum(0, d_clean - smooth_focus_dist)
                    thresh_dof = bg_penalty * 0.05

                    # 5. Combined Spatially Adaptive Thresholding.
                    local_thresh = thresh_phys + thresh_dof

                    # 6. Spatially Adaptive Gain Modulation (Divisor).
                    # Combines physical distance attenuation with focal plane priority.
                    divisor_phys = 0.01 + (d_clean * 0.005)
                    divisor_dof = bg_penalty * 0.05
                    local_divisor = np.clip(divisor_phys + divisor_dof, 0.05, 0.50)

                    # 7. Edge Calculation (Matrix Operation).
                    raw_edges = np.clip((grad_mag - local_thresh) / local_divisor, 0, 1)
                    
                    depth_edges_norm = cv2.morphologyEx(raw_edges, cv2.MORPH_OPEN, kernel)

                # --- B. FUSIONE IBRIDA (RGB Texture + LiDAR Structure) ---
                # Apply proximity to the RGB image
                body_fill = frame_gray_norm * proximity
                
                
                # We're only applying dimming to the body_fill, not the LiDAR edges.
                # Reducing the global intensity (0.7) to prevent too many phosphenes from activating all at once, which can be visually overwhelming.
                background_dimming_factor = 0.7
                body_fill_dimmed = body_fill * background_dimming_factor

                # 2. Structure: LiDAR edges take priority (Pure White) - they're crucial for spatial awareness and object definition.
                # overlaying the precise LiDAR edges onto the RGB texture to combine both types of information effectively.
                scene_map = np.maximum(body_fill_dimmed, depth_edges_norm)
        
        # AUDIO LOGIC: only activate audio feedback if a target is actually detected and its distance is known.
        if target_dist_detected is not None:
            current_audio_dist = target_dist_detected

        # Updating the audio system with the current distance to the target for feedback.
        audio_sys.update_distance(current_audio_dist)

        # --- 3. SEMANTIC TARGET (YOLO Override) ---
        final_spv_input = scene_map.copy()

        if final_box is not None:
            x, y, w_b, h_b = map(int, final_box)
            
            # Clipping coordinates to ensure they stay within the square_frame boundaries.
            y1, y2 = max(0, y), min(square_frame.shape[0], y + h_b)
            x1, x2 = max(0, x), min(square_frame.shape[1], x + w_b)
            
            if (x2 - x1) > 0 and (y2 - y1) > 0:
                # Fixed frequency (stable, reduces visual load)
                pulse_freq = pulse_frequency # 2.0 Hz
                
                # Phase accumulation (frequency integration) for smooth transitions
                pulse_phase += 2 * np.pi * pulse_freq * dt
                pulse = (np.sin(pulse_phase) + 1) / 2  # Oscilla 0.0 -> 1.0
                target_intensity = 0.5 + (0.8 * pulse) # Oscilla 0.8 -> 1.6
                # --- SEGMENTATION-BASED ILLUMINATION --- - Lighting uptour target's precise shape.
                if final_polygon is not None: # If we have a precise segmentation mask, let's use it!
                    # 1. Binary Mask (This gives us the actual shape of the object).
                    mask_h, mask_w = final_spv_input.shape[:2]
                    binary_mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
                    cv2.fillPoly(binary_mask, [final_polygon.astype(np.int32)], 1)
                    
                    # 2. Semantic Contour Extraction (Smart Edge) - Finding the outline of the object.
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Drawing the contour on a separate layer for clarity.
                    semantic_edge_layer = np.zeros_like(final_spv_input)
                    
                    # --- 1. Outer Dark Halo (Contrast Booster) ---
                    dilated_mask = cv2.dilate(binary_mask, np.ones((5,5), np.uint8), iterations=8)
                    outer_ring = cv2.subtract(dilated_mask, binary_mask)
                    final_spv_input = np.where(outer_ring > 0, final_spv_input * 0.2, final_spv_input)

                    # --- 2. Body vs Edge Rendering ---
                    cv2.drawContours(semantic_edge_layer, contours, -1, 1.0, thickness=5) # Bordo netto
                    
                    # 3. "Internal Glow" Effect (Synthetic Salience) - Making the object feel more solid.
                    # Body at 30% (0.3), Edges at 100% (1.0) - This combination generally looks better for phosphenes.
                    salient_object = np.maximum(binary_mask.astype(float) * 0.3, semantic_edge_layer)

                    # Apply the pulse and overwrite the object in the scene
                    final_spv_input = np.where(binary_mask > 0, salient_object * target_intensity, final_spv_input)
                    
                else:
                    # Fallback to Bounding Box (if segmentation isn't available or fails for some reason).
                    if depth_perceptual is not None:
                        object_tex = depth_perceptual[y1:y2, x1:x2]
                        if np.max(object_tex) > np.min(object_tex):
                            object_tex = cv2.normalize(object_tex, None, 0.5, 1.0, cv2.NORM_MINMAX)
                    else: # If no depth info is available, just use the grayscale texture from the RGB frame.
                        object_tex = frame_gray_norm[y1:y2, x1:x2] 
                    
                    final_spv_input[y1:y2, x1:x2] = object_tex * target_intensity # Applying the pulsating texture to the object's region.
        
        # Final clipping to ensure all values are within the valid 0-1 range.
        spv_input = np.clip(final_spv_input, 0, 1).astype(np.float32)

        # --- TEMPORAL SMOOTHING (Reduces flickering) 
        if prev_spv_input is not None and prev_spv_input.shape == spv_input.shape:
            spv_input = cv2.addWeighted(spv_input, 1.0 - smoothing_factor, prev_spv_input, smoothing_factor, 0)
        prev_spv_input = spv_input
        
        # --- E. PHOSPHENE RENDERING --- - This is where the actual phosphene generation happens.
        # We pass the processed scene directly to the renderer.
        # The renderer handles the final downsampling to the 32x32 electrode grid internally.
        phosphene_frame = spv_renderer.render(spv_input)
        
        # --- F. VISUALIZATION ---

        # Normalizing the phosphene frame for display.
        phosphene_display = cv2.resize(phosphene_frame, (min_dim, min_dim))
        phosphene_display = np.clip(phosphene_display, 0, 1)
        
        # Drawing bounding boxes on the Real World view for context and user feedback.
        display_real = square_frame.copy()
        
        if show_all_classes:
            for (bx, by, bw, bh, bcls) in all_valid_detections:
                cv2.rectangle(display_real, (int(bx), int(by)), (int(bx+bw), int(by+bh)), (255, 100, 0), 1)
                cv2.putText(display_real, model.names[bcls], (int(bx), int(by)-5), # Displaying the class name above the bounding box.
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

        if final_box is not None:
            x, y, w_b, h_b = map(int, final_box)
            cv2.rectangle(display_real, (x, y), (x+w_b, y+h_b), (0, 255, 0), 3)
            
            # Also drawing the segmentation contour if it's available, for a more precise visual outline.
            if final_polygon is not None:
                cv2.polylines(display_real, [final_polygon.astype(np.int32)], True, (0, 255, 0), 2)
            
            # Show the detected object name (e.g. "cup", "person", "cell phone")
            label = "TARGET"
            if active_class_id is not None:
                label = f"TARGET: {model.names[active_class_id]}"
            
            cv2.putText(display_real, label, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Drawing a small circle at the center of the box to visually indicate where we're sampling the distance from.
            cx = x + w_b // 2
            cy = y + h_b // 2
            cv2.circle(display_real, (cx, cy), 5, (0, 0, 255), -1) # A little red dot for clarity.
        # Drawing the FOCUS ROI for debugging purposes.
        # Draw the FOCUS region (ROI) for debugging
        f_cx, f_cy = min_dim // 2, min_dim // 2
        f_r = int(min_dim * focus_roi_percentage) // 2
        cv2.rectangle(display_real, (f_cx-f_r, f_cy-f_r), (f_cx+f_r, f_cy+f_r), (255, 255, 255), 1)

        # Horizontal stack - Arranging our two views side-by-side.
        # Converting the phosphene frame to BGR so we can concatenate it with the real view.
        phosphene_bgr = cv2.cvtColor((phosphene_display * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        # Drawing the ROI on the phosphene view too (using gray so it doesn't distract too much from the phosphenes).
        cv2.rectangle(phosphene_bgr, (f_cx-f_r, f_cy-f_r), (f_cx+f_r, f_cy+f_r), (80, 80, 80), 1)
        
        if substitute_vision:
            # We want to show the input to the phosphene generator.
            spv_input_display = (spv_input * 255).astype(np.uint8)
            display_left = cv2.cvtColor(spv_input_display, cv2.COLOR_GRAY2BGR)
        else:
            display_left = display_real

        # --- INTERFACCIA UTENTE (UI) ---
        ui_y = 30
        ui_gap = 30 # Increased gap for larger font
        ui_font = cv2.FONT_HERSHEY_DUPLEX # A slightly more 'blocky' font
        ui_scale = 0.6 # Increased font size
        ui_thick = 1
        ui_thick_on = 2 # Bolder text for 'ON' states

        # 1. FPS Display
        fps = 1.0 / dt if dt > 0 else 0
        cv2.putText(display_left, f"FPS: {int(fps)}", (10, ui_y), ui_font, 0.7, (0, 255, 0), 2)
        ui_y += ui_gap

        # 2. Detection Status (D)
        det_color = (0, 255, 0) if detection_enabled else (0, 0, 255)
        cv2.putText(display_left, f"Detection (D): {'ON' if detection_enabled else 'OFF'}", (10, ui_y), ui_font, ui_scale, det_color, ui_thick_on if detection_enabled else ui_thick)
        ui_y += ui_gap
        
        # 3. Selected Target (T)
        target_name = SELECTABLE_TARGETS[current_target_idx][1]
        cv2.putText(display_left, f"Target (T): {target_name}", (10, ui_y), ui_font, ui_scale, (255, 255, 0), ui_thick_on)
        ui_y += ui_gap

        # 4. Mode (A)
        mode_text = "ALL" if show_all_classes else "SINGLE"
        mode_color = (255, 0, 255) if show_all_classes else (200, 200, 200)
        cv2.putText(display_left, f"Mode (A): {mode_text}", (10, ui_y), ui_font, ui_scale, mode_color, ui_thick_on if show_all_classes else ui_thick)
        ui_y += ui_gap

        # 5. Phosphene Version (R)
        cv2.putText(display_left, f"Renderer (R): {current_phosphene_version}", (10, ui_y), ui_font, ui_scale, (200, 200, 200), ui_thick)
        ui_y += ui_gap

        # 6. Audio Feedback (O)
        aud_color = (0, 255, 0) if enable_audio else (0, 0, 255)
        cv2.putText(display_left, f"Audio (O): {'ON' if enable_audio else 'OFF'}", (10, ui_y), ui_font, ui_scale, aud_color, ui_thick_on if enable_audio else ui_thick)
        ui_y += ui_gap

        # --- Display Modes ---
        # 7. SPV Only Mode (V)
        spv_color = (0, 255, 0) if show_only_spv else (0, 0, 255)
        cv2.putText(display_left, f"SPV Only (V): {'ON' if show_only_spv else 'OFF'}", (10, ui_y), ui_font, ui_scale, spv_color, ui_thick_on if show_only_spv else ui_thick)
        ui_y += ui_gap

        # 8. Substitute Vision (S)
        sub_color = (0, 255, 0) if substitute_vision else (0, 0, 255)
        cv2.putText(display_left, f"Substitute (S): {'ON' if substitute_vision else 'OFF'}", (10, ui_y), ui_font, ui_scale, sub_color, ui_thick_on if substitute_vision else ui_thick)
        ui_y += ui_gap
        
        # 9. Pure RGB (G)
        prgb_color = (0, 255, 0) if pure_rgb_mode else (0, 0, 255)
        cv2.putText(display_left, f"Pure RGB (G): {'ON' if pure_rgb_mode else 'OFF'}", (10, ui_y), ui_font, ui_scale, prgb_color, ui_thick_on if pure_rgb_mode else ui_thick)
        ui_y += ui_gap

        # 10. Pure Depth (P)
        pdepth_color = (0, 255, 0) if pure_depth_mode else (0, 0, 255)
        cv2.putText(display_left, f"Pure Depth (P): {'ON' if pure_depth_mode else 'OFF'}", (10, ui_y), ui_font, ui_scale, pdepth_color, ui_thick_on if pure_depth_mode else ui_thick)
        ui_y += ui_gap
        
        # --- Bottom Info ---
        # Focus Distance Info
        cv2.putText(display_left, f"Focus: {smooth_focus_dist:.2f}m", (10, display_left.shape[0] - 10), ui_font, 0.6, (255, 255, 255), 1)

        if show_only_spv:
            cv2.imshow("BIONIC SENSES - Python Demo", phosphene_bgr)
        else:
            combined = np.hstack((display_left, phosphene_bgr))
            cv2.imshow("BIONIC SENSES - Python Demo", combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            show_all_classes = not show_all_classes
            print(f" >> Show All Classes: {show_all_classes}")
        elif key == ord('d'):
            detection_enabled = not detection_enabled
            print(f" >> Detection Enabled: {detection_enabled}")
        elif key == ord('t'):
            current_target_idx = (current_target_idx + 1) % len(SELECTABLE_TARGETS)
            audio_sys.play_target_change_sound()

        elif key == ord('r'):
            # swithc phosphene version
            if current_phosphene_version == "v4":
                current_phosphene_version = "v3"
            elif current_phosphene_version == "v3":
                current_phosphene_version = "v2"
            else:
                current_phosphene_version = "v4"
            
            print(f" >> Switching to Phosphene Processing {current_phosphene_version}")
            # Reloading the module to get the new renderer class.
            phosphene_module = importlib.import_module(f"phosphene_processing_{current_phosphene_version}")
            PhospheneRenderer = getattr(phosphene_module, 'PhospheneRenderer')
            # Re-initializing the renderer with our newly selected class and parameters to apply the change.
            spv_renderer = PhospheneRenderer(electrode_grid=(32, 32), 
                                             lattice_structure=(1, 0.5, 1),
                                             dropout_rate=0.1, 
                                             intensity_levels=PHOSPHENE_INTENSITY_LEVELS)
        elif key == ord('v'):
            show_only_spv = not show_only_spv
            print(f" >> Show Only SPV: {show_only_spv}")
        elif key == ord('g'):
            pure_rgb_mode = not pure_rgb_mode
            print(f" >> Pure RGB Mode: {pure_rgb_mode}")
        elif key == ord('p'):
            pure_depth_mode = not pure_depth_mode
            print(f" >> Pure Depth Mode: {pure_depth_mode}")
        elif key == ord('o'):
            enable_audio = not enable_audio
            audio_sys.active = enable_audio
            print(f" >> Audio Feedback: {enable_audio}")
        elif key == ord('s'):
            substitute_vision = not substitute_vision
            print(f" >> Substitute Vision: {'ON' if substitute_vision else 'OFF'}")

    if cap:
        cap.release()
    if lidar_stream:
        lidar_stream.close()
    detection_thread.stop()
    audio_sys.close()
    voice_sys.close()
    cv2.destroyAllWindows()












if __name__ == "__main__":
    main()