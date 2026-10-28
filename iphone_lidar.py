import numpy as np
import cv2
from threading import Event

try:
    from record3d import Record3DStream
    HAS_RECORD3D = True
except ImportError:
    HAS_RECORD3D = False # Flag to indicate if the record3d library is available
    print("WARNING: 'record3d' module not found. Make sure it's installed or in the project folder.")
    class Record3DStream: pass

class IPhoneLiDARStream:
    def __init__(self, port=7100):
        # The port is not used with the USB library, but we keep the signature for consistency.
        self.event = Event() # Event for signaling new frames
        self.session = None
        self.connected = False
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1
        # Buffer for Temporal Smoothing (Moving Average)
        self.depth_buffer = []
        self.buffer_size = 1  # REDUCED TO 1: Removes latency (Zero-Lag)

    def on_new_frame(self):
        """Callback invoked by the Record3D thread when a new frame arrives."""
        self.event.set()

    def on_stream_stopped(self):
        self.connected = False
        print("Stream interrupted by the device.")
        self.event.set() # Sblocca eventuali wait() in corso

    def connect(self, host='127.0.0.1'):
        if not HAS_RECORD3D:
            return False
            
        print("Trying to connect to the iPhone via USB (record3d library)...")
        try:
            devs = Record3DStream.get_connected_devices()
            print(f'{len(devs)} device(s) found')
            for dev in devs:
                print(f'\tID: {dev.product_id}\n\tUDID: {dev.udid}')

            if not devs:
                print("No device found.")
                return False

            dev_idx = 0
            dev = devs[dev_idx]
            
            self.session = Record3DStream()
            self.session.on_new_frame = self.on_new_frame
            self.session.on_stream_stopped = self.on_stream_stopped # Register callback for stream stop
            self.session.connect(dev)
            self.connected = True
            print(" >> iPhone Connected successfully (USB)!")
            return True
        except Exception as e:
            print(f" >> USB connection error: {e}")
            self.connected = False
            return False

    def read(self):
        if not self.connected or not HAS_RECORD3D:
            return False, None, None, None
        
        # Following the record3d demo: we wait without timeout.
        # If the app is slow to start, timeout=1.0 would close the program.
        self.event.wait()
        # Safety check: if we woke up due to a disconnection, exit cleanly
        # Safety check: if we woke up because of a disconnection, exit cleanly
        if not self.connected:
            return False, None, None, None

        depth = self.session.get_depth_frame()
        rgb = self.session.get_rgb_frame()
        confidence = self.session.get_confidence_frame()
        
        # --- TEMPORAL SMOOTHING & THRESHOLDING ---
        if depth is not None:
            # Reset the buffer if the resolution changes (e.g. screen rotation)
            if len(self.depth_buffer) > 0 and self.depth_buffer[0].shape != depth.shape:
                self.depth_buffer = []

            # 1. Append to the buffer
            self.depth_buffer.append(depth)
            if len(self.depth_buffer) > self.buffer_size:
                self.depth_buffer.pop(0)
            
            # 2. Compute the mean (reduces random noise)
            if len(self.depth_buffer) > 1:
                stack = np.stack(self.depth_buffer, axis=0)
                depth = np.mean(stack, axis=0).astype(np.float32)
                
                # 3. Threshold filter
                depth[depth < 0.2] = 0.0  # Ignore very-close-range noise (< 20cm)
                depth[depth > 5.0] = 5.0  # Ignora oltre 5m (limite fisico affidabile)

        # Record3D invia RGB, OpenCV vuole BGR
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # Clear the event after reading, as in the record3d demo
        self.event.clear()
        
        # FIX: FaceID (TrueDepth) may return an empty confidence map.
        # Return None to avoid crashes in downstream resizes.
        if confidence is not None and confidence.size == 0:
            confidence = None

        return True, rgb, depth, confidence

    def get_device_type(self):
        if self.connected and self.session:
            return self.session.get_device_type()
        return -1

    def close(self):
        self.connected = False
        # The library handles shutdown, but we can force the flag
        pass

# --- MOCK CLASS FOR TESTING WITHOUT AN IPHONE ---
# If the Record3D app is not available, this class simulates a LiDAR
# by creating a fake depth map centered on the image.
class MockLiDARStream:
    def __init__(self, cap_ref):
        self.cap = cap_ref # Reference to the standard webcam

    def read(self):
        ret, frame = self.cap.read()
        if not ret: return False, None, None, None
        
        h, w = frame.shape[:2]
        
        # Simulated depth: near center (1.0), far edges (0.0)
        # Create a depth "bubble" in the center
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xv, yv = np.meshgrid(x, y)
        depth = np.exp(-(xv**2 + yv**2) * 2) # Gaussiana
        
        # Depth in meters (simulated: 0.5m at the center, 5m at the edges)
        # Invert: 1.0 = near, 0.0 = far for visualization
        return True, frame, depth, None

    def get_device_type(self):
        return 1 # Simulates the rear LiDAR (1) by default

    def close(self):
        pass