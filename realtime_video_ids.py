"""
realtime_video_ids.py (with attack simulation)
- Reads frames from webcam/RTSP and extracts features.
- Two attack simulation options:
    1) Press 'a' in the OpenCV window to toggle ATTACK mode (manual).
    2) Press 'i' to start an automated injected attack (runs for ATTACK_DURATION secs).
- Press 'q' to quit.
"""

import cv2
import time
import numpy as np
import joblib
from collections import deque
import threading

# CONFIG
VIDEO_SOURCE = 0          # 0 = webcam or "rtsp://..."
WINDOW_SECONDS = 5        # sliding window
MODEL_PATH = "rf_model.pkl"
ATTACK_DURATION = 8       # seconds for automated injection
ATTACK_INJECT_RATE = 60   # synthetic frames per second during injection
ATTACK_SIZE_MULT = 3.0    # multiply JPEG size to simulate large payloads

# Load model
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded:", MODEL_PATH)
except Exception as e:
    print("Failed to load model:", e)
    raise SystemExit(1)

# Helpers
def frame_to_jpeg_size(frame, quality=70):
    _, enc = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return enc.size

def motion_score(prev_gray, cur_gray):
    if prev_gray is None:
        return 0.0
    diff = cv2.absdiff(prev_gray, cur_gray)
    return float(np.mean(diff))

# Shared buffer (timestamp, size_bytes, motion)
buffer = deque()
buffer_lock = threading.Lock()

# Attack control flags
manual_attack = False    # toggled by keypress 'a'
inject_attack = False    # toggled by keypress 'i' -> starts automated injection thread

def automated_injector(duration=ATTACK_DURATION):
    """Inject synthetic high-rate, large-size entries into buffer for `duration` seconds."""
    global inject_attack
    inject_attack = True
    print(f"🔴 Automated injection started for {duration}s")
    start = time.time()
    while time.time() - start < duration:
        tstamp = time.time()
        # produce many synthetic "frames" per second
        # we simulate large sizes & high motion by multiplying typical values
        synthetic_size = 5000 * ATTACK_SIZE_MULT   # bytes (very large)
        synthetic_motion = 50.0                     # high motion
        with buffer_lock:
            buffer.append((tstamp, synthetic_size, synthetic_motion))
            # keep buffer limited in memory by trimming older entries
            while buffer and (tstamp - buffer[0][0] > WINDOW_SECONDS * 3):
                buffer.popleft()
        # sleep a little to simulate many packets (adjust to control intensity)
        time.sleep(1.0 / ATTACK_INJECT_RATE)
    inject_attack = False
    print("🟢 Automated injection finished")

# Video capture
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("❌ Unable to open video source:", VIDEO_SOURCE)
    raise SystemExit(1)

prev_gray = None
last_report = 0
print("🚀 Starting real-time video IDS. Press 'a' to toggle manual attack, 'i' to inject, 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame not received — retrying...")
            time.sleep(0.5)
            continue

        timestamp = time.time()
        frame_small = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        # If manual_attack is set, alter the frame size / motion to reflect attack
        if manual_attack:
            # simulate attack: increase encoded size by lowering JPEG compression and add noise
            noisy = frame_small.copy()
            noise = (np.random.randn(*noisy.shape) * 20).astype(np.uint8)
            noisy = cv2.add(noisy, noise)
            size_bytes = frame_to_jpeg_size(noisy, quality=95) * ATTACK_SIZE_MULT
            motion = motion_score(prev_gray, gray) + 30.0
        else:
            size_bytes = frame_to_jpeg_size(frame_small, quality=70)
            motion = motion_score(prev_gray, gray)

        prev_gray = gray

        # push into buffer
        with buffer_lock:
            buffer.append((timestamp, size_bytes, motion))
            # drop old entries beyond window
            while buffer and (timestamp - buffer[0][0] > WINDOW_SECONDS):
                buffer.popleft()

        # compute detection every 1 second
        if time.time() - last_report >= 1.0:
            last_report = time.time()
            with buffer_lock:
                if len(buffer) > 0:
                    duration = buffer[-1][0] - buffer[0][0] if len(buffer) > 1 else 1.0
                    frame_count = len(buffer)
                    packet_rate = frame_count / max(duration, 1e-6)
                    avg_pkt_size = float(np.mean([b for (_, b, _) in buffer]))
                    avg_motion = float(np.mean([m for (_, _, m) in buffer]))
                else:
                    packet_rate = 0.0
                    avg_pkt_size = 0.0
                    avg_motion = 0.0

            # Use same features your model expects (here: packet_rate, avg_pkt_size)
            features = np.array([[packet_rate, avg_pkt_size]])
            pred = model.predict(features)[0]
            label = "attack" if pred == 1 else "normal"

            ts_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            attack_flags = []
            if manual_attack: attack_flags.append("MANUAL")
            if inject_attack: attack_flags.append("INJECT")
            flags = (" | " + ",".join(attack_flags)) if attack_flags else ""
            print(f"[{ts_readable}] {label}{flags} | rate={packet_rate:.2f}fps size={avg_pkt_size:.1f}B motion={avg_motion:.2f}")

        # display small frame and overlay status
        status_text = "ATTACK" if manual_attack or inject_attack else "NORMAL"
        color = (0,0,255) if (manual_attack or inject_attack) else (0,255,0)
        cv2.putText(frame_small, status_text, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("IoT Camera (press 'a' 'i' 'q')", frame_small)

        # key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            manual_attack = not manual_attack
            print("🔁 Manual attack toggled:", manual_attack)
        elif key == ord('i') and not inject_attack:
            # start automated injection in a thread
            t = threading.Thread(target=automated_injector, args=(ATTACK_DURATION,), daemon=True)
            t.start()

except KeyboardInterrupt:
    print("\n🛑 Stopped by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
