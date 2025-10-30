import random
import time
import joblib
import numpy as np
import sys

# Load the trained model
try:
    model = joblib.load("rf_model.pkl")
    print("✅ Model loaded successfully!\n")
except Exception as e:
    print("❌ Error loading model:", e)
    sys.exit(1)

print("🚀 Starting IoT Camera Simulation (Press Ctrl + C to stop)\n")

# Main loop
try:
    while True:
        # Generate random IoT traffic
        packet_rate = random.uniform(1, 50)
        avg_packet_size = random.uniform(200, 2000)
        features = np.array([[packet_rate, avg_packet_size]])

        # Predict attack or normal
        pred = model.predict(features)[0]

        if pred == 1:
            print(f"🚨 Attack detected! [Rate={packet_rate:.2f}, Size={avg_packet_size:.2f}]")
        else:
            print(f"✅ Normal traffic [Rate={packet_rate:.2f}, Size={avg_packet_size:.2f}]")

        # Flush output immediately (so it appears live)
        sys.stdout.flush()
        time.sleep(1)

except KeyboardInterrupt:
    print("\n🛑 Simulation stopped by user.")
