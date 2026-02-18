import time
from datetime import datetime
from picamera2 import Picamera2

# 1. Initialize camera
picam2 = Picamera2()

# 2. Create timestamped filename
# Format: 20260218-150530.mp4
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
filename = f"{timestamp}.mp4"

# 3. Record for 30 seconds
# This function handles the 'output' argument internally
print(f"🎬 Recording 30 seconds to: {filename}")
picam2.start_and_record_video(filename, duration=30)

print(f"✅ Video saved successfully as {filename}")
