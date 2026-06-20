import cv2
import numpy as np
import subprocess as sp
import time
import static_ffmpeg  # Add this import

WIDTH = 640
HEIGHT = 480

# This downloads and adds FFmpeg to your PATH automatically
static_ffmpeg.add_paths()  # Downloads ffmpeg binary on first run

# Now ffmpeg will be available!
ffmpeg_cmd = [
    "ffmpeg",  # This will now work!
    "-protocol_whitelist", "file,udp,rtp",
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-rtbufsize", "100M",
    "-i", "stream.sdp",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-vsync", "0",
    "-"
]

print("Starting stream...")
pipe = sp.Popen(ffmpeg_cmd, stdout=sp.PIPE, stderr=sp.DEVNULL)
frame_size = WIDTH * HEIGHT * 3

# Clear initial buffer
for _ in range(3):
    pipe.stdout.read(frame_size)

frame_count = 0
start_time = time.time()

while True:
    raw = pipe.stdout.read(frame_size)
    if len(raw) != frame_size:
        break
    
    frame = np.frombuffer(raw, np.uint8).reshape((HEIGHT, WIDTH, 3))
    frame_count += 1
    
    if frame_count % 30 == 0:
        fps = frame_count / (time.time() - start_time)
        print(f"FPS: {fps:.1f}")
    
    cv2.imshow("Stream", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipe.terminate()
cv2.destroyAllWindows()