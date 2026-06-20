import subprocess as sp
import time

ffplay_cmd = [
    "ffplay",
    "-protocol_whitelist", "file,udp,rtp",
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-rtbufsize", "100M",
    "-i", "stream.sdp"
]

print("Starting stream with ffplay...")
pipe = sp.Popen(ffplay_cmd)

# Keep the script running until ffplay is closed
pipe.wait()