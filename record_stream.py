import subprocess as sp
import static_ffmpeg

STREAM_SDP = "stream.sdp"
OUTPUT_FILE = "dataset_capture.mp4"

def record_stream():
    static_ffmpeg.add_paths()

    cmd = [
        "ffmpeg",
        "-protocol_whitelist", "file,udp,rtp",
        "-i", STREAM_SDP,
        "-t", "120",              # record 2 minutes (change this)
        "-vf", "scale=640:480",   # match your training resolution
        "-c:v", "libx264",
        "-crf", "35",             # compression → simulate low bandwidth
        "-preset", "veryfast",
        OUTPUT_FILE
    ]

    sp.run(cmd)

if __name__ == "__main__":
    record_stream()
