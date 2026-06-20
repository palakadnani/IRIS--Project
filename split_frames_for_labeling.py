import cv2
import os
import math

# ================= CONFIG =================
VIDEO_PATH = "dataset_capture.mp4"
OUTPUT_DIR = "dataset_chunks"

FPS_EXTRACT = 25            # frames per second to extract
RESIZE = (640, 480)       # match your training resolution
NUM_SPLITS = 6            # number of labelers / folders
JPEG_QUALITY = 85         # lower = more compression (simulate stream)
# ==========================================


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    ensure_dir(OUTPUT_DIR)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Cannot open video")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(video_fps / FPS_EXTRACT))

    print(f"[INFO] Video FPS: {video_fps}")
    print(f"[INFO] Extracting every {frame_interval} frames")

    frame_idx = 0
    saved_idx = 0

    # Create split folders
    split_dirs = []
    for i in range(NUM_SPLITS):
        d = os.path.join(OUTPUT_DIR, f"chunk_{i}")
        ensure_dir(d)
        split_dirs.append(d)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Resize
            if RESIZE is not None:
                frame = cv2.resize(frame, RESIZE)

            # Choose which chunk this frame goes to
            chunk_id = saved_idx % NUM_SPLITS
            out_dir = split_dirs[chunk_id]

            filename = f"img_{saved_idx:06d}.jpg"
            out_path = os.path.join(out_dir, filename)

            # Save with compression
            cv2.imwrite(
                out_path,
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )

            saved_idx += 1

        frame_idx += 1

    cap.release()

    print(f"[DONE] Extracted {saved_idx} images")
    print(f"[DONE] Split into {NUM_SPLITS} folders:")
    for d in split_dirs:
        print(" -", d)


if __name__ == "__main__":
    main()
