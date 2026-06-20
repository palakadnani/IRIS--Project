import cv2
import json
import numpy as np
import socket
import subprocess as sp
import threading
import time
import static_ffmpeg
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"
STREAM_SDP = "stream.sdp"
STREAM_WIDTH = 640
STREAM_HEIGHT = 480

# ========================= NETWORK SETTINGS =========================
# Set these to the correct IP addresses on your network.
PI_IP = "192.168.137.157"          # Raspberry Pi Zero IP
PC_BIND_IP = "0.0.0.0"          # Listen on all interfaces for mode updates
PI_HAPTIC_PORT = 5005           # Pi listens here for haptic commands
PC_MODE_PORT = 5006             # PC listens here for Pi mode packets
HAPTIC_SEND_HZ = 20.0           # How often to send motor updates to Pi
ALLOW_KEYBOARD_MODE_OVERRIDE = True
# ===================================================================

# ── GPU: loads model onto CUDA device 0; falls back to CPU automatically ──
model = YOLO(MODEL_PATH)
try:
    model.to("cuda")
except Exception:
    model.to("cpu")

#======================= Ball Model ==============================
BALL_MODEL_PATH = "best.pt"   # your trained model
ball_model = YOLO(BALL_MODEL_PATH)
ball_model.to("cuda")
BALL_CONF_THRES = 0.25
# =================================================================

PERSON_CLASS = 0
mode = "BALL"
mode_lock = threading.Lock()
last_mode_msg_time = 0.0

REAL_BALL_DIAMETER_CM = 22.0
REAL_PERSON_WIDTH_CM = 45.0

FOCAL_LENGTH_BALL = 650.0
FOCAL_LENGTH_PERSON = 700.0

CAMERA_FOV_DEG = 53.5

PERSON_CONF_THRES = 0.35
DETECT_EVERY = 2
INFER_SIZE = 640

BOX_ALPHA = 0.35
DIST_ALPHA = 0.25
SMOOTH_ALPHA_BALL = 0.28

# ── Ball tracker tuning ──────────────────────────────────────────────
BALL_CONFIRM_FRAMES = 2   # consecutive hits required before ball is shown
BALL_DROP_FRAMES    = 10  # consecutive misses required before ball is dropped
BALL_ROI_GATE_PX    = 120 # max pixel distance from last center to accept a new hit
# ─────────────────────────────────────────────────────────────────────

prev_ball_dist = None
prev_ball_angle = None
prev_people = []

frame_count = 0
cached_result = {
    "ball": None,
    "people": []
}

last_haptic_send_time = 0.0
last_sent_packet = None

MOTOR_NAMES = ["LEFT", "SLIGHT_LEFT", "MIDDLE", "SLIGHT_RIGHT", "RIGHT"]


class BallTracker:
    """
    Combines three layers of robustness on top of the raw color detector:

      1. Kalman filter  — models position + velocity so the tracked box keeps
                          moving smoothly during brief detection gaps instead of
                          freezing or vanishing.

      2. Hysteresis     — requires BALL_CONFIRM_FRAMES consecutive hits before
                          the ball is shown, and BALL_DROP_FRAMES consecutive
                          misses before it is hidden. Eliminates single-frame
                          flickers in both directions.

      3. ROI gating     — once the ball is known, only detections within
                          BALL_ROI_GATE_PX of the last center are accepted.
                          Kills false-positive flashes from distant red objects.
    """

    def __init__(self):
        self.kf = None
        self.kalman_ready = False
        self.confirmed   = False
        self.hit_count   = 0
        self.miss_count  = 0
        self.last_cx     = None   # last Kalman-estimated center x
        self.last_cy     = None   # last Kalman-estimated center y
        self.last_w      = None   # last measured width (for distance estimation)

    # ── internal ──────────────────────────────────────────────────────

    def _make_kalman(self, cx: float, cy: float):
        """Initialise a constant-velocity Kalman filter seeded at (cx, cy)."""
        kf = cv2.KalmanFilter(4, 2)   # 4 state dims, 2 measurement dims
        # State transition: x' = x + vx,  y' = y + vy,  vx'=vx, vy'=vy
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        # We only measure position (not velocity)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        # Process noise: how much we trust the motion model
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 5e-2
        # Measurement noise: how noisy the detector output is
        # Higher value = smoother but slower to react
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        seed = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
        kf.statePre  = seed.copy()
        kf.statePost = seed.copy()
        return kf

    def _reset(self):
        self.kf           = None
        self.kalman_ready = False
        self.confirmed    = False
        self.hit_count    = 0
        self.miss_count   = 0
        self.last_cx      = None
        self.last_cy      = None
        self.last_w       = None

    # ── public ────────────────────────────────────────────────────────

    def update(self, raw_box, frame_w: int, frame_h: int):
        """
        Feed one frame's raw detection result.

        raw_box : (x, y, w, h) from detect_full_orange_ball, or None.
        Returns : (x, y, w, h) box to use for display / haptics, or None.
        """
        # ── ROI gate: discard detections too far from last known center ──
        if raw_box is not None and self.last_cx is not None:
            rx, ry, rw, rh = raw_box
            rcx = rx + rw // 2
            rcy = ry + rh // 2
            if np.hypot(rcx - self.last_cx, rcy - self.last_cy) > BALL_ROI_GATE_PX:
                raw_box = None   # treat as a miss this frame

        # ── HIT branch ───────────────────────────────────────────────────
        if raw_box is not None:
            rx, ry, rw, rh = raw_box
            rcx = float(rx + rw // 2)
            rcy = float(ry + rh // 2)

            if not self.kalman_ready:
                self.kf = self._make_kalman(rcx, rcy)
                self.kalman_ready = True

            # Kalman predict → correct cycle
            self.kf.predict()
            state = self.kf.correct(
                np.array([[rcx], [rcy]], dtype=np.float32)
            )

            self.last_cx = int(state[0, 0])
            self.last_cy = int(state[1, 0])
            self.last_w  = rw

            self.miss_count  = 0
            self.hit_count  += 1
            if self.hit_count >= BALL_CONFIRM_FRAMES:
                self.confirmed = True

        # ── MISS branch ──────────────────────────────────────────────────
        else:
            self.hit_count   = 0
            self.miss_count += 1

            if self.miss_count > BALL_DROP_FRAMES:
                self._reset()
                return None

            # Kalman predict-only: box keeps drifting on inertia during gap
            if self.kalman_ready:
                state = self.kf.predict()
                self.last_cx = int(state[0, 0])
                self.last_cy = int(state[1, 0])

        # ── Emit box only once confirmed ─────────────────────────────────
        if not self.confirmed or self.last_cx is None or self.last_w is None:
            return None

        half = self.last_w // 2
        bx = max(0, self.last_cx - half)
        by = max(0, self.last_cy - half)
        bw = min(frame_w - bx, self.last_w)
        bh = min(frame_h - by, self.last_w)   # keep box square (ball is round)
        return (bx, by, bw, bh)


ball_tracker = BallTracker()   # single shared instance used in the main loop


class PiModeReceiver:
    def __init__(self, bind_ip: str, bind_port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((bind_ip, bind_port))
        self.sock.settimeout(0.2)
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        global mode, last_mode_msg_time
        while self.running:
            try:
                data, _ = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                msg = json.loads(data.decode("utf-8"))
            except Exception:
                continue

            if msg.get("type") == "mode":
                new_mode = str(msg.get("mode", "BALL")).upper()
                if new_mode in ("BALL", "HUMAN"):
                    with mode_lock:
                        mode = new_mode
                        last_mode_msg_time = time.time()
                    print(f"[MODE] Pi set mode -> {new_mode}")

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass


class FrameReader:
    # Continuously drains the pipe in the background so the main loop always
    # gets the freshest frame instead of processing a queue of stale ones.
    def __init__(self, pipe, frame_size, width, height):
        self.pipe = pipe
        self.frame_size = frame_size
        self.width = width
        self.height = height
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            if self.pipe.stdout is None:
                break
            raw = self.pipe.stdout.read(self.frame_size)
            if len(raw) != self.frame_size:
                break
            frame = np.frombuffer(raw, np.uint8).reshape((self.height, self.width, 3))
            with self.lock:
                self.frame = frame  # always overwrite — keeps only the latest

    def read(self):
        with self.lock:
            return self.frame is not None, (self.frame.copy() if self.frame is not None else None)

    def stop(self):
        self.running = False


def get_mode():
    with mode_lock:
        return mode


def set_mode(new_mode: str):
    global mode
    new_mode = str(new_mode).upper()
    if new_mode not in ("BALL", "HUMAN"):
        return
    with mode_lock:
        mode = new_mode


def start_stream(stream_sdp, width, height):
    static_ffmpeg.add_paths()

    ffmpeg_cmd = [
        "ffmpeg",
        "-protocol_whitelist", "file,udp,rtp",
        "-probesize", "1M",
        "-analyzeduration", "100000",
        "-fflags", "nobuffer+discardcorrupt",
        "-flags", "low_delay",
        "-avioflags", "direct",
        "-rtbufsize", "512k",
        "-i", stream_sdp,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-vsync", "drop",
        "-"
    ]

    pipe = sp.Popen(ffmpeg_cmd, stdout=sp.PIPE, stderr=sp.DEVNULL, bufsize=10**8)
    frame_size = width * height * 3
    return pipe, frame_size


def smooth_box(prev_box, new_box, alpha=0.35):
    if prev_box is None:
        return new_box
    px, py, pw, ph = prev_box
    x, y, w, h = new_box
    sx = int(alpha * x + (1 - alpha) * px)
    sy = int(alpha * y + (1 - alpha) * py)
    sw = int(alpha * w + (1 - alpha) * pw)
    sh = int(alpha * h + (1 - alpha) * ph)
    return (sx, sy, sw, sh)


def smooth_value(prev_val, new_val, alpha=0.25):
    if prev_val is None:
        return new_val
    if new_val is None:
        return prev_val
    return alpha * new_val + (1 - alpha) * prev_val


def get_direction(center_x, frame_width):
    norm = (center_x - frame_width / 2) / (frame_width / 2)
    if norm < -0.6:
        return "LEFT"
    elif norm < -0.2:
        return "SLIGHT LEFT"
    elif norm < 0.2:
        return "FRONT"
    elif norm < 0.6:
        return "SLIGHT RIGHT"
    else:
        return "RIGHT"


def get_ball_direction(center_x, frame_width):
    if center_x < frame_width / 3:
        return "LEFT"
    elif center_x < 2 * frame_width / 3:
        return "CENTER"
    else:
        return "RIGHT"


def get_angle_deg(center_x, frame_width, camera_fov_deg=70.0):
    norm = (center_x - frame_width / 2) / (frame_width / 2)
    return norm * (camera_fov_deg / 2.0)


def angle_text(angle_deg):
    if angle_deg is None:
        return "N/A"
    return f"{angle_deg:+.1f} deg"


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return x1, y1, x2, y2


def box_iou(boxA, boxB):
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    areaA = aw * ah
    areaB = bw * bh
    union = areaA + areaB - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def estimate_distance_from_width(real_width_cm, focal_length, pixel_width):
    if pixel_width <= 0:
        return None
    return (real_width_cm * focal_length) / pixel_width


def classify_bib_color(frame, person_box):
    x, y, w, h = person_box
    H, W = frame.shape[:2]

    rx1 = x + int(0.30 * w)
    ry1 = y + int(0.22 * h)
    rx2 = x + int(0.70 * w)
    ry2 = y + int(0.60 * h)

    rx1, ry1, rx2, ry2 = clamp_box(rx1, ry1, rx2, ry2, W, H)
    if rx2 <= rx1 or ry2 <= ry1:
        return "UNKNOWN", None

    roi = frame[ry1:ry2, rx1:rx2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    roi_area = roi.shape[0] * roi.shape[1]
    if roi_area == 0:
        return "UNKNOWN", (rx1, ry1, rx2, ry2)

    lower_red1 = np.array([0, 70, 60])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 60])
    upper_red2 = np.array([179, 255, 255])
    lower_blue = np.array([90, 60, 40])
    upper_blue = np.array([135, 255, 255])

    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    red_ratio = np.count_nonzero(red_mask) / roi_area
    blue_ratio = np.count_nonzero(blue_mask) / roi_area
    mean_sat = float(np.mean(hsv[:, :, 1]))

    if red_ratio >= 0.18 and red_ratio > blue_ratio + 0.08 and mean_sat >= 60:
        return "OPPONENT", (rx1, ry1, rx2, ry2)
    if blue_ratio >= 0.18 and blue_ratio > red_ratio + 0.08 and mean_sat >= 50:
        return "TEAMMATE", (rx1, ry1, rx2, ry2)
    return "UNKNOWN", (rx1, ry1, rx2, ry2)


def detect_ball_ml(frame):
    results = ball_model(frame, imgsz=416, verbose=False)

    boxes = results[0].boxes
    if boxes is None:
        return None

    best_box = None
    best_conf = 0.0

    for box in boxes:
        conf = float(box.conf[0].item())
        if conf < BALL_CONF_THRES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            continue

        if conf > best_conf:
            best_conf = conf
            best_box = (x1, y1, w, h)

    return best_box



def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def distance_to_intensity(distance_cm, near_cm=70.0, far_cm=350.0):
    if distance_cm is None:
        return 0.0
    if distance_cm <= near_cm:
        return 1.0
    if distance_cm >= far_cm:
        return 0.0
    t = 1.0 - ((distance_cm - near_cm) / (far_cm - near_cm))
    return clamp01(0.18 + 0.82 * t)


def angle_to_motor_mix(angle_deg, base_intensity):
    """
    Converts a horizontal angle into a 5-motor intensity pattern.
    Motors:
    [LEFT, SLIGHT_LEFT, MIDDLE, SLIGHT_RIGHT, RIGHT]
    """
    if angle_deg is None or base_intensity <= 0.0:
        return [0.0] * 5

    angle_deg = float(np.clip(angle_deg, -CAMERA_FOV_DEG / 2.0, CAMERA_FOV_DEG / 2.0))

    centers = np.array([-28.0, -14.0, 0.0, 14.0, 28.0], dtype=np.float32)
    widths = np.array([14.0, 14.0, 14.0, 14.0, 14.0], dtype=np.float32)

    weights = []
    for c, w in zip(centers, widths):
        wt = max(0.0, 1.0 - abs(angle_deg - c) / w)
        weights.append(wt)

    weights = np.array(weights, dtype=np.float32)

    if float(weights.max()) <= 0.0:
        idx = 0 if angle_deg < 0 else 4
        weights[idx] = 1.0

    weights = weights / max(float(weights.max()), 1e-6)

    # Per-motor gain — tune these independently without touching the mix logic.
    # Values above 1.0 amplify, below 1.0 attenuate. Output is still clamped to [0, 1].
    # Order: [LEFT, SLIGHT_LEFT, MIDDLE, SLIGHT_RIGHT, RIGHT]
    motor_gains = np.array([1.5, 1.4, 1.5, 1.0, 1.0], dtype=np.float32)

    motors = [clamp01(float(base_intensity * w * g)) for w, g in zip(weights, motor_gains)]
    return motors


def build_haptic_packet(current_mode, ball, teammate_list):
    """
    Returns:
        motors: list of 5 values [0.0 .. 1.0]
        target_label: str
        target_angle: float | None
        target_distance: float | None
    """
    motors = [0.0] * 5
    target_label = "NONE"
    target_angle = None
    target_distance = None

    if current_mode == "BALL":
        if ball is not None and ball.get("distance") is not None:
            target_angle = ball.get("angle")
            target_distance = ball.get("distance")
            base = distance_to_intensity(target_distance)
            motors = angle_to_motor_mix(target_angle, base)
            target_label = "BALL"

    elif current_mode == "HUMAN":
        if teammate_list:
            nearest_teammate = min(teammate_list, key=lambda p: p["distance"])
            target_angle = nearest_teammate.get("angle")
            target_distance = nearest_teammate.get("distance") / 2
            base = distance_to_intensity(target_distance)
            motors = angle_to_motor_mix(target_angle, base)
            target_label = "TEAMMATE"

    return motors, target_label, target_angle, target_distance


def send_haptic_packet(sock, motors, current_mode, target_label, target_angle, target_distance):
    packet = {
        "type": "haptic",
        "mode": current_mode,
        "target": target_label,
        "angle_deg": None if target_angle is None else round(float(target_angle), 2),
        "distance_cm": None if target_distance is None else round(float(target_distance), 2),
        "motors": [round(float(m), 3) for m in motors],
        "ts": round(time.time(), 3),
    }
    data = json.dumps(packet).encode("utf-8")
    sock.sendto(data, (PI_IP, PI_HAPTIC_PORT))
    return packet


pipe, frame_size = start_stream(STREAM_SDP, STREAM_WIDTH, STREAM_HEIGHT)
stream_start_time = time.time()

reader = FrameReader(pipe, frame_size, STREAM_WIDTH, STREAM_HEIGHT)
mode_receiver = PiModeReceiver(PC_BIND_IP, PC_MODE_PORT)
haptic_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cv2.namedWindow("IRIS", cv2.WINDOW_NORMAL)
cv2.namedWindow("Ball Mask", cv2.WINDOW_NORMAL)

try:
    while True:
        ret, frame = reader.read()
        if not ret or frame is None:
            time.sleep(0.001)
            continue
        
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # rotates input 90 degrees
        
        frame_count += 1
        H, W = frame.shape[:2]

        if frame_count % 30 == 0:
            fps = frame_count / max(time.time() - stream_start_time, 1e-6)
            print(f"FPS: {fps:.1f}")

        raw_ball_box = detect_ball_ml(frame)
        ball_mask = frame  # placeholder (you can remove window if you want)


        tracked_box = ball_tracker.update(raw_ball_box, W, H)

        if tracked_box is not None:
            bx, by, bw, bh = tracked_box
            ball_dist = estimate_distance_from_width(REAL_BALL_DIAMETER_CM, FOCAL_LENGTH_BALL, bw)
            if ball_dist is not None:
                ball_dist = smooth_value(prev_ball_dist, ball_dist, DIST_ALPHA)
            center_x = bx + bw // 2
            ball_angle = get_angle_deg(center_x, W, CAMERA_FOV_DEG)
            ball_angle = smooth_value(prev_ball_angle, ball_angle, DIST_ALPHA)
            cached_result["ball"] = {
                "box": tracked_box,
                "distance": ball_dist,
                "angle": ball_angle
            }
            prev_ball_dist = ball_dist
            prev_ball_angle = ball_angle
        else:
            cached_result["ball"] = None
            prev_ball_dist = None
            prev_ball_angle = None

        run_detect = (frame_count % DETECT_EVERY == 0)

        if run_detect:
            results = model(frame, imgsz=INFER_SIZE, verbose=False)
            boxes = results[0].boxes
            detected_people = []

            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H)
                    bw = x2 - x1
                    bh = y2 - y1
                    if bw <= 0 or bh <= 0:
                        continue
                    if cls == PERSON_CLASS and conf >= PERSON_CONF_THRES:
                        role, roi_box = classify_bib_color(frame, (x1, y1, bw, bh))
                        detected_people.append({
                            "box": (x1, y1, bw, bh),
                            "conf": conf,
                            "role": role,
                            "roi_box": roi_box
                        })

            new_people = []
            used_prev = set()

            for person in detected_people:
                best_match_idx = -1
                best_iou = 0.0
                for i, prev_person in enumerate(prev_people):
                    if i in used_prev:
                        continue
                    iou = box_iou(person["box"], prev_person["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_match_idx = i

                if best_match_idx != -1 and best_iou > 0.2:
                    used_prev.add(best_match_idx)
                    prev_person = prev_people[best_match_idx]
                    smoothed_box_person = smooth_box(prev_person["box"], person["box"], BOX_ALPHA)
                    distance = estimate_distance_from_width(
                        REAL_PERSON_WIDTH_CM,
                        FOCAL_LENGTH_PERSON,
                        smoothed_box_person[2]
                    )
                    distance = smooth_value(prev_person.get("distance"), distance, DIST_ALPHA)
                    center_x = smoothed_box_person[0] + smoothed_box_person[2] // 2
                    angle = get_angle_deg(center_x, W, CAMERA_FOV_DEG)
                    angle = smooth_value(prev_person.get("angle"), angle, DIST_ALPHA)
                    new_people.append({
                        "box": smoothed_box_person,
                        "conf": person["conf"],
                        "role": person["role"],
                        "roi_box": person["roi_box"],
                        "distance": distance,
                        "angle": angle
                    })
                else:
                    distance = estimate_distance_from_width(
                        REAL_PERSON_WIDTH_CM,
                        FOCAL_LENGTH_PERSON,
                        person["box"][2]
                    )
                    center_x = person["box"][0] + person["box"][2] // 2
                    angle = get_angle_deg(center_x, W, CAMERA_FOV_DEG)
                    new_people.append({
                        "box": person["box"],
                        "conf": person["conf"],
                        "role": person["role"],
                        "roi_box": person["roi_box"],
                        "distance": distance,
                        "angle": angle
                    })

            prev_people = new_people
            cached_result["people"] = new_people

        display = frame.copy()
        cv2.line(display, (W // 2, 0), (W // 2, H), (0, 255, 255), 2)

        ball = cached_result["ball"]
        people = cached_result["people"]
        teammate_list = [p for p in people if p["role"] == "TEAMMATE" and p["distance"] is not None]

        current_mode = get_mode()
        cv2.putText(display, f"MODE: {current_mode}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if current_mode == "BALL":
            if ball is not None:
                x, y, w, h = ball["box"]
                cx = x + w // 2
                direction = get_ball_direction(cx, W)
                dist_txt = "N/A" if ball["distance"] is None else f"{ball['distance']:.1f} cm"
                ang_txt = angle_text(ball.get("angle"))
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    display,
                    f"BALL | {direction} | {ang_txt} | {dist_txt}",
                    (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                cv2.putText(display, f"BALL: {direction}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display, f"ANGLE: {ang_txt}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(display, f"DIST: {dist_txt}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            else:
                cv2.putText(display, "BALL: NOT FOUND", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        elif current_mode == "HUMAN":
            for p in people:
                x, y, w, h = p["box"]
                cx = x + w // 2
                direction = get_direction(cx, W)
                if p["role"] == "TEAMMATE":
                    color, label, text_color = (0, 0, 255), "TEAMMATE", (0, 0, 0)
                elif p["role"] == "OPPONENT":
                    color, label, text_color = (255, 0, 0), "OPPONENT", (0, 0, 0)
                else:
                    color, label, text_color = (180, 180, 180), "UNKNOWN", (0, 0, 0)

                cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                if p["roi_box"] is not None:
                    rx1, ry1, rx2, ry2 = p["roi_box"]
                    cv2.rectangle(display, (rx1, ry1), (rx2, ry2), color, 1)

                dist_txt = "N/A" if p["distance"] is None else f"{p['distance']:.1f} cm"
                ang_txt = angle_text(p.get("angle"))
                cv2.putText(
                    display,
                    f"{label} | {direction} | {ang_txt} | {dist_txt}",
                    (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    text_color,
                    2
                )

            if teammate_list:
                nearest_teammate = min(teammate_list, key=lambda p: p["distance"])
                x, y, w, h = nearest_teammate["box"]
                direction = get_direction(x + w // 2, W)
                dist_txt = f"{nearest_teammate['distance']:.1f} cm"
                ang_txt = angle_text(nearest_teammate.get("angle"))
                cv2.putText(
                    display,
                    f"CLOSEST TEAMMATE: {direction}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    2
                )
                cv2.putText(display, f"ANGLE: {ang_txt}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(display, f"DIST: {dist_txt}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            else:
                cv2.putText(
                    display,
                    "CLOSEST TEAMMATE: NOT FOUND",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

        motors, target_label, target_angle, target_distance = build_haptic_packet(
            current_mode,
            ball,
            teammate_list
        )

        now = time.time()
        if now - last_haptic_send_time >= (1.0 / HAPTIC_SEND_HZ):
            last_sent_packet = send_haptic_packet(
                haptic_sock,
                motors,
                current_mode,
                target_label,
                target_angle,
                target_distance
            )
            last_haptic_send_time = now

        motor_txt = " | ".join([f"{name}:{m:.2f}" for name, m in zip(MOTOR_NAMES, motors)])
        cv2.putText(display, f"TARGET: {target_label}", (20, H - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, motor_txt, (20, H - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, "Press B=Ball, H=Human, Q=Quit", (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("IRIS", display)
        cv2.imshow("Ball Mask", ball_mask)

        key = cv2.waitKey(1) & 0xFF
        if ALLOW_KEYBOARD_MODE_OVERRIDE:
            if key == ord("b"):
                set_mode("BALL")
            elif key == ord("h"):
                set_mode("HUMAN")
        if key == 27 or key == ord("q"):
            break

finally:
    reader.stop()
    mode_receiver.stop()
    try:
        haptic_sock.close()
    except Exception:
        pass
    if pipe.poll() is None:
        pipe.terminate()
    cv2.destroyAllWindows()
