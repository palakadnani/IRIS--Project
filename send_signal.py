import cv2
import numpy as np
from ultralytics import YOLO
import socket

# =====================================
# SOCKET SETUP (SEND TO RASPBERRY PI)
# =====================================
PI_IP = "192.168.137.157"   # CHANGE THIS to your Pi IP
PI_PORT = 5000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_signal(mode, direction):
    message = f"{mode},{direction}"
    sock.sendto(message.encode(), (PI_IP, PI_PORT))
    print("Sent:", message)

# =====================================
# FILES
# =====================================
MODEL_PATH = "yolov8n.pt"
VIDEO_SOURCE = 0

model = YOLO(MODEL_PATH)
PERSON_CLASS = 0

# =====================================
# CAMERA
# =====================================
cap = cv2.VideoCapture(VIDEO_SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# =====================================
# SETTINGS
# =====================================
mode = "BALL"

REAL_BALL_DIAMETER_CM = 22.0
REAL_PERSON_WIDTH_CM = 45.0

FOCAL_LENGTH_BALL = 650.0
FOCAL_LENGTH_PERSON = 700.0

CAMERA_FOV_DEG = 70.0

PERSON_CONF_THRES = 0.35
DETECT_EVERY = 2
INFER_SIZE = 640

prev_people = []
frame_count = 0

# =====================================
# HELPERS
# =====================================
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

def detect_neon_orange_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([8, 170, 170])
    upper_orange = np.array([22, 255, 255])

    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 150:
            x, y, w, h = cv2.boundingRect(cnt)
            if area > max_area:
                max_area = area
                best_box = (x, y, w, h)

    return best_box

# =====================================
# MAIN LOOP
# =====================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]

    ball_box = detect_neon_orange_ball(frame)

    # =====================================
    # PERSON DETECTION
    # =====================================
    people = []
    if frame_count % DETECT_EVERY == 0:
        results = model(frame, imgsz=INFER_SIZE, verbose=False)

        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == PERSON_CLASS and conf > PERSON_CONF_THRES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    people.append((x1, y1, x2 - x1, y2 - y1))

    frame_count += 1

    # =====================================
    # DECISION LOGIC
    # =====================================
    direction = "NONE"

    if mode == "BALL":
        if ball_box is not None:
            x, y, w, h = ball_box
            cx = x + w // 2
            direction = get_direction(cx, W)

            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)

    elif mode == "HUMAN":
        if people:
            # choose nearest (largest box)
            best = max(people, key=lambda p: p[2]*p[3])
            x, y, w, h = best

            cx = x + w // 2
            direction = get_direction(cx, W)

            cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)

    # =====================================
    # SEND SIGNAL TO PI
    # =====================================
    if direction != "NONE":
        send_signal(mode, direction)
# =====================================
    # DISPLAY
    # =====================================
    cv2.putText(frame, f"MODE: {mode}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.putText(frame, f"DIRECTION: {direction}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("IRIS", frame)

    key = cv2.waitKey(1) & 0xFF

    # 🔁 SIMULATE BUTTON (FOR NOW)
    if key == ord('m'):
        if mode == "BALL":
            mode = "HUMAN"
            print("Switched to HUMAN")
        else:
            mode = "BALL"
            print("Switched to BALL")

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()