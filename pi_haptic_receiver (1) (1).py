import json
import socket
import threading
import time
from gpiozero import Button, PWMOutputDevice

# ========================= NETWORK SETTINGS =========================
PC_IP = "192.168.137.1"     # Your laptop/desktop IP
PI_BIND_IP = "0.0.0.0"
PI_HAPTIC_PORT = 5005       # Pi listens here for haptic commands from PC
PC_MODE_PORT = 5006         # PC listens here for mode packets from Pi
# ===================================================================

# GPIO pins for vibration motors:
# [left, slight left, middle, slight right, right]
MOTOR_PINS = [17, 22, 23, 24, 27]
BUTTON_PIN = 5

PWM_FREQUENCY = 180
HAPTIC_TIMEOUT_S = 0.50
MODE_DEBOUNCE_S = 0.25

motors = [
    PWMOutputDevice(pin, frequency=PWM_FREQUENCY, initial_value=0.0)
    for pin in MOTOR_PINS
]

button = Button(BUTTON_PIN, pull_up=True, bounce_time=0.08)

current_mode = "BALL"
last_packet_time = 0.0
last_toggle_time = 0.0
state_lock = threading.Lock()
running = True


def set_all_motors(values):
    for motor, value in zip(motors, values):
        motor.value = max(0.0, min(1.0, float(value)))


def stop_all_motors():
    set_all_motors([0.0] * len(motors))


def send_mode(sock):
    packet = {
        "type": "mode",
        "mode": current_mode,
        "ts": round(time.time(), 3),
    }
    sock.sendto(json.dumps(packet).encode("utf-8"), (PC_IP, PC_MODE_PORT))


def on_button_pressed():
    global current_mode, last_toggle_time
    now = time.time()
    if now - last_toggle_time < MODE_DEBOUNCE_S:
        return
    last_toggle_time = now

    with state_lock:
        current_mode = "HUMAN" if current_mode == "BALL" else "BALL"
        local_mode = current_mode

    print(f"[BUTTON] Mode changed to {local_mode}")
    send_mode(mode_sock)


def haptic_listener():
    global last_packet_time

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((PI_BIND_IP, PI_HAPTIC_PORT))
    sock.settimeout(0.2)

    while running:
        try:
            data, _ = sock.recvfrom(4096)
        except socket.timeout:
            continue
        except OSError:
            break

        try:
            msg = json.loads(data.decode("utf-8"))
        except Exception:
            continue

        if msg.get("type") != "haptic":
            continue

        packet_mode = str(msg.get("mode", "")).upper()
        packet_motors = msg.get("motors", [])

        if len(packet_motors) != 5:
            continue

        with state_lock:
            local_mode = current_mode

        # Ignore stale commands from the wrong mode.
        if packet_mode != local_mode:
            continue

        set_all_motors(packet_motors)
        last_packet_time = time.time()

        angle = msg.get("angle_deg")
        distance = msg.get("distance_cm")
        target = msg.get("target", "NONE")
        print(f"[HAPTIC] mode={packet_mode} target={target} angle={angle} dist={distance} motors={packet_motors}")

    sock.close()


def watchdog_loop():
    while running:
        if time.time() - last_packet_time > HAPTIC_TIMEOUT_S:
            stop_all_motors()
        time.sleep(0.05)


def mode_heartbeat():
    while running:
        try:
            send_mode(mode_sock)
        except Exception:
            pass
        time.sleep(1.0)


mode_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
button.when_pressed = on_button_pressed

listener_thread = threading.Thread(target=haptic_listener, daemon=True)
watchdog_thread = threading.Thread(target=watchdog_loop, daemon=True)
heartbeat_thread = threading.Thread(target=mode_heartbeat, daemon=True)

listener_thread.start()
watchdog_thread.start()
heartbeat_thread.start()

send_mode(mode_sock)
print(f"Started. Current mode = {current_mode}")
print("Motors =", MOTOR_PINS)
print("Button =", BUTTON_PIN)

try:
    while True:
        time.sleep(0.2)
except KeyboardInterrupt:
    pass
finally:
    running = False
    stop_all_motors()
    for motor in motors:
        motor.close()
    mode_sock.close()
