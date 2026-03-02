import os, time, cv2, threading, subprocess, requests, base64
import numpy as np
from datetime import datetime
from queue import Queue, Empty
from picamera2 import Picamera2

# ===================== CONFIG =====================
MODEL_PATH = "best_nano.onnx"

CLASS_NAMES = ["giant_squirrel", "monkey", "wild_boar"]
TARGET_CLASSES = {"monkey", "wild_boar", "giant_squirrel"}

MODEL_SIZE = 640

INFER_CONF = 0.25
TRIGGER_CONF = 0.75

MOTION_RESIZE_W = 320
MOTION_THRESH = 18
MIN_AREA = 900

INFER_INTERVAL_SEC = 0.25
TRIGGER_COOLDOWN_SEC = 2.0
TRIGGER_TEXT_SEC = 2.0

SAVE_DIR = "captures"
os.makedirs(SAVE_DIR, exist_ok=True)

WINDOW_NAME = "PiCam + Motion + YOLO (q to quit)"

# RTSP stream settings
RTSP_URL = "rtsp://127.0.0.1:8554/live"
STREAM_WIDTH = 640
STREAM_HEIGHT = 480
STREAM_FPS = 15

# ── Servo settings ──────────────────────────────────────────────
SERVO_GPIO_PIN    = 18   # GPIO 18 = physical pin 12
SERVO_IDLE_ANGLE  = 180  # degrees — resting position (door open / ready, opposite side)
SERVO_CLOSE_ANGLE = 90   # degrees — trigger position (rotates 90° from idle to close)
# Duty cycle tuning: most SG90/MG996R servos work with 2.5% (0°) to 12.5% (180°)
# If your servo only moves partially, adjust SERVO_DUTY_MIN/MAX inside _angle_to_duty()

# ── API settings ─────────────────────────────────────────────────
# !! Change this to your FastAPI server's local IP address !!
API_BASE_URL = "http://0.0.0.0:8000"
API_ENDPOINT = f"{API_BASE_URL}/api/trap-event"
# ==================================================

# ---------- SERVO SETUP (RPi.GPIO — pre-installed on Raspberry Pi OS) ----------
try:
    import RPi.GPIO as GPIO

    # Use BCM numbering so GPIO 18 = physical pin 12 (your orange wire)
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(SERVO_GPIO_PIN, GPIO.OUT)

    # 50 Hz is standard for hobby servos
    _pwm = GPIO.PWM(SERVO_GPIO_PIN, 50)
    _pwm.start(0)   # start with signal off

    def _angle_to_duty(angle):
        """
        Convert 0-180° to duty cycle for 50 Hz PWM.
        Most SG90/MG996R servos: 0° = 2.5%, 180° = 12.5%
        Tweak SERVO_DUTY_MIN / SERVO_DUTY_MAX if movement is partial.
        """
        SERVO_DUTY_MIN = 2.5   # duty % for 0°
        SERVO_DUTY_MAX = 12.5  # duty % for 180°
        angle = max(0, min(180, angle))
        return SERVO_DUTY_MIN + (angle / 180.0) * (SERVO_DUTY_MAX - SERVO_DUTY_MIN)

    def servo_set_angle(angle):
        _pwm.ChangeDutyCycle(_angle_to_duty(angle))
        time.sleep(0.6)          # wait for servo to physically reach position
        _pwm.ChangeDutyCycle(0)  # stop signal to prevent jitter

    def servo_trigger():
        """
        One full trigger cycle — loops on every detection:
          1. Rotate 0° → 90°  (door closes)
          2. Immediately reverse 90° → 0°  (door opens, ready for next)
        Each detection fires this same sequence.
        """
        print(f"[SERVO] Trigger: rotating to {SERVO_CLOSE_ANGLE}°")
        servo_set_angle(SERVO_CLOSE_ANGLE)   # 0° → 90°
        print(f"[SERVO] Reversing back to {SERVO_IDLE_ANGLE}°")
        servo_set_angle(SERVO_IDLE_ANGLE)    # 90° → 0°  (back to idle, ready for next)
        print("[SERVO] Cycle complete — ready for next detection")

    def servo_open_door():
        """Reset servo to idle (0°) — called on exit."""
        print(f"[SERVO] Reset to idle {SERVO_IDLE_ANGLE}°")
        servo_set_angle(SERVO_IDLE_ANGLE)

    # Initialise: make sure servo starts at idle (0°)
    servo_set_angle(SERVO_IDLE_ANGLE)
    print(f"[SERVO] Ready on GPIO {SERVO_GPIO_PIN} (pin 12) — idle at {SERVO_IDLE_ANGLE}°")
    SERVO_AVAILABLE = True

except Exception as e:
    print(f"[SERVO] WARNING: {e}")
    print("[SERVO] Continuing WITHOUT servo — detection still works.")
    SERVO_AVAILABLE = False
    def servo_trigger():    pass
    def servo_open_door():  pass

# ---------- LOAD ONNX MODEL ----------
net = cv2.dnn.readNetFromONNX(MODEL_PATH)

# ---------- CAMERA ----------
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(1.0)

# ---------- FFMPEG → RTSP STREAM ----------
ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-f", "rawvideo", "-vcodec", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", f"{STREAM_WIDTH}x{STREAM_HEIGHT}",
    "-r", str(STREAM_FPS),
    "-i", "-",
    "-vcodec", "libx264",
    "-pix_fmt", "yuv420p",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-profile:v", "baseline",
    "-b:v", "800k",
    "-f", "rtsp",
    "-rtsp_transport", "tcp",
    RTSP_URL
]
ffmpeg_proc = subprocess.Popen(
    ffmpeg_cmd, stdin=subprocess.PIPE,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)
print(f"[RTSP] Streaming to {RTSP_URL}")

stream_q = Queue(maxsize=2)

def stream_writer():
    while True:
        frame = stream_q.get()
        if frame is None:
            break
        try:
            if frame.shape[1] != STREAM_WIDTH or frame.shape[0] != STREAM_HEIGHT:
                frame = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT))
            ffmpeg_proc.stdin.write(frame.tobytes())
            ffmpeg_proc.stdin.flush()
        except Exception as e:
            print(f"[RTSP] Write error: {e}")
            break

stream_thread = threading.Thread(target=stream_writer, daemon=True)
stream_thread.start()

# ---------- SEND EVENT TO FLUTTER APP ----------
def send_capture_to_app(animal_name, confidence, capture_time_iso, image_path):
    """
    POST trap event to FastAPI server in a background thread.
    Payload includes animal name, confidence, datetime, and JPEG image as base64.
    The Flutter app fetches this from the server.
    """
    def _send():
        try:
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")

            payload = {
                "animal_name":     animal_name,
                "confidence":      round(float(confidence), 4),
                "captured_at":     capture_time_iso,      # e.g. "2026-02-22T11:07:31"
                "image_base64":    image_b64,              # full JPEG encoded as base64
                "image_filename":  os.path.basename(image_path)
            }

            response = requests.post(API_ENDPOINT, json=payload, timeout=10)

            if response.status_code == 200:
                print(f"[API] ✓ Event sent — {animal_name} ({confidence:.2f})")
            else:
                print(f"[API] Server error {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            print(f"[API] ✗ Cannot reach {API_BASE_URL} — check server IP/port in CONFIG.")
        except Exception as e:
            print(f"[API] ✗ Unexpected error: {e}")

    # Run in background thread — never blocks the detection loop
    threading.Thread(target=_send, daemon=True).start()


# ---------- LETTERBOX ----------
def letterbox(image, new_size):
    h, w = image.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (nw, nh))
    canvas = np.zeros((new_size, new_size, 3), dtype=np.uint8)
    top  = (new_size - nh) // 2
    left = (new_size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas, scale, left, top

# ---------- YOLO ONNX DETECT ----------
def yolo_onnx_detect(frame_bgr):
    img, scale, left, top = letterbox(frame_bgr, MODEL_SIZE)
    blob = cv2.dnn.blobFromImage(
        img, 1/255.0, (MODEL_SIZE, MODEL_SIZE), swapRB=True, crop=False
    )
    net.setInput(blob)
    out = net.forward()
    if isinstance(out, (list, tuple)):
        out = out[0]

    preds = out
    if preds.ndim == 3:
        preds = preds[0]
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T

    boxes, scores, class_ids = [], [], []
    nc = len(CLASS_NAMES)
    num_cols = preds.shape[1]
    H, W = frame_bgr.shape[:2]

    for p in preds:
        x, y, w, h = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        if num_cols == 4 + nc:
            obj = 1.0
            cls_scores = p[4:]
        else:
            obj = float(p[4])
            cls_scores = p[5:]

        cls_id   = int(np.argmax(cls_scores))
        cls_conf = float(cls_scores[cls_id])
        conf     = obj * cls_conf
        if conf < INFER_CONF:
            continue

        x1 = (x - w/2 - left) / scale;  y1 = (y - h/2 - top)  / scale
        x2 = (x + w/2 - left) / scale;  y2 = (y + h/2 - top)  / scale
        x1 = max(0, min(W-1, x1));       y1 = max(0, min(H-1, y1))
        x2 = max(0, min(W-1, x2));       y2 = max(0, min(H-1, y2))

        boxes.append([x1, y1, x2-x1, y2-y1])
        scores.append(conf)
        class_ids.append(cls_id)

    idxs = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=INFER_CONF, nms_threshold=0.45)

    dets = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            dets.append((CLASS_NAMES[class_ids[i]], float(scores[i]), boxes[i]))
    return dets

def draw_dets(frame, dets):
    for name, conf, box in dets:
        x, y, w, h = map(int, box)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x, max(20, y-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# ---------- FAST MOTION ----------
prev_small = None
def motion_detect_fast(frame_bgr):
    global prev_small
    h, w = frame_bgr.shape[:2]
    sc   = MOTION_RESIZE_W / float(w)
    small = cv2.resize(frame_bgr, (MOTION_RESIZE_W, int(h * sc)))
    gray  = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (11, 11), 0)

    if prev_small is None:
        prev_small = gray
        return False

    diff = cv2.absdiff(prev_small, gray)
    prev_small = gray

    _, th = cv2.threshold(diff, MOTION_THRESH, 255, cv2.THRESH_BINARY)
    th = cv2.dilate(th, None, iterations=2)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return any(cv2.contourArea(c) > MIN_AREA for c in cnts)

# ---------- INFERENCE THREAD ----------
frame_q  = Queue(maxsize=1)
result_q = Queue(maxsize=1)

def infer_worker():
    while True:
        item = frame_q.get()
        if item is None:
            break
        try:
            dets = yolo_onnx_detect(item)
            while not result_q.empty():
                try: result_q.get_nowait()
                except: break
            result_q.put(dets)
        except Exception as e:
            print("Inference error:", e)

threading.Thread(target=infer_worker, daemon=True).start()

# ---------- MAIN LOOP ----------
last_dets  = []
last_status = "READY"
last_infer_time = last_save_time = 0.0
trigger_show_until = 0.0
trigger_msg = ""
fps_t0 = time.time();  fps_count = 0;  fps = 0.0
last_stream_time = 0.0
STREAM_INTERVAL  = 1.0 / STREAM_FPS

print("Running... press 'q' to quit.")
try:
    while True:
        frame = picam2.capture_array()   # BGR already

        fps_count += 1
        if time.time() - fps_t0 >= 1.0:
            fps = fps_count / (time.time() - fps_t0)
            fps_t0 = time.time();  fps_count = 0

        try:
            last_dets = result_q.get_nowait()
        except Empty:
            pass

        moved = motion_detect_fast(frame)
        now   = time.time()

        # Motion → send to YOLO
        if moved and (now - last_infer_time) >= INFER_INTERVAL_SEC:
            if frame_q.full():
                try: frame_q.get_nowait()
                except: pass
            frame_q.put(frame.copy())
            last_infer_time = now
            last_status = "MOTION -> YOLO RUN"

        # ─── TRIGGER ────────────────────────────────────────────────
        best = None
        for name, conf, box in last_dets:
            if name in TARGET_CLASSES and conf >= TRIGGER_CONF:
                if best is None or conf > best[1]:
                    best = (name, conf)

        if best and (now - last_save_time) >= TRIGGER_COOLDOWN_SEC:
            animal, conf = best
            capture_dt   = datetime.now()
            ts           = capture_dt.strftime("%Y-%m-%d_%H-%M-%S")
            iso_time     = capture_dt.isoformat()
            out_path     = os.path.join(SAVE_DIR, f"{ts}_{animal}_{conf:.2f}.jpg")

            # Step 1 — save captured image immediately
            cv2.imwrite(out_path, frame)
            print(f"[TRIGGER] {animal} ({conf:.2f}) detected — image saved: {out_path}")

            # Step 2 — fire servo trigger cycle in background thread
            # Rotates 0°→90° then immediately reverses 90°→0°, ready for next detection
            # Runs in background so it never blocks detection loop or API call
            threading.Thread(target=servo_trigger, daemon=True).start()

            # Step 3 — send full capture details to Flutter app via FastAPI
            # animal_name, confidence, ISO datetime, image path — all sent as JSON + base64 image
            send_capture_to_app(animal, conf, iso_time, out_path)

            trigger_msg        = f"TRIGGERED: {animal} ({conf:.2f}) — servo cycling"
            trigger_show_until = now + TRIGGER_TEXT_SEC
            last_status        = f"DOOR CLOSED | {animal} {conf:.2f}"
            last_save_time     = now
            print(f"[TRIGGER] Servo fired + API notified — {animal} @ {iso_time}")

        # ─── Display ────────────────────────────────────────────────
        display = frame.copy()
        draw_dets(display, last_dets)

        cv2.putText(display, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(display, f"MOTION: {'YES' if moved else 'NO'} | {last_status}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        if now < trigger_show_until:
            cv2.putText(display, trigger_msg, (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # Push to RTSP stream (throttled)
        if now - last_stream_time >= STREAM_INTERVAL:
            if not stream_q.full():
                stream_q.put_nowait(display.copy())
            last_stream_time = now

        cv2.imshow(WINDOW_NAME, display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    frame_q.put(None)
    stream_q.put(None)
    stream_thread.join(timeout=2)
    try:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait(timeout=3)
    except:
        ffmpeg_proc.kill()
    if SERVO_AVAILABLE:
        servo_open_door()          # reset door to open position on exit
        _pwm.stop()                # stop PWM signal
        GPIO.cleanup()             # release GPIO pins
    cv2.destroyAllWindows()
    picam2.stop()
    print("Done.")
