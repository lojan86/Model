from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, uuid, socket, base64, threading
import requests
from datetime import datetime
from typing import List

APP_PORT = 8000
# !! Change this to your Node.js backend IP:port !!
NODE_BACKEND_URL = "http://10.153.9.168:3000"
TRAP_ID_FILE = "/home/nexus/trap_id.txt"

# ── New: image storage & event store ────────────────────────────
IMAGES_DIR = "/home/nexus/trap_images"
MAX_EVENTS  = 100
os.makedirs(IMAGES_DIR, exist_ok=True)

trap_events: List[dict] = []   # in-memory, newest first
# ────────────────────────────────────────────────────────────────

app = FastAPI(title="Trap Device API")

# Allow Flutter app on local network
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve captured images statically
# Flutter loads them via: http://<pi-ip>:8000/images/<filename>
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

# ===============================
# Trap Activation State
# ===============================
trap_active = False

# ===============================
# Utilities
# ===============================
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "0.0.0.0"

def load_or_create_trap_id() -> str:
    if os.path.exists(TRAP_ID_FILE):
        with open(TRAP_ID_FILE, "r") as f:
            tid = f.read().strip()
            if tid:
                return tid
    tid = "TRAP-" + uuid.uuid4().hex[:6].upper()
    with open(TRAP_ID_FILE, "w") as f:
        f.write(tid)
    return tid

TRAP_ID = load_or_create_trap_id()

# ===============================
# Existing Endpoints (UNCHANGED)
# ===============================
@app.get("/health")
def health():
    return {"ok": True, "trap_id": TRAP_ID}

@app.get("/device/info")
def device_info():
    ip = get_local_ip()
    return JSONResponse({
        "trap_id":      TRAP_ID,
        "device":       "Raspberry Pi 4",
        "api_version":  "1.0",
        "server_time":  datetime.now().isoformat(),
        "local_ip":     ip,
        "rtsp_url":     f"rtsp://{ip}:8554/live"
    })

class TrapControl(BaseModel):
    status: str  # "active" or "inactive"

@app.post("/device/set-status")
def set_status(data: TrapControl):
    global trap_active
    if data.status == "active":
        trap_active = True
        print("Trap ACTIVATED")
        # 👉 TODO: GPIO HIGH here (servo enable / system start)
    else:
        trap_active = False
        print("Trap DEACTIVATED")
        # 👉 TODO: GPIO LOW here (servo disable / system stop)
    return {
        "message": "Trap status updated",
        "trap_id": TRAP_ID,
        "active":  trap_active
    }

@app.get("/device/status")
def get_status():
    return {
        "trap_id": TRAP_ID,
        "active":  trap_active
    }

# ===============================
# NEW: Trap Event Endpoints
# ===============================

class TrapEventIn(BaseModel):
    """Payload posted by trap_detector.py when an animal is captured."""
    animal_name:    str
    confidence:     float
    captured_at:    str    # ISO datetime e.g. "2026-02-22T11:07:31"
    image_base64:   str    # full JPEG encoded as base64
    image_filename: str

@app.post("/api/trap-event", status_code=200)
def receive_trap_event(event: TrapEventIn):
    """
    Called by trap_detector.py immediately after:
      1. Image is saved locally on the Pi
      2. Servo closes the trap door
    Steps:
      a. Decode + save base64 image to disk
      b. Store event in local memory (for RTSP viewer overlay)
      c. Forward full event to Node.js backend (persists to SQLite + notifies Flutter)
    """
    # a. Decode and save image to disk
    try:
        image_bytes = base64.b64decode(event.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    safe_name  = event.image_filename.replace("/", "_").replace("\\", "_")
    image_path = os.path.join(IMAGES_DIR, safe_name)
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    event_id  = str(uuid.uuid4())
    received  = datetime.now().isoformat()
    # image_url is the Pi-local static path — Flutter can load via Pi IP
    image_url = f"/images/{safe_name}"

    # b. Store in local memory
    record = {
        "id":           event_id,
        "trap_id":      TRAP_ID,
        "animal_name":  event.animal_name,
        "confidence":   round(event.confidence, 4),
        "captured_at":  event.captured_at,
        "image_url":    image_url,
        "received_at":  received,
    }
    trap_events.insert(0, record)
    if len(trap_events) > MAX_EVENTS:
        trap_events.pop()

    # c. Forward to Node.js backend in background thread
    #    Node.js saves to SQLite + logs notification for Flutter to fetch
    def _forward_to_node():
        try:
            node_payload = {
                "trap_id":      TRAP_ID,
                "animal_name":  event.animal_name,
                "confidence":   round(event.confidence, 4),
                "captured_at":  event.captured_at,
                "image_url":    image_url,    # Pi-hosted static URL
            }
            resp = requests.post(
                f"{NODE_BACKEND_URL}/api/events/trap-event",
                json=node_payload,
                timeout=8
            )
            if resp.status_code == 200:
                print(f"[NODE] ✓ Event forwarded — {event.animal_name} ({event.confidence:.2f})")
            else:
                print(f"[NODE] ✗ Backend returned {resp.status_code}: {resp.text}")
        except requests.exceptions.ConnectionError:
            print(f"[NODE] ✗ Cannot reach Node.js backend at {NODE_BACKEND_URL}")
        except Exception as e:
            print(f"[NODE] ✗ Forward error: {e}")

    threading.Thread(target=_forward_to_node, daemon=True).start()

    print(f"[EVENT] {event.animal_name} ({event.confidence:.2f}) @ {event.captured_at} — saved {safe_name}")
    return record


@app.get("/api/trap-events")
def get_all_events(limit: int = 50):
    """
    Flutter polls this to show the capture history list.
    GET /api/trap-events?limit=20
    """
    return trap_events[:limit]


@app.get("/api/trap-events/{event_id}")
def get_event(event_id: str):
    """Get a single event by ID for the Flutter detail screen."""
    for e in trap_events:
        if e["id"] == event_id:
            return e
    raise HTTPException(status_code=404, detail="Event not found")


@app.delete("/api/trap-events", status_code=200)
def clear_events():
    """Clear all events — useful for testing / resetting from the app."""
    trap_events.clear()
    return {"message": "All events cleared", "trap_id": TRAP_ID}
