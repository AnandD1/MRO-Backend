from flask import Flask, jsonify, render_template_string, send_file, Response, session
import asyncio
import websockets
import logging
import io
import socket
import threading
from datetime import datetime
import time
import json
import os
from dotenv import load_dotenv

# Add blueprint imports
from auth_routes import bp_auth
from scan_routes import bp_scan


# ----------------------------
# Config
# ----------------------------
HTTP_HOST = "0.0.0.0"
HTTP_PORT = 5000

WS_HOST = "0.0.0.0"
WS_PORT = 8765                   # inbound WS from Unity (raw frames)
WS_MAX_SIZE = 25 * 1024 * 1024   # 25 MB
WS_PING_INTERVAL = None          # minimal chatter
WS_CLOSE_TIMEOUT = 1

# --- NEW: outbound WS (AI → Unity) ---
WS_OUT_PORT = 8766               # outbound WS to Unity (annotated frames)
AI_OUT_FPS  = 30                 # push cadence to Unity

# --- NEW: outbound JSON WS (AI meta → Unity) ---
WS_JSON_OUT_PORT = 8767          # detection metadata JSON

# --- AI config knobs ---
AI_DEVICE = "cuda:0"    # use "cuda:0" if available; falls back to CPU automatically
AI_FP16   = True        # half precision on GPU
AI_IMGSZ  = 640         # inference size
AI_CONF   = 0.25        # confidence threshold
AI_EVERY_N = 3          # run AI on 1 out of every N frames (e.g., 3 => ~1/3)

# --- Orientation fixes ---
ROTATE_180 = True               # rotate incoming camera image by 180°
FLIP_X     = True               # horizontal mirror fix (left–right flip)

# ----------------------------
# Flask app & logging
# ----------------------------
# Load env
load_dotenv()
app = Flask(__name__)

# === Simple Session Configuration (Replaced JWT) ===
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET", "dev-secret-change-me")

# Simple in-memory logged-in users store
logged_in_users = set()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")
logger.setLevel(logging.INFO)

# ----------------------------
# Shared state (latest raw frame only)
# ----------------------------
latest_unity_jpeg: bytes | None = None
latest_unity_timestamp_iso: str | None = None
frames_in = 0
_ws_clients = 0
_lock = threading.Lock()
last_frame_len = 0
last_frame_is_jpeg = False

# ----------------------------
# YOLO worker (latest-only processing)
# ----------------------------
import cv2
import numpy as np
from ultralytics import YOLO
import torch

# Safe torch speedups
torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)

# AI state
_ai_lock = threading.Lock()         # protects AI buffers/counters
_ai_event = threading.Event()       # signals a new frame is pending
_ai_pending_jpeg: bytes | None = None
latest_ai_jpeg: bytes | None = None
ai_frames_out = 0
ai_last_infer_ms = 0.0
_model: YOLO | None = None

# --- carry timestamps through AI pipeline & out socket ---
_ai_pending_ts_ms: int | None = None
latest_ai_ts_ms: int | None = None

# --- NEW: JSON meta state ---
_ai_json_clients = set()            # used only inside JSON WS loop
latest_ai_json = None               # set by AI worker under _ai_lock

# ============================
# Minimal scratch tracker (no durations, no HUD)
# ============================
def _iou_xyxy(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (a[2]-a[0])) * max(0.0, (a[3]-a[1]))
    area_b = max(0.0, (b[2]-b[0])) * max(0.0, (b[3]-b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0

def _center_xyxy(bb):
    return ((bb[0]+bb[2])*0.5, (bb[1]+bb[3])*0.5)

class ScratchTracker:
    def __init__(self,
                 max_age_ms=8000,         # keep tracks this long after last seen (off-screen persistence)
                 base_gate_px=60,         # optimize for slow camera motion
                 px_per_s_gate=220,       # small growth with dt to tolerate slow nets / jitter
                 iou_thresh=0.1):
        self.max_age_ms = max_age_ms
        self.base_gate_px = base_gate_px
        self.px_per_s_gate = px_per_s_gate
        self.iou_thresh = iou_thresh

        self.next_id = 1
        self.tracks = {}   # id -> dict(bbox, cx, cy, last_ms, counted)
        self.total_unique = 0
        self._last_update_ms = None

    def _gate_px(self, dt_ms):
        return self.base_gate_px + (self.px_per_s_gate * (max(0, dt_ms) / 1000.0))

    def _prune(self, now_ms):
        to_del = []
        for tid, t in self.tracks.items():
            if now_ms - t["last_ms"] > self.max_age_ms:
                to_del.append(tid)
        for tid in to_del:
            del self.tracks[tid]

    def update(self, dets_xyxy, now_ms):
        if not isinstance(dets_xyxy, np.ndarray):
            dets_xyxy = np.array(dets_xyxy, dtype=np.float32).reshape(-1, 4)
        else:
            dets_xyxy = dets_xyxy.astype(np.float32)

        dt_ms = 33 if self._last_update_ms is None else (now_ms - self._last_update_ms)
        self._last_update_ms = now_ms
        gate = self._gate_px(dt_ms)

        unmatched_track_ids = set(self.tracks.keys())
        assignments = []  # (tid, det_idx)

        # greedy match with distance + IoU gating
        for di, db in enumerate(dets_xyxy):
            dcx, dcy = _center_xyxy(db)
            best = None
            best_cost = 1e9
            for tid in list(unmatched_track_ids):
                tb = self.tracks[tid]["bbox"]
                tcx, tcy = self.tracks[tid]["cx"], self.tracks[tid]["cy"]
                dist = ((dcx - tcx)**2 + (dcy - tcy)**2) ** 0.5
                if dist > gate:
                    continue
                iou = _iou_xyxy(db, tb)
                if iou < self.iou_thresh:
                    continue
                cost = dist - (iou * 10.0)
                if cost < best_cost:
                    best_cost = cost
                    best = tid
            if best is not None:
                assignments.append((best, di))
                unmatched_track_ids.discard(best)

        # new tracks for unmatched detections
        matched_det_idxs = set([di for _, di in assignments])
        for di, db in enumerate(dets_xyxy):
            if di in matched_det_idxs:
                continue
            dcx, dcy = _center_xyxy(db)
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                "bbox": db.copy(),
                "cx": dcx, "cy": dcy,
                "last_ms": now_ms,
                "counted": False
            }
            self.total_unique += 1
            assignments.append((tid, di))

        # update matched tracks
        active_out = []
        for tid, di in assignments:
            db = dets_xyxy[di]
            dcx, dcy = _center_xyxy(db)
            t = self.tracks[tid]
            t["bbox"] = db.copy()
            t["cx"] = dcx; t["cy"] = dcy
            t["last_ms"] = now_ms
            if not t["counted"]:
                t["counted"] = True  # counted once
            active_out.append({"tid": tid, "bbox": db.copy()})

        # keep unmatched for grace period (off-screen)
        self._prune(now_ms)

        return active_out

# ============================

def _ai_worker():
    """
    Background thread:
    - Wait for a signal that a new frame is pending (_ai_event).
    - Always process the *latest* pending JPEG (drop intermediate frames).
    - Save annotated JPEG into latest_ai_jpeg (+ timestamp).
    """
    global latest_ai_jpeg, ai_frames_out, ai_last_infer_ms, _model, _ai_pending_jpeg
    global latest_ai_ts_ms, _ai_pending_ts_ms, latest_ai_json  # <-- added

    # Load once (GPU if available; otherwise CPU).
    try:
        dev = AI_DEVICE if (AI_DEVICE.startswith("cuda") and torch.cuda.is_available()) else "cpu"
        _model = YOLO("best.pt")
        if dev != "cpu":
            _model.to(dev)

        # small speedup & deterministic first-run
        try:
            _model.fuse()
        except Exception:
            pass

        # warm up the model so first real frame is fast
        try:
            _model.warmup(imgsz=(1, 3, AI_IMGSZ, AI_IMGSZ))
        except Exception:
            dummy = np.zeros((AI_IMGSZ, AI_IMGSZ, 3), dtype=np.uint8)
            _ = _model.predict(
                dummy,
                device=(0 if dev != "cpu" else "cpu"),
                imgsz=AI_IMGSZ,
                conf=AI_CONF,
                half=(AI_FP16 and dev != "cpu"),
                verbose=False
            )
        if dev != "cpu":
            torch.cuda.synchronize()

        logger.info(f"✅ YOLO model loaded (best.pt) on {dev}, fp16={AI_FP16 and dev!='cpu'} (warmed up)")

    except Exception as e:
        logger.exception(f"Failed to load YOLO model: {e}")
        return  # stop worker if model can't load

    # Precompute scratch class filter once (if names are available)
    scratch_class_ids = None
    try:
        names = getattr(_model, "names", None)
        if isinstance(names, dict):
            ids = [i for i, n in names.items() if "scratch" in str(n).lower()]
            scratch_class_ids = set(ids) if ids else None
        elif isinstance(names, (list, tuple)):
            ids = [i for i, n in enumerate(names) if "scratch" in str(n).lower()]
            scratch_class_ids = set(ids) if ids else None
    except Exception:
        scratch_class_ids = None  # accept all if anything odd

    tracker = ScratchTracker(  # optimized for slow camera & net
        max_age_ms=8000,
        base_gate_px=60,
        px_per_s_gate=220,
        iou_thresh=0.1
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        _ai_event.wait()
        _ai_event.clear()

        # Get the latest pending JPEG atomically (drop backlog)
        with _ai_lock:
            buf = _ai_pending_jpeg
            ts_src = _ai_pending_ts_ms
            _ai_pending_jpeg = None
            _ai_pending_ts_ms = None

        if not buf:
            continue

        t0 = time.perf_counter()
        try:
            arr = np.frombuffer(buf, np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue

            # --- Orientation fixes before inference ---
            if ROTATE_180:
                img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_180)
            if FLIP_X:
                img_bgr = cv2.flip(img_bgr, 1)  # horizontal flip (mirror fix)

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Inference (GPU if available, FP16 if GPU)
            with torch.inference_mode():
                results = _model.predict(
                    img_rgb,
                    device=(0 if dev != "cpu" else "cpu"),
                    imgsz=AI_IMGSZ,
                    conf=AI_CONF,
                    half=(AI_FP16 and dev != "cpu"),
                    verbose=False
                )
            res = results[0]

            # --------- Extract "scratch" detections ----------
            det_xyxy = np.empty((0, 4), dtype=np.float32)
            if hasattr(res, "boxes") and res.boxes is not None and res.boxes.xyxy is not None:
                xyxy = res.boxes.xyxy
                cls  = getattr(res.boxes, "cls", None)
                try:
                    xyxy = xyxy.detach().cpu().numpy().astype(np.float32)
                except Exception:
                    xyxy = np.array(xyxy, dtype=np.float32)

                if scratch_class_ids is not None and cls is not None:
                    try:
                        cls_np = cls.detach().cpu().numpy().astype(int)
                        mask = np.array([c in scratch_class_ids for c in cls_np], dtype=bool)
                    except Exception:
                        mask = np.ones((xyxy.shape[0],), dtype=bool)
                else:
                    mask = np.ones((xyxy.shape[0],), dtype=bool)

                det_xyxy = xyxy[mask] if xyxy.size else np.empty((0, 4), dtype=np.float32)

            now_ms = ts_src or int(time.time() * 1000)

            # Update tracker (no durations returned)
            active_tracks = tracker.update(det_xyxy, now_ms)

            # --------- Minimal annotated view: just boxes + ID ----------
            annotated_bgr = img_bgr.copy()
            for tr in active_tracks:
                x1, y1, x2, y2 = tr["bbox"].astype(int)
                tid = tr["tid"]
                cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.putText(annotated_bgr, f"ID#{tid}", (x1, max(20, y1 - 8)),
                            font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # --- NEW: Build minimal detection JSON (post-rotation/flip) ---
            h, w = img_bgr.shape[:2]
            det_list = []
            for tr in active_tracks:
                x1, y1, x2, y2 = tr["bbox"].tolist()
                det_list.append({
                    "tid": int(tr["tid"]),
                    "xyxy": [float(x1), float(y1), float(x2), float(y2)]
                })
            msg = {
                "ts_ms": int(now_ms),
                "w": int(w),
                "h": int(h),
                "detections": det_list
            }
            with _ai_lock:
                latest_ai_json = msg  # global set under lock

            # Encode back to JPEG
            ok2, out = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok2:
                continue

            with _ai_lock:
                latest_ai_jpeg = out.tobytes()
                ai_frames_out += 1
                ai_last_infer_ms = (time.perf_counter() - t0) * 1000.0
                latest_ai_ts_ms = now_ms

        except Exception as e:
            logger.exception(f"AI worker error: {e}")
            # continue loop to process the next incoming frame

def start_ai_worker():
    t = threading.Thread(target=_ai_worker, daemon=True)
    t.start()
    logger.info("AI worker thread started")

# ----------------------------
# Helpers
# ----------------------------
def get_local_ip() -> str:
    """Best-effort LAN IP to show the correct URLs in logs."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        try:
            s.close()
        except Exception:
            pass
    return ip

# ----------------------------
# WebSocket handler (inbound Unity -> backend)
# Message: [8-byte little-endian int64 timestamp(ms)] + [JPEG bytes]
# ----------------------------
async def unity_websocket_handler(websocket):
    global latest_unity_jpeg, latest_unity_timestamp_iso, frames_in, _ws_clients
    global last_frame_len, last_frame_is_jpeg, _ai_pending_jpeg, _ai_pending_ts_ms

    _ws_clients += 1
    path = getattr(websocket, "path", "/")
    logger.info(f"WS client connected: {websocket.remote_address} path={path}")

    try:
        async for message in websocket:
            if not isinstance(message, (bytes, bytearray)) or len(message) < 8:
                continue

            ts_ms = int.from_bytes(message[:8], "little", signed=True)
            jpeg_bytes = message[8:]

            # quick signature check: 0xFFD8 ... 0xFFD9
            is_jpeg = len(jpeg_bytes) >= 4 and jpeg_bytes[0] == 0xFF and jpeg_bytes[1] == 0xD8 \
                      and jpeg_bytes[-2] == 0xFF and jpeg_bytes[-1] == 0xD9

            with _lock:
                latest_unity_jpeg = bytes(jpeg_bytes)
                last_frame_len = len(jpeg_bytes)
                last_frame_is_jpeg = bool(is_jpeg)
                latest_unity_timestamp_iso = datetime.utcfromtimestamp(ts_ms / 1000.0).isoformat() + "Z"
                frames_in += 1

            # Hand the latest JPEG to the AI worker at 1-in-N cadence (latest-only)
            if frames_in % AI_EVERY_N == 0:
                with _ai_lock:
                    _ai_pending_jpeg = jpeg_bytes
                    _ai_pending_ts_ms = ts_ms
                    _ai_event.set()

            # log occasionally
            if frames_in % 60 == 0:
                logger.info(f"Frames: {frames_in}, size={last_frame_len}B, jpeg={last_frame_is_jpeg}")

    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"WS closed: {e.code} {e.reason}")
    except Exception as e:
        logger.exception(f"WS handler error: {e}")
    finally:
        _ws_clients -= 1

async def run_websocket_server():
    """Start the inbound WebSocket server and wait until closed."""
    logger.info(f"Starting WebSocket server on {WS_HOST}:{WS_PORT}")
    server = await websockets.serve(
        unity_websocket_handler,
        WS_HOST,
        WS_PORT,
        max_size=WS_MAX_SIZE,
        ping_interval=WS_PING_INTERVAL,
        close_timeout=WS_CLOSE_TIMEOUT,
    )
    logger.info("✅ WebSocket server is listening")
    await server.wait_closed()

def start_websocket_server():
    """Run the inbound WS server in a background thread with its own event loop."""
    def _runner():
        try:
            asyncio.run(run_websocket_server())
        except Exception as e:
            logger.exception(f"WebSocket server crashed: {e}")

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    logger.info("WS server thread started")
    return t

# ----------------------------
# NEW: AI-out WebSocket (backend -> Unity)
# Broadcast latest annotated frame at AI_OUT_FPS
# ----------------------------
_ai_clients = set()  # only used inside the out server's event loop

async def ai_ws_handler(websocket):
    _ai_clients.add(websocket)
    logger.info(f"AI-WS client connected: {getattr(websocket, 'remote_address', '?')}")
    try:
        await websocket.wait_closed()
    finally:
        _ai_clients.discard(websocket)
        logger.info("AI-WS client disconnected")

async def ai_broadcast_loop():
    interval = 1.0 / max(1, AI_OUT_FPS)
    while True:
        await asyncio.sleep(interval)

        # snapshot buffers under lock
        with _ai_lock:
            ai_buf = latest_ai_jpeg
            ts_ms  = latest_ai_ts_ms

        if not ai_buf or not ts_ms:
            continue

        header = int(ts_ms).to_bytes(8, "little", signed=True)
        packet = header + ai_buf

        if not _ai_clients:
            continue

        dead = []
        for ws in list(_ai_clients):
            try:
                await ws.send(packet)
            except Exception:
                dead.append(ws)
        for ws in dead:
            _ai_clients.discard(ws)

async def run_websocket_server_out():
    """Start the AI-out WebSocket server and its broadcaster task."""
    logger.info(f"Starting AI-out WebSocket on {WS_HOST}:{WS_OUT_PORT}")
    server = await websockets.serve(
        ai_ws_handler,
        WS_HOST,
        WS_OUT_PORT,
        max_size=WS_MAX_SIZE,
        ping_interval=WS_PING_INTERVAL,
        close_timeout=WS_CLOSE_TIMEOUT,
    )
    logger.info("✅ AI-out WebSocket listening")
    asyncio.create_task(ai_broadcast_loop())
    await server.wait_closed()

def start_websocket_server_out():
    def _runner():
        try:
            asyncio.run(run_websocket_server_out())
        except Exception as e:
            logger.exception(f"AI-out WebSocket crashed: {e}")
    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    logger.info("AI-out WS server thread started")
    return t

# ----------------------------
# NEW: AI-JSON WebSocket (backend -> Unity)
# Broadcast latest detection metadata JSON at AI_OUT_FPS
# ----------------------------
async def ai_json_ws_handler(websocket):
    _ai_json_clients.add(websocket)
    logger.info(f"AI-JSON WS connected: {getattr(websocket, 'remote_address', '?')}")
    try:
        await websocket.wait_closed()
    finally:
        _ai_json_clients.discard(websocket)
        logger.info("AI-JSON WS disconnected")

async def ai_json_broadcast_loop():
    interval = 1.0 / max(1, AI_OUT_FPS)
    while True:
        await asyncio.sleep(interval)
        if not _ai_json_clients:
            continue
        with _ai_lock:
            msg = latest_ai_json
        if not msg:
            continue
        payload = json.dumps(msg, separators=(',', ':'))
        dead = []
        for ws in list(_ai_json_clients):
            try:
                await ws.send(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            _ai_json_clients.discard(ws)

async def run_websocket_server_json_out():
    """Start the AI-JSON WebSocket server and its broadcaster task."""
    logger.info(f"Starting AI-JSON WebSocket on {WS_HOST}:{WS_JSON_OUT_PORT}")
    server = await websockets.serve(
        ai_json_ws_handler,
        WS_HOST,
        WS_JSON_OUT_PORT,
        max_size=WS_MAX_SIZE,
        ping_interval=WS_PING_INTERVAL,
        close_timeout=WS_CLOSE_TIMEOUT,
    )
    logger.info("✅ AI-JSON WebSocket listening")
    asyncio.create_task(ai_json_broadcast_loop())
    await server.wait_closed()

def start_websocket_server_json_out():
    def _runner():
        try:
            asyncio.run(run_websocket_server_json_out())
        except Exception as e:
            logger.exception(f"AI-JSON WebSocket crashed: {e}")
    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    logger.info("AI-JSON WS server thread started")
    return t

# ----------------------------
# HTTP routes (unchanged except raw fallback transform)
# ----------------------------
@app.get("/")
def index():
    """Minimal viewer that pulls the latest AI-annotated frame ~30fps."""
    html = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>Unity Stream Preview</title>
      <style>
        body { margin:0; background:#111; color:#eee; font-family:sans-serif; }
        header { padding:12px 16px; background:#1b1b1b; position:sticky; top:0; }
        main { display:flex; justify-content:center; align-items:center; height:calc(100vh - 56px); }
        img { max-width:100%; max-height:100%; object-fit:contain; background:#000; }
        .meta { font-size:12px; opacity:.85; }
      </style>
    </head>
    <body>
      <header>
        <div>Unity → Backend preview (AI annotated)</div>
        <div class="meta" id="meta">Waiting for frames…</div>
      </header>
      <main><img id="feed" alt="stream will appear here"/></main>
      <script>
        const img = document.getElementById('feed');
        const meta = document.getElementById('meta');

        function tick(){ img.src = '/latest_unity_frame?t=' + Date.now(); }
        async function status(){
          try {
            const r = await fetch('/api/status?x=' + Date.now());
            const s = await r.json();
            const ms = (typeof s.ai_last_infer_ms === 'number') ? s.ai_last_infer_ms.toFixed(1) : s.ai_last_infer_ms;
            meta.textContent = `Frames: ${s.frames_in} | Clients: ${s.clients} | Last: ${s.latest_timestamp_iso || '—'} | ` +
                               `${s.last_frame_len}B | jpeg=${s.last_frame_is_jpeg} | AI out=${s.ai_frames_out} | infer=${ms}ms`;
          } catch(e) { meta.textContent = 'Status error'; }
        }
        setInterval(tick, 33);     // ~30 fps pull
        setInterval(status, 500);  // update status
      </script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.get("/latest_unity_frame")
def latest_unity_frame_route():
    """
    Return the latest AI-annotated JPEG if available; otherwise fall back to the raw frame.
    """
    with _ai_lock:
        ai_buf = latest_ai_jpeg
    if ai_buf:
        resp = send_file(io.BytesIO(ai_buf), mimetype="image/jpeg", as_attachment=False, download_name="ai.jpg")
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

    # Fallback: raw frame (apply same orientation fixes if enabled)
    with _lock:
        buf = latest_unity_jpeg
    if not buf:
        return Response("No frame yet", status=404)

    if ROTATE_180 or FLIP_X:
        try:
            arr = np.frombuffer(buf, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                if ROTATE_180:
                    img = cv2.rotate(img, cv2.ROTATE_180)
                if FLIP_X:
                    img = cv2.flip(img, 1)  # horizontal flip
                ok2, out = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok2:
                    buf = out.tobytes()
        except Exception:
            # if transform fails for any reason, fall back to original bytes
            pass

    resp = send_file(io.BytesIO(buf), mimetype="image/jpeg", as_attachment=False, download_name="frame.jpg")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

@app.get("/latest_raw_frame")
def latest_raw_frame_route():
    """Optional: raw frame for debugging."""
    with _lock:
        buf = latest_unity_jpeg
    if not buf:
        return Response("No frame yet", status=404)
    resp = send_file(io.BytesIO(buf), mimetype="image/jpeg", as_attachment=False, download_name="raw.jpg")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp

@app.get("/checked_frame")
def checked_frame():
    with _lock:
        buf = latest_unity_jpeg
        ts = latest_unity_timestamp_iso
        size = last_frame_len
        ok = last_frame_is_jpeg

    if not buf:
        return Response("No frame yet", status=404)

    arr = np.frombuffer(buf, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({
            "error": "decode_failed",
            "bytes": size,
            "looks_like_jpeg": ok,
            "ts": ts
        }), 500

    h, w = img.shape[:2]
    cv2.putText(img, f"{w}x{h} jpeg={ok} bytes={size}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    if ts:
        cv2.putText(img, ts, (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    ok2, out = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok2:
        return Response("re-encode failed", status=500)

    resp = send_file(io.BytesIO(out.tobytes()), mimetype="image/jpeg", as_attachment=False, download_name="checked.jpg")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp

@app.get("/api/status")
def api_status():
    with _lock:
        frames = frames_in
        clients = _ws_clients
        ts = latest_unity_timestamp_iso
        last_len = last_frame_len
        last_is_jpg = last_frame_is_jpeg
    with _ai_lock:
        ai_out = ai_frames_out
        ai_ms = ai_last_infer_ms
    return jsonify({
        "frames_in": frames,
        "clients": clients,
        "latest_timestamp_iso": ts,
        "last_frame_len": last_len,
        "last_frame_is_jpeg": last_is_jpg,
        "ai_frames_out": ai_out,
        "ai_last_infer_ms": ai_ms
    })

# Register blueprints
app.register_blueprint(bp_auth)
app.register_blueprint(bp_scan)

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    ip = get_local_ip()
    logger.info("========================================")
    logger.info(f"HTTP preview    : http://{ip}:{HTTP_PORT}/")
    logger.info(f"WS in  (Unity)  : ws://{ip}:{WS_PORT}")
    logger.info(f"WS out (to HMD) : ws://{ip}:{WS_OUT_PORT}")
    logger.info(f"WS JSON (meta)  : ws://{ip}:{WS_JSON_OUT_PORT}")  # <-- added
    logger.info("========================================")
    from db import get_db
    db = get_db()
    logger.info(f"Mongo connected DB={db.name}")

    # Start AI worker (loads best.pt once; processes latest-only)
    start_ai_worker()

    # Start inbound WS (Unity -> backend)
    start_websocket_server()

    # Start outbound WS (backend -> Unity)
    start_websocket_server_out()

    # Start outbound JSON WS (backend -> Unity)
    start_websocket_server_json_out()  # <-- added

    # Run HTTP
    app.run(host=HTTP_HOST, port=HTTP_PORT, debug=False, use_reloader=False)
