from flask import Flask, jsonify, send_file, render_template_string
import asyncio
import websockets
import numpy as np
import cv2
import io
import logging
from datetime import datetime
import threading

# ----------------------------
# Flask app (HTTP for previews)
# ----------------------------
app = Flask(__name__)

# Quiet logs (keep error-level)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bw-backend")

# ----------------------------
# Global state
# ----------------------------
latest_unity_frame = None           # np.ndarray (BGR)
latest_unity_timestamp_iso = None
latest_gray_frame = None            # np.ndarray (BGR, 3-channel grayscale for easy JPEG)
latest_gray_timestamp_iso = None

frames_in = 0
clients_stream = set()              # connected /stream websocket clients

# ----------------------------
# WebSocket server (frames + stream)
# ----------------------------

async def handle_frames(ws):
    """Receive [8-byte little-endian timestamp][JPEG] from Unity, decode, gray, broadcast."""
    global latest_unity_frame, latest_unity_timestamp_iso
    global latest_gray_frame, latest_gray_timestamp_iso, frames_in, clients_stream

    try:
        async for message in ws:
            if not isinstance(message, (bytes, bytearray)) or len(message) < 9:
                continue

            # Split timestamp + JPEG
            ts_ms = int.from_bytes(message[:8], byteorder="little", signed=False)
            jpeg = message[8:]

            # Decode original (BGR)
            nparr = np.frombuffer(jpeg, np.uint8)
            color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if color is None:
                continue

            # Store original for HTML preview
            latest_unity_frame = color
            latest_unity_timestamp_iso = datetime.fromtimestamp(ts_ms / 1000.0).isoformat()
            frames_in += 1

            # Convert to grayscale (single channel), then back to 3-channel for safe JPEG/Unity decode
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # Encode grayscale JPEG
            ok, gray_jpg_buf = cv2.imencode('.jpg', gray_bgr, [
                cv2.IMWRITE_JPEG_QUALITY, 75,
                cv2.IMWRITE_JPEG_OPTIMIZE, 0
            ])
            if not ok:
                continue

            latest_gray_frame = gray_bgr
            latest_gray_timestamp_iso = latest_unity_timestamp_iso

            # Build outbound payload: same 8-byte timestamp + JPEG
            payload = ts_ms.to_bytes(8, byteorder='little', signed=False) + gray_jpg_buf.tobytes()

            # Broadcast to all /stream clients (remove dead ones)
            dead = []
            for client in clients_stream:
                try:
                    await client.send(payload)
                except Exception:
                    dead.append(client)
            for d in dead:
                clients_stream.discard(d)

    except websockets.ConnectionClosed:
        pass
    except Exception as e:
        logger.error(f"/frames handler error: {e}")


async def handle_stream(ws):
    """Register a client to receive grayscale frames broadcast from /frames."""
    clients_stream.add(ws)
    try:
        await ws.wait_closed()
    finally:
        clients_stream.discard(ws)


async def ws_router(ws, path):
    """Route based on URL path."""
    if path == "/frames":
        await handle_frames(ws)
    elif path == "/stream":
        await handle_stream(ws)
    else:
        # Close unknown paths
        await ws.close(code=1008, reason="Unsupported path")


def start_ws_server():
    logger.info("Starting WebSocket server thread")
    
    async def runner():
        try:
            # Check if WebSocket server is starting
            logger.info("Starting the WebSocket server on port 8765")
            server = await websockets.serve(
                ws_router,  # <-- FIXED: use ws_router, not unity_websocket_handler
                "0.0.0.0",  # Listen on all addresses (0.0.0.0)
                8765,
                ping_interval=None,
                ping_timeout=None,
                max_size=10**7,
                close_timeout=1
            )
            logger.info("🚀 WebSocket server started on port 8765")
            await server.wait_closed()
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")

    def thread_main():
        try:
            logger.info("Starting WebSocket server thread (Thread Main)")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(runner())
        except Exception as e:
            logger.error(f"Thread error: {e}")

    t = threading.Thread(target=thread_main, daemon=True)
    t.start()
    logger.info("WebSocket server thread initialized")
    return t

# ----------------------------
# Flask routes for simple HTML preview + JPG endpoints
# ----------------------------

@app.route("/")
def index():
    """Streaming interface with double-window display"""
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Unity Stream Analysis</title>
        <meta charset="UTF-8">
        <style>
            /* Add styling here */
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Main stream (all frames) -->
            <div class="stream-container">
                <div class="stream-title">
                    <span>UNITY STREAM (ALL FRAMES)</span>
                    <span id="main-fps">0 FPS</span>
                </div>
                <div class="stream-window">
                    <img id="main-stream" src="/latest_unity_frame" alt="Unity Stream">
                </div>
            </div>
            
            <!-- Grayscale stream -->
            <div class="grayscale-container">
                <div class="stream-title">
                    <span>Grayscale (echo to Unity)</span>
                    <span id="fpsOut">0 FPS</span>
                </div>
                <div class="stream-window">
                    <img id="grayscale-stream" src="/grayscale_frame" alt="Grayscale Frame">
                </div>
            </div>
        </div>
    </body>
    <script>
        let mainStream = document.getElementById('main-stream');
        let grayscaleStream = document.getElementById('grayscale-stream');
        let fpsOut = document.getElementById('fpsOut');

        function updateMainStream() {
            mainStream.src = '/latest_unity_frame?' + Date.now();
        }

        function updateGrayscaleStream() {
            grayscaleStream.src = '/grayscale_frame?' + Date.now();
        }

        setInterval(updateMainStream, 100);
        setInterval(updateGrayscaleStream, 100);
    </script>
    </html>
    '''
    return render_template_string(html_template)


@app.route("/latest_unity_frame")
def http_latest_unity():
    global latest_unity_frame

    if latest_unity_frame is None:
        return ("", 404)

    ok, buf = cv2.imencode('.jpg', latest_unity_frame, [
        cv2.IMWRITE_JPEG_QUALITY, 70,
        cv2.IMWRITE_JPEG_OPTIMIZE, 0
    ])

    if not ok:
        return ("", 500)

    resp = send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.route("/grayscale_frame")
def http_gray():
    global latest_gray_frame

    if latest_gray_frame is None:
        return ("", 404)

    ok, buf = cv2.imencode('.jpg', latest_gray_frame, [
        cv2.IMWRITE_JPEG_QUALITY, 75,
        cv2.IMWRITE_JPEG_OPTIMIZE, 0
    ])

    if not ok:
        return ("", 500)

    resp = send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.route("/api/status")
def api_status():
    return jsonify({
        "frames_in": frames_in,
        "clients_stream": len(clients_stream),
        "unity_timestamp": latest_unity_timestamp_iso,
        "gray_timestamp": latest_gray_timestamp_iso,
        "unity_shape": (latest_unity_frame.shape if latest_unity_frame is not None else None),
        "gray_shape": (latest_gray_frame.shape if latest_gray_frame is not None else None)
    })

# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    try:
        logger.info("Starting WebSocket server thread")
        # Start WebSocket server
        ws_thread = start_ws_server()

        # Start Flask server
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True,
            use_reloader=False
        )
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        exit(1)
