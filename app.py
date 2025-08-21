from flask import Flask, request, jsonify, render_template_string, send_file
import asyncio
import websockets
import base64
import numpy as np
import cv2
import os
from datetime import datetime
import logging
import io
import socket
import threading
import json
# Add ultralytics for YOLOv8
from ultralytics import YOLO


app = Flask(__name__)


# Configure logging - temporarily enable werkzeug for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Disable all logging for maximum performance
log = logging.getLogger('werkzeug')
log.disabled = True
app.logger.disabled = True
logging.getLogger().setLevel(logging.ERROR)


# Create directories only for non-frame data if needed
os.makedirs('received_images/color', exist_ok=True)
os.makedirs('received_images/depth', exist_ok=True)


# Store latest images for web display
latest_unity_frame = None
latest_unity_timestamp = None
latest_color_image = None
latest_depth_image = None
latest_color_timestamp = None
latest_depth_timestamp = None
websocket_clients = 0
unity_frames_received = 0
start_time = datetime.now()


# Add global variables for tracking selected frames
latest_unity_frame = None
latest_unity_timestamp = None
latest_selected_frame = None  # Store the selected 1-out-of-3 frames
latest_selected_timestamp = None  # Timestamp for selected frames
frame_counter = 0  # Counter to track frame selection
selected_frames_count = 0  # Counter for selected frames


# Add new global variables for AI processing
latest_ai_processed_frame = None  # Store the frame with AI detections
latest_ai_timestamp = None  # Timestamp for AI processed frames
ai_processed_count = 0  # Counter for AI processed frames
ai_model = None  # YOLOv8 model instance


# Load YOLOv8 model at startup
def load_yolo_model():
    """Load the YOLOv8 model from the saved weights file"""
    global ai_model
    try:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
        logger.info(f"Loading YOLOv8 model from {model_path}")
        ai_model = YOLO(model_path)
        logger.info("YOLOv8 model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading YOLOv8 model: {e}")
        return False


# Function to run inference on an image
def process_frame_with_ai(frame, timestamp_ms=None, websocket=None):
    """Process the frame with YOLOv8 and return annotated image"""
    global ai_model, ai_processed_count
    
    if ai_model is None:
        return frame, []  # Return original frame if model not loaded
    
    try:
        # Run YOLOv8 inference
        results = ai_model.predict(frame, conf=0.25)  # Adjust confidence threshold as needed
        
        # Get the first result (only one image)
        result = results[0]
        
        # Create a copy of the frame to draw on
        annotated_frame = frame.copy()
        
        # Prepare detections list
        detections = []
        unity_detections = []
        
        # Draw bounding boxes and labels on the frame
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            # Get class names
            class_names = result.names
            
            # Process each detection
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                conf = confidences[i]
                cls_id = class_ids[i]
                cls_name = class_names.get(cls_id, f"Class {cls_id}")
                
                # Calculate Unity-required values
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                
                # Add detection info to list for backend use
                detections.append({
                    "class": cls_name,
                    "confidence": float(conf),
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "center": [center_x, center_y],
                    "size": [width, height]
                })
                
                # Add to Unity-formatted detection list
                unity_detections.append({
                    "time": timestamp_ms,
                    "xScreen": center_x,
                    "yScreen": center_y,
                    "width": width,
                    "height": height,
                    "class": cls_name,
                    "confidence": float(conf)
                })
                
                # Draw rectangle and label
                color = (0, 255, 0)  # Green box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add label with class name and confidence
                label = f"{cls_name}: {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Send detections to Unity if websocket is provided and we have detections
        if websocket is not None and len(unity_detections) > 0:
            asyncio.create_task(send_detections_to_unity(websocket, unity_detections))
        
        ai_processed_count += 1
        return annotated_frame, detections
    
    except Exception as e:
        logger.error(f"Error in AI processing: {e}")
        return frame, []  # Return original frame on error


# Add function to send detections back to Unity
async def send_detections_to_unity(websocket, detections):
    """Send detection data back to Unity via WebSocket with frame dims"""
    try:
        # Normalize per‑detection timestamp
        for d in detections:
            if "time" in d and d["time"] is not None:
                d["time"] = int(d["time"])
        # Get AI frame dimensions from the latest selected frame
        global latest_selected_frame
        if latest_selected_frame is not None:
            ai_h, ai_w = latest_selected_frame.shape[:2]
        else:
            ai_h, ai_w = 0, 0  # Unity will ignore zero sizes

        # Inline message construction
        message = {
            "type": "detections",
            "count": len(detections),
            "detections": detections,
            "timestamp": int(datetime.now().timestamp() * 1000),  # server send time (ms)
            "aiWidth": ai_w,
            "aiHeight": ai_h,
            "aiYOriginTopLeft": True
        }

        await websocket.send(json.dumps(message))

        # Logging
        if detections:
            logger.info(f"Sent {len(detections)} detections to Unity client (ai {ai_w}x{ai_h})")
            logger.info(
                f"Sample detection center=({detections[0]['xScreen']:.1f},{detections[0]['yScreen']:.1f}) "
                f"w={detections[0]['width']:.1f} h={detections[0]['height']:.1f} cls={detections[0].get('class','?')}"
            )
        else:
            logger.info(f"Sent empty detections set (ai {ai_w}x{ai_h})")

    except Exception as e:
        logger.error(f"Error sending detections to Unity: {e}")


# Simple WebSocket handler - treat all connections as Unity clients for simplicity
# Modify the unity_websocket_handler function in your backend

async def unity_websocket_handler(websocket):
    """Handle Unity WebSocket connections with frame selection logic"""
    global websocket_clients, unity_frames_received, latest_unity_frame, latest_unity_timestamp
    global frame_counter, latest_selected_frame, latest_selected_timestamp, selected_frames_count
    global latest_ai_processed_frame, latest_ai_timestamp
   
    websocket_clients += 1
   
    try:
        # Main frame receiving loop
        while True:
            try:
                # Receive data from Unity
                frame_data = await websocket.recv()
               
                # Handle binary frame data (JPEG from Unity)
                if isinstance(frame_data, bytes) and len(frame_data) > 16:  # Ensure we have enough data
                    # Extract timestamp (first 8 bytes)
                    timestamp_bytes = frame_data[:8]
                    image_data = frame_data[8:]  # Rest is the JPEG data
                    
                    # Convert timestamp bytes to long/int64 (milliseconds since epoch)
                    timestamp_ms = int.from_bytes(timestamp_bytes, byteorder='little')
                    
                    # Convert to datetime
                    timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0)
                    
                    # Decode frame
                    nparr = np.frombuffer(image_data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                   
                    if image is not None:
                        # Store latest frame for main display
                        latest_unity_frame = image
                        latest_unity_timestamp = timestamp.isoformat()
                        unity_frames_received += 1
                       
                        # Frame selection logic (1 out of 3)
                        frame_counter += 1
                        if frame_counter % 3 == 0:
                            # Store selected frame (make a copy to avoid reference issues)
                            latest_selected_frame = image.copy()
                            latest_selected_timestamp = timestamp.isoformat()
                            selected_frames_count += 1
                            
                            # Process the selected frame with AI model and pass websocket for response
                            if ai_model is not None:
                                processed_frame, detections = process_frame_with_ai(
                                    latest_selected_frame, 
                                    timestamp_ms, 
                                    websocket
                                )
                                latest_ai_processed_frame = processed_frame
                                latest_ai_timestamp = timestamp.isoformat()
                                
                                # Log occasionally (every 10 frames)
                                if selected_frames_count % 10 == 0:
                                    logger.info(f"AI processed frame #{selected_frames_count}, " 
                                                f"timestamp: {timestamp}, detections: {len(detections)}")
                           
                # Handle text messages (possibly commands from Unity)
                elif isinstance(frame_data, str):
                    try:
                        message = json.loads(frame_data)
                        # Handle command messages if needed
                        if 'command' in message:
                            logger.info(f"Received command from Unity: {message['command']}")
                            # Handle different commands here
                    except json.JSONDecodeError:
                        logger.warning(f"Received invalid JSON message: {frame_data[:100]}")
                   
            except Exception as e:
                # Log the error but continue processing
                logger.error(f"Error processing frame: {e}")
                continue
               
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}")
    finally:
        websocket_clients -= 1
# Start WebSocket server - optimized settings
def start_websocket_server():
    """Start optimized WebSocket server for minimal latency"""
    async def run_server():
        try:
            # Optimized server settings for low latency
            server = await websockets.serve(
                unity_websocket_handler,
                "0.0.0.0",
                8765,
                ping_interval=None,  # Disable ping for minimal overhead
                ping_timeout=None,
                close_timeout=1,     # Fast close
                max_size=10**7      # 10MB max frame size
            )
            logger.info("🚀 Optimized WebSocket server started on port 8765")
           
            await server.wait_closed()
           
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
   
    # High priority thread for WebSocket
    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_server())
   
    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    return thread


# Flask routes
@app.route('/')
def index():
    """Streaming interface with triple-window display"""
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Unity Stream Analysis</title>
        <meta charset="UTF-8">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { background: #000; overflow: hidden; font-family: Arial, sans-serif; color: white; }
            .container {
                display: flex;
                width: 100vw;
                height: 100vh;
            }
            .stream-container {
                flex: 1;
                display: flex;
                flex-direction: column;
                padding: 10px;
                border-right: 1px solid #333;
            }
            .selected-container {
                flex: 1;
                display: flex;
                flex-direction: column;
                padding: 10px;
                border-right: 1px solid #333;
            }
            .ai-container {
                flex: 1;
                display: flex;
                flex-direction: column;
                padding: 10px;
            }
            .stream-title {
                text-align: center;
                padding: 5px;
                font-size: 14px;
                font-weight: bold;
                margin-bottom: 5px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .stream-title .indicator {
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background-color: #0f0;
                display: inline-block;
                margin-right: 5px;
            }
            .stream-window {
                flex: 1;
                display: flex;
                justify-content: center;
                align-items: center;
                background: #111;
                border-radius: 5px;
                position: relative;
            }
            .stream-window img {
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
            }
            .stream-info {
                position: absolute;
                bottom: 5px;
                left: 5px;
                background: rgba(0,0,0,0.6);
                color: white;
                padding: 3px 6px;
                border-radius: 3px;
                font-size: 10px;
                font-family: monospace;
            }
            .stats {
                color: #0f0;
                font: 10px monospace;
                background: rgba(0,0,0,0.7);
                padding: 5px;
                border-radius: 3px;
                margin-top: 5px;
                display: flex;
                justify-content: space-between;
            }
            .selected-stats {
                color: #0ff;
                font: 10px monospace;
                background: rgba(0,0,0,0.7);
                padding: 5px;
                border-radius: 3px;
                margin-top: 5px;
            }
            .ai-stats {
                color: #f0f;
                font: 10px monospace;
                background: rgba(0,0,0,0.7);
                padding: 5px;
                border-radius: 3px;
                margin-top: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Main stream (all frames) -->
            <div class="stream-container">
                <div class="stream-title">
                    <span><span class="indicator" id="main-indicator"></span>UNITY STREAM (ALL FRAMES)</span>
                    <span id="main-fps">0 FPS</span>
                </div>
                <div class="stream-window">
                    <img id="main-stream" src="/test_image" alt="Unity Stream">
                    <div class="stream-info" id="main-info">Resolution: -</div>
                </div>
                <div class="stats">
                    <div>Total Frames: <span id="frame-count">0</span></div>
                    <div>Client FPS: <span id="fps">0</span></div>
                </div>
            </div>
            
            <!-- Selected stream (1 of 3 frames) -->
            <div class="selected-container">
                <div class="stream-title">
                    <span><span class="indicator" id="selected-indicator"></span>SELECTED FRAMES (1 OF 3)</span>
                    <span id="selected-fps">0 FPS</span>
                </div>
                <div class="stream-window">
                    <img id="selected-stream" src="/test_image" alt="Selected Frames">
                    <div class="stream-info" id="selected-info">Selecting: 1 of 3 frames</div>
                </div>
                <div class="selected-stats">
                    <div>Selected Frames: <span id="selected-count">0</span> (<span id="selection-ratio">0%</span> of total)</div>
                    <div>Last Selected: <span id="last-selected-time">-</span></div>
                </div>
            </div>
            
            <!-- AI processed stream -->
            <div class="ai-container">
                <div class="stream-title">
                    <span><span class="indicator" id="ai-indicator"></span>AI DETECTION (YOLOv8)</span>
                    <span id="ai-fps">0 FPS</span>
                </div>
                <div class="stream-window">
                    <img id="ai-stream" src="/test_image" alt="AI Processed">
                    <div class="stream-info" id="ai-info">Processing: YOLOv8</div>
                </div>
                <div class="ai-stats">
                    <div>AI Processed: <span id="ai-count">0</span></div>
                    <div>Last Detected: <span id="last-ai-time">-</span></div>
                </div>
            </div>
        </div>
       
        <script>
            const mainStream = document.getElementById('main-stream');
            const selectedStream = document.getElementById('selected-stream');
            const aiStream = document.getElementById('ai-stream');
            const fpsEl = document.getElementById('fps');
            const frameCountEl = document.getElementById('frame-count');
            const selectedCountEl = document.getElementById('selected-count');
            const selectionRatioEl = document.getElementById('selection-ratio');
            const mainInfoEl = document.getElementById('main-info');
            const selectedInfoEl = document.getElementById('selected-info');
            const aiInfoEl = document.getElementById('ai-info');
            const lastSelectedTimeEl = document.getElementById('last-selected-time');
            const lastAiTimeEl = document.getElementById('last-ai-time');
            const mainIndicator = document.getElementById('main-indicator');
            const selectedIndicator = document.getElementById('selected-indicator');
            const aiIndicator = document.getElementById('ai-indicator');
            const mainFpsEl = document.getElementById('main-fps');
            const selectedFpsEl = document.getElementById('selected-fps');
            const aiFpsEl = document.getElementById('ai-fps');
            const aiCountEl = document.getElementById('ai-count');
           
            let frameCount = 0;
            let selectedCount = 0;
            let aiCount = 0;
            let lastMainTime = performance.now();
            let lastSelectedTime = performance.now();
            let lastAiTime = performance.now();
            let mainFpsValue = 0;
            let selectedFpsValue = 0;
            let aiFpsValue = 0;
            let mainFramesReceived = 0;
            let selectedFramesReceived = 0;
            let aiFramesReceived = 0;
           
            // Format time difference
            function formatTimeDiff(isoTime) {
                if (!isoTime) return "-";
                const timeMs = new Date(isoTime).getTime();
                const now = Date.now();
                const diffMs = now - timeMs;
               
                if (diffMs < 1000) return "just now";
                if (diffMs < 60000) return Math.floor(diffMs/1000) + "s ago";
                return Math.floor(diffMs/60000) + "m " + Math.floor((diffMs % 60000)/1000) + "s ago";
            }
           
            // Update main stream
            function updateMainStream() {
                const now = performance.now();
                mainStream.src = '/latest_unity_frame?' + now + Math.random().toString(36).slice(2);
                frameCount++;
                mainFramesReceived++;
               
                // Calculate FPS - fixed to use proper time difference
                if (mainFramesReceived % 10 === 0) {
                    const elapsed = now - lastMainTime;
                    mainFpsValue = Math.round(10 * 1000 / elapsed);
                    mainFpsEl.textContent = mainFpsValue + " FPS";
                    lastMainTime = now;
                   
                    // Update indicator color based on FPS
                    if (mainFpsValue > 20) {
                        mainIndicator.style.backgroundColor = "#0f0"; // Green
                    } else if (mainFpsValue > 10) {
                        mainIndicator.style.backgroundColor = "#ff0"; // Yellow
                    } else {
                        mainIndicator.style.backgroundColor = "#f00"; // Red
                    }
                }
               
                // Update stats every 30 frames
                if (frameCount % 30 === 0) {
                    const clientFps = Math.round(30 * 1000 / (now - lastMainTime + 0.1));
                    fpsEl.textContent = clientFps;
                   
                    // Fetch server stats
                    fetch('/api/status?' + Date.now())
                        .then(r => r.json())
                        .then(d => {
                            frameCountEl.textContent = d.frames;
                            selectedCountEl.textContent = d.selected_frames;
                            aiCountEl.textContent = d.ai_processed || 0;
                           
                            // Fix selection ratio calculation - correct percentage
                            const ratio = d.selected_frames > 0 ?
                                Math.round((d.selected_frames / d.frames) * 100) : 0;
                            selectionRatioEl.textContent = ratio + "%";
                           
                            // Update resolution info
                            if (d.frame_resolution) {
                                mainInfoEl.textContent =
                                    `Resolution: ${d.frame_resolution[1]}x${d.frame_resolution[0]}`;
                            }
                        })
                        .catch(() => {});
                }
               
                requestAnimationFrame(updateMainStream);
            }
           
            // Update selected stream
            function updateSelectedStream() {
                const now = performance.now();
                selectedStream.src = '/selected_frame?' + now + Math.random().toString(36).slice(2);
                selectedFramesReceived++;
               
                // Calculate FPS for selected stream - fixed calculation
                if (selectedFramesReceived % 5 === 0) {
                    const elapsed = now - lastSelectedTime;
                    selectedFpsValue = Math.round(5 * 1000 / (elapsed + 0.1)); // Add 0.1 to avoid division by zero
                    selectedFpsEl.textContent = selectedFpsValue + " FPS";
                    lastSelectedTime = now;
                   
                    // Update indicator color
                    if (selectedFpsValue > 5) {
                        selectedIndicator.style.backgroundColor = "#0f0"; // Green
                    } else if (selectedFpsValue > 2) {
                        selectedIndicator.style.backgroundColor = "#ff0"; // Yellow
                    } else {
                        selectedIndicator.style.backgroundColor = "#f00"; // Red
                    }
                   
                    // Fetch more detailed info about selected frames
                    fetch('/debug?' + Date.now())
                        .then(r => r.json())
                        .then(d => {
                            if (d.latest_selected_timestamp) {
                                lastSelectedTimeEl.textContent =
                                    formatTimeDiff(d.latest_selected_timestamp);
                               
                                // Extract the ratio correctly
                                const ratioValue = d.selection_ratio ?
                                    d.selection_ratio.split(':')[1] : "3";
                                selectedInfoEl.textContent =
                                    `Selecting: 1 of ${ratioValue} frames`;
                            }
                        })
                        .catch(() => {});
                }
            }
            
            // Update AI stream
           // In the script section of your HTML template, modify the updateAiStream function:
function updateAiStream() {
    const now = performance.now();
    aiStream.src = '/ai_processed_frame?' + now + Math.random().toString(36).slice(2);
    aiFramesReceived++;
    
    // Calculate FPS for AI stream
    if (aiFramesReceived % 5 === 0) {
        const elapsed = now - lastAiTime;
        aiFpsValue = Math.round(5 * 1000 / (elapsed + 0.1));
        aiFpsEl.textContent = aiFpsValue + " FPS";
        lastAiTime = now;
        
        // Update indicator color
        if (aiFpsValue > 5) {
            aiIndicator.style.backgroundColor = "#0f0"; // Green
        } else if (aiFpsValue > 2) {
            aiIndicator.style.backgroundColor = "#ff0"; // Yellow
        } else {
            aiIndicator.style.backgroundColor = "#f00"; // Red
        }
        
        // Fetch AI info
        fetch('/debug?' + Date.now())
            .then(r => r.json())
            .then(d => {
                if (d.latest_ai_timestamp) {
                    lastAiTimeEl.textContent = 
                        formatTimeDiff(d.latest_ai_timestamp);
                    
                    // Add Unity timestamp to the AI info display
                    aiInfoEl.textContent = 
                        `YOLOv8: ${d.ai_detection_count || 0} objects | Unity timestamp: ${d.unity_timestamp_readable || "unknown"}`;
                }
            })
            .catch(() => {});
            }
        }
                
            // Start streams
            requestAnimationFrame(updateMainStream);
           
            // Update selected stream at ~10fps
            setInterval(updateSelectedStream, 100);
            
            // Update AI stream at ~10fps
            setInterval(updateAiStream, 100);
           
            // Prevent delays
            mainStream.onload = () => {};
            mainStream.onerror = () => {};
            selectedStream.onload = () => {};
            selectedStream.onerror = () => {};
            aiStream.onload = () => {};
            aiStream.onerror = () => {};
        </script>
    </body>
    </html>
    '''
    return render_template_string(html_template)


@app.route('/latest_unity_frame')
def latest_unity_frame_route():
    """Ultra-optimized frame serving - zero overhead"""
    global latest_unity_frame
   
    if latest_unity_frame is not None:
        # Fastest possible JPEG encoding
        _, buffer = cv2.imencode('.jpg', latest_unity_frame, [
            cv2.IMWRITE_JPEG_QUALITY, 70,  # Lower quality for speed
            cv2.IMWRITE_JPEG_OPTIMIZE, 0,  # No optimization
            cv2.IMWRITE_JPEG_PROGRESSIVE, 0  # No progressive
        ])
       
        # Minimal response
        response = send_file(io.BytesIO(buffer.tobytes()), mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'no-store'
        return response
    else:
        return "", 404


# Add endpoint to serve selected frames
@app.route('/selected_frame')
def selected_frame_route():
    """Serve the selected 1-out-of-3 frames"""
    global latest_selected_frame
   
    if latest_selected_frame is not None:
        # Encode frame
        _, buffer = cv2.imencode('.jpg', latest_selected_frame, [
            cv2.IMWRITE_JPEG_QUALITY, 80,  # Better quality for selected frames
            cv2.IMWRITE_JPEG_OPTIMIZE, 0,  # No optimization for speed
        ])
       
        # Create response
        response = send_file(io.BytesIO(buffer.tobytes()), mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'no-store'
        return response
    else:
        return "", 404


# Add endpoint to serve AI-processed frames
@app.route('/ai_processed_frame')
def ai_processed_frame_route():
    """Serve the AI-processed frames with detections"""
    global latest_ai_processed_frame
   
    if latest_ai_processed_frame is not None:
        # Encode frame
        _, buffer = cv2.imencode('.jpg', latest_ai_processed_frame, [
            cv2.IMWRITE_JPEG_QUALITY, 85,  # Higher quality for AI-processed frames
            cv2.IMWRITE_JPEG_OPTIMIZE, 0,  # No optimization for speed
        ])
       
        # Create response
        response = send_file(io.BytesIO(buffer.tobytes()), mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'no-store'
        return response
    else:
        return "", 404


# Remove all other routes except essential ones
@app.route('/api/status')
def api_status():
    return jsonify({
        'frames': unity_frames_received,
        'clients': websocket_clients,
        'selected_frames': selected_frames_count,
        'ai_processed': ai_processed_count,
        'frame_resolution': latest_unity_frame.shape if latest_unity_frame is not None else None
    })

@app.route('/debug')
def debug_info():
    """Debug endpoint to check server state"""
    # Get original timestamp from Unity in a readable format
    unity_timestamp_readable = "None"
    if latest_unity_timestamp:
        try:
            # Convert ISO format to a more readable format
            dt = datetime.fromisoformat(latest_unity_timestamp)
            unity_timestamp_readable = dt.strftime("%H:%M:%S.%f")[:-3]
        except:
            unity_timestamp_readable = latest_unity_timestamp
            
    return jsonify({
        'websocket_clients': websocket_clients,
        'unity_frames_received': unity_frames_received,
        'selected_frames_count': selected_frames_count,
        'selection_ratio': f"1:{frame_counter // selected_frames_count if selected_frames_count > 0 else 0}",
        'ai_processed_count': ai_processed_count,
        'latest_unity_frame_available': latest_unity_frame is not None,
        'latest_unity_timestamp': latest_unity_timestamp,
        'unity_timestamp_readable': unity_timestamp_readable,
        'latest_selected_timestamp': latest_selected_timestamp,
        'latest_ai_timestamp': latest_ai_timestamp,
        'time_since_last_selected': (datetime.now() - datetime.fromisoformat(latest_selected_timestamp)).total_seconds() if latest_selected_timestamp else None,
        'time_since_last_ai': (datetime.now() - datetime.fromisoformat(latest_ai_timestamp)).total_seconds() if latest_ai_timestamp else None,
        'frame_shape': latest_unity_frame.shape if latest_unity_frame is not None else None,
        'server_time': datetime.now().isoformat(),
        'ai_model_loaded': ai_model is not None,
        'ai_detection_count': len(latest_ai_processed_frame) if latest_ai_processed_frame is not None else 0
    })

# Add a simple test image endpoint
@app.route('/test_image')
def test_image():
    """Test endpoint to serve a simple test image"""
    # Create a simple test image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    test_img[:, :] = (0, 255, 0)  # Green image
   
    # Add some text
    cv2.putText(test_img, 'TEST IMAGE', (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
   
    _, buffer = cv2.imencode('.jpg', test_img)
   
    response = send_file(
        io.BytesIO(buffer.tobytes()),
        mimetype='image/jpeg',
        as_attachment=False
    )
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


if __name__ == '__main__':
    try:
        # Load the YOLOv8 model first
        load_yolo_model()
        
        # Start WebSocket server first
        ws_thread = start_websocket_server();
       
        # Start Flask with maximum performance settings
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True,
            use_reloader=False,
            processes=1  # Single process for minimal overhead
        )
       
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        exit(1)

