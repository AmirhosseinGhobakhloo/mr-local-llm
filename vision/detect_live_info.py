# Import OpenCV for camera capture and drawing
import cv2
# Import requests for synchronous HTTP calls to the AI service
import requests
# Import time for cache TTL and small sleeps
import time
# Import threading so capture / inference / LLM fetch can run in parallel
import threading
# Import YOLO detector from Ultralytics
from ultralytics import YOLO

# Path to the local YOLOv8 weights file
MODEL_PATH = "yolov8n.pt"
# FastAPI chat endpoint used to fetch object descriptions
API_URL = "http://127.0.0.1:8000/chat"
# How long a cached description remains valid (seconds)
CACHE_TTL = 30
# Resize width used during inference for better FPS
INFERENCE_W = 320
# Minimum detector confidence to keep a box
MIN_CONF = 0.35
# Max characters drawn on the overlay for each description
MAX_DESC_CHARS = 90

# Load YOLO model once at startup
model = YOLO(MODEL_PATH)
# cache[label] = (description_text, timestamp)
cache: dict[str, tuple[str, float]] = {}
# Labels currently being fetched so we do not spam the API
fetching: set[str] = set()
# Protects cache / fetching shared state
cache_lock = threading.Lock()
# Protects the latest camera frame
frame_lock = threading.Lock()
# Protects the latest inference result package
results_lock = threading.Lock()

# Newest raw camera frame shared across threads
latest_frame = None
# Newest pair of (frame_copy, boxes)
latest_data = None
# Global run flag used to stop worker loops cleanly
running = True


def capture_loop(cap):
    """Continuously grab frames from the webcam into latest_frame."""
    # Allow writing to the shared frame variable
    global latest_frame
    # Keep looping until main thread asks us to stop
    while running:
        # Read one frame from the camera
        ret, f = cap.read()
        # Only publish valid frames
        if ret:
            # Lock while replacing the shared frame reference
            with frame_lock:
                # Store the newest frame
                latest_frame = f
        # Small sleep to avoid busy-spinning the CPU
        time.sleep(0.03)


def inference_loop():
    """Run YOLO on the latest frame and publish boxes in latest_data."""
    # Allow writing to the shared inference result variable
    global latest_data
    # Keep looping until shutdown
    while running:
        # Safely copy the current frame reference
        with frame_lock:
            # Local alias of the shared frame
            f = latest_frame
        # If no frame is ready yet, wait and retry
        if f is None:
            # Avoid tight loop when camera is not ready
            time.sleep(0.05)
            # Continue waiting for frames
            continue

        # Compute scaled height while keeping aspect ratio
        new_h = int(f.shape[0] * INFERENCE_W / f.shape[1])
        # Resize frame for faster inference
        small = cv2.resize(f, (INFERENCE_W, new_h))
        # X scale used to map boxes back to full-resolution frame
        sx = f.shape[1] / small.shape[1]
        # Y scale used to map boxes back to full-resolution frame
        sy = f.shape[0] / small.shape[0]
        # Run YOLO on the downscaled frame
        res = model(small, verbose=False)[0]

        # Collected boxes for this frame
        boxes = []
        # Only parse when detector returned boxes
        if res.boxes is not None and len(res.boxes):
            # Iterate over each detection
            for box in res.boxes:
                # Read confidence score
                conf = float(box.conf[0])
                # Drop weak detections
                if conf < MIN_CONF:
                    # Skip low-confidence box
                    continue
                # Read xyxy coordinates on the small frame
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Store box mapped back to original frame coordinates
                boxes.append(
                    {
                        # Integer pixel box on full-resolution frame
                        "xyxy": (
                            int(x1 * sx),
                            int(y1 * sy),
                            int(x2 * sx),
                            int(y2 * sy),
                        ),
                        # Detection confidence
                        "conf": conf,
                        # Class id used to resolve the label name
                        "cls": int(box.cls[0]),
                    }
                )

        # Publish a copy of the frame plus its boxes
        with results_lock:
            # Copy frame so drawing later cannot race capture
            latest_data = (f.copy(), boxes)
        # Pace inference to leave CPU for other work
        time.sleep(0.05)


def fetch_llm(label: str) -> None:
    """Fetch a short description for one label from the AI service (RAG-aware)."""
    try:
        # IMPORTANT: send the bare label so RAG can match knowledge filenames
        r = requests.post(
            API_URL,
            # JSON body expected by /chat
            json={"message": label},
            # Allow slow local LLM responses
            timeout=60,
        )
        # Parse JSON response body
        data = r.json()
        # Prefer the model response text
        answer = str(data.get("response", "")).strip()
        # Hard-limit overlay text length
        answer = answer[:100] if answer else "No description."
    except Exception as e:
        # Keep overlay informative if the HTTP call fails
        answer = f"error: {e}"[:100]

    # Store result in cache and clear the in-flight marker
    with cache_lock:
        # Save description + fresh timestamp
        cache[label] = (answer, time.time())
        # Allow future refreshes after TTL
        fetching.discard(label)


def draw_text(frame, text, x, y, scale=0.45, color=(0, 220, 255), thickness=1):
    """Draw text clamped inside the frame bounds."""
    # Use a simple OpenCV Hershey font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Measure text extent for clamping and background sizing
    (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
    # Read frame size
    h, w = frame.shape[:2]
    # Clamp X so text does not leave the right/left edges
    x = max(0, min(x, w - tw - 2))
    # Clamp Y so text does not leave the top/bottom edges
    y = max(th + bl, min(y, h - bl - 2))
    # Draw a dark filled rectangle behind text for readability
    cv2.rectangle(
        frame,
        (x - 2, y - th - bl),
        (x + tw + 2, y + bl),
        (0, 0, 0),
        -1,
    )
    # Draw the actual text on top of the rectangle
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


# Open default webcam with DirectShow backend on Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Warm up the camera with a few discarded frames
for _ in range(5):
    # Read and ignore startup frames
    cap.read()

# Start camera reader thread
threading.Thread(target=capture_loop, args=(cap,), daemon=True).start()
# Start YOLO inference thread
threading.Thread(target=inference_loop, daemon=True).start()

# Main UI / overlay loop
while True:
    # Safely read the latest inference package
    with results_lock:
        # Local copy of shared reference
        data = latest_data

    # If inference has not produced anything yet, just wait for quit key
    if data is None:
        # Exit when user presses q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            # Break main loop
            break
        # Continue waiting
        continue

    # Unpack frame and detections
    frame, boxes = data
    # Track labels already annotated in this frame (one overlay line each)
    seen = set()
    # Vertical cursor for the left-side description list
    y = 22

    # Draw every detection box
    for box in boxes:
        # Unpack coordinates
        x1, y1, x2, y2 = box["xyxy"]
        # Resolve class id to YOLO label string
        label = model.names[box["cls"]]
        # Read confidence
        conf = box["conf"]

        # Draw green bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw label + confidence above the box
        draw_text(frame, f"{label} {conf:.2f}", x1, y1 - 6, color=(0, 255, 0))

        # Only fetch/show one description line per unique label per frame
        if label not in seen:
            # Remember this label for the current frame
            seen.add(label)
            # Current timestamp for TTL checks
            now = time.time()
            # Read cache state under lock
            with cache_lock:
                # Existing cache entry if any
                entry = cache.get(label)
                # Whether a fetch is already running
                is_fetching = label in fetching

            # Refresh when missing or stale and not already in flight
            if not is_fetching and (entry is None or now - entry[1] >= CACHE_TTL):
                # Mark label as in-flight
                with cache_lock:
                    # Prevent duplicate workers for the same label
                    fetching.add(label)
                # Start background HTTP/LLM worker
                threading.Thread(target=fetch_llm, args=(label,), daemon=True).start()

            # Choose overlay text from cache or a waiting placeholder
            if entry:
                # Use cached description, truncated for the overlay
                desc = entry[0][:MAX_DESC_CHARS]
            else:
                # Shown while the first fetch is still running
                desc = "loading knowledge..."

            # Draw left-side description line
            draw_text(frame, f"{label}: {desc}", 10, y)
            # Move downward for the next unique label
            y += 22

    # Show the annotated frame
    cv2.imshow("MR Local-LLM", frame)
    # Quit when user presses q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        # Leave main loop
        break

# Signal worker threads to stop
running = False
# Release camera device
cap.release()
# Close OpenCV windows
cv2.destroyAllWindows()