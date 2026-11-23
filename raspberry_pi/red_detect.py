
import cv2
import numpy as np
import os
import time
import json
import requests
import threading
from queue import Queue
from datetime import datetime, timezone

# ==================================================
# CAMERA, STORAGE AND MANUAL LOCATION
# ==================================================
CAMERA = "/dev/video0"

BASE_DIR = os.path.expanduser("~/roadbot")
PHOTO_DIR = os.path.join(BASE_DIR, "photos")
STATE_FILE = os.path.join(BASE_DIR, "object_state.json")
KEY_FILE = os.path.expanduser("~/.roadbot.env")

os.makedirs(PHOTO_DIR, exist_ok=True)

# Fixed example coordinates, NOT live GPS readings.
LATITUDE = 10.7735
LONGITUDE = 76.3775

# ==================================================
# SUPABASE UPLOAD
# ==================================================
SUPABASE_UPLOAD_URL = (
    "https://iglcowsagtkasxwmsqvt.supabase.co/"
    "functions/v1/robot-upload"
)

UPLOAD_RETRY_SECONDS = 10
REQUEST_TIMEOUT = 15

# ==================================================
# RED DETECTION SETTINGS
# ==================================================
MIN_RED_AREA = 500
STABLE_FRAMES = 3
MATCH_DISTANCE = 140

MAX_PHOTOS_PER_OBJECT = 2
SECOND_PHOTO_DELAY_SECONDS = 2.0

LABEL = "CRACK DETECTED"
BOX_THICKNESS = 1

# HSV red ranges. These are starting values; tune for your lighting.
LOWER_RED_1 = np.array([0, 80, 50], dtype=np.uint8)
UPPER_RED_1 = np.array([12, 255, 255], dtype=np.uint8)

LOWER_RED_2 = np.array([168, 80, 50], dtype=np.uint8)
UPPER_RED_2 = np.array([179, 255, 255], dtype=np.uint8)

KERNEL = np.ones((3, 3), np.uint8)

# ==================================================
# SHARED STATE
# ==================================================
tracks = []
next_id = 1

state_lock = threading.RLock()
upload_queue = Queue()


def load_robot_key():
    """Read the secret key from ~/.roadbot.env or the environment."""
    if os.path.isfile(KEY_FILE):
        try:
            with open(KEY_FILE, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()

                    if line.startswith("ROBOT_UPLOAD_KEY="):
                        key = line.split("=", 1)[1].strip()

                        if (
                            len(key) >= 2
                            and key[0] == key[-1]
                            and key[0] in ("'", '"')
                        ):
                            key = key[1:-1]

                        if key:
                            return key.strip()

        except OSError as error:
            print("WARNING: Cannot read key file:", error)

    return os.environ.get("ROBOT_UPLOAD_KEY", "").strip()


ROBOT_UPLOAD_KEY = load_robot_key()


def save_state():
    """Save tracking records safely, using an atomic file replacement."""
    temp_path = STATE_FILE + ".tmp"

    with state_lock:
        try:
            with open(temp_path, "w", encoding="utf-8") as file:
                json.dump(
                    {"next_id": next_id, "tracks": tracks},
                    file,
                    indent=2
                )

            os.replace(temp_path, STATE_FILE)

        except (OSError, TypeError) as error:
            print("STATE SAVE ERROR:", error)

            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass


def load_state():
    """Restore previously saved track IDs and pending uploads."""
    global tracks, next_id

    if not os.path.isfile(STATE_FILE):
        return

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as file:
            state = json.load(file)

        with state_lock:
            tracks = state.get("tracks", [])
            next_id = state.get("next_id", 1)

            for track in tracks:
                track["matched"] = False
                track["stable_frames"] = 0
                track.setdefault("photo_count", 0)
                track.setdefault("last_photo_time", 0.0)
                track.setdefault("uploads", [])

                # Ignore invalid or incomplete saved upload entries.
                track["uploads"] = [
                    item for item in track["uploads"]
                    if isinstance(item, dict)
                    and "path" in item
                    and "timestamp" in item
                ]

                # Queue saved photos which have not uploaded successfully.
                for item in track["uploads"]:
                    item.setdefault("uploaded", False)
                    item.setdefault("next_attempt", 0.0)

                    if not item["uploaded"]:
                        upload_queue.put((track["id"], item["path"]))

        print("Loaded saved tracks:", len(tracks))
        print("Pending uploads:", upload_queue.qsize())

    except (OSError, json.JSONDecodeError, TypeError, KeyError) as error:
        print("WARNING: Could not load tracking state:", error)


# ==================================================
# RED DETECTION
# ==================================================
def detect_red(frame):
    """Return red regions which exceed the minimum area."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)

    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, KERNEL, iterations=1
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, KERNEL, iterations=2
    )

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_RED_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        detections.append({
            "box": (x, y, w, h),
            "center": (x + w // 2, y + h // 2),
            "area": area
        })

    return detections


# ==================================================
# PHOTO CAPTURE
# ==================================================
def capture_photo(frame, track, detection):
    """Save one annotated photo and queue it for upload."""
    with state_lock:
        if track["photo_count"] >= MAX_PHOTOS_PER_OBJECT:
            return

        photo_number = track["photo_count"] + 1
        object_id = track["id"]

    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")

    photo_path = os.path.join(
        PHOTO_DIR,
        f"crack_{object_id}_photo{photo_number}_{stamp}.jpg"
    )

    # IMPORTANT: frame.copy() must be inside this function.
    output = frame.copy()

    x, y, w, h = detection["box"]

    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        BOX_THICKNESS
    )

    cv2.putText(
        output,
        LABEL,
        (x, max(y - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )

    if not cv2.imwrite(photo_path, output):
        print("PHOTO SAVE ERROR:", photo_path)
        return

    with state_lock:
        # Reserve the photo number before saving the state.
        track["photo_count"] = photo_number
        track["last_photo_time"] = time.time()

        item = {
            "path": photo_path,
            "timestamp": timestamp,
            "uploaded": False,
            "next_attempt": 0.0
        }

        track.setdefault("uploads", []).append(item)

        save_state()

    report_path = os.path.splitext(photo_path)[0] + ".json"

    report = {
        "detection": LABEL,
        "object_id": object_id,
        "photo_number": photo_number,
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "timestamp": timestamp,
        "photo": photo_path
    }

    try:
        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
    except OSError as error:
        print("REPORT SAVE ERROR:", error)

    upload_queue.put((object_id, photo_path))

    print(
        f"\n{LABEL}! Object #{object_id}, "
        f"photo {photo_number}/{MAX_PHOTOS_PER_OBJECT}"
    )
    print("Saved:", photo_path)


# ==================================================
# SUPABASE UPLOAD WORKER
# ==================================================
def upload_report(photo_path, timestamp):
    """Upload a saved photo. Never capture another photo on retry."""
    if not ROBOT_UPLOAD_KEY:
        print("UPLOAD ERROR: Missing ROBOT_UPLOAD_KEY in", KEY_FILE)
        return False

    if not os.path.isfile(photo_path):
        print("UPLOAD ERROR: Photo file not found:", photo_path)
        return False

    try:
        with open(photo_path, "rb") as photo:
            response = requests.post(
                SUPABASE_UPLOAD_URL,
                headers={"x-robot-key": ROBOT_UPLOAD_KEY},
                data={
                    "latitude": str(LATITUDE),
                    "longitude": str(LONGITUDE),
                    "detected_at": timestamp,
                    "detection": LABEL
                },
                files={
                    "photo": (
                        os.path.basename(photo_path),
                        photo,
                        "image/jpeg"
                    )
                },
                timeout=REQUEST_TIMEOUT
            )

        if response.ok:
            print("UPLOAD SUCCESS:", os.path.basename(photo_path))
            return True

        print(
            "UPLOAD FAILED:",
            response.status_code,
            response.text[:300]
        )
        return False

    except (requests.RequestException, OSError) as error:
        print("UPLOAD CONNECTION ERROR:", error)
        return False


def upload_worker():
    """Upload in a separate thread so detection remains responsive."""
    while True:
        object_id, photo_path = upload_queue.get()

        try:
            while True:
                with state_lock:
                    track = next(
                        (
                            t for t in tracks
                            if t["id"] == object_id
                        ),
                        None
                    )

                    item = None

                    if track:
                        item = next(
                            (
                                u for u in track.get("uploads", [])
                                if u["path"] == photo_path
                            ),
                            None
                        )

                    if item is None or item.get("uploaded", False):
                        break

                    wait = item.get("next_attempt", 0.0) - time.time()

                if wait > 0:
                    time.sleep(min(wait, 1.0))
                    continue

                with state_lock:
                    # Recheck before starting the request.
                    if item.get("uploaded", False):
                        break

                    timestamp = item["timestamp"]

                success = upload_report(photo_path, timestamp)

                with state_lock:
                    item["uploaded"] = success

                    if not success:
                        item["next_attempt"] = (
                            time.time() + UPLOAD_RETRY_SECONDS
                        )

                    save_state()

                if success:
                    break

                print(
                    f"Retrying upload in {UPLOAD_RETRY_SECONDS} seconds:",
                    os.path.basename(photo_path)
                )

        except Exception as error:
            print("UPLOAD WORKER ERROR:", error)

            with state_lock:
                for track in tracks:
                    for item in track.get("uploads", []):
                        if item["path"] == photo_path:
                            item["next_attempt"] = (
                                time.time() + UPLOAD_RETRY_SECONDS
                            )
                            break

                save_state()

        finally:
            upload_queue.task_done()


# ==================================================
# TRACKING
# ==================================================
def find_matching_track(detection, used_ids):
    """Match a detection to the closest unused tracked center."""
    cx, cy = detection["center"]
    best_track = None
    best_distance = MATCH_DISTANCE

    for track in tracks:
        if track["id"] in used_ids:
            continue

        tx, ty = track["center"]

        distance = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5

        if distance < best_distance:
            best_distance = distance
            best_track = track

    return best_track


def process_detections(frame, detections):
    global next_id

    used_ids = set()

    for detection in detections:
        with state_lock:
            track = find_matching_track(detection, used_ids)

            if track is None:
                track = {
                    "id": next_id,
                    "center": list(detection["center"]),
                    "stable_frames": 0,
                    "photo_count": 0,
                    "last_photo_time": 0.0,
                    "matched": False,
                    "uploads": []
                }

                tracks.append(track)
                next_id += 1

                print("New red-region track:", track["id"])

            track["center"] = list(detection["center"])
            track["matched"] = True
            track["stable_frames"] = (
                track.get("stable_frames", 0) + 1
            )

            used_ids.add(track["id"])

            photo_count = track.get("photo_count", 0)
            stable = track["stable_frames"] >= STABLE_FRAMES

            delay_passed = (
                time.time() - track.get("last_photo_time", 0.0)
                >= SECOND_PHOTO_DELAY_SECONDS
            )

            should_capture = (
                stable
                and photo_count < MAX_PHOTOS_PER_OBJECT
                and (
                    photo_count == 0
                    or delay_passed
                )
            )

        if should_capture:
            capture_photo(frame, track, detection)


# ==================================================
# MAIN CAMERA LOOP
# ==================================================
def main():
    load_state()

    # Start uploader independently from the camera loop.
    threading.Thread(
        target=upload_worker,
        daemon=True,
        name="RoadBotUploader"
    ).start()

    camera = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA)

    if not camera.isOpened():
        print("ERROR: Cannot open camera:", CAMERA)
        print("Check that the webcam works with /dev/video0.")
        return

    # Reduce buffering where the camera driver supports it.
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("\n======================================")
    print("          ROADBOT DETECTION")
    print("======================================")
    print("Camera:", CAMERA)
    print("Photos:", PHOTO_DIR)
    print("Manual location:", LATITUDE, LONGITUDE)
    print("Robot key:", "Loaded" if ROBOT_UPLOAD_KEY else "MISSING")
    print("Red area threshold:", MIN_RED_AREA)
    print("Stable frames:", STABLE_FRAMES)
    print("Max photos per track:", MAX_PHOTOS_PER_OBJECT)
    print("Camera preview: OFF")
    print("Stop with Ctrl+C")
    print("======================================\n")

    frame_count = 0
    last_state_save = time.time()

    try:
        while True:
            ok, frame = camera.read()

            if not ok:
                print("WARNING: No camera frame.")
                time.sleep(0.1)
                continue

            frame_count += 1

            detections = detect_red(frame)

            with state_lock:
                for track in tracks:
                    track["matched"] = False

            process_detections(frame, detections)

            # Periodically persist track centers and IDs.
            if time.time() - last_state_save >= 2.0:
                save_state()
                last_state_save = time.time()

            # Print a heartbeat occasionally, without opening a window.
            if frame_count % 150 == 0:
                print(
                    f"Camera running | frames={frame_count} "
                    f"| red regions={len(detections)} "
                    f"| pending uploads={upload_queue.qsize()}"
                )

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nRoadBot stopped by user.")

    finally:
        save_state()
        camera.release()
        print("Camera released. Tracking state saved.")


if __name__ == "__main__":
    main()
