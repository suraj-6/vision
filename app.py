import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import threading
import time
import os
import tempfile
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# ========== Optional TTS (local vs browser) ==========
try:
    import pyttsx3
    USE_LOCAL_TTS = True
except ImportError:
    from gtts import gTTS
    USE_LOCAL_TTS = False

# ========== Load .env (for GOOGLE_API_KEY) ==========
load_dotenv()

# --- CONFIGURATION ---
FOCAL_LENGTH = 682
KNOWN_WIDTHS = {"car": 1.8, "person": 0.5, "bus": 2.5,
                "truck": 2.6, "motorcycle": 0.7, "bicycle": 0.6}
MAX_DISTANCE_METERS = 30.0
DISAPPEARED_GRACE_PERIOD = 15


# --- Streamlit Session State ---
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'scene_description_requested' not in st.session_state:
    st.session_state.scene_description_requested = False


# --- AUDIO HANDLING ---
def speak_text(text: str):
    """TTS: use pyttsx3 locally if available, otherwise gTTS+st.audio in browser."""
    if USE_LOCAL_TTS:
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print("Local TTS error:", e)
    else:
        try:
            tts = gTTS(text=text, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                st.audio(fp.name, format="audio/mp3", autoplay=True)
        except Exception as e:
            st.warning(f"TTS error: {e}")


def make_announcement(text):
    """Threaded announcements, skip if scene description is active."""
    if os.path.exists("audio.lock"):
        return
    thread = threading.Thread(target=speak_text, args=(text,))
    thread.daemon = True
    thread.start()


# --- GEMINI SCENE DESCRIPTION ---
def describe_scene(image_path: str):
    """Send frame to Gemini API and narrate description."""
    lock_file = "audio.lock"
    try:
        with open(lock_file, "w") as f:
            f.write("locked")

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            make_announcement("Sorry, the API key is not configured.")
            return

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        img = Image.open(image_path)

        prompt = (
            "Describe this scene for a visually impaired person. "
            "Focus on obstacles, vehicles, pathways, and important context."
        )

        response = model.generate_content([prompt, img])
        if response and response.text:
            make_announcement(response.text)
        else:
            make_announcement("Sorry, I could not understand the scene.")

    except Exception as e:
        make_announcement("An error occurred with scene description.")
        print("Scene description error:", e)
    finally:
        if os.path.exists(lock_file):
            os.remove(lock_file)


# --- HELPERS ---
def estimate_distance(object_pixel_width, class_name):
    if class_name in KNOWN_WIDTHS and object_pixel_width > 0:
        return (KNOWN_WIDTHS[class_name] * FOCAL_LENGTH) / object_pixel_width
    return float('inf')


# --- YOLO VIDEO/WEBCAM LOOP ---
def process_video_source(video_source):
    if not st.session_state.model_loaded:
        with st.spinner("Loading YOLO model..."):
            model = YOLO("yolov8n.pt")
            st.session_state.model = model
            st.session_state.model_loaded = True
    else:
        model = st.session_state.model

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error("Error: could not open video source")
        return

    tracked_objects = {}
    focus_tid = None
    frame_count = 0
    video_placeholder = st.empty()
    status_placeholder = st.empty()

    while st.session_state.processing and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        h, w, _ = frame.shape

        results = model.track(frame, persist=True,
                              tracker="botsort.yaml", verbose=False)

        detected_objects_in_frame = []
        if results[0].boxes.id is not None:
            for box in results[0].boxes:
                track_id = int(box.id[0])
                class_name = model.names[int(box.cls[0])]
                if class_name not in KNOWN_WIDTHS:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]
                pixel_width = x2 - x1
                distance = estimate_distance(pixel_width, class_name)
                if distance > MAX_DISTANCE_METERS:
                    continue

                # Position description
                cx = (x1 + x2) / 2
                if cx < w/3: direction = "on your left"
                elif cx > 2*w/3: direction = "on your right"
                else: direction = "ahead"

                if track_id not in tracked_objects:
                    status_placeholder.info(
                        f"[NEW] A {class_name} appeared {direction}, {distance:.1f}m")
                    tracked_objects[track_id] = {"class_name": class_name,
                                                 "last_seen": frame_count}
                else:
                    tracked_objects[track_id]["last_seen"] = frame_count
                detected_objects_in_frame.append(
                    {"tid": track_id, "distance": distance})

        # remove disappeared
        disappeared = [tid for tid, d in tracked_objects.items()
                       if frame_count - d["last_seen"] > DISAPPEARED_GRACE_PERIOD]
        for tid in disappeared:
            status_placeholder.warning(
                f"[DISAPPEARED] The {tracked_objects[tid]['class_name']} left view")
            del tracked_objects[tid]

        # announcements
        if detected_objects_in_frame:
            nearest = min(detected_objects_in_frame, key=lambda x: x["distance"])
            if focus_tid != nearest["tid"]:
                obj = tracked_objects[nearest["tid"]]
                make_announcement(
                    f"Nearest obstacle is a {obj['class_name']}, at {nearest['distance']:.1f} meters")
                focus_tid = nearest["tid"]
        elif focus_tid is not None:
            make_announcement("The way ahead appears clear.")
            focus_tid = None

        # display frame
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        video_placeholder.image(annotated_rgb,
                                channels="RGB", use_column_width=True)

        time.sleep(0.03)

    cap.release()
    status_placeholder.info("Processing stopped.")


# --- STREAMLIT UI ---
st.set_page_config(page_title="Smart Glasses", layout="wide")
st.title("👁️ Smart Glasses for Visually Impaired")
st.markdown("Real-time object detection with audio guidance.")

with st.sidebar:
    st.header("Controls")
    if st.button("📸 Describe Scene"):
        st.session_state.scene_description_requested = True
    if st.button("⏹ Stop"):
        st.session_state.processing = False
    st.subheader("Settings")
    st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)


tab1, tab2 = st.tabs(["📹 Camera", "📁 Upload Video"])

# --- Camera Mode ---
with tab1:
    st.subheader("Camera Mode")

    # Check if we're on Streamlit Cloud
    if os.environ.get("STREAMLIT_RUNTIME", None):
        # In the cloud → use browser camera
        st.warning("⚠️ Webcam not available on server. Using your browser camera instead.")
        img_file = st.camera_input("Take a picture")
        if img_file is not None:
            bytes_data = img_file.getvalue()
            np_img = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            if not st.session_state.model_loaded:
                st.session_state.model = YOLO("yolov8n.pt")
                st.session_state.model_loaded = True

            results = st.session_state.model(frame)
            annotated = results[0].plot()
            st.image(annotated, channels="BGR")
    else:
        # Local mode → use real webcam
        if st.toggle("Start Webcam"):
            st.session_state.processing = True
            process_video_source(0)    # Safe: only runs locally

# --- Video Upload Mode ---
with tab2:
    st.subheader("Video Upload Mode")
    upl = st.file_uploader("Upload", type=["mp4", "avi", "mov", "mkv"])
    if upl is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(upl.read())
        st.video(tfile.name)
        if st.button("▶ Process Video"):
            st.session_state.processing = True
            process_video_source(tfile.name)
        os.unlink(tfile.name)

# --- Scene Description ---
if st.session_state.scene_description_requested:
    st.session_state.scene_description_requested = False
    temp_frame = "scene.jpg"
    cv2.imwrite(temp_frame, np.zeros((480, 640, 3), np.uint8))  # dummy frame
    threading.Thread(target=describe_scene, args=(temp_frame,), daemon=True).start()
    st.success("Scene description requested.")

