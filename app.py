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

# ========== Optional TTS ==========
try:
    import pyttsx3
    USE_LOCAL_TTS = True
except ImportError:
    from gtts import gTTS
    USE_LOCAL_TTS = False

load_dotenv()

# ========== CONFIG ==========
FOCAL_LENGTH = 682
KNOWN_WIDTHS = {"car": 1.8, "person": 0.5, "bus": 2.5,
                "truck": 2.6, "motorcycle": 0.7, "bicycle": 0.6}
MAX_DISTANCE_METERS = 30.0
DISAPPEARED_GRACE_PERIOD = 15

if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'scene_description_requested' not in st.session_state:
    st.session_state.scene_description_requested = False

# ========== AUDIO ==========
def speak_text(text: str):
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
    if os.path.exists("audio.lock"):  # block during Gemini
        return
    threading.Thread(target=speak_text, args=(text,), daemon=True).start()

# ========== GEMINI ==========
def describe_scene(image_path: str):
    lock_file = "audio.lock"
    try:
        with open(lock_file, "w") as f:
            f.write("locked")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            make_announcement("Sorry, no API key configured.")
            return
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        img = Image.open(image_path)
        prompt = "Describe this scene for a visually impaired person in concise terms."
        response = model.generate_content([prompt, img])
        if response and response.text:
            make_announcement(response.text)
        else:
            make_announcement("I could not understand the scene.")
    except Exception as e:
        print("Scene description failed:", e)
        make_announcement("Error with scene understanding.")
    finally:
        if os.path.exists(lock_file):
            os.remove(lock_file)

# ========== HELPERS ==========
def estimate_distance(object_pixel_width, class_name):
    if class_name in KNOWN_WIDTHS and object_pixel_width > 0:
        return (KNOWN_WIDTHS[class_name] * FOCAL_LENGTH) / object_pixel_width
    return float('inf')

# ========== YOLO LOOP (local only) ==========
def process_video_source(video_source):
    if not st.session_state.model_loaded:
        st.session_state.model = YOLO("yolov8n.pt")
        st.session_state.model_loaded = True
    model = st.session_state.model

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error("Error: Could not open video source")
        return

    tracked_objects = {}
    focus_tid = None
    frame_count = 0
    placeholder = st.empty()

    while st.session_state.processing and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        h, w, _ = frame.shape

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

        detected = []
        if results[0].boxes.id is not None:
            for box in results[0].boxes:
                track_id = int(box.id[0])
                cname = model.names[int(box.cls[0])]
                if cname not in KNOWN_WIDTHS: continue
                x1,y1,x2,y2 = box.xyxy[0]
                pixel_width = x2 - x1
                dist = estimate_distance(pixel_width, cname)
                if dist > MAX_DISTANCE_METERS: continue
                cx = (x1+x2)/2
                if cx < w/3: direction = "on your left"
                elif cx > 2*w/3: direction = "on your right"
                else: direction = "ahead"
                tracked_objects[track_id] = {"class_name": cname, "last_seen": frame_count}
                detected.append((track_id, cname, dist))
        
        if detected:
            nearest_tid, cname, dist = min(detected, key=lambda x: x[2])
            if focus_tid != nearest_tid:
                make_announcement(f"Nearest object is a {cname}, {dist:.1f}m {direction}")
                focus_tid = nearest_tid
        elif focus_tid is not None:
            make_announcement("The way ahead appears clear.")
            focus_tid = None

        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        placeholder.image(annotated_rgb, channels="RGB", use_column_width=True)

        time.sleep(0.03)

    cap.release()

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="Smart Glasses", layout="wide")
st.title("👁️ Smart Glasses for Visually Impaired")
st.write("Dual-mode: works locally with webcam, or in cloud with browser camera/video upload.")

with st.sidebar:
    st.header("Controls")
    if st.button("📸 Describe Scene"):
        st.session_state.scene_description_requested = True
    if st.button("⏹ Stop"):
        st.session_state.processing = False

tabs = st.tabs(["📹 Camera", "📁 Upload Video"])

# --- Camera Tab ---
with tabs[0]:
    st.subheader("Camera")
    if os.environ.get("STREAMLIT_RUNTIME"):   # Cloud mode
        st.info("Running on Cloud: use browser camera 📷")
        img_file = st.camera_input("Take a picture")
        if img_file:
            np_img = np.frombuffer(img_file.getvalue(), np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            if not st.session_state.model_loaded:
                st.session_state.model = YOLO("yolov8n.pt")
                st.session_state.model_loaded = True
            results = st.session_state.model(frame)
            annotated = results[0].plot()
            st.image(annotated, channels="BGR")
    else:  # Local laptop
        if st.toggle("Start Webcam"):
            st.session_state.processing = True
            process_video_source(0)

# --- Video Upload Tab ---
with tabs[1]:
    upl = st.file_uploader("Upload a video", type=["mp4","avi","mov","mkv"])
    if upl:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(upl.read()); tfile.flush()
        st.video(tfile.name)
        if st.button("▶ Process Uploaded Video"):
            st.session_state.processing = True
            process_video_source(tfile.name)

# --- Scene description trigger ---
if st.session_state.scene_description_requested:
    st.session_state.scene_description_requested = False
    tmp = "scene.jpg"
    cv2.imwrite(tmp, np.zeros((480,640,3), np.uint8))
    threading.Thread(target=describe_scene, args=(tmp,), daemon=True).start()
    st.success("Scene description requested")
