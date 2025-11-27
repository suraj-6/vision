import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import threading
import time
import pyttsx3
import os
import tempfile
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
from streamlit_extras.stylable_container import stylable_container

# Load environment variables
load_dotenv()

# --- CONFIGURATION & INITIALIZATION ---
FOCAL_LENGTH = 682 
KNOWN_WIDTHS = {# --- People & Animals ---
    "person": 0.5,
    "dog": 0.35,
    "street dog": 0.4,
    "cat": 0.25,
    "cow": 0.8,
    "goat": 0.35,
    "horse": 0.6,
    "sheep": 0.4,
    "bird": 0.15,

    # --- Vehicles ---
    "car": 1.8,
    "bus": 2.5,
    "truck": 2.6,
    "motorcycle": 0.7,
    "bicycle": 0.6,
    "scooter": 0.65,
    "autorickshaw": 1.3,
    "van": 1.9,
    "ambulance": 2.0,
    "tractor": 2.2,
    "train": 3.2,

    # --- Large Outdoor Objects & Street Infrastructure ---
    "traffic light": 0.6,
    "stop sign": 0.6,
    "school zone sign": 0.9,
    "hospital sign": 1.0,
    "pole": 0.15,
    "street light": 0.3,
    "sign": 0.8,
    "traffic sign": 0.6,
    "traffic cone": 0.3,
    "barricade": 1.2,
    "traffic barricade": 1.2,
    "road barrier": 1.0,
    "bus stop": 2.0,
    "auto stand": 1.5,
    "railway track": 1.6,
    "bridge": 5.0,

    # --- Hazards (Critical for Visually Impaired) ---
    "pothole": 0.7,
    "open manhole": 0.6,
    "manhole": 0.6,
    "speed bump": 1.0,
    "stairs": 1.0,
    "step": 0.3,
    "ramp": 1.0,

    # --- Furniture / Indoor Objects ---
    "chair": 0.5,
    "table": 1.1,
    "dining table": 1.2,
    "desk": 1.0,
    "sofa": 2.0,
    "couch": 2.0,
    "bed": 1.6,
    "wardrobe": 1.2,
    "cupboard": 1.1,
    "bookshelf": 0.9,
    "tv": 1.1,
    "monitor": 0.55,

    # --- Personal Items ---
    "smartphone": 0.08,
    "cell phone": 0.08,
    "laptop": 0.33,
    "wallet": 0.12,
    "id card": 0.085,
    "backpack": 0.35,
    "handbag": 0.28,
    "suitcase": 0.45,
    "tiffin": 0.18,
    "bottle": 0.07,
    "cup": 0.08,

    # --- Household & Hygiene Items ---
    "toothbrush": 0.025,
    "toothpaste": 0.04,
    "soap": 0.06,
    "shampoo bottle": 0.08,
    "hairbrush": 0.08,
    "comb": 0.03,
    "towel": 0.4,
    "bucket": 0.30,
    "mug": 0.10,
    "plate": 0.25,
    "bowl": 0.18,
    "spoon": 0.03,
    "fork": 0.025,
    "knife": 0.03,
    "cooking pot": 0.28,
    "pan": 0.30,
    "frying pan": 0.32,
    "kettle": 0.20,
    "pressure cooker": 0.30,
    "gas cylinder": 0.32,
    "milk packet": 0.12,
    "egg tray": 0.25,
    "rice bag": 0.40,
    "flour bag": 0.40,

    # --- Cleaning Tools ---
    "broom": 0.03,
    "mop": 0.15,
    "cleaning cloth": 0.25,
    "scrub pad": 0.12,
    "detergent bottle": 0.1,
    "phenyl bottle": 0.1,
    "floor wiper": 0.35,

    # --- Kitchen Appliances ---
    "microwave": 0.50,
    "blender": 0.15,
    "mixer grinder": 0.35,
    "toaster": 0.30,
    "induction stove": 0.35,
    "gas stove": 0.60,
    "fridge": 0.75,
    "refrigerator": 0.75,
    "washing machine": 0.60,

    # --- Office/Home Small Items ---
    "remote": 0.05,
    "charger": 0.06,
    "power bank": 0.07,
    "flashlight": 0.04,
    "scissors": 0.12,
    "tape": 0.06,

    # --- Natural Objects ---
    "tree": 1.5,
    "plant": 0.4,
    "potted plant": 0.4,
    "bush": 1.0,

    # --- Industrial / Other Objects ---
    "water tank": 1.8,
    "solar panel": 1.6,
    "solar panels": 1.6,
    "transformer": 1.5,
    "pipeline": 0.5,

    # --- Shopping / Public Area Items ---
    "shopping cart": 1.0,
    "trolley": 1.0,
    "basket": 0.35,
    "box": 0.40,
    "package": 0.35,}
MAX_DISTANCE_METERS = 30.0
DISAPPEARED_GRACE_PERIOD = 15
AUTO_DESCRIBE_INTERVAL = 10  # seconds

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'scene_description_requested' not in st.session_state:
    st.session_state.scene_description_requested = False
if 'temp_files' not in st.session_state:
    st.session_state.temp_files = []
if 'latest_frame' not in st.session_state:
    st.session_state.latest_frame = None
if 'description_history' not in st.session_state:
    st.session_state.description_history = []
if 'auto_describe' not in st.session_state:
    st.session_state.auto_describe = False
if 'last_auto_describe_time' not in st.session_state:
    st.session_state.last_auto_describe_time = 0

# --- AUDIO HANDLING ---
def speak_text_threaded(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 180)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"Audio Error: {e}")

def make_announcement(text):
    if os.path.exists("audio.lock"):
        print(f"[INFO] Audio locked. Skipping: {text}")
        return
    thread = threading.Thread(target=speak_text_threaded, args=(text,))
    thread.daemon = True
    thread.start()

# --- SCENE DESCRIPTION FUNCTION ---
def describe_scene(image_path, return_text=False):
    lock_file = "audio.lock"
    try:
        with open(lock_file, "w") as f:
            f.write("locked")

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            msg = "API key not configured."
            if return_text: return msg
            else: speak_text_threaded(msg)
            return

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        img = Image.open(image_path)
        prompt = (
            "You are the eyes of a visually impaired person walking. The image shows detected objects with labels. "
            "Describe ONLY what matters for safe navigation:\n"
            "- Obstacles within 5 meters and their direction (left/center/right)\n"
            "- Moving threats (cars, bikes) and their movement\n"
            "- Safe paths, doors, stairs, curbs, crosswalks, traffic lights\n"
            "- Urgent warnings if danger is close (<2m)\n"
            "Be concise, calm, directional, and practical. Example: 'Caution: bicycle approaching fast from right at 2 meters.'"
        )
        
        response = model.generate_content([prompt, img])
        
        if response and response.text:
            description = response.text.strip()
            timestamp = time.strftime("%H:%M:%S")
            record = f"[{timestamp}] {description}"
            st.session_state.description_history.append(record)
            
            if return_text:
                return description
            else:
                speak_text_threaded(description)
                return description
        else:
            msg = "No description available."
            if return_text: return msg
            else: speak_text_threaded(msg)
            return msg

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        if return_text: return error_msg
        else: speak_text_threaded("Sorry, scene understanding failed.")
        return error_msg
    finally:
        if os.path.exists(lock_file):
            try: os.remove(lock_file)
            except: pass

# --- HELPER FUNCTIONS ---
def estimate_distance(pixel_width, class_name):
    if class_name in KNOWN_WIDTHS and pixel_width > 0:
        return (KNOWN_WIDTHS[class_name] * FOCAL_LENGTH) / pixel_width
    return float('inf')

def cleanup_temp_files():
    for temp_file in st.session_state.temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except: pass
    st.session_state.temp_files = []
    if os.path.exists("audio.lock"):
        try: os.remove("audio.lock")
        except: pass

# --- MAIN PROCESSING FUNCTION ---
def process_video_source(video_source, use_camera=False):
    if not st.session_state.model_loaded:
        with st.spinner("Loading YOLO model..."):
            model = YOLO('yolov8n.pt')
            st.session_state.model = model
            st.session_state.model_loaded = True
    else:
        model = st.session_state.model

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error("Could not open video source")
        return

    tracked_objects = {}
    focus_tid = None
    frame_count = 0
    last_frame_time = time.time()

    video_placeholder = st.empty()
    status_placeholder = st.empty()

    while st.session_state.processing and cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame_count += 1
        image_height, image_width, _ = frame.shape
        results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)

        detected_objects_in_frame = []
        dangerous_objects = []

        if results[0].boxes.id is not None:
            for box in results[0].boxes:
                track_id = int(box.id[0])
                class_name = model.names[int(box.cls[0])]
                if class_name not in KNOWN_WIDTHS: continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pixel_width = x2 - x1
                distance = estimate_distance(pixel_width, class_name)
                if distance > MAX_DISTANCE_METERS: continue
                
                direction = "in front of you"
                box_center_x = (x1 + x2) / 2
                if box_center_x < image_width / 3:
                    direction = "from your left"
                elif box_center_x > image_width * 2 / 3:
                    direction = "from your right"

                if track_id not in tracked_objects:
                    msg = f"[NEW] {class_name} {direction}, {distance:.1f}m"
                    status_placeholder.info(msg)
                    tracked_objects[track_id] = {'class_name': class_name, 'last_seen': frame_count}
                else:
                    tracked_objects[track_id]['last_seen'] = frame_count

                detected_objects_in_frame.append({"tid": track_id, "distance": distance, "class": class_name, "dir": direction})

                # Flag dangerous objects (< 2m)
                if distance < 2.0:
                    dangerous_objects.append(f"{class_name} {direction} at {distance:.1f}m")

        # Handle disappeared objects
        disappeared_ids = []
        for tid, data in list(tracked_objects.items()):
            if frame_count - data['last_seen'] > DISAPPEARED_GRACE_PERIOD:
                msg = f"[GONE] {data['class_name']} disappeared"
                status_placeholder.warning(msg)
                disappeared_ids.append(tid)
        for tid in disappeared_ids:
            tracked_objects.pop(tid, None)

        # Audio announcement (nearest object)
        if detected_objects_in_frame:
            nearest = min(detected_objects_in_frame, key=lambda x: x['distance'])
            if focus_tid != nearest['tid']:
                obj = tracked_objects.get(nearest['tid'])
                if obj:
                    announcement = f"Nearest: {obj['class_name']} {nearest['dir']}, {nearest['distance']:.1f} meters"
                    make_announcement(announcement)
                    focus_tid = nearest['tid']
        elif focus_tid is not None:
            make_announcement("Path ahead is clear.")
            focus_tid = None

        # Auto-describe every N seconds
        current_time = time.time()
        if (st.session_state.auto_describe and 
            current_time - st.session_state.last_auto_describe_time >= AUTO_DESCRIBE_INTERVAL):
            st.session_state.last_auto_describe_time = current_time
            if st.session_state.latest_frame is not None:
                temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
                st.session_state.temp_files.append(temp_path)
                cv2.imwrite(temp_path, st.session_state.latest_frame)
                auto_thread = threading.Thread(target=describe_scene, args=(temp_path,))
                auto_thread.start()
                st.toast("🤖 Auto-description triggered!", icon="⏱️")

        # Save latest frame for scene description
        annotated_frame = results[0].plot()
        st.session_state.latest_frame = annotated_frame.copy()

        # Convert for Streamlit
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)

        time.sleep(0.03)

    cap.release()
    status_placeholder.info("⏹️ Video processing stopped.")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); color: #ffffff; }
    h1, h2, h3, h4, h5, h6 { color: #ffffff !important; font-family: 'Segoe UI', sans-serif; }
    .stButton>button {
        border-radius: 12px; padding: 12px 24px; font-weight: bold; border: none;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2); transition: all 0.3s ease;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0,0,0,0.3); }
    .stSlider>div>div>div { background: #4CAF50 !important; }
    .css-1d391kg { background-color: #1a2a3a !important; }
    .stAlert { border-radius: 12px; }
    .stVideo { border-radius: 16px; box-shadow: 0 8px 32px rgba(0,0,0,0.3); }
    .block-container { padding-top: 2rem; }
    footer {visibility: hidden;}
    .history-item { background: rgba(255,255,255,0.1); padding: 10px; margin: 5px 0; border-radius: 8px; }
    .made-with-love {
        text-align: center; padding: 20px; font-size: 14px; color: #aaa;
        border-top: 1px solid #333; margin-top: 40px;
    }
    .vibrate { animation: vibrate 0.3s linear infinite; }
    @keyframes vibrate { 0% { transform: translateX(0); } 25% { transform: translateX(2px); } 50% { transform: translateX(-2px); } 75% { transform: translateX(2px); } 100% { transform: translateX(0); } }
</style>
""", unsafe_allow_html=True)

# --- STREAMLIT UI ---
st.markdown("<h1 style='text-align: center;'>👁️ AI-Powered Smart Navigator : Real-Time Object Detection And Voice Assistance For The Visually Impaired </h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 1.2rem; margin-bottom: 2rem;'>
    Real-time object detection + AI scene understanding to navigate safely.
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='color: #4CAF50;'>🛠️ Controls</h2>", unsafe_allow_html=True)
    
    with stylable_container(key="scene_desc_btn", css_styles="button { background-color: #FF6B6B; color: white; border-radius: 12px; }"):
        if st.button("📸 Describe Current Scene", use_container_width=True):
            st.session_state.scene_description_requested = True
            st.toast("✅ Capturing current frame...", icon="📷")

    with stylable_container(key="auto_desc_btn", css_styles="button { background-color: #5D3FD3; color: white; border-radius: 12px; }"):
        st.session_state.auto_describe = st.toggle("⏱️ Auto-Describe Every 10s", value=st.session_state.auto_describe, help="Hands-free periodic scene updates")

    with stylable_container(key="stop_btn", css_styles="button { background-color: #e74c3c; color: white; border-radius: 12px; }"):
        if st.button("⏹️ Stop Processing", use_container_width=True):
            st.session_state.processing = False
            st.session_state.latest_frame = None
            st.toast("⏹️ Stopped.", icon="✋")

    with stylable_container(key="cleanup_btn", css_styles="button { background-color: #95a5a6; color: white; border-radius: 12px; }"):
        if st.button("🧹 Cleanup Files", use_container_width=True):
            cleanup_temp_files()
            st.toast("✨ Cleaned up!", icon="✅")
    
    st.markdown("---")
    st.markdown("<h3 style='color: #4CAF50;'>⚙️ Settings</h3>", unsafe_allow_html=True)
    confidence_threshold = st.slider("🎯 Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📹 Camera", "📁 Video", "🖼️ Image", "📜 History"])

# Tab 1: Live Camera
with tab1:
    st.markdown("### 👁️ Real-Time Detection via Webcam")
    st.info("🔊 Audio announces nearest obstacle. Toggle below to start.")

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        camera_on = st.toggle("🟢 Start Camera", value=False, key="camera_toggle")

    if camera_on:
        st.session_state.processing = True
        with st.spinner("Initializing camera..."):
            time.sleep(1)
        st.success("🎥 Camera active. Point toward your path!")

        video_placeholder = st.empty()
        status_placeholder = st.empty()
        process_video_source(0, use_camera=True)

        st.info("ℹ️ Click 'Stop Processing' in sidebar when done.")
    else:
        st.session_state.processing = False
        st.markdown("""
        <div style='background: rgba(255,255,255,0.1); padding: 30px; border-radius: 16px; text-align: center;'>
            <h3>👉 Toggle above to start camera</h3>
            <p>Avoid obstacles with real-time audio guidance.</p>
        </div>
        """, unsafe_allow_html=True)

# Tab 2: Upload Video
with tab2:
    st.markdown("### 📁 Analyze Pre-recorded Videos")
    st.info("Upload MP4/AVI/MOV to detect objects frame-by-frame.")

    uploaded_file = st.file_uploader("📤 Drop video here", type=['mp4','avi','mov','mkv'])

    if uploaded_file:
        try:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.close()
            video_path = tfile.name
            st.session_state.temp_files.append(video_path)
            st.video(video_path)

            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                if st.button("▶️ Process Video", use_container_width=True, type="primary"):
                    st.session_state.processing = True
                    with st.spinner("Processing..."):
                        process_video_source(video_path, use_camera=False)
                    st.balloons()
                    st.success("✅ Analysis complete!")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Tab 3: Upload Image
with tab3:
    st.markdown("### 🖼️ Describe Any Image")
    st.info("Upload JPG/PNG. Gemini describes scene for safe navigation.")

    uploaded_image = st.file_uploader("Choose image...", type=["jpg","jpeg","png"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        temp_img_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
        st.session_state.temp_files.append(temp_img_path)
        image.save(temp_img_path)

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("🧠 Describe This Image", use_container_width=True, type="primary"):
                with st.spinner("Analyzing with Gemini..."):
                    desc = describe_scene(temp_img_path, return_text=True)
                    st.markdown("### 📝 AI Description")
                    if "Caution:" in desc or "Danger:" in desc:
                        st.error(desc, icon="⚠️")
                    else:
                        st.success(desc)
                    speak_thread = threading.Thread(target=speak_text_threaded, args=(desc,))
                    speak_thread.start()
                    st.toast("🔊 Description playing", icon="💬")

# Tab 4: Description History
with tab4:
    st.markdown("### 📜 Scene Description History")
    if st.session_state.description_history:
        for i, record in enumerate(reversed(st.session_state.description_history)):
            if i >= 20: break  # Show last 20
            if "Caution:" in record or "Danger:" in record:
                st.markdown(f"<div class='history-item vibrate'>{record}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='history-item'>{record}</div>", unsafe_allow_html=True)
    else:
        st.info("No descriptions yet. Use camera or upload image.")

# Handle Scene Description Request (Live Frame)
if st.session_state.scene_description_requested:
    st.session_state.scene_description_requested = False

    if st.session_state.latest_frame is None:
        msg = "Start camera first."
        st.warning(f"⚠️ {msg}")
        speak_text_threaded(msg)
    else:
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
        st.session_state.temp_files.append(temp_path)
        cv2.imwrite(temp_path, st.session_state.latest_frame)

        with st.sidebar:
            st.image(st.session_state.latest_frame, caption="📸 Captured Frame", channels="BGR", use_column_width=True)

        scene_thread = threading.Thread(target=describe_scene, args=(temp_path,))
        scene_thread.start()
        st.toast("🧠 Analyzing scene... Audio coming soon.", icon="👁️")

# Footer
st.markdown("""
<div class="made-with-love">
    Made with ❤️ and ♿ for independent, safe navigation.<br>
    <small>v2.0 • YOLOv8 + Gemini Flash • Real-time Awareness</small>
</div>
""", unsafe_allow_html=True)

# Auto-cleanup on rerun
if 'last_run_cleanup' not in st.session_state:
    st.session_state.last_run_cleanup = True
    cleanup_temp_files()
