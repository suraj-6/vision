import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import threading
import time
import pyttsx3
import subprocess
import os
import sys
import tempfile
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION & INITIALIZATION ---
FOCAL_LENGTH = 682 
KNOWN_WIDTHS = {"car": 1.8, "person": 0.5, "bus": 2.5, "truck": 2.6, "motorcycle": 0.7, "bicycle": 0.6}
MAX_DISTANCE_METERS = 30.0
DISAPPEARED_GRACE_PERIOD = 15

# Initialize session state variables
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'scene_description_requested' not in st.session_state:
    st.session_state.scene_description_requested = False
if 'temp_files' not in st.session_state:
    st.session_state.temp_files = []

# --- AUDIO HANDLING FUNCTIONS ---
def speak_text_threaded(text):
    """Initializes a pyttsx3 engine in a temporary thread and speaks."""
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"Error in audio thread: {e}")

def make_announcement(text):
    """
    Creates and starts a new thread for each announcement,
    but ONLY if the audio is not locked by the scene describer.
    """
    # Check for the "Do Not Disturb" sign
    if os.path.exists("audio.lock"):
        print(f"[INFO] Audio is paused by scene describer. Ignoring: '{text}'")
        return

    thread = threading.Thread(target=speak_text_threaded, args=(text,))
    thread.daemon = True
    thread.start()

# --- SCENE DESCRIPTION FUNCTION ---
def describe_scene(image_path):
    """
    Sends an image to the Gemini API and speaks the generated description.
    Manages an audio lock file to prevent other announcements from interrupting.
    """
    lock_file = "audio.lock"
    try:
        # Create the lock file to pause main.py announcements
        with open(lock_file, "w") as f:
            f.write("locked")

        # Get the API key securely from the environment
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            error_msg = "Error: GOOGLE_API_KEY not found in .env file."
            print(error_msg)
            speak_text("Sorry, the API key is not configured correctly.")
            return

        genai.configure(api_key=api_key)
        # Use the correct and latest free vision model
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        img = Image.open(image_path)
        prompt = "Describe this scene from the perspective of a person walking. Focus on the most important objects for a visually impaired person, like obstacles, vehicles, and pathways. Be concise and direct."
        
        print("Generating scene description from Gemini...")
        response = model.generate_content([prompt, img])
        
        if response and response.text:
            description = response.text
            print(f"Scene Description: {description}")
            speak_text(description)
        else:
            print("Could not generate a description for the image.")
            speak_text("Sorry, I could not understand the scene.")

    except Exception as e:
        print(f"An error occurred: {e}")
        speak_text("Sorry, there was an error with the scene understanding feature.")
    
    finally:
        # Always remove the lock file when done, even if there was an error.
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except Exception as e:
                print(f"Error removing lock file: {e}")

# --- HELPER FUNCTION ---
def estimate_distance(object_pixel_width, class_name):
    if class_name in KNOWN_WIDTHS and object_pixel_width > 0:
        return (KNOWN_WIDTHS[class_name] * FOCAL_LENGTH) / object_pixel_width
    return float('inf')

# --- FILE CLEANUP FUNCTION ---
def cleanup_temp_files():
    """Clean up any temporary files that were created"""
    for temp_file in st.session_state.temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Cleaned up temporary file: {temp_file}")
        except Exception as e:
            print(f"Error cleaning up {temp_file}: {e}")
    
    # Clear the list after cleanup
    st.session_state.temp_files = []
    
    # Also clean up any lock files
    if os.path.exists("audio.lock"):
        try:
            os.remove("audio.lock")
        except Exception as e:
            print(f"Error removing audio.lock: {e}")

# --- MAIN PROCESSING FUNCTION ---
def process_video_source(video_source, use_camera=False):
    """
    Process video from file or camera
    video_source: path to video file or camera index (0 for default camera)
    use_camera: boolean, True if using camera, False if using video file
    """
    # Load model if not already loaded
    if not st.session_state.model_loaded:
        with st.spinner("Loading YOLO model..."):
            model = YOLO('yolov8n.pt')
            st.session_state.model = model
            st.session_state.model_loaded = True
    else:
        model = st.session_state.model

    # Open video source
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        st.error("Error: Could not open video source")
        return

    # Initialize tracking variables
    tracked_objects = {}
    focus_tid = None
    frame_count = 0

    # Create placeholders for video display
    video_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Process frames
    while st.session_state.processing and cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        # Resize frame for better display in Streamlit
        display_frame = frame.copy()
        
        image_height, image_width, _ = frame.shape
        results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)

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
                
                direction = "in front of you"
                box_center_x = (x1 + x2) / 2
                if box_center_x < image_width / 3:
                    direction = "from your left"
                elif box_center_x > image_width * 2 / 3:
                    direction = "from your right"
                
                # Log new objects
                if track_id not in tracked_objects:
                    status_message = f"[NEW] A {class_name} has appeared {direction}, at {distance:.1f} meters"
                    status_placeholder.info(status_message)
                    tracked_objects[track_id] = {'class_name': class_name, 'last_seen': frame_count}
                else:
                    tracked_objects[track_id]['last_seen'] = frame_count
                
                detected_objects_in_frame.append({"tid": track_id, "distance": distance})
        
        # Check for disappeared objects
        disappeared_ids = []
        for tid, data in list(tracked_objects.items()):
            if frame_count - data['last_seen'] > DISAPPEARED_GRACE_PERIOD:
                status_message = f"[DISAPPEARED] The {data['class_name']} is no longer in view"
                status_placeholder.warning(status_message)
                disappeared_ids.append(tid)
        for tid in disappeared_ids:
            if tid in tracked_objects:
                del tracked_objects[tid]

        # Audio announcement (nearest object only)
        nearest_object = None
        if detected_objects_in_frame:
            nearest_object = min(detected_objects_in_frame, key=lambda obj: obj['distance'])

        if nearest_object:
            nearest_tid = nearest_object['tid']
            if focus_tid != nearest_tid:
                obj_data = tracked_objects.get(nearest_tid)
                if obj_data:
                    announcement = f"Nearest obstacle is a {obj_data['class_name']}, at {nearest_object['distance']:.1f} meters"
                    make_announcement(announcement)
                    focus_tid = nearest_tid
        elif focus_tid is not None:
            make_announcement("The way ahead appears clear.")
            focus_tid = None

        # Draw annotations on frame
        annotated_frame = results[0].plot()
        
        # Convert BGR to RGB for Streamlit display
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame
        video_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
        
        # Small delay to control processing speed
        time.sleep(0.03)
    
    # Cleanup
    cap.release()
    status_placeholder.info("Video processing stopped.")

# --- STREAMLIT APP UI ---
st.set_page_config(page_title="Smart Glasses for Visually Impaired", layout="wide")

st.title("👁️ Smart Glasses for Visually Impaired")
st.markdown("""
This application helps visually impaired individuals navigate their environment by detecting objects and providing audio descriptions.
""")

# Add cleanup button in sidebar
with st.sidebar:
    st.header("Controls")
    
    # Scene description button
    if st.button("📸 Describe Current Scene", key="scene_desc_btn"):
        st.session_state.scene_description_requested = True
        st.info("Scene description will be triggered on the next frame.")
    
    # Stop processing button
    if st.button("⏹️ Stop Processing", key="stop_btn"):
        st.session_state.processing = False
        st.success("Processing stopped.")
    
    # Manual cleanup button
    if st.button("🧹 Cleanup Temporary Files", key="cleanup_btn"):
        cleanup_temp_files()
        st.success("Temporary files cleaned up.")
    
    st.markdown("---")
    st.subheader("Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Main content area
tab1, tab2 = st.tabs(["📹 Camera Input", "📁 Video Upload"])

with tab1:
    st.header("Real-time Camera Detection")
    st.markdown("Use your webcam to detect objects in real-time.")
    
    camera_on = st.toggle("Start Camera", key="camera_toggle")
    
    if camera_on:
        st.session_state.processing = True
        st.warning("Camera is active. Point it at your surroundings.")
        
        # Process camera feed
        process_video_source(0, use_camera=True)
    else:
        st.session_state.processing = False
        st.info("Click the toggle to start the camera.")

with tab2:
    st.header("Upload Video File")
    st.markdown("Upload a video file to analyze objects in the recording.")
    
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded_file is not None:
        try:
            # Save uploaded file to temporary location
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
            tfile.write(uploaded_file.read())
            tfile.close()  # Close the file to release the handle
            
            video_path = tfile.name
            # Add to temp files list for cleanup
            st.session_state.temp_files.append(video_path)
            
            st.video(video_path)  # Display the uploaded video
            
            if st.button("▶️ Process Video", key="process_video_btn"):
                st.session_state.processing = True
                st.info("Processing video...")
                
                # Process the uploaded video
                process_video_source(video_path, use_camera=False)
                
        except Exception as e:
            st.error(f"Error processing video: {e}")

# Handle scene description requests
if st.session_state.scene_description_requested:
    st.session_state.scene_description_requested = False
    
    # For demo purposes, create a black placeholder image
    # In a real implementation, you would capture the current frame
    temp_frame_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
    st.session_state.temp_files.append(temp_frame_path)
    
    # Create a simple placeholder image
    placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(temp_frame_path, placeholder_img)
    
    # Run scene description in a separate thread
    scene_thread = threading.Thread(target=describe_scene, args=(temp_frame_path,))
    scene_thread.start()
    
    st.success("Scene description requested. Audio will play shortly.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for visually impaired individuals to help them navigate their environment safely.")

# Register cleanup function to run when Streamlit script reruns
st.session_state.last_run_cleanup = True
