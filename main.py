import cv2
from ultralytics import YOLO
import numpy as np
import threading
import time
import pyttsx3
import subprocess
import os
import sys

# --- 1. CONFIGURATION & INITIALIZATION ---

FOCAL_LENGTH = 682 
KNOWN_WIDTHS = {"car": 1.8, "person": 0.5, "bus": 2.5, "truck": 2.6, "motorcycle": 0.7, "bicycle": 0.6}
MAX_DISTANCE_METERS = 30.0
DISAPPEARED_GRACE_PERIOD = 15

# --- The Reliable Audio Handling Method ---
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
    # NEW: Check for the "Do Not Disturb" sign
    if os.path.exists("audio.lock"):
        print(f"[INFO] Audio is paused by scene describer. Ignoring: '{text}'")
        return

    thread = threading.Thread(target=speak_text_threaded, args=(text,))
    thread.daemon = True
    thread.start()

# --- System Initialization ---
model = YOLO('yolov8n.pt')
video_path = r"./sample_video.mp4"
cap = cv2.VideoCapture(video_path)

# --- Tracking & Logic Variables ---
tracked_objects = {} 
focus_tid = None

# --- 2. HELPER FUNCTION ---
def estimate_distance(object_pixel_width, class_name):
    if class_name in KNOWN_WIDTHS and object_pixel_width > 0:
        return (KNOWN_WIDTHS[class_name] * FOCAL_LENGTH) / object_pixel_width
    return float('inf')

# --- 3. MAIN PROCESSING LOOP ---
try:
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame_count += 1
        
        # --- SCENE UNDERSTANDING & QUIT TRIGGER ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            temp_frame_path = "temp_frame.jpg"
            cv2.imwrite(temp_frame_path, frame)
            print("\n[INFO] Scene description requested. Pausing regular announcements.")
            subprocess.Popen([sys.executable, "scene_describer.py", temp_frame_path])
        
        if key == ord('q'):
            break

        image_height, image_width, _ = frame.shape
        results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)

        detected_objects_in_frame = []
        if results[0].boxes.id is not None:
            for box in results[0].boxes:
                track_id = int(box.id[0])
                class_name = model.names[int(box.cls[0])]
                if class_name not in KNOWN_WIDTHS: continue
                
                x1, y1, x2, y2 = box.xyxy[0]
                pixel_width = x2 - x1
                distance = estimate_distance(pixel_width, class_name)
                if distance > MAX_DISTANCE_METERS: continue
                
                direction = "in front of you"
                box_center_x = (x1 + x2) / 2
                if box_center_x < image_width / 3: direction = "from your left"
                elif box_center_x > image_width * 2 / 3: direction = "from your right"
                
                # --- DETAILED TERMINAL LOGGING ---
                if track_id not in tracked_objects:
                    print(f"[NEW] A {class_name} has appeared {direction}, at {distance:.1f} meters")
                    tracked_objects[track_id] = {'class_name': class_name, 'last_seen': frame_count}
                else:
                    tracked_objects[track_id]['last_seen'] = frame_count
                
                detected_objects_in_frame.append({ "tid": track_id, "distance": distance })
        
        # --- Check for disappeared objects for logging ---
        disappeared_ids = []
        for tid, data in list(tracked_objects.items()):
            if frame_count - data['last_seen'] > DISAPPEARED_GRACE_PERIOD:
                print(f"[DISAPPEARED] The {data['class_name']} is no longer in view")
                disappeared_ids.append(tid)
        for tid in disappeared_ids:
            if tid in tracked_objects:
                del tracked_objects[tid]

        # --- AUDIO ANNOUNCEMENT (NEAREST OBJECT ONLY) ---
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

        annotated_frame = results[0].plot()
        cv2.imshow("Intelligent Tracking System", annotated_frame)
        
finally:
    print("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()
    # Clean up any leftover files
    if os.path.exists("temp_frame.jpg"):
        os.remove("temp_frame.jpg")
    if os.path.exists("audio.lock"):
        os.remove("audio.lock")






































# import cv2
# from ultralytics import YOLO
# import numpy as np
# import time
# import subprocess
# import os
# import sys
# import socket

# # --- 1. CONFIGURATION ---
# FOCAL_LENGTH = 682 
# KNOWN_WIDTHS = {"car": 1.8, "person": 0.5, "bus": 2.5, "truck": 2.6, "motorcycle": 0.7, "bicycle": 0.6}
# MAX_DISTANCE_METERS = 30.0

# # --- NEW: Function to send text to the audio server ---
# def make_announcement(text):
#     """Connects to the audio server and sends text to be spoken."""
#     HOST, PORT = '127.0.0.1', 65432
#     try:
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#             s.connect((HOST, PORT))
#             s.sendall(text.encode('utf-8'))
#     except ConnectionRefusedError:
#         # This is expected if the server is not running or busy, so we can ignore it silently
#         pass
#     except Exception as e:
#         print(f"Error connecting to audio server: {e}")

# # --- System Initialization ---
# model = YOLO('yolov8n.pt')
# video_path = r"D:\DOWN LOADS\Smart_Glasses_For_Blind_People-main\Gemini\Bangalore city walk HMT main road - Citywalker (360p, h264).mp4"
# cap = cv2.VideoCapture(video_path)

# # --- Tracking & Logic Variables ---
# tracked_objects = {} 
# focus_tid = None
# ANNOUNCEMENT_COOLDOWN = 4.0
# DISAPPEARED_GRACE_PERIOD = 5
# audio_paused_until = 0 # NEW: Timestamp for pausing audio

# # --- 2. HELPER FUNCTION ---
# def estimate_distance(object_pixel_width, class_name):
#     if class_name in KNOWN_WIDTHS and object_pixel_width > 0:
#         return (KNOWN_WIDTHS[class_name] * FOCAL_LENGTH) / object_pixel_width
#     return float('inf')

# # --- 3. MAIN PROCESSING LOOP ---
# try:
#     frame_count = 0
#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success: break
        
#         frame_count += 1
#         current_time = time.time()
        
#         # --- SCENE UNDERSTANDING TRIGGER ---
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('c'):
#             temp_frame_path = "temp_frame.jpg"
#             cv2.imwrite(temp_frame_path, frame)
#             print("\n[INFO] Scene description requested. Pausing tracking announcements for 15 seconds.")
#             audio_paused_until = current_time + 15 # NEW: Pause audio for 15 seconds
#             # Use sys.executable to ensure we use the same Python interpreter
#             subprocess.Popen([sys.executable, "scene_describer.py", temp_frame_path])
        
#         if key == ord('q'):
#             break

#         image_height, image_width, _ = frame.shape
#         results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)

#         detected_objects_in_frame = []
#         if results[0].boxes.id is not None:
#             for box in results[0].boxes:
#                 track_id = int(box.id[0])
#                 class_name = model.names[int(box.cls[0])]
#                 if class_name not in KNOWN_WIDTHS: continue
                
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 pixel_width = x2 - x1
#                 distance = estimate_distance(pixel_width, class_name)
#                 if distance > MAX_DISTANCE_METERS: continue
                
#                 direction = "in front of you"
#                 box_center_x = (x1 + x2) / 2
#                 if box_center_x < image_width / 3: direction = "from your left"
#                 elif box_center_x > image_width * 2 / 3: direction = "from your right"
                
#                 if track_id not in tracked_objects:
#                     print(f"[NEW] A {class_name} has appeared {direction}, at {distance:.1f} meters")
#                     tracked_objects[track_id] = {'class_name': class_name, 'last_seen': frame_count, 'distance_history': [distance]}
#                 else:
#                     tracked_objects[track_id]['last_seen'] = frame_count
#                     history = tracked_objects[track_id]['distance_history']
#                     history.append(distance)
#                     if len(history) > 10: history.pop(0)
                
#                 detected_objects_in_frame.append({ "tid": track_id, "distance": distance })
        
#         disappeared_ids = []
#         for tid, data in list(tracked_objects.items()):
#             if frame_count - data['last_seen'] > DISAPPEARED_GRACE_PERIOD:
#                 print(f"[DISAPPEARED] The {data['class_name']} is no longer in view")
#                 disappeared_ids.append(tid)
#         for tid in disappeared_ids:
#             if tid in tracked_objects:
#                 del tracked_objects[tid]

#         # --- AUDIO ANNOUNCEMENT (only if not paused) ---
#         if current_time > audio_paused_until:
#             nearest_object = None
#             if detected_objects_in_frame:
#                 nearest_object = min(detected_objects_in_frame, key=lambda obj: obj['distance'])

#             if nearest_object:
#                 nearest_tid = nearest_object['tid']
#                 obj_data = tracked_objects.get(nearest_tid)
                
#                 if obj_data and current_time - obj_data.get('last_announced_time', 0) > ANNOUNCEMENT_COOLDOWN:
#                     announcement = ""
#                     history = obj_data['distance_history']
#                     if len(history) > 5:
#                         avg_past_dist = np.mean(history[:4])
#                         avg_current_dist = np.mean(history[-4:])
#                         change_in_dist = avg_past_dist - avg_current_dist
#                         motion_desc = ""
#                         if change_in_dist > 2.5: motion_desc = "is approaching swiftly"
#                         elif change_in_dist > 0.8: motion_desc = "is approaching"
#                         elif change_in_dist < -2.5: motion_desc = "is moving away swiftly"
#                         elif change_in_dist < -0.8: motion_desc = "is moving away"
#                         if motion_desc:
#                             announcement = f"The {obj_data['class_name']} {motion_desc}, now at {nearest_object['distance']:.1f} meters"
                    
#                     if not announcement and focus_tid != nearest_tid:
#                         announcement = f"Nearest obstacle is a {obj_data['class_name']}, at {nearest_object['distance']:.1f} meters"
                    
#                     if announcement:
#                         make_announcement(announcement)
#                         obj_data['last_announced_time'] = current_time
#                         focus_tid = nearest_tid

#             elif focus_tid is not None:
#                 make_announcement("The way ahead appears clear.")
#                 focus_tid = None

#         annotated_frame = results[0].plot()
#         cv2.imshow("Intelligent Tracking System - Press 'c' for Scene Description", annotated_frame)
        
# finally:
#     print("Shutting down...")
#     cap.release()
#     cv2.destroyAllWindows()
#     if os.path.exists("temp_frame.jpg"):
#         os.remove("temp_frame.jpg")



























# import cv2
# from ultralytics import YOLO
# import pyttsx3

# # --- 1. INITIALIZATION ---

# # Load the pre-trained YOLOv8 model (yolov8n.pt is the smallest and fastest)
# model = YOLO('yolov8n.pt')

# # Initialize the text-to-speech engine
# engine = pyttsx3.init()

# # --- 2. IMAGE PROCESSING ---

# # Load the image you captured
# image_path = 'D:\DOWN LOADS\Smart_Glasses_For_Blind_People-main\Gemini\image.png' # Replace with your image path
# img = cv2.imread(image_path)

# # Perform object detection on the image
# results = model(img)

# # --- 3. LOGIC & AUDIO OUTPUT ---

# # Get image dimensions for location logic
# image_height, image_width, _ = img.shape

# # A list to hold descriptions of detected objects
# descriptions = []

# # Process the detection results
# for result in results:
#     for box in result.boxes:
#         # Get the class name of the detected object
#         class_id = int(box.cls[0])
#         class_name = model.names[class_id]

#         # Get the bounding box coordinates
#         x1, y1, x2, y2 = box.xyxy[0]

#         # --- This is the "Smart" Logic Part ---
#         # Determine the object's horizontal position
#         box_center_x = (x1 + x2) / 2
        
#         position = ""
#         if box_center_x < image_width / 3:
#             position = f"to your left"
#         elif box_center_x > image_width * 2 / 3:
#             position = f"to your right"
#         else:
#             position = f"in front of you"
            
#         # Optional: Determine proximity based on box size (a simple heuristic)
#         box_area = (x2 - x1) * (y2 - y1)
#         proximity = ""
#         if box_area > (image_width * image_height) * 0.2: # If object takes > 20% of the screen
#             proximity = "close"

#         # Create a human-readable description
#         description = f"{proximity} {class_name} {position}"
#         descriptions.append(description.strip())


# # --- 4. GENERATE FINAL AUDIO COMMAND ---

# if not descriptions:
#     final_output = "The way ahead is clear."
# else:
#     # Join all descriptions into a single sentence
#     final_output = ". ".join(descriptions)

# print(f"Generated Text: {final_output}")

# # Convert the final text to speech
# engine.say(final_output)

# engine.runAndWait()
