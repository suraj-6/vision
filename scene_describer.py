import google.generativeai as genai
from PIL import Image
import pyttsx3
import os
import sys
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# --- Audio Handling ---
def speak_text(text):
    """Initializes a pyttsx3 engine and speaks the given text."""
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"Error in audio engine: {e}")

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
        # CORRECTED: Use the correct and latest free vision model
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
        # IMPORTANT: Always remove the lock file when done, even if there was an error.
        if os.path.exists(lock_file):
            os.remove(lock_file)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_file = sys.argv[1]
        if os.path.exists(image_file):
            describe_scene(image_file)
        else:
            print(f"Error: Image file not found at {image_file}")
    else:
        print("Usage: python scene_describer.py <path_to_image>")

