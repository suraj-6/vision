# VisionAI: Intelligent Navigation Software for the Visually Impaired

VisionAI is a computer vision-based software application designed to assist visually impaired individuals with environmental awareness and navigation. This software prototype uses advanced AI models and real-time video processing to provide intelligent audio feedback about the user's surroundings, serving as the foundation for future smart assistive device implementations.

## Features

- **Real-Time Object Detection**: Identifies and tracks objects in the user's environment with audio alerts for the nearest obstacles
- **Spatial Awareness**: Provides distance estimates and directional information through voice announcements
- **AI-Powered Scene Analysis**: On-demand comprehensive scene descriptions for complex environments
- **Smart Audio Management**: Intelligent prioritization ensures clear communication without information overload
- **Development Tools**: Console logging and diagnostic features for testing and optimization

## Installation

### Prerequisites
- Python 3.8+
- Webcam or video file for testing
- Google AI Studio API key (for scene analysis feature)
- streamlit
- google genai

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/suraj-6/vision.git
   cd vision-ai
   ```

2. **Install Required Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Access**
   - Create a file named `.env` in the project directory
   - Add your Google AI Studio API key (required for scene analysis):
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

4. **Camera Calibration (Optional)**
   For improved distance accuracy, adjust the FOCAL_LENGTH parameter in main.py based on your camera specifications. Default values work for most standard webcams.

## Usage

### Launch the Application
Run the main script to start the vision system:
```bash
streamlit run app.py
```

### Application Controls
- **Automatic Detection**: Continuous monitoring with audio alerts for nearest obstacles
- **Scene Analysis**: Press 'c' for detailed AI-generated scene descriptions  
- **Exit Program**: Press 'q' to close the application

## How It Works

VisionAI employs a multi-layered approach combining computer vision and natural language processing:

1. **Video Processing Pipeline**: OpenCV captures and processes video frames in real-time, providing the foundation for object detection and analysis.

2. **Object Detection & Tracking**: YOLOv8 neural network identifies objects in each frame, while custom algorithms calculate spatial relationships and movement patterns.

3. **Audio Feedback System**: Text-to-speech engine (pyttsx3) converts detection results into spoken announcements. Threading ensures audio doesn't interrupt video processing.

4. **Scene Understanding Module**: When triggered, the current frame is analyzed by Google's Gemini AI to generate contextually rich descriptions of complex scenes.

5. **Coordination Layer**: File-based locking mechanism prevents audio conflicts, ensuring users receive clear, prioritized information.

## Project Structure

```
vision-ai/
├── main.py                 # Core application with video processing and object detection
├── scene_describer.py      # AI-powered scene analysis module
├── requirements.txt        # Python package dependencies
├── .env                    # Configuration file for API keys (create this)
└── README.md              # Project documentation
```

## Technologies Used

- **Python**: Core programming language
- **OpenCV**: Video processing and computer vision
- **YOLOv8**: Object detection neural network
- **pyttsx3**: Text-to-speech synthesis
- **Google Gemini API**: Scene understanding and analysis
- **python-dotenv**: Environment configuration management


## Future Improvements

- **Hardware Integration**: Embedded system implementation for portable devices
- **Enhanced Detection**: Custom models for navigation-specific objects (stairs, curbs, hazards)
- **Text Recognition**: OCR capabilities for signs and documents
- **Multi-modal Feedback**: Haptic and spatial audio integration
- **Mobile App**: Companion application for configuration
- **Offline Mode**: Reduced cloud dependency for improved reliability

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

