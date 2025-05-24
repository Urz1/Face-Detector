# Human Detection with DeepFace

This project uses a Flask server and DeepFace to detect whether an image contains an unknown human (not in a custom dataset of known faces). It’s designed to process images sent via HTTP POST requests, such as from an ESP32-CAM or Postman.

## Prerequisites

- Python 3.8+
- Dependencies listed in `requirements.txt`
- A `known_faces/` folder with images of known people (e.g., family members) in JPEG or PNG format

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Urz1/Face-Detector.git
   cd human-detection