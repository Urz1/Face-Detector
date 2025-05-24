from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2
import os
import numpy as np
import logging
import pickle
from functools import wraps
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Path to custom database of known faces
DATABASE_PATH = "./known_faces"
EMBEDDINGS_PATH = "./known_faces_embeddings.pkl"

# Pre-compute embeddings for known faces
def load_or_compute_embeddings():
    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, 'rb') as f:
            logger.info("Loading pre-computed embeddings")
            return pickle.load(f)
    
    logger.info("Computing embeddings for known faces...")
    embeddings = {}
    known_faces = [os.path.join(DATABASE_PATH, f) for f in os.listdir(DATABASE_PATH) if f.endswith(('.jpg', '.png'))]
    for face_path in known_faces:
        try:
            embedding = DeepFace.represent(img_path=face_path, model_name='VGG-Face', detector_backend='retinaface')
            embeddings[face_path] = embedding
        except ValueError as e:
            logger.warning(f"Skipping {face_path}: {str(e)}")
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(embeddings, f)
    return embeddings

# Load embeddings at startup
known_embeddings = load_or_compute_embeddings()

# Timeout decorator to prevent hanging
def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            if time.time() - start_time > seconds:
                logger.error(f"Function {func.__name__} timed out after {seconds} seconds")
                raise TimeoutError(f"Operation timed out after {seconds} seconds")
            return result
        return wrapper
    return decorator

# API key authentication
def require_api_key(func):
    @wraps(func)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key and api_key == os.getenv('API_KEY', 'your-secret-key'):
            return func(*args, **kwargs)
        else:
            return jsonify({'error': 'Invalid or missing API key'}), 401
    return decorated_function

@app.route('/process_image', methods=['POST'])
@require_api_key
def process_image():
    try:
        logger.info("Received image for processing")
        # Save received image
        image_data = request.get_data()
        temp_image_path = "profile.jpg"
        with open(temp_image_path, 'wb') as f:
            f.write(image_data)

        # Load image with OpenCV
        img = cv2.imread(temp_image_path)
        if img is None:
            logger.error("Invalid image received")
            os.remove(temp_image_path)
            return jsonify({'result': False, 'error': 'Invalid image'}), 400

        # Step 1: Detect face
        @timeout(10)
        def detect_face():
            return DeepFace.detectFace(img_path=temp_image_path, detector_backend='retinaface')
        
        try:
            detect_face()
            logger.info("Face detected in image")
        except (ValueError, TimeoutError) as e:
            logger.error(f"Face detection failed: {str(e)}")
            os.remove(temp_image_path)
            return jsonify({'result': False, 'message': 'No human face detected'})

        # Step 2: Compare against known faces using pre-computed embeddings
        @timeout(10)
        def compute_input_embedding():
            return DeepFace.represent(img_path=temp_image_path, model_name='VGG-Face', detector_backend='retinaface')
        
        try:
            input_embedding = compute_input_embedding()
            is_known = False
            for known_face_path, known_embedding in known_embeddings.items():
                try:
                    distance = np.linalg.norm(np.array(input_embedding) - np.array(known_embedding))
                    if distance < 0.4:  # VGG-Face cosine threshold
                        logger.info(f"Match found with {known_face_path}")
                        is_known = True
                        break
                except Exception as e:
                    logger.warning(f"Error comparing with {known_face_path}: {str(e)}")
                    continue
        except (ValueError, TimeoutError) as e:
            logger.error(f"Embedding computation failed: {str(e)}")
            os.remove(temp_image_path)
            return jsonify({'result': False, 'message': 'Error processing face'})

        # Step 3: Determine result
        result = not is_known
        logger.info(f"Result: {'Unknown human' if result else 'Known person'}")

        # Clean up
        os.remove(temp_image_path)
        return jsonify({'result': result, 'message': 'Unknown human' if result else 'Known person or no human'})

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        return jsonify({'result': False, 'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)