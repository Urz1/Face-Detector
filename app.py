from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2
import os
import numpy as np
import logging
import pickle
from functools import wraps
import time
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

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
        try:
            with open(EMBEDDINGS_PATH, 'rb') as f:
                logger.info("Loading pre-computed embeddings")
                embeddings = pickle.load(f)
                logger.info(f"Loaded {len(embeddings)} embeddings")
                return embeddings
        except Exception as e:
            logger.warning(f"Error loading embeddings: {str(e)}. Recomputing embeddings.")
    
    logger.info("Computing embeddings for known faces...")
    embeddings = {}
    
    if not os.path.exists(DATABASE_PATH):
        logger.warning(f"Directory {DATABASE_PATH} does not exist. Creating it.")
        os.makedirs(DATABASE_PATH)
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(embeddings, f)
        return embeddings
    
    try:
        known_faces = [os.path.join(DATABASE_PATH, f) for f in os.listdir(DATABASE_PATH) if f.endswith(('.jpg', '.png'))]
        logger.info(f"Found {len(known_faces)} images in {DATABASE_PATH}")
    except Exception as e:
        logger.error(f"Error accessing {DATABASE_PATH}: {str(e)}")
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(embeddings, f)
        return embeddings
    
    if not known_faces:
        logger.warning(f"No valid images found in {DATABASE_PATH}. Using empty embeddings.")
    else:
        for face_path in known_faces:
            try:
                embedding = DeepFace.represent(img_path=face_path, model_name='VGG-Face', detector_backend='mtcnn', enforce_detection=True)[0]['embedding']
                embeddings[face_path] = embedding
                logger.info(f"Computed embedding for {face_path}")
            except Exception as e:
                logger.warning(f"Skipping {face_path}: {str(e)}")
    
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(embeddings, f)
    logger.info(f"Saved {len(embeddings)} embeddings to {EMBEDDINGS_PATH}")
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
        if api_key and api_key == os.getenv('API_KEY', '5gl5TTvpUaX3J9K-muC3pSiTfGHy8S_Vn3ruNk8Vnqw'):
            return func(*args, **kwargs)
        else:
            return jsonify({'error': 'Invalid or missing API key'}), 401
    return decorated_function

@app.route('/process_image', methods=['POST'])
@require_api_key
def process_image():
    try:
        logger.info("Received image for processing")
        # Save received image to a temporary file
        image_data = request.get_data()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_image_path = temp_file.name
            temp_file.write(image_data)

        # Load image with OpenCV
        img = cv2.imread(temp_image_path)
        if img is None:
            logger.error("Invalid image received")
            os.remove(temp_image_path)
            return jsonify({'result': False, 'error': 'Invalid image'}), 400

        # Step 1: Detect face
        @timeout(10)
        def detect_face():
            faces = DeepFace.extract_faces(img_path=temp_image_path, detector_backend='mtcnn', enforce_detection=False)
            # Validate faces
            valid_faces = [face for face in faces if face['confidence'] > 0.9]
            logger.info(f"Detected {len(faces)} faces, {len(valid_faces)} valid (confidence > 0.9)")
            return valid_faces

        face_detected = False
        try:
            valid_faces = detect_face()
            face_detected = len(valid_faces) > 0
            if face_detected:
                logger.info("Valid face detected in image")
            else:
                logger.info("No valid face detected in image")
                os.remove(temp_image_path)
                response = {'result': False, 'message': 'No human face detected'}
                print(f"Flask Response: {response}")
                return jsonify(response)
        except Exception as e:
            logger.info(f"Face detection failed: {str(e)}")
            os.remove(temp_image_path)
            response = {'result': False, 'message': 'No human face detected'}
            print(f"Flask Response: {response}")
            return jsonify(response)

        # Step 2: Compare against known faces
        @timeout(10)
        def compute_input_embedding():
            return DeepFace.represent(img_path=temp_image_path, model_name='VGG-Face', detector_backend='mtcnn', enforce_detection=True)

        try:
            input_embedding = compute_input_embedding()[0]['embedding']
            is_known = False
            if not known_embeddings:
                logger.info("No known embeddings available. Treating face as unknown.")
            else:
                for known_face_path, known_embedding in known_embeddings.items():
                    try:
                        distance = np.linalg.norm(np.array(input_embedding) - np.array(known_embedding))
                        logger.info(f"Distance to {known_face_path}: {distance}")
                        if distance < 0.6:  # Adjusted threshold
                            logger.info(f"Match found with {known_face_path}")
                            is_known = True
                            break
                    except Exception as e:
                        logger.warning(f"Error comparing with {known_face_path}: {str(e)}")
                        continue
            
            result = not is_known
            response = {'result': result, 'message': 'Unknown human' if result else 'Known person'}
            logger.info(f"Result: {'Unknown human' if result else 'Known person'}")
            print(f"Flask Response: {response}")
            os.remove(temp_image_path)
            return jsonify(response)

        except (ValueError, TimeoutError) as e:
            logger.error(f"Embedding computation failed: {str(e)}")
            os.remove(temp_image_path)
            response = {'result': False, 'message': 'Error processing face'}
            print(f"Flask Response: {response}")
            return jsonify(response)

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        response = {'result': False, 'error': str(e)}
        print(f"Flask Response: {response}")
        return jsonify(response)

if __name__ == '__main__':
    import os
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)




# from flask import Flask, request, jsonify
# from deepface import DeepFace
# import cv2
# import os
# import numpy as np
# import logging
# import pickle
# from functools import wraps
# import time
# from dotenv import load_dotenv
# import tempfile

# # Load environment variables
# load_dotenv()

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# app = Flask(__name__)

# # Path to custom database of known faces
# DATABASE_PATH = "./known_faces"
# EMBEDDINGS_PATH = "./known_faces_embeddings.pkl"

# # Pre-compute embeddings for known faces
# def load_or_compute_embeddings():
#     if os.path.exists(EMBEDDINGS_PATH):
#         try:
#             with open(EMBEDDINGS_PATH, 'rb') as f:
#                 logger.info("Loading pre-computed embeddings")
#                 return pickle.load(f)
#         except Exception as e:
#             logger.warning(f"Error loading embeddings: {str(e)}. Recomputing embeddings.")
    
#     logger.info("Computing embeddings for known faces...")
#     embeddings = {}
    
#     if not os.path.exists(DATABASE_PATH):
#         logger.warning(f"Directory {DATABASE_PATH} does not exist. No known faces available.")
#         with open(EMBEDDINGS_PATH, 'wb') as f:
#             pickle.dump(embeddings, f)
#         return embeddings
    
#     try:
#         known_faces = [os.path.join(DATABASE_PATH, f) for f in os.listdir(DATABASE_PATH) if f.endswith(('.jpg', '.png'))]
#     except Exception as e:
#         logger.error(f"Error accessing {DATABASE_PATH}: {str(e)}")
#         with open(EMBEDDINGS_PATH, 'wb') as f:
#             pickle.dump(embeddings, f)
#         return embeddings
    
#     if not known_faces:
#         logger.warning(f"No valid images found in {DATABASE_PATH}. Using empty embeddings.")
#     else:
#         for face_path in known_faces:
#             try:
#                 embedding = DeepFace.represent(img_path=face_path, model_name='VGG-Face', detector_backend='retinaface', enforce_detection=False)[0]['embedding']
#                 embeddings[face_path] = embedding
#             except ValueError as e:
#                 logger.warning(f"Skipping {face_path}: {str(e)}")
    
#     with open(EMBEDDINGS_PATH, 'wb') as f:
#         pickle.dump(embeddings, f)
#     return embeddings

# # Load embeddings at startup
# known_embeddings = load_or_compute_embeddings()

# # Timeout decorator to prevent hanging
# def timeout(seconds):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             start_time = time.time()
#             result = func(*args, **kwargs)
#             if time.time() - start_time > seconds:
#                 logger.error(f"Function {func.__name__} timed out after {seconds} seconds")
#                 raise TimeoutError(f"Operation timed out after {seconds} seconds")
#             return result
#         return wrapper
#     return decorator

# # API key authentication
# def require_api_key(func):
#     @wraps(func)
#     def decorated_function(*args, **kwargs):
#         api_key = request.headers.get('X-API-Key')
#         if api_key and api_key == os.getenv('API_KEY', 'your-secret-key'):
#             return func(*args, **kwargs)
#         else:
#             return jsonify({'error': 'Invalid or missing API key'}), 401
#     return decorated_function

# @app.route('/process_image', methods=['POST'])
# @require_api_key
# def process_image():
#     try:
#         logger.info("Received image for processing")
#         # Save received image to a temporary file
#         image_data = request.get_data()
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
#             temp_image_path = temp_file.name
#             temp_file.write(image_data)

#         # Load image with OpenCV
#         img = cv2.imread(temp_image_path)
#         if img is None:
#             logger.error("Invalid image received")
#             os.remove(temp_image_path)
#             return jsonify({'result': False, 'error': 'Invalid image'}), 400

#         # Step 1: Detect face
#         @timeout(10)
#         def detect_face():
#             faces = DeepFace.extract_faces(img_path=temp_image_path, detector_backend='retinaface', enforce_detection=False)
#             return len(faces) > 0
        
#         face_detected = False
#         try:
#             face_detected = detect_face()
#             if face_detected:
#                 logger.info("Face detected in image")
#             else:
#                 logger.info("No face detected in image")
#                 os.remove(temp_image_path)
#                 response = {'result': False, 'message': 'No human face detected'}
#                 print(f"Flask Response: {response}")
#                 return jsonify(response)
#         except Exception as e:
#             logger.info(f"Face detection failed: {str(e)}")
#             os.remove(temp_image_path)
#             response = {'result': False, 'message': 'No human face detected'}
#             print(f"Flask Response: {response}")
#             return jsonify(response)

#         # Step 2: Compare against known faces
#         @timeout(10)
#         def compute_input_embedding():
#             return DeepFace.represent(img_path=temp_image_path, model_name='VGG-Face', detector_backend='retinaface', enforce_detection=False)
        
#         try:
#             input_embedding = compute_input_embedding()[0]['embedding']
#             is_known = False
#             if not known_embeddings:
#                 logger.info("No known embeddings available. Treating face as unknown.")
#             else:
#                 for known_face_path, known_embedding in known_embeddings.items():
#                     try:
#                         distance = np.linalg.norm(np.array(input_embedding) - np.array(known_embedding))
#                         if distance < 0.4:  # VGG-Face cosine threshold
#                             logger.info(f"Match found with {known_face_path}")
#                             is_known = True
#                             break
#                     except Exception as e:
#                         logger.warning(f"Error comparing with {known_face_path}: {str(e)}")
#                         continue
            
#             result = not is_known
#             response = {'result': result, 'message': 'Unknown human' if result else 'Known person'}
#             logger.info(f"Result: {'Unknown human' if result else 'Known person'}")
#             print(f"Flask Response: {response}")
#             os.remove(temp_image_path)
#             return jsonify(response)

#         except (ValueError, TimeoutError) as e:
#             logger.error(f"Embedding computation failed: {str(e)}")
#             os.remove(temp_image_path)
#             response = {'result': False, 'message': 'Error processing face'}
#             print(f"Flask Response: {response}")
#             return jsonify(response)

#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}")
#         if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
#             os.remove(temp_image_path)
#         response = {'result': False, 'error': str(e)}
#         print(f"Flask Response: {response}")
#         return jsonify(response)

# if __name__ == '__main__':
#     import os
#     port = int(os.getenv('PORT', 5000))
#     app.run(host='0.0.0.0', port=port)