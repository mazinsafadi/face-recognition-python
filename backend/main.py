from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import json
import os
import numpy as np
from typing import Dict
from deepface import DeepFace
import logging
from datetime import datetime
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configure TensorFlow to use less memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = FastAPI()

# Configure CORS
CORS_ORIGIN = os.getenv("CORS_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

USERS_FILE = os.getenv("USERS_FILE", "/tmp/users.json")

# Initialize DeepFace model at startup
logger.info("Initializing face recognition model...")
try:
    # Pre-load the model
    DeepFace.represent(
        img_path=np.zeros((112, 112, 3), dtype=np.uint8),
        model_name="Facenet",
        enforce_detection=False
    )
    logger.info("Face recognition model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing face recognition model: {e}")

def load_users() -> Dict:
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
        return {}
    except Exception as e:
        logger.error(f"Error loading users: {str(e)}")
        return {}

def save_users(users: Dict):
    try:
        os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f)
    except Exception as e:
        logger.error(f"Error saving users: {str(e)}")
        raise

def get_face_embedding(image_array):
    try:
        # Resize image to reduce memory usage
        max_size = 640
        height, width = image_array.shape[:2]
        if height > max_size or width > max_size:
            scale = max_size / max(height, width)
            image_array = cv2.resize(image_array, None, fx=scale, fy=scale)

        embedding = DeepFace.represent(
            img_path=image_array,
            model_name="Facenet",  # Using Facenet model which is faster and lighter
            enforce_detection=True,
            detector_backend="retinaface",  # Using RetinaFace for better detection
            align=True
        )

        if not embedding:
            raise HTTPException(status_code=400, detail="No face detected in the image")

        return embedding[0]["embedding"]
    except Exception as e:
        logger.error(f"Error in face embedding: {str(e)}")
        if "Face could not be detected" in str(e):
            raise HTTPException(status_code=400, detail="No face detected in the image")
        raise HTTPException(status_code=400, detail="Failed to process face in image")

def compare_faces(embedding1, embedding2, threshold=0.4):
    try:
        distance = np.linalg.norm(np.array(embedding1) - np.array(embedding2))
        similarity = 1 / (1 + distance)
        logger.info(f"Face similarity score: {similarity} (threshold: {threshold})")
        return similarity > threshold
    except Exception as e:
        logger.error(f"Error comparing faces: {str(e)}")
        raise

@app.post("/api/signup")
async def signup(photo: UploadFile = File(...), name: str = Form(...)):
    logger.info(f"Signup attempt for user: {name}")
    try:
        contents = await photo.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        face_embedding = get_face_embedding(image)
        users = load_users()

        # Clear memory
        del image
        del contents
        del nparr

        for existing_user in users.values():
            if compare_faces(existing_user["embedding"], face_embedding):
                logger.warning(f"Face already registered under name: {existing_user['name']}")
                raise HTTPException(status_code=400, detail="Face already registered")

        users[name] = {
            "name": name,
            "embedding": face_embedding,
            "created_at": datetime.utcnow().isoformat()
        }
        save_users(users)
        logger.info(f"Successfully registered new user: {name}")

        return {"message": f"Successfully registered {name}", "name": name}

    except HTTPException as he:
        raise
    except Exception as e:
        logger.error(f"Error during signup: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/signin")
async def signin(photo: UploadFile = File(...)):
    logger.info("Signin attempt")
    try:
        contents = await photo.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        face_embedding = get_face_embedding(image)

        # Clear memory
        del image
        del contents
        del nparr

        users = load_users()

        for user in users.values():
            if compare_faces(user["embedding"], face_embedding):
                logger.info(f"Successful login for user: {user['name']}")
                return {"message": f"Welcome back, {user['name']}", "name": user["name"]}

        logger.warning("Face not recognized")
        raise HTTPException(status_code=401, detail="Face not recognized")

    except HTTPException as he:
        raise
    except Exception as e:
        logger.error(f"Error during signin: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users")
async def get_users():
    try:
        users = load_users()
        user_list = list(users.keys())
        return {"users": user_list}
    except Exception as e:
        logger.error(f"Error getting users: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve users")

@app.get("/")
async def root():
    return {"message": "API is running"}