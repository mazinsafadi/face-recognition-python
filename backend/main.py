# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import json
import os
import numpy as np
from typing import Dict
import mediapipe as mp
import logging
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize MediaPipe Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

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
        # Convert to RGB
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        # Get face landmarks
        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            raise HTTPException(status_code=400, detail="No face detected in the image")

        # Extract face landmarks as embedding
        face_landmarks = results.multi_face_landmarks[0]
        embedding = []

        # Convert landmarks to flat array
        for landmark in face_landmarks.landmark:
            embedding.extend([landmark.x, landmark.y, landmark.z])

        return embedding
    except Exception as e:
        logger.error(f"Error in face embedding: {str(e)}")
        raise HTTPException(status_code=400, detail="Failed to process face in image")

def compare_faces(embedding1, embedding2, threshold=0.85):
    try:
        # Reshape embeddings for sklearn
        emb1 = np.array(embedding1).reshape(1, -1)
        emb2 = np.array(embedding2).reshape(1, -1)

        # Calculate similarity
        similarity = cosine_similarity(emb1, emb2)[0][0]
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