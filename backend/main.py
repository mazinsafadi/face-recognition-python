# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import json
import os
import numpy as np
from typing import Dict
import base64
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Configure CORS
CORS_ORIGIN = os.getenv("CORS_ORIGIN", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info(f"CORS configured with origin: {CORS_ORIGIN}")

# Data storage
USERS_FILE = os.getenv("USERS_FILE", "users.json")
logger.info(f"Using users file: {USERS_FILE}")

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    logger.error("Error: Could not load face cascade classifier")
else:
    logger.info("Face cascade classifier loaded successfully")

def load_users() -> Dict:
    """Load users from JSON file"""
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r') as f:
                users = json.load(f)
                logger.info(f"Loaded {len(users)} users from file")
                return users
        logger.info("No existing users file, starting with empty user list")
        return {}
    except Exception as e:
        logger.error(f"Error loading users: {str(e)}")
        return {}

def save_users(users: Dict):
    """Save users to JSON file"""
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f)
            logger.info(f"Saved {len(users)} users to file")
    except Exception as e:
        logger.error(f"Error saving users: {str(e)}")
        raise

def get_face_encoding(image_array):
    """Extract face features using OpenCV"""
    try:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            logger.warning("No face detected in the image")
            raise HTTPException(status_code=400, detail="No face detected in the image")

        # Get the largest face
        (x, y, w, h) = max(faces, key=lambda x: x[2] * x[3])
        face_image = gray[y:y+h, x:x+w]
        logger.info(f"Face detected at position ({x}, {y}) with size {w}x{h}")

        # Resize to standard size
        face_image = cv2.resize(face_image, (100, 100))

        # Flatten the image to create a feature vector
        return face_image.flatten().tolist()
    except Exception as e:
        logger.error(f"Error in face encoding: {str(e)}")
        raise

def compare_faces(face1, face2, threshold=5000):
    """Compare two face encodings"""
    try:
        distance = np.linalg.norm(np.array(face1) - np.array(face2))
        logger.info(f"Face comparison distance: {distance} (threshold: {threshold})")
        return distance < threshold
    except Exception as e:
        logger.error(f"Error comparing faces: {str(e)}")
        raise

@app.get("/")
async def read_root():
    logger.info("Root endpoint accessed")
    return {"message": "Face Recognition Auth API"}

@app.post("/api/signup")
async def signup(
        photo: UploadFile = File(...),
        name: str = Form(...)  # Changed to Form parameter
):
    logger.info(f"Signup attempt for user: {name}")
    try:
        # Log file details
        logger.info(f"Received file: {photo.filename}, content-type: {photo.content_type}")

        # Read and process the uploaded image
        contents = await photo.read()
        logger.info(f"Read file contents, size: {len(contents)} bytes")

        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            logger.error("Failed to decode image")
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Get face encoding
        face_encoding = get_face_encoding(image)
        logger.info("Successfully generated face encoding")

        # Load existing users
        users = load_users()

        # Check if face already exists
        for existing_user in users.values():
            if compare_faces(existing_user["face_encoding"], face_encoding):
                logger.warning(f"Face already registered under name: {existing_user['name']}")
                raise HTTPException(status_code=400, detail="Face already registered")

        # Store new user
        users[name] = {
            "name": name,
            "face_encoding": face_encoding
        }
        save_users(users)
        logger.info(f"Successfully registered new user: {name}")

        return {"message": f"Successfully registered {name}", "name": name}

    except HTTPException as he:
        logger.warning(f"HTTP Exception during signup: {str(he.detail)}")
        raise
    except Exception as e:
        logger.error(f"Error during signup: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/signin")
async def signin(photo: UploadFile = File(...)):
    logger.info("Signin attempt")
    try:
        # Log file details
        logger.info(f"Received file: {photo.filename}, content-type: {photo.content_type}")

        # Read and process the uploaded image
        contents = await photo.read()
        logger.info(f"Read file contents, size: {len(contents)} bytes")

        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            logger.error("Failed to decode image")
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Get face encoding
        face_encoding = get_face_encoding(image)
        logger.info("Successfully generated face encoding")

        users = load_users()

        # Check against all stored faces
        for user in users.values():
            if compare_faces(user["face_encoding"], face_encoding):
                logger.info(f"Successful login for user: {user['name']}")
                return {"message": f"Welcome back, {user['name']}", "name": user["name"]}

        logger.warning("Face not recognized")
        raise HTTPException(status_code=401, detail="Face not recognized")

    except HTTPException as he:
        logger.warning(f"HTTP Exception during signin: {str(he.detail)}")
        raise
    except Exception as e:
        logger.error(f"Error during signin: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users")
async def get_users():
    """Get list of registered users (names only)"""
    try:
        users = load_users()
        user_list = list(users.keys())
        logger.info(f"Retrieved user list, count: {len(user_list)}")
        return {"users": user_list}
    except Exception as e:
        logger.error(f"Error getting users: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve users")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)