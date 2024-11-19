# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import json
import os
import numpy as np
from typing import Dict
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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

def get_face_encoding(image_bytes):
    try:
        # Convert image bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)

        # Load the image using face_recognition
        image = face_recognition.load_image_file(nparr)

        # Find all faces in the image
        face_locations = face_recognition.face_locations(image, model="hog")

        if not face_locations:
            raise HTTPException(status_code=400, detail="No face detected in the image")

        # Get the encoding for the first face found
        face_encoding = face_recognition.face_encodings(image, face_locations)[0]

        return face_encoding.tolist()
    except Exception as e:
        logger.error(f"Error in face encoding: {str(e)}")
        raise HTTPException(status_code=400, detail="Failed to process face in image")

@app.post("/api/signup")
async def signup(photo: UploadFile = File(...), name: str = Form(...)):
    logger.info(f"Signup attempt for user: {name}")
    try:
        contents = await photo.read()
        face_encoding = get_face_encoding(contents)
        users = load_users()

        # Check for existing faces
        for existing_user in users.values():
            if face_recognition.compare_faces([np.array(existing_user["encoding"])],
                                              np.array(face_encoding))[0]:
                logger.warning(f"Face already registered under name: {existing_user['name']}")
                raise HTTPException(status_code=400, detail="Face already registered")

        users[name] = {
            "name": name,
            "encoding": face_encoding,
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
        face_encoding = get_face_encoding(contents)
        users = load_users()

        for user in users.values():
            if face_recognition.compare_faces([np.array(user["encoding"])],
                                              np.array(face_encoding))[0]:
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