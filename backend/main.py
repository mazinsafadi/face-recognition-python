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

USERS_FILE = os.getenv("USERS_FILE", "users.json")

def load_users() -> Dict:
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r') as f:
                users = json.load(f)
                # Convert face encodings back to numpy arrays
                for user in users.values():
                    user["face_encoding"] = np.array(user["face_encoding"])
                return users
        return {}
    except Exception as e:
        logger.error(f"Error loading users: {str(e)}")
        return {}

def save_users(users: Dict):
    try:
        # Convert numpy arrays to lists for JSON serialization
        users_json = {
            name: {
                "name": user["name"],
                "face_encoding": user["face_encoding"].tolist()
            }
            for name, user in users.items()
        }
        with open(USERS_FILE, 'w') as f:
            json.dump(users_json, f)
    except Exception as e:
        logger.error(f"Error saving users: {str(e)}")
        raise

def get_face_encoding(image_bytes):
    """Extract face features using face_recognition library"""
    try:
        # Convert image bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_image)

        if not face_locations:
            logger.warning("No face detected in the image")
            raise HTTPException(status_code=400, detail="No face detected in the image")

        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        if not face_encodings:
            logger.warning("Could not compute face encoding")
            raise HTTPException(status_code=400, detail="Could not compute face encoding")

        return face_encodings[0]  # Return encoding of first face found

    except Exception as e:
        logger.error(f"Error in face encoding: {str(e)}")
        raise

@app.post("/api/signup")
async def signup(
        photo: UploadFile = File(...),
        name: str = Form(...)
):
    logger.info(f"Signup attempt for user: {name}")
    try:
        contents = await photo.read()
        face_encoding = get_face_encoding(contents)

        users = load_users()

        # Check if face already exists with a more reliable comparison
        for existing_user in users.values():
            # face_recognition's compare_faces uses a better threshold
            if face_recognition.compare_faces([existing_user["face_encoding"]], face_encoding)[0]:
                logger.warning(f"Face already registered under name: {existing_user['name']}")
                raise HTTPException(status_code=400, detail="Face already registered")

        users[name] = {
            "name": name,
            "face_encoding": face_encoding
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

        # Check against all stored faces using face_recognition's compare_faces
        for user in users.values():
            if face_recognition.compare_faces([user["face_encoding"]], face_encoding)[0]:
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