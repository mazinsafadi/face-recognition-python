# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
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

load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("CORS_ORIGIN", "http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data storage
USERS_FILE = os.getenv("USERS_FILE", "users.json")

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_users() -> Dict:
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users: Dict):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

def get_face_encoding(image_array):
    """Extract face features using OpenCV"""
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="No face detected in the image")

    # Get the largest face
    (x, y, w, h) = max(faces, key=lambda x: x[2] * x[3])
    face_image = gray[y:y+h, x:x+w]

    # Resize to standard size
    face_image = cv2.resize(face_image, (100, 100))

    # Flatten the image to create a feature vector
    return face_image.flatten().tolist()

def compare_faces(face1, face2, threshold=5000):
    """Compare two face encodings"""
    return np.linalg.norm(np.array(face1) - np.array(face2)) < threshold

@app.get("/")
async def read_root():
    return {"message": "Face Recognition Auth API"}

@app.post("/api/signup")
async def signup(name: str, photo: UploadFile = File(...)):
    try:
        # Read and process the uploaded image
        contents = await photo.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Get face encoding
        face_encoding = get_face_encoding(image)

        # Load existing users
        users = load_users()

        # Check if face already exists
        for existing_user in users.values():
            if compare_faces(existing_user["face_encoding"], face_encoding):
                raise HTTPException(status_code=400, detail="Face already registered")

        # Store new user
        users[name] = {
            "name": name,
            "face_encoding": face_encoding
        }
        save_users(users)

        return {"message": f"Successfully registered {name}", "name": name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/signin")
async def signin(photo: UploadFile = File(...)):
    try:
        # Read and process the uploaded image
        contents = await photo.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Get face encoding
        face_encoding = get_face_encoding(image)

        users = load_users()

        # Check against all stored faces
        for user in users.values():
            if compare_faces(user["face_encoding"], face_encoding):
                return {"message": f"Welcome back, {user['name']}", "name": user["name"]}

        raise HTTPException(status_code=401, detail="Face not recognized")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users")
async def get_users():
    """Get list of registered users (names only)"""
    users = load_users()
    return {"users": list(users.keys())}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)