# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
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

@app.get("/")
async def read_root():
    return {"message": "Face Recognition Auth API"}

@app.post("/api/signup")
async def signup(name: str, photo: UploadFile = File(...)):
    """
    Register a new user with their photo and name
    """
    try:
        # Read and process the uploaded image
        contents = await photo.read()
        image = face_recognition.load_image_file(BytesIO(contents))
        face_encodings = face_recognition.face_encodings(image)

        if not face_encodings:
            raise HTTPException(status_code=400, detail="No face detected in the image")

        # Convert face encoding to list for JSON serialization
        face_encoding = face_encodings[0].tolist()

        # Load existing users
        users = load_users()

        # Check if face already exists
        for existing_user in users.values():
            existing_encoding = np.array(existing_user["face_encoding"])
            if len(face_recognition.compare_faces([existing_encoding], np.array(face_encoding))[0]):
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
    """
    Authenticate user using their photo
    """
    try:
        # Read and process the uploaded image
        contents = await photo.read()
        image = face_recognition.load_image_file(BytesIO(contents))
        face_encodings = face_recognition.face_encodings(image)

        if not face_encodings:
            raise HTTPException(status_code=400, detail="No face detected in the image")

        face_encoding = face_encodings[0]
        users = load_users()

        # Check against all stored faces
        for user in users.values():
            stored_encoding = np.array(user["face_encoding"])
            if face_recognition.compare_faces([stored_encoding], face_encoding)[0]:
                return {"message": f"Welcome back, {user['name']}", "name": user["name"]}

        raise HTTPException(status_code=401, detail="Face not recognized")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users")
async def get_users():
    """
    Get list of registered users (names only)
    """
    users = load_users()
    return {"users": list(users.keys())}

if __name__ == "__main__":
    import uvicorn

    # Get port from environment variable or use default
    port = int(os.getenv("PORT", 8000))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True  # Enable auto-reload during development
    )