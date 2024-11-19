from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import cv2
import face_recognition
import os
import json
from typing import List

app = FastAPI()

USERS_DB = "users.json"

def load_users():
    if os.path.exists(USERS_DB):
        with open(USERS_DB, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_DB, "w") as f:
        json.dump(users, f)

def get_face_encoding(image_array):
    # Convert BGR (OpenCV) to RGB (face_recognition)
    rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_image)
    if not encodings:
        raise HTTPException(status_code=400, detail="No face detected in the image")
    return encodings[0]

@app.post("/api/signup")
async def signup(name: str, photo: UploadFile = File(...)):
    users = load_users()
    if name in users:
        raise HTTPException(status_code=400, detail="User already exists")

    image_data = await photo.read()
    image_array = np.fromstring(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    try:
        encoding = get_face_encoding(image)
        users[name] = encoding.tolist()
        save_users(users)
        return {"message": f"User {name} registered successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/signin")
async def signin(photo: UploadFile = File(...)):
    users = load_users()
    if not users:
        raise HTTPException(status_code=400, detail="No registered users")

    image_data = await photo.read()
    image_array = np.fromstring(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    try:
        encoding = get_face_encoding(image)
        for name, user_encoding in users.items():
            matches = face_recognition.compare_faces([np.array(user_encoding)], encoding)
            if matches[0]:
                return {"message": f"Welcome back, {name}!"}
        raise HTTPException(status_code=401, detail="Face not recognized")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
