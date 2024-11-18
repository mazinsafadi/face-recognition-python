from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import cv2
import json
import os
import numpy as np
from typing import Dict
import logging
from datetime import datetime

# ... (keep your existing imports and setup code)

def get_face_encoding(image_bytes):
    """Extract face embeddings using DeepFace"""
    try:
        # Convert image bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Get embeddings
        embedding = DeepFace.represent(
            image,
            model_name="VGG-Face",
            enforce_detection=True,
            detector_backend="opencv"
        )

        if not embedding:
            logger.warning("No face detected in the image")
            raise HTTPException(status_code=400, detail="No face detected in the image")

        return embedding[0]["embedding"]

    except Exception as e:
        logger.error(f"Error in face encoding: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

def compare_faces(face1, face2, threshold=0.6):
    """Compare two face embeddings"""
    try:
        distance = np.float64(DeepFace.dst.findCosineDistance(face1, face2))
        logger.info(f"Face comparison distance: {distance} (threshold: {threshold})")
        return distance < threshold
    except Exception as e:
        logger.error(f"Error comparing faces: {str(e)}")
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

        # Check if face already exists
        for existing_user in users.values():
            if compare_faces(existing_user["face_encoding"], face_encoding):
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

        # Check against all stored faces
        for user in users.values():
            if compare_faces(user["face_encoding"], face_encoding):
                logger.info(f"Successful login for user: {user['name']}")
                return {"message": f"Welcome back, {user['name']}", "name": user["name"]}

        logger.warning("Face not recognized")
        raise HTTPException(status_code=401, detail="Face not recognized")

    except HTTPException as he:
        raise
    except Exception as e:
        logger.error(f"Error during signin: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))