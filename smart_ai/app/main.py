"""
main.py — FastAPI Server for Smart Classroom Face Recognition
==============================================================
Provides REST API endpoints for registering students with their
face embeddings and recognizing faces against stored data.
"""

import os
import shutil
import uuid

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.face_service import generate_embedding, find_match
from app.database import (
    add_student,
    add_embedding,
    load_embeddings,
    get_student_by_id,
)


# ─── App Initialization ──────────────────────────────────────────────────────

app = FastAPI(
    title="Smart Classroom — Face Recognition API",
    description=(
        "Register students with face images and recognize them in real-time. "
        "Powered by DeepFace (ArcFace model)."
    ),
    version="1.0.0",
)

# Temporary folder for uploaded images
TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)


# ─── Helper ──────────────────────────────────────────────────────────────────

def _save_temp_image(upload_file: UploadFile) -> str:
    """Save an uploaded image to a temporary file and return its path."""
    ext = os.path.splitext(upload_file.filename or "image.jpg")[1] or ".jpg"
    temp_filename = f"{uuid.uuid4().hex}{ext}"
    temp_path = os.path.join(TEMP_DIR, temp_filename)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return temp_path


def _cleanup(path: str) -> None:
    """Remove a temporary file if it exists."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.post("/register-student", summary="Register a new student with a face image")
async def register_student(
    image: UploadFile = File(..., description="Face image of the student"),
    name: str = Form(..., description="Full name of the student"),
    student_id: str = Form(..., description="Unique student identifier"),
    department: str = Form("General", description="Department name"),
):
    """
    Register a student in the system:
    1. Accept an image file along with student metadata.
    2. Generate a face embedding from the image.
    3. Store the student info and embedding in JSON files.
    """
    temp_path = _save_temp_image(image)

    try:
        # Generate face embedding
        embedding = generate_embedding(temp_path)

        # Prepare student info
        student_info = {
            "student_id": student_id,
            "name": name,
            "department": department,
        }

        # Save to database
        add_student(student_info)
        add_embedding(student_id, embedding)

        return JSONResponse(
            status_code=201,
            content={
                "status": "success",
                "message": f"Student '{name}' registered successfully.",
                "student": student_info,
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    finally:
        _cleanup(temp_path)


@app.post("/recognize", summary="Recognize a face from an uploaded image")
async def recognize_face(
    image: UploadFile = File(..., description="Face image to recognize"),
    threshold: float = Form(0.6, description="Cosine similarity threshold (0–1)"),
):
    """
    Recognize a student from a face image:
    1. Accept an image file.
    2. Generate a face embedding.
    3. Compare against all stored embeddings.
    4. Return the matched student's information (if any).
    """
    temp_path = _save_temp_image(image)

    try:
        # Generate face embedding for the query image
        query_embedding = generate_embedding(temp_path)

        # Load stored embeddings and find the best match
        stored_embeddings = load_embeddings()

        if not stored_embeddings:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "no_match",
                    "message": "No students registered yet.",
                },
            )

        match = find_match(query_embedding, stored_embeddings, threshold=threshold)

        if match is None:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "no_match",
                    "message": "No matching student found above the similarity threshold.",
                },
            )

        matched_id, similarity_score = match
        student = get_student_by_id(matched_id)

        return JSONResponse(
            status_code=200,
            content={
                "status": "matched",
                "similarity_score": round(similarity_score, 4),
                "student": student,
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    finally:
        _cleanup(temp_path)


# ─── Health Check ─────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
async def health_check():
    """Simple health-check endpoint."""
    return {
        "status": "running",
        "service": "Smart Classroom — Face Recognition API",
        "version": "1.0.0",
    }
