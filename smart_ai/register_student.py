"""
register_student.py — Student Registration Script
====================================================
Registers a student into the Smart Classroom Face Recognition
system by generating a face embedding and saving it to
data/embeddings.json.

Usage:
    python register_student.py
"""

import os
import sys
import json
from deepface import DeepFace


# ─── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.json")

# Hardcoded student details
STUDENT_NAME = "Rishit"
STUDENT_CLASS = "11th"
STUDENT_ID = "122cs0060"

MODEL_NAME = "ArcFace"
IMAGE_PATH = os.path.join(SCRIPT_DIR, "Rishit.jpeg")


# ─── Functions ────────────────────────────────────────────────────────────────

def generate_embedding(image_path: str) -> list:
    """
    Generate a face embedding from the given image using DeepFace (ArcFace).

    Args:
        image_path: Absolute or relative path to a face image.

    Returns:
        A list of floats representing the face embedding vector.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If no face is detected in the image.
    """
    # Validate image path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Generate embedding
    results = DeepFace.represent(
        img_path=image_path,
        model_name=MODEL_NAME,
        enforce_detection=True,
    )

    if not results:
        raise ValueError("No face detected in the provided image.")

    # Return embedding as a plain list
    embedding = list(results[0]["embedding"])
    return embedding


def save_student(student_id: str, name: str, student_class: str, embedding: list) -> None:
    """
    Save or update a student record in data/embeddings.json.

    - Creates the data/ directory if it doesn't exist.
    - Creates embeddings.json as an empty list if it doesn't exist.
    - Updates the embedding if the student_id already exists (no duplicates).

    Args:
        student_id: Unique identifier for the student.
        name: Full name of the student.
        student_class: Class/grade of the student.
        embedding: Face embedding vector.
    """
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Load existing data or create empty list
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # Handle case where file contains a dict instead of a list
                if isinstance(data, dict):
                    data = list(data.values()) if data else []
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Build the student record
    student_record = {
        "student_id": student_id,
        "name": name,
        "class": student_class,
        "embedding": embedding,
    } 


    # Check for duplicate — update if exists, append if new
    updated = False
    for i, record in enumerate(data):
        if record.get("student_id") == student_id:
            data[i] = student_record
            updated = True
            break

    if not updated:
        data.append(student_record)

    # Write back to file
    with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    """Main entry point — prompts for image path and registers the student."""
    print("=" * 55)
    print("  Smart Classroom — Student Registration")
    print("=" * 55)
    print(f"\n  Student  : {STUDENT_NAME}")
    print(f"  Class    : {STUDENT_CLASS}")
    print(f"  ID       : {STUDENT_ID}")
    print("-" * 55)

    # Use hardcoded image path
    image_path = IMAGE_PATH

    if not os.path.exists(image_path):
        print(f"\n  [ERROR] Image not found: {image_path}\n")
        sys.exit(1)

    print(f"\n  Processing: {image_path}")
    print("  Generating face embedding (ArcFace)...\n")

    try:
        # Step 1: Generate embedding
        embedding = generate_embedding(image_path)
        print(f"  Embedding generated ({len(embedding)} dimensions)")

        # Step 2: Save to database
        save_student(STUDENT_ID, STUDENT_NAME, STUDENT_CLASS, embedding)
        print(f"  Data saved to: {EMBEDDINGS_FILE}")

        print("-" * 55)
        print(f"\n  {STUDENT_NAME} successfully registered!\n")

    except FileNotFoundError as e:
        print(f"\n  [ERROR] {e}")
        print("  Please check the image path and try again.\n")
        sys.exit(1)

    except ValueError as e:
        print(f"\n  [ERROR] Face detection failed: {e}")
        print("  Make sure the image contains a clearly visible face.\n")
        sys.exit(1)

    except Exception as e:
        print(f"\n  [ERROR] Unexpected error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
