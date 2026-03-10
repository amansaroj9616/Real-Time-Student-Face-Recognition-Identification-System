"""
live_recognition.py — Live Classroom Face Recognition
======================================================
Real-time face recognition for Smart Classroom using webcam,
DeepFace (ArcFace), and cosine similarity matching.

Usage:
    python live_recognition.py

Press 'Q' to exit.
"""

import cv2
import json
import os
import sys
import time
import numpy as np
from numpy.linalg import norm
from deepface import DeepFace


# ─── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_FILE = os.path.join(SCRIPT_DIR, "data", "embeddings.json")

MODEL_NAME = "ArcFace"
MATCH_THRESHOLD = 0.40  # Lower threshold for webcam-to-photo matching
FRAME_SKIP = 10  # Run recognition every N frames

# Colors (BGR)
GREEN = (0, 200, 100)
RED = (0, 0, 220)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_BG = (40, 40, 40)
HEADER_BG = (80, 50, 20)


# ─── Functions ────────────────────────────────────────────────────────────────

def load_students() -> list:
    """
    Load registered student embeddings from data/embeddings.json.

    Returns:
        A list of student dicts, each with student_id, name, class, embedding.
    """
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"  [ERROR] Embeddings file not found: {EMBEDDINGS_FILE}")
        print("  Please register a student first using register_student.py")
        sys.exit(1)

    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("  [ERROR] Invalid JSON in embeddings.json")
            sys.exit(1)

    if not data:
        print("  [WARNING] No students registered yet. embeddings.json is empty.")
        return []

    # Handle both list and dict formats
    if isinstance(data, dict):
        # Convert dict format {id: embedding} to list format
        return [{"student_id": k, "embedding": v} for k, v in data.items()]

    return data


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = norm(a)
    norm_b = norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def find_match(new_embedding: np.ndarray, students: list) -> tuple:
    """
    Find the best matching student for the given embedding.

    Args:
        new_embedding: Face embedding from live frame.
        students: List of registered student records.

    Returns:
        Tuple of (student_dict, confidence_score) or (None, 0.0).
    """
    best_score = 0.0
    best_student = None

    for student in students:
        stored_embedding = np.array(student["embedding"])
        score = cosine_similarity(new_embedding, stored_embedding)

        if score > best_score:
            best_score = score
            best_student = student

    if best_score >= MATCH_THRESHOLD and best_student is not None:
        return best_student, best_score

    return None, best_score


def draw_overlay(frame, matched_student, confidence, fps):
    """
    Draw the recognition overlay on the frame.

    Args:
        frame: The OpenCV frame to draw on.
        matched_student: The matched student dict, or None.
        confidence: Cosine similarity score.
        fps: Current frames per second.
    """
    h, w = frame.shape[:2]

    # ── Header bar ────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 45), HEADER_BG, -1)
    cv2.putText(
        frame, "Smart Classroom Recognition",
        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2
    )

    # ── FPS counter (top right) ───────────────────────────────────────────
    fps_text = f"FPS: {fps:.0f}"
    cv2.putText(
        frame, fps_text,
        (w - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2
    )

    # ── Info panel (bottom left) ──────────────────────────────────────────
    panel_x, panel_y = 10, h - 130
    panel_w, panel_h = 300, 120

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), DARK_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Border color based on match
    border_color = GREEN if matched_student else RED
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), border_color, 2)

    if matched_student:
        name = matched_student.get("name", "Unknown")
        student_class = matched_student.get("class", "N/A")

        cv2.putText(
            frame, f"Name: {name}",
            (panel_x + 15, panel_y + 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2
        )
        cv2.putText(
            frame, f"Class: {student_class}",
            (panel_x + 15, panel_y + 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2
        )
        cv2.putText(
            frame, f"Confidence: {confidence:.2f}",
            (panel_x + 15, panel_y + 95),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2
        )
    else:
        cv2.putText(
            frame, "Unknown Student",
            (panel_x + 15, panel_y + 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2
        )
        cv2.putText(
            frame, f"Score: {confidence:.2f}",
            (panel_x + 15, panel_y + 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1
        )


def draw_face_box(frame, matched):
    """
    Attempt to detect face and draw a bounding rectangle.

    Args:
        frame: The OpenCV frame.
        matched: Whether a match was found (affects box color).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    color = GREEN if matched else RED

    for (x, y, fw, fh) in faces:
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)


def main():
    """Main entry point — starts webcam and runs live recognition."""
    print("=" * 55)
    print("  Smart Classroom — Live Recognition")
    print("=" * 55)

    # Load registered students
    students = load_students()
    print(f"\n  Loaded {len(students)} registered student(s)")

    if not students:
        print("  No students to match against. Exiting.")
        sys.exit(0)

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("\n  [ERROR] Could not open webcam.")
        print("  Make sure your camera is connected and not in use.\n")
        sys.exit(1)

    print("  Webcam opened successfully")
    print("  Press 'Q' to exit\n")
    print("-" * 55)

    # State variables
    frame_count = 0
    matched_student = None
    confidence = 0.0
    fps = 0.0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("  [ERROR] Failed to read frame from webcam.")
            break

        frame_count += 1

        # Calculate FPS
        current_time = time.time()
        time_diff = current_time - prev_time
        if time_diff > 0:
            fps = 1.0 / time_diff
        prev_time = current_time

        # Run recognition every N frames (performance optimization)
        if frame_count % FRAME_SKIP == 0:
            try:
                results = DeepFace.represent(
                    frame,
                    model_name=MODEL_NAME,
                    enforce_detection=False,
                )

                if results:
                    embedding = np.array(results[0]["embedding"])
                    matched_student, confidence = find_match(embedding, students)
                    # Debug: print similarity score to terminal
                    if matched_student:
                        print(f"  MATCHED: {matched_student.get('name')} | Score: {confidence:.4f}")
                    else:
                        print(f"  No match | Best score: {confidence:.4f}")

            except Exception:
                # Face detection failed on this frame, keep last result
                pass

        # Draw face bounding box
        draw_face_box(frame, matched_student is not None)

        # Draw overlay with student info, confidence, and FPS
        draw_overlay(frame, matched_student, confidence, fps)

        # Show frame
        cv2.imshow("Smart Classroom Live Recognition", frame)

        # Exit on 'Q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n  Live recognition stopped.\n")


if __name__ == "__main__":
    main()