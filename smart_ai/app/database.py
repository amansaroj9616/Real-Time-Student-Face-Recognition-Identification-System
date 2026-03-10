"""
database.py — JSON-Based Database Module
==========================================
Provides functions to load, save, and query student information
and face embeddings stored in local JSON files.
"""

import json
import os
from typing import Optional

# ─── File Paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
STUDENTS_FILE = os.path.join(DATA_DIR, "students.json")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.json")


def _ensure_data_dir():
    """Create the data directory if it doesn't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)


# ─── Student Operations ──────────────────────────────────────────────────────

def load_students() -> list:
    """
    Load all students from students.json.

    Returns:
        A list of student dictionaries.
    """
    _ensure_data_dir()

    if not os.path.exists(STUDENTS_FILE):
        return []

    with open(STUDENTS_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_students(students: list) -> None:
    """
    Save the full students list to students.json.

    Args:
        students: List of student dictionaries to persist.
    """
    _ensure_data_dir()

    with open(STUDENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(students, f, indent=2, ensure_ascii=False)


def get_student_by_id(student_id: str) -> Optional[dict]:
    """
    Look up a student by their unique ID.

    Args:
        student_id: The student's unique identifier.

    Returns:
        The student dict if found, otherwise None.
    """
    students = load_students()

    for student in students:
        if student.get("student_id") == student_id:
            return student

    return None


def add_student(student_info: dict) -> dict:
    """
    Add a new student record. Prevents duplicate student IDs.

    Args:
        student_info: Dict with keys like name, student_id, department.

    Returns:
        The added student dict.

    Raises:
        ValueError: If a student with the same ID already exists.
    """
    students = load_students()

    # Check for duplicate
    for s in students:
        if s.get("student_id") == student_info.get("student_id"):
            raise ValueError(
                f"Student with ID '{student_info['student_id']}' already exists."
            )

    students.append(student_info)
    save_students(students)
    return student_info


# ─── Embedding Operations ────────────────────────────────────────────────────

def load_embeddings() -> dict:
    """
    Load all embeddings from embeddings.json.

    Returns:
        A dict mapping student_id -> embedding vector (list of floats).
    """
    _ensure_data_dir()

    if not os.path.exists(EMBEDDINGS_FILE):
        return {}

    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def save_embeddings(embeddings: dict) -> None:
    """
    Save the full embeddings dict to embeddings.json.

    Args:
        embeddings: Dict mapping student_id -> embedding vector.
    """
    _ensure_data_dir()

    with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, indent=2, ensure_ascii=False)


def add_embedding(student_id: str, embedding: list) -> None:
    """
    Store (or overwrite) the embedding for a given student.

    Args:
        student_id: The student's unique identifier.
        embedding: The face embedding vector (list of floats).
    """
    embeddings = load_embeddings()
    embeddings[student_id] = embedding
    save_embeddings(embeddings)
