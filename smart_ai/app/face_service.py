"""
face_service.py — Face Recognition Service Module
===================================================
Handles face embedding generation, cosine similarity computation,
and threshold-based matching using DeepFace with the ArcFace model.
"""

import numpy as np
from deepface import DeepFace
from typing import Optional, Tuple


# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "opencv"
DEFAULT_THRESHOLD = 0.6  # Cosine similarity threshold for a positive match


def generate_embedding(image_path: str) -> list:
    """
    Generate a face embedding vector from an image file.

    Args:
        image_path: Path to the image file.

    Returns:
        A list of floats representing the face embedding.

    Raises:
        ValueError: If no face is detected in the image.
    """
    try:
        # DeepFace.represent returns a list of dicts, one per detected face
        results = DeepFace.represent(
            img_path=image_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
        )

        if not results:
            raise ValueError("No face detected in the provided image.")

        # Use the first detected face's embedding
        embedding = results[0]["embedding"]
        return embedding

    except Exception as e:
        raise ValueError(f"Face embedding generation failed: {str(e)}")


def cosine_similarity(embedding_a: list, embedding_b: list) -> float:
    """
    Compute the cosine similarity between two embedding vectors.

    Args:
        embedding_a: First embedding vector.
        embedding_b: Second embedding vector.

    Returns:
        A float between -1 and 1, where 1 means identical.
    """
    a = np.array(embedding_a, dtype=np.float64)
    b = np.array(embedding_b, dtype=np.float64)

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Guard against zero-length vectors
    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = dot_product / (norm_a * norm_b)
    return float(similarity)


def find_match(
    query_embedding: list,
    stored_embeddings: dict,
    threshold: float = DEFAULT_THRESHOLD,
) -> Optional[Tuple[str, float]]:
    """
    Compare a query embedding against all stored embeddings and return
    the best match that exceeds the similarity threshold.

    Args:
        query_embedding: The embedding to search for.
        stored_embeddings: Dict mapping student_id -> embedding vector.
        threshold: Minimum cosine similarity to consider a match.

    Returns:
        A tuple of (student_id, similarity_score) for the best match,
        or None if no match exceeds the threshold.
    """
    best_match_id: Optional[str] = None
    best_score: float = -1.0

    for student_id, stored_emb in stored_embeddings.items():
        score = cosine_similarity(query_embedding, stored_emb)

        if score > best_score:
            best_score = score
            best_match_id = student_id

    # Only return a match if the best score exceeds the threshold
    if best_match_id and best_score >= threshold:
        return (best_match_id, best_score)

    return None
