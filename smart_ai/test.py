"""
test.py — DeepFace Setup Verification Script
==============================================
Verifies that DeepFace is correctly installed and working
by running a face verification on two sample images:
  - photo_aman.jpg
  - srajan.jpg
"""

import os
import sys
from deepface import DeepFace


# ─── Default image paths (in the same directory as this script) ───────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_1 = os.path.join(SCRIPT_DIR, "photo_aman.jpg")
IMAGE_2 = os.path.join(SCRIPT_DIR, "srajan.jpg")


def verify_faces(img1_path: str, img2_path: str) -> None:
    """
    Verify whether two face images belong to the same person.
    """
    # Check files exist
    for path in [img1_path, img2_path]:
        if not os.path.exists(path):
            print(f"  ERROR: Image not found -> {path}")
            sys.exit(1)

    print("=" * 60)
    print("  Smart Classroom — DeepFace Verification Test")
    print("=" * 60)
    print(f"\n  Image 1 : {img1_path}")
    print(f"  Image 2 : {img2_path}")
    print(f"  Model   : ArcFace")
    print("-" * 60)

    try:
        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name="ArcFace",
            detector_backend="opencv",
            enforce_detection=True,
        )

        verified = result.get("verified", False)
        distance = result.get("distance", None)
        threshold = result.get("threshold", None)
        similarity_metric = result.get("similarity_metric", "cosine")

        print(f"\n  Verified       : {verified}")
        if distance is not None:
            print(f"  Distance       : {distance:.6f}")
        if threshold is not None:
            print(f"  Threshold      : {threshold}")
        print(f"  Model          : ArcFace")
        print(f"  Metric         : {similarity_metric}")
        print("-" * 60)

        if verified:
            print("\n  RESULT: The two faces belong to the SAME person.\n")
        else:
            print("\n  RESULT: The two faces are DIFFERENT people.\n")

    except Exception as e:
        print(f"\n  Verification failed: {str(e)}\n")
        sys.exit(1)


def main():
    """Entry point — uses default images or command-line args."""
    # Use command-line args if provided, otherwise use defaults
    if len(sys.argv) >= 3:
        img1 = sys.argv[1]
        img2 = sys.argv[2]
    else:
        print("\nUsing default images: photo_aman.jpg & srajan.jpg\n")
        img1 = IMAGE_1
        img2 = IMAGE_2

    verify_faces(img1, img2)


if __name__ == "__main__":
    main()