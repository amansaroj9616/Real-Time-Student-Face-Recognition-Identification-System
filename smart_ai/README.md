# 🎓 Smart Classroom — Face Recognition System

A production-style face recognition API for smart classroom attendance and student identification, powered by **DeepFace (ArcFace)** and **FastAPI**.

---

## 📁 Project Structure

```
smart_ai/
│
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI server & endpoints
│   ├── face_service.py    # Face embedding & matching logic
│   └── database.py        # JSON-based student & embedding storage
│
├── data/
│   ├── students.json      # Registered student records
│   └── embeddings.json    # Face embedding vectors
│
├── test.py                # DeepFace verification test script
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## 🚀 Quick Start

### 1. Create Conda Environment

```bash
conda create -n face python=3.9 -y
conda activate face
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Server

```bash
uvicorn app.main:app --reload
```

### 4. Access the API Docs

Open your browser and navigate to:

```
http://127.0.0.1:8000/docs
```

---

## 📡 API Endpoints

### `POST /register-student`

Register a new student with a face image.

| Parameter     | Type   | Description                  |
|---------------|--------|------------------------------|
| `image`       | File   | Face image of the student    |
| `name`        | String | Full name of the student     |
| `student_id`  | String | Unique student identifier    |
| `department`  | String | Department name (optional)   |

### `POST /recognize`

Recognize a face from an uploaded image.

| Parameter   | Type  | Description                           |
|-------------|-------|---------------------------------------|
| `image`     | File  | Face image to recognize               |
| `threshold` | Float | Similarity threshold (default: 0.6)   |

### `GET /`

Health check endpoint.

---

## 🧪 Test DeepFace Setup

```bash
# Quick library check (no images needed)
python test.py

# Verify with two face images
python test.py path/to/image1.jpg path/to/image2.jpg
```

---

## 🧠 Tech Stack

| Component        | Technology          |
|------------------|---------------------|
| Face Recognition | DeepFace (ArcFace)  |
| Face Detection   | OpenCV              |
| API Framework    | FastAPI             |
| Server           | Uvicorn             |
| Data Storage     | JSON files          |
| Language         | Python 3.9          |

---

## ⚙️ Configuration

- **Model**: ArcFace (via DeepFace)
- **Detector Backend**: OpenCV
- **Similarity Threshold**: 0.6 (adjustable per request)
- **Temp Uploads**: Auto-cleaned after each request

---

## 📝 Notes

- Face images should contain a clearly visible face for best results.
- The first run may take a moment as DeepFace downloads the ArcFace model weights.
- All data is stored locally in the `data/` directory as JSON files.
