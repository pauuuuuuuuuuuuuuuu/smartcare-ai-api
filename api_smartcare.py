import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

# =========================
# REDUCE TENSORFLOW LOG
# =========================

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
import uvicorn
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import tempfile

# =========================
# FASTAPI SETUP
# =========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOAD MODEL
# =========================

print("⏳ Loading SmartCare AI Model...")

model_cv = load("smartcare_final_model.pkl")

print("✅ Model Loaded!")

# =========================
# MEDIAPIPE SETUP
# =========================

mp_face_detection = mp.solutions.face_detection

# =========================
# ROOT ENDPOINT
# =========================

@app.get("/")
def home():

    return {
        "message": "SmartCare AI API Running"
    }

# =========================
# VIDEO PREDICTION
# =========================

@app.post("/predict")
async def predict_video(
    file: UploadFile = File(...)
):

    # =========================
    # SAVE TEMP VIDEO
    # =========================

    temp_video = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    temp_video.write(
        await file.read()
    )

    temp_video.close()

    video_path = temp_video.name

    # =========================
    # VIDEO CAPTURE
    # =========================

    cap = cv2.VideoCapture(video_path)

    total_frames = 0
    frames_fokus = 0
    frames_lari = 0

    with mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    ) as face_detection:

        while cap.isOpened():

            success, image = cap.read()

            if not success:
                break

            total_frames += 1

            image_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            # FACE DETECTION

            face_results = face_detection.process(
                image_rgb
            )

            if face_results.detections:
                frames_fokus += 1
            else:
                frames_lari += 1

    cap.release()

    # =========================
    # FEATURE ENGINEERING
    # =========================

    p_fokus = (
        (frames_fokus / total_frames) * 100
        if total_frames > 0 else 0
    )

    p_lari = (
        (frames_lari / total_frames) * 100
        if total_frames > 0 else 0
    )

    # Lightweight movement approximation

    total_movement = p_lari * 2
    avg_movement = total_movement / 10
    direction_changes = int(p_lari / 5)

    # =========================
    # DATAFRAME
    # =========================

    data_webcam = pd.DataFrame([{

        'ID_Anak': 999,
        'Total_Waktu_Detik': total_frames / 30,
        'Persentase_Fokus': p_fokus,
        'Persentase_Lari_Mata': p_lari,
        'Persentase_Berkedip': 5.0,
        'Fokus_Manusia': p_fokus,
        'Fokus_Benda': p_lari,
        'Menatap_Kosong': 0.0,
        'Rata_Pupil_Fokus': 3.5,
        'frames': total_frames,
        'total_movement': total_movement,
        'avg_movement': avg_movement,
        'direction_changes': direction_changes,
        'action': 'upload'

    }])

    # =========================
    # ENCODING
    # =========================

    data_webcam = pd.get_dummies(
        data_webcam
    )

    expected_cols = model_cv.feature_names_in_

    for col in expected_cols:

        if col not in data_webcam.columns:
            data_webcam[col] = 0

    data_webcam = data_webcam[
        expected_cols
    ]

    # =========================
    # AI PREDICTION
    # =========================

    prediction = model_cv.predict(
        data_webcam
    )[0]

    probabilities = model_cv.predict_proba(
        data_webcam
    )[0]

    confidence = float(
        np.max(probabilities) * 100
    )

    # =========================
    # DELETE TEMP VIDEO
    # =========================

    os.unlink(video_path)

    # =========================
    # RETURN RESULT
    # =========================

    return {

        "prediction": prediction,

        "confidence": round(
            confidence,
            2
        ),

        "focus_percentage": round(
            p_fokus,
            2
        ),

        "eye_movement_percentage": round(
            p_lari,
            2
        ),

        "total_frames": total_frames,

        "status": "success"
    }

# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":

    uvicorn.run(
        "api_smartcare:app",
        host="0.0.0.0",
        port=8000
    )