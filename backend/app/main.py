from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import librosa
import numpy as np
import tempfile
import json
import os

from model.cnn_model import CNNClassifier

app = FastAPI()

# CORS para frontend local o Dockerizado
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512

# Cargar modelo y label map
device = torch.device("cpu")

model = CNNClassifier(num_classes=50)  # O ajusta automáticamente abajo

# model_path = "model/best_model.pth"
# model_path = "model/final_pink_noise.pth"
model_path = "model/final_pitch_shift.pth"

model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.to(device)
model.eval()

with open("model/label_map.json") as f:
    label_map = json.load(f)
idx_to_label = {v: k for k, v in label_map.items()}


class PredictionResponse(BaseModel):
    predictions: list[str]
    confidences: list[float]


def extract_mel(y, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = (mel_spec_db - np.mean(mel_spec_db)) / (np.std(mel_spec_db) + 1e-9)  # Normalizar
    return mel_spec_db.astype(np.float32)


@app.post("/predict", response_model=PredictionResponse)
async def predict_audio(file: UploadFile = File(...)):
    try:
        # Guardar audio temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        y, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        os.remove(tmp_path)

        # Extraer features
        mel = extract_mel(y)
        mel_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, n_mels, time]

        # Predicción
        with torch.no_grad():
            output = model(mel_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]

        # Top 3
        top_indices = np.argsort(probs)[-3:][::-1]
        top_labels = [idx_to_label[i] for i in top_indices]
        top_confidences = [float(probs[i]) for i in top_indices]

        return PredictionResponse(predictions=top_labels, confidences=top_confidences)

    except Exception as e:
        return {"error": str(e)}
