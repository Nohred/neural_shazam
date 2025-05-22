from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import tempfile
import os
from .utils.audio_utils import extract_features, load_model_and_encoder
from pydantic import BaseModel

app = FastAPI()

# Permitir CORS localmente (opcional si ya lo sirves junto)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo y codificador
model, label_encoder = load_model_and_encoder()

class PredictionResponse(BaseModel):
    predictions: list[str]
    confidences: list[float]

@app.post("/predict", response_model=PredictionResponse)
async def predict_audio(file: UploadFile = File(...)):
    try:
        # Guardar archivo temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Cargar audio
        y, sr = librosa.load(tmp_path, sr=22050, mono=True)
        os.remove(tmp_path)

        # Extraer características
        features = extract_features(y)

        # Hacer predicción
        preds = model.predict(features[np.newaxis, ...], verbose=0)[0]
        top_indices = np.argsort(preds)[-3:][::-1]
        top_songs = [label_encoder[idx] for idx in top_indices]
        top_confs = [float(preds[i]) for i in top_indices]

        return PredictionResponse(predictions=top_songs, confidences=top_confs)

    except Exception as e:
        return {"error": str(e)}
