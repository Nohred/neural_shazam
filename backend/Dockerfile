# Imagen base oficial de Python
FROM python:3.10-slim

# Evita prompts
ENV PYTHONUNBUFFERED=1

# Instala dependencias del sistema necesarias para librosa
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Crea directorio de trabajo
WORKDIR /app

# Copia archivos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app
COPY ./model ./model

# Expone el puerto para FastAPI
EXPOSE 8000

# Comando para lanzar la app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
