import os
import numpy as np
import librosa
from pathlib import Path
import json

def add_white_noise(y, noise_level=0.05):
    noise = np.random.normal(0, noise_level, size=y.shape)
    return y + noise

def add_pink_noise(y, noise_level=0.5):
    uneven = y.shape[0] % 2
    X = np.random.randn(y.shape[0] // 2 + 1 + uneven) + 1j * np.random.randn(y.shape[0] // 2 + 1 + uneven)
    S = np.sqrt(np.arange(len(X)) + 1.)
    y_noise = np.fft.irfft(X / S).real
    y_noise = y_noise[:len(y)]
    y_noise = noise_level * y_noise / np.max(np.abs(y_noise))
    return y + y_noise

def add_pitch_shift(y, sr, steps=1):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

def get_active_chunks(y, sr, chunk_duration=10, top_db=30):
    intervals = librosa.effects.split(y, top_db=top_db)
    chunk_samples = sr * chunk_duration
    active_chunks = []

    for start, end in intervals:
        segment = y[start:end]
        for i in range(0, len(segment) - chunk_samples + 1, chunk_samples):
            chunk = segment[i:i + chunk_samples]
            if len(chunk) == chunk_samples:
                active_chunks.append(chunk)
    return active_chunks

def validate_and_process_audio(raw_dir="data/raw", processed_dir="data/processed", max_chunks_per_song=12): ### Ajusta el número máximo de chunks por canción
    os.makedirs(processed_dir, exist_ok=True)

    valid_files = []
    invalid_files = []

    SAMPLE_RATE = 22050
    DURATION = 10
    HOP_LENGTH = 512
    N_MELS = 128

    for audio_file in Path(raw_dir).glob("*.mp3"):
        try:
            y, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
            # chunk_samples = SAMPLE_RATE * DURATION
            chunks = get_active_chunks(y, SAMPLE_RATE, DURATION)

            # Limitar a los primeros N chunks activos
            if len(chunks) > max_chunks_per_song:
                chunks = chunks[:max_chunks_per_song]

            features = []

            for i, chunk in enumerate(chunks):
                augmentations = {
                    "clean": chunk,
                    "white_noise": add_white_noise(chunk),
                    "pink_noise": add_pink_noise(chunk),
                    "pitch_shift": add_pitch_shift(chunk, SAMPLE_RATE)
                }

                for version, y_proc in augmentations.items():
                    mel_spec = librosa.feature.melspectrogram(
                        y=y_proc,
                        sr=SAMPLE_RATE,
                        n_mels=N_MELS,
                        hop_length=HOP_LENGTH
                    )
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

                    filename = f"{audio_file.stem}_chunk_{i}_{version}.npy"
                    np.save(os.path.join(processed_dir, filename), mel_spec_db)
                    features.append(filename)

            valid_files.append({
                "song_name": audio_file.stem,
                "chunks": features
            })
            print(f"✓ Procesado: {audio_file.name}")

        except Exception as e:
            print(f"✗ Error con {audio_file.name}: {e}")
            invalid_files.append(str(audio_file))

    metadata = {
        "valid_files": valid_files,
        "invalid_files": invalid_files,
        "processing_params": {
            "sample_rate": SAMPLE_RATE,
            "duration": DURATION,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS,
            "noise_augmented": ["white_noise", "pink_noise", "pitch_shift"],
            "max_chunks_per_song": max_chunks_per_song
        }
    }

    with open("data/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    validate_and_process_audio()
