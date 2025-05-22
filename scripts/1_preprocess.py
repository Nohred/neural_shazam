import os
import numpy as np
import librosa
from pathlib import Path
import json

def validate_and_process_audio(raw_dir="data/raw", processed_dir="data/processed"):
    """Validate audio files and process them into features."""
    
    # Create processed directory
    os.makedirs(processed_dir, exist_ok=True)
    
    # Track valid and invalid files
    valid_files = []
    invalid_files = []
    
    # Parameters for processing
    SAMPLE_RATE = 22050
    DURATION = 10  # seconds per chunk
    HOP_LENGTH = 512
    N_MELS = 128
    
    for audio_file in Path(raw_dir).glob("*.mp3"):
        try:
            # Load audio file
            y, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
            
            # Split into 5-second chunks
            chunk_samples = SAMPLE_RATE * DURATION
            chunks = [y[i:i + chunk_samples] for i in range(0, len(y), chunk_samples)]
            
            # Process each chunk
            features = []
            for i, chunk in enumerate(chunks):
                if len(chunk) >= chunk_samples:  # Only process complete chunks
                    # Convert to mel spectrogram
                    mel_spec = librosa.feature.melspectrogram(
                        y=chunk, 
                        sr=SAMPLE_RATE,
                        n_mels=N_MELS,
                        hop_length=HOP_LENGTH
                    )
                    
                    # Convert to log scale
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    
                    # Save features
                    chunk_filename = f"{audio_file.stem}_chunk_{i}.npy"
                    np.save(os.path.join(processed_dir, chunk_filename), mel_spec_db)
                    features.append(chunk_filename)
            
            valid_files.append({
                "song_name": audio_file.stem,
                "chunks": features
            })
            
            print(f"Successfully processed: {audio_file.name}")
            
        except Exception as e:
            print(f"Error processing {audio_file.name}: {e}")
            invalid_files.append(str(audio_file))
    
    # Save metadata
    metadata = {
        "valid_files": valid_files,
        "invalid_files": invalid_files,
        "processing_params": {
            "sample_rate": SAMPLE_RATE,
            "duration": DURATION,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS
        }
    }
    
    with open("data/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    validate_and_process_audio()