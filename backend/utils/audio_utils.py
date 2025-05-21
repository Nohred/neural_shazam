import numpy as np
import librosa
import tensorflow as tf

def extract_features(audio_chunk, sr=22050):
    """Extract mel spectrogram features from audio chunk."""
    # Parameters matching the training preprocessing
    mel_bands = 128
    
    # Convert to mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_chunk,
        sr=sr,
        n_mels=mel_bands,
        hop_length=512
    )
    
    # Convert to log scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    mel_spec_norm = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
    
    # Add channel dimension
    mel_spec_norm = mel_spec_norm[..., np.newaxis]
    
    return mel_spec_norm

def load_model_and_encoder(model_path='model/final_model.h5', encoder_path='label_encoder.npy'):
    """Load the trained model and label encoder."""
    model = tf.keras.models.load_model(model_path)
    label_encoder = np.load(encoder_path, allow_pickle=True)
    return model, label_encoder