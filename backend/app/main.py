import pyaudio
import numpy as np
import wave
import threading
import queue
import time
from ..utils.audio_utils import extract_features, load_model_and_encoder 

class AudioPredictor:
    def __init__(self):
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 22050
        self.RECORD_SECONDS = 10
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # Load model and encoder
        self.model, self.label_encoder = load_model_and_encoder()
        
        # Create a queue for audio chunks
        self.audio_queue = queue.Queue()
        
        # Initialize recording state
        self.is_recording = False
    
    def start_recording(self):
        """Start recording audio from microphone."""
        self.is_recording = True
        
        def callback(in_data, frame_count, time_info, status):
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_queue.put(audio_data)
            return (in_data, pyaudio.paContinue)
        
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=callback
        )
        
        print("* Recording started")
        self.stream.start_stream()
    
    def process_audio(self):
        """Process audio chunks and make predictions."""
        chunks = []
        samples_needed = self.RATE * self.RECORD_SECONDS
        
        while self.is_recording:
            # Collect chunks until we have 5 seconds
            while len(chunks) * self.CHUNK < samples_needed:
                if not self.is_recording:
                    return
                if not self.audio_queue.empty():
                    chunks.append(self.audio_queue.get())
            
            # Concatenate chunks
            audio_data = np.concatenate(chunks)
            
            # Extract features
            features = extract_features(audio_data)
            
            # Make prediction
            predictions = self.model.predict(features[np.newaxis, ...], verbose=0)[0]
            # Get top 3 predictions
            top_indices = np.argsort(predictions)[-3:][::-1]
            top_confidences = predictions[top_indices]
            top_songs = [self.label_encoder[idx] for idx in top_indices]
            
            # Print top 3 results
            print("\nTop 3 predicted songs:")
            for i, (song, conf) in enumerate(zip(top_songs, top_confidences), 1):
                print(f"{i}. {song} (Confidence: {conf:.2f})")
            # Clear chunks but keep the excess
            excess_samples = len(chunks) * self.CHUNK - samples_needed
            chunks = [chunks[-1][-excess_samples:]] if excess_samples > 0 else []
    
    def stop_recording(self):
        """Stop recording and clean up."""
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

def main():
    predictor = AudioPredictor()
    
    try:
        # Start recording in a separate thread
        predictor.start_recording()
        
        # Start processing in main thread
        predictor.process_audio()
        
    except KeyboardInterrupt:
        print("\n* Stopping recording...")
        predictor.stop_recording()

if __name__ == "__main__":
    main()