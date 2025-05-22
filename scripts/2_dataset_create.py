import os
import numpy as np
import pandas as pd
import json
from pathlib import Path

def create_dataset(processed_dir="data/processed", output_file="data/dataset.csv"):
    """
    Create a balanced dataset from processed audio features.
    Uses balanced weighting where each song contributes equally regardless of length.
    Saves as CSV with features and labels and returns sample weights.
    """
    # Load metadata
    with open("data/metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Initialize lists to store features and labels
    all_features = []
    labels = []
    song_chunk_counts = {}  # Track chunks per song
    
    # First pass: count chunks per song
    for file_info in metadata["valid_files"]:
        song_name = file_info["song_name"]
        song_chunk_counts[song_name] = len(file_info["chunks"])
    
    print("Chunks per song:")
    for song, count in song_chunk_counts.items():
        print(f"  {song}: {count} chunks")
    
    # Calculate balanced weights (each song gets equal total weight)
    total_chunks = sum(song_chunk_counts.values())
    num_songs = len(song_chunk_counts)
    
    song_weights = {
        song: (total_chunks / num_songs) / count
        for song, count in song_chunk_counts.items()
    }
    
    print(f"\nUsing balanced weighting (each song gets equal total weight)")
    print("Song weights:")
    for song, weight in song_weights.items():
        chunks = song_chunk_counts[song]
        total_weight = weight * chunks
        print(f"  {song}: {weight:.4f} per chunk × {chunks} chunks = {total_weight:.4f} total weight")
    
    # Second pass: Process chunks with weights
    for file_info in metadata["valid_files"]:
        song_name = file_info["song_name"]
        weight = song_weights[song_name]
        
        # Load and process each chunk
        for chunk_file in file_info["chunks"]:
            # Load the mel spectrogram
            feature_path = os.path.join(processed_dir, chunk_file)
            
            try:
                mel_spec = np.load(feature_path)
            except FileNotFoundError:
                print(f"Warning: Could not find feature file {feature_path}")
                continue
            except Exception as e:
                print(f"Warning: Error loading {feature_path}: {e}")
                continue
            
            # Flatten the 2D mel spectrogram into 1D array
            flattened_features = mel_spec.flatten()
            
            # Append features and label
            all_features.append(flattened_features)
            labels.append(song_name)
    
    if not all_features:
        raise ValueError("No features were loaded. Check your processed_dir and file paths.")
    
    print(f"\nLoaded {len(all_features)} chunks total")
    
    # Check memory usage before conversion
    if all_features:
        first_feature_size = len(all_features[0])
        total_features = len(all_features)
        estimated_memory_mb = (first_feature_size * total_features * 8) / (1024 * 1024)  # 8 bytes per float64
        print(f"Feature vector size: {first_feature_size}")
        print(f"Estimated memory usage: {estimated_memory_mb:.1f} MB")
        
        if estimated_memory_mb > 1000:  # More than 1GB
            print("WARNING: Large dataset detected. This may take a while or run out of memory.")
            print("Consider processing in batches or reducing feature size.")
    
    print("Converting to numpy arrays...")
    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(labels)
    print("✓ Numpy conversion complete")
    
    print("Creating DataFrame...")
    # Create DataFrame more efficiently
    try:
        # For very large datasets, create DataFrame without explicit column names first
        if X.shape[1] > 10000:  # If more than 10k features
            print("Large feature set detected - creating DataFrame efficiently...")
            df = pd.DataFrame(X)
            df.columns = [f'feature_{i}' for i in range(X.shape[1])]
        else:
            feature_columns = [f'feature_{i}' for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=feature_columns)
        
        df['song_name'] = y
        print("✓ DataFrame creation complete")
        
    except MemoryError:
        print("ERROR: Not enough memory to create DataFrame")
        print("Try reducing the number of chunks or feature dimensions")
        raise
    
    print("Calculating sample weights...")
    # Calculate sample weights for each chunk
    sample_weights = np.array([song_weights[song] for song in y])
    
    # Add weights to DataFrame
    df['sample_weight'] = sample_weights
    print("✓ Sample weights added")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Saving to CSV: {output_file}")
    print("This may take a while for large datasets...")
    # Save to CSV
    df.to_csv(output_file, index=False)
    print("✓ CSV saved successfully")
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Dataset shape: {df.shape}")
    print(f"Number of unique songs: {len(df['song_name'].unique())}")
    print(f"Feature vector size: {X.shape[1]}")
    
    print("\nSong distribution before weighting:")
    chunk_counts = df['song_name'].value_counts()
    for song, count in chunk_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {song}: {count} chunks ({percentage:.1f}%)")
    
    print("\nEffective song distribution after weighting:")
    weighted_dist = df.groupby('song_name')['sample_weight'].sum()
    total_weight = weighted_dist.sum()
    for song, weight in weighted_dist.items():
        percentage = (weight / total_weight) * 100
        print(f"  {song}: {weight:.4f} total weight ({percentage:.1f}%)")
    
    # Verify weighting is working correctly
    print(f"\nWeighting verification:")
    print(f"Standard deviation of total weights: {weighted_dist.std():.6f}")
    print(f"(Lower values indicate better balance - should be close to 0)")
    
    print(f"\nDataset saved to: {output_file}")
    
    return df

if __name__ == "__main__":
    create_dataset()