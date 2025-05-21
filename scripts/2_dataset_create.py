import os
import numpy as np
import pandas as pd
import json
from pathlib import Path

def create_dataset(processed_dir="data/processed", output_file="data/dataset.csv"):
    """
    Create a complete dataset from processed audio features.
    Saves as CSV with features and song labels.
    """
    # Load metadata
    with open(os.path.join(processed_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    # Initialize lists to store features and labels
    all_features = []
    labels = []

    # Process each valid file
    for file_info in metadata["valid_files"]:
        song_name = file_info["song_name"]
        
        # Load and process each chunk
        for chunk_file in file_info["chunks"]:
            # Load the mel spectrogram
            feature_path = os.path.join(processed_dir, chunk_file)
            mel_spec = np.load(feature_path)
            
            # Flatten the 2D mel spectrogram into 1D array
            flattened_features = mel_spec.flatten()
            
            # Append features and label
            all_features.append(flattened_features)
            labels.append(song_name)

    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(labels)

    # Create DataFrame
    feature_columns = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_columns)
    df['song_name'] = y

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Dataset created with shape: {df.shape}")
    print(f"Number of unique songs: {len(df['song_name'].unique())}")
    print(f"Dataset saved to: {output_file}")

    return df

if __name__ == "__main__":
    create_dataset()