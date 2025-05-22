import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
from cnn_model import create_model


def load_and_prepare_data(dataset_path="data/dataset.csv"):
    """Load and prepare data for training"""
    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    
    # Get sample weights if they exist
    sample_weights = df['sample_weight'].values if 'sample_weight' in df.columns else None
    
    # Separate features and labels
    X = df.drop(['song_name', 'sample_weight'] if 'sample_weight' in df.columns else ['song_name'], axis=1).values
    y = df['song_name'].values
    
    # Reshape features back to 3D (samples, mel_bands, time_steps)
    mel_bands = 128
    time_steps = X.shape[1] // mel_bands
    X = X.reshape(-1, mel_bands, time_steps)
    
    # Normalize features
    X = (X - X.mean()) / (X.std() + 1e-8)
    
    # Add channel dimension for CNN
    X = X[..., np.newaxis]
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split the data
    if sample_weights is not None:
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y_encoded, sample_weights, 
            test_size=0.2,  
            stratify=y_encoded,
            shuffle=True
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=0.2,  
            stratify=y_encoded,
            shuffle=True
        )
        w_train, w_test = None, None
    
    return X_train, X_test, y_train, y_test, w_train, w_test, label_encoder


def train():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available for training")
    else:
        print("No GPU found, using CPU")
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, w_train, w_test, label_encoder = load_and_prepare_data()
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"Input shape: {X_train.shape[1:]}")
    print(f"Sample weights: {w_train is not None}")
    print(f"Sample weights shape: {w_train.shape if w_train is not None else 'N/A'}")
    print(f"Sample weights test shape: {w_test.shape if w_test is not None else 'N/A'}")
    print(f"Label encoder classes: {label_encoder.classes_}")

    
    # Create model
    model = create_model(
        input_shape=(X_train.shape[1], X_train.shape[2], 1),
        num_classes=num_classes
    )
    
    # Compile model with better learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Enhanced callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'model/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=15,
            min_lr=0.001
        )
    ]
    
    # Train with data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomTranslation(0.05, 0.05),
        tf.keras.layers.GaussianNoise(0.05)
    ])
    
    # Train model
    print("Starting training...")
    history = model.fit(
        data_augmentation(X_train, training=True),
        y_train,
        sample_weight=w_train,  # Add sample weights
        batch_size=32,
        epochs=250,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model with sample weights
    print("\nEvaluating model on test set...")
    test_loss, test_acc = model.evaluate(
        X_test, 
        y_test, 
        sample_weight=w_test,
        verbose=1
    )
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Save model and label encoder
    model.save('model/final_model.h5')
    np.save('model/label_encoder.npy', label_encoder.classes_)
    print("\nModel and label encoder saved")

    return history, model, label_encoder

def predict_song(audio_chunk, model, label_encoder):
    """Predict song from an audio chunk"""
    # Ensure audio chunk has correct shape
    if audio_chunk.ndim == 3:
        audio_chunk = audio_chunk[np.newaxis, ...]
    
    # Get prediction
    predictions = model.predict(audio_chunk)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    
    # Get song name
    predicted_song = label_encoder.inverse_transform([predicted_idx])[0]
    
    return predicted_song, confidence

if __name__ == "__main__":
    history, model, label_encoder = train()
    # Plot training history
    # import matplotlib.pyplot as plt

    # # Plot accuracy
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(history.history['accuracy'], label='Training Accuracy')
    # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    # plt.title('Model Accuracy')
    
    # plt.subplot(1, 2, 2)
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss') 
    
    # plt.show()
