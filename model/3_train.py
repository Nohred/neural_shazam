import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from cnn_model import CNNClassifier
from audio_dataset import MelSpectrogramDataset

def train_model(
    dataset_json="data/dataset_lists/clean_dataset.json",
    # dataset_json="data/dataset_lists/white_noise_dataset.json",
    # dataset_json="data/dataset_lists/pitch_shift_dataset.json",
    # dataset_json="data/dataset_lists/pink_noise_dataset.json",
    label_map_path="data/dataset_lists/label_map.json",
    model_dir="model",
    batch_size=32,
    epochs=100,
    lr=3e-4,
    val_split=0.2
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load label encoder
    with open(label_map_path) as f:
        label_map = json.load(f)
    num_classes = len(label_map)

    # Load dataset
    full_dataset = MelSpectrogramDataset(dataset_json)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Get input shape from first sample
    sample_input, _ = full_dataset[0]
    input_shape = sample_input.shape

    model = CNNClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0
    os.makedirs(model_dir, exist_ok=True)

    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for mel, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            mel, labels = mel.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(mel)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * mel.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = running_loss / total

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for mel, labels in val_loader:
                mel, labels = mel.to(device), labels.to(device)
                outputs = model(mel)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * mel.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        val_loss /= total

        print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
            print("✓ New best model saved")

    # Save final model
    torch.save(model.state_dict(), os.path.join(model_dir, "final_model.pth"))
    print("\n✓ Training complete. Final model saved.")

    # Save label map for decoding
    with open(os.path.join(model_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)
    print("✓ Label map saved")

def predict_song(model, mel_tensor, label_map_path="model/label_map.json"):
    model.eval()
    with open(label_map_path) as f:
        index_to_label = {int(v): k for k, v in json.load(f).items()}

    with torch.no_grad():
        logits = model(mel_tensor.unsqueeze(0))
        probs = F.softmax(logits, dim=1)[0]
        top3_idx = torch.topk(probs, 3).indices.tolist()
        top3_conf = torch.topk(probs, 3).values.tolist()
        top3_labels = [index_to_label[i] for i in top3_idx]

    return top3_labels, top3_conf

if __name__ == "__main__":
    train_model()
