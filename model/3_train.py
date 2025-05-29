import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from cnn_model import CNNMel
from audio_dataset import MelSpectrogramDataset


def train_model():
    dataset_paths = [
        "data/dataset_lists/clean_dataset.json",
        "data/dataset_lists/white_noise_dataset.json",
        "data/dataset_lists/pitch_shift_dataset.json",
        "data/dataset_lists/pink_noise_dataset.json",
    ]
    label_map_path = "data/dataset_lists/label_map.json"
    model_dir = "model"
    batch_size = 32
    epochs = 100
    lr = 3e-4
    test_split = 0.2

    # Cargar y unir todos los datos
    all_entries = []
    for path in dataset_paths:
        with open(path, "r") as f:
            for entry in json.load(f):
                all_entries.append({
                    "file": entry.get("file", entry.get("path")),  # usar file o path
                    "label": entry["label"]
                })

    # División estratificada
    labels = [entry["label"] for entry in all_entries]
    train_entries, test_entries = train_test_split(
        all_entries, test_size=test_split, stratify=labels, random_state=42
    )

    train_dataset = MelSpectrogramDataset(entries=train_entries)
    test_dataset = MelSpectrogramDataset(entries=test_entries)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Cargar label map
    with open(label_map_path) as f:
        label_map = json.load(f)
    num_classes = len(label_map)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNMel(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0
    os.makedirs(model_dir, exist_ok=True)

    print("\n🚀 Entrenando modelo...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for mel, label in tqdm(train_loader, desc=f"Epoch {epoch}"):
            mel, label = mel.to(device), label.to(device)

            optimizer.zero_grad()
            outputs = model(mel)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * mel.size(0)
            _, pred = torch.max(outputs, 1)
            correct += (pred == label).sum().item()
            total += label.size(0)

        acc = correct / total
        loss_avg = train_loss / total
        print(f"📊 Epoch {epoch}: Loss={loss_avg:.4f} | Acc={acc:.4f}")

        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
            print("💾 Modelo mejor guardado")

    # Evaluación final
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for mel, label in test_loader:
            mel, label = mel.to(device), label.to(device)
            outputs = model(mel)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    # Métricas generales
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print("\n📋 Resultados en validación:")
    print(f"🔹 Precisión (macro): {precision:.4f}")
    print(f"🔹 Recall (macro): {recall:.4f}")
    print(f"🔹 F1-score (macro): {f1:.4f}")

    # Reporte completo (opcional)
    report = classification_report(
        all_labels, all_preds, target_names=list(label_map.keys()), zero_division=0
    )
    print("\n🧾 Classification report:")
    print(report)

    # Guardar reporte
    with open(os.path.join(model_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    print("📄 Reporte de clasificación guardado como 'classification_report.txt'")

    torch.save(model.state_dict(), os.path.join(model_dir, "final_model.pth"))
    print("🎉 Modelo final guardado")

    with open(os.path.join(model_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)


if __name__ == "__main__":
    train_model()
