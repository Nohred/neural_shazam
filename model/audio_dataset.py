import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class MelSpectrogramDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.paths = [item['path'] for item in self.data]
        self.labels = [item['label'] for item in self.data]

        self.encoder = LabelEncoder()
        self.encoded_labels = self.encoder.fit_transform(self.labels)
        self.label_map = {label: int(idx) for idx, label in enumerate(self.encoder.classes_)}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        mel_path = self.paths[idx]
        label = self.encoded_labels[idx]

        mel = np.load(mel_path)
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 128, time)

        return mel_tensor, label

    def get_label_map(self):
        return self.label_map
