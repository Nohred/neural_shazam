import os
import json
from pathlib import Path

def create_dataset_lists(processed_dir="data/processed", output_dir="data/dataset_lists", include_versions=["clean"]):
    """
    Crea archivos .json para PyTorch Dataset con lista de paths y etiquetas.
    Un archivo por cada versión seleccionada: clean, white_noise, pink_noise, pitch_shift
    """
    os.makedirs(output_dir, exist_ok=True)

    # Cargar metadata
    with open("data/metadata.json", "r") as f:
        metadata = json.load(f)

    # Crear índice de clases
    all_song_names = sorted([entry["song_name"] for entry in metadata["valid_files"]])
    label_map = {name: idx for idx, name in enumerate(all_song_names)}
    print(f"Clases encontradas: {label_map}")

    for version in include_versions:
        dataset = []
        for entry in metadata["valid_files"]:
            label = label_map[entry["song_name"]]
            for chunk_file in entry["chunks"]:
                if f"_{version}.npy" in chunk_file:
                    dataset.append({
                        "path": os.path.join(processed_dir, chunk_file),
                        "label": label
                    })

        print(f"✓ {len(dataset)} ejemplos con versión: {version}")
        output_path = os.path.join(output_dir, f"{version}_dataset.json")
        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2)

    # Guardar también el mapa de clases para decodificar resultados
    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    print("✓ Listas de dataset generadas correctamente")

if __name__ == "__main__":
    create_dataset_lists(include_versions=["clean", "white_noise", "pink_noise", "pitch_shift"])
