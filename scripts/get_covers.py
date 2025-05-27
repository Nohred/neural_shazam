import os
import requests
from ytmusicapi import YTMusic

# Directorios
RAW_DIR = "data/raw"
COVER_DIR = "data/covers"
os.makedirs(COVER_DIR, exist_ok=True)

# Inicializar API
ytmusic = YTMusic()

def sanitize_filename(name):
    return name.replace(".mp3", "").strip()

def download_image(url, output_path):
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"Guardada: {output_path}")
        else:
            print(f"Falló la descarga: {url}")
    except Exception as e:
        print(f"Error descargando imagen: {e}")

def fetch_and_save_cover(song_query, output_path):
    try:
        results = ytmusic.search(query=song_query, filter="songs")
        if results:
            thumbnail_url = results[0]['thumbnails'][-1]['url']
            download_image(thumbnail_url, output_path)
        else:
            print(f"No se encontró portada para {song_query}")
    except Exception as e:
        print(f"Error con {song_query}: {e}")

def main():
    for filename in os.listdir(RAW_DIR):
        if filename.endswith(".mp3"):
            song_name = sanitize_filename(filename)
            cover_path = os.path.join(COVER_DIR, f"{song_name}.jpg")
            if os.path.exists(cover_path):
                print(f"Ya existe: {cover_path}")
                continue
            print(f"Buscando portada para: {song_name}")
            fetch_and_save_cover(song_name.replace("_", " "), cover_path)

if __name__ == "__main__":
    main()
