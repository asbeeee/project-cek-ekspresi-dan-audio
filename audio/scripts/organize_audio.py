import os
import shutil
from pathlib import Path

# --- SESUAIKAN PATH INI ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAVDESS_DIR = PROJECT_ROOT / "audio" / "datasets" / "ravdess" / "AudioWAV"
CREMA_DIR = PROJECT_ROOT / "audio" / "datasets" / "crema-d" / "AudioWAV"
OUTPUT_DIR = PROJECT_ROOT / "audio" / "datasets" / "audio_emotion"
# ----------------------------

# Mapping kode RAVDESS -> label standar
ravdess_map = {
    "01": "neutral", "02": "neutral",  # calm digabung ke neutral (opsional)
    "03": "happy", "04": "sad",
    "05": "angry", "06": "fear",
    "07": "disgust", "08": "surprise"
}

# Mapping kode CREMA-D -> label standar
crema_map = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fear",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad"
}

def organize_ravdess():
    count = 0
    for actor_folder in RAVDESS_DIR.rglob("*.wav"):
        parts = actor_folder.stem.split("-")
        if len(parts) < 3:
            continue
        code = parts[2]
        label = ravdess_map.get(code)
        if label is None:
            continue
        
        dest = OUTPUT_DIR / "raw" / label
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(actor_folder, dest / f"ravdess_{actor_folder.name}")
        count += 1
    print(f"RAVDESS: {count} file diproses")

def organize_crema():
    count = 0
    for f in CREMA_DIR.glob("*.wav"):
        parts = f.stem.split("_")
        if len(parts) < 3:
            continue
        code = parts[2]
        label = crema_map.get(code)
        if label is None:
            continue
        
        dest = OUTPUT_DIR / "raw" / label
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, dest / f"crema_{f.name}")
        count += 1
    print(f"CREMA-D: {count} file diproses")

if __name__ == '__main__':
    organize_ravdess()
    organize_crema()
    
    # Ringkasan akhir
    print("\n=== Ringkasan per kelas emosi ===")
    for emo_folder in sorted((OUTPUT_DIR / "raw").iterdir()):
        n = len(list(emo_folder.glob("*.wav")))
        print(f"{emo_folder.name}: {n} file")