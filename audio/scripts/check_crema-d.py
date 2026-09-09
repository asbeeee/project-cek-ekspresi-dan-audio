from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
f = PROJECT_ROOT / "audio" / "datasets" / "audio_emotion" / "train" / "fear" / "crema_1053_ITH_FEA_XX.wav"
print(f"Ukuran: {f.stat().st_size} bytes")

# Cek isi awal file
with open(f, "rb") as file:
    head = file.read(100)
print(f"Isi awal: {head}")