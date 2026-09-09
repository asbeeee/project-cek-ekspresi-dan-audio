import shutil
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "audio" / "datasets" / "audio_emotion" / "raw"
DST = PROJECT_ROOT / "audio" / "datasets" / "audio_emotion"

random.seed(42)

if __name__ == '__main__':
    for emo_folder in sorted(SRC.iterdir()):
        if not emo_folder.is_dir():
            continue
        
        files = list(emo_folder.glob("*.wav"))
        random.shuffle(files)
        
        n = len(files)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        
        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:]
        }
        
        for split_name, split_files in splits.items():
            dest = DST / split_name / emo_folder.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy(f, dest / f.name)
        
        print(f"{emo_folder.name}: {n_train} train, {n_val} val, {n - n_train - n_val} test")
    
    print("\nSplit selesai!")