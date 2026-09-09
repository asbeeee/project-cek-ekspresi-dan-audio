import os
import shutil
import random
from pathlib import Path

# Path FER2013 Bee
src_test = Path("fer2013/test")
dst_val = Path("fer2013/val")
dst_test_new = Path("fer2013/test_new")

# Buat folder val dan test baru per kelas
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
for emo in emotions:
    (dst_val / emo).mkdir(parents=True, exist_ok=True)
    (dst_test_new / emo).mkdir(parents=True, exist_ok=True)

# Split 50:50 dari test asli → val + test baru
random.seed(42)  # untuk reproducibility
for emo in emotions:
    files = list((src_test / emo).glob("*.jpg")) + list((src_test / emo).glob("*.png"))
    random.shuffle(files)
    
    mid = len(files) // 2
    val_files = files[:mid]
    test_files = files[mid:]
    
    for f in val_files:
        shutil.copy(f, dst_val / emo / f.name)
    for f in test_files:
        shutil.copy(f, dst_test_new / emo / f.name)
    
    print(f"{emo}: {len(val_files)} val, {len(test_files)} test")

print("Split selesai!")