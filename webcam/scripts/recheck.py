from pathlib import Path

root = Path("fer2013")
for split in ["train", "val", "test"]:
    print(f"\n{split}/")
    for emo_folder in sorted((root / split).iterdir()):
        if emo_folder.is_dir():
            count = len(list(emo_folder.iterdir()))
            print(f"  {emo_folder.name}: {count} gambar")