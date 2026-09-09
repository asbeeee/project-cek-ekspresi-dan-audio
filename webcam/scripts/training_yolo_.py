from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if __name__ == '__main__':
    # Load pretrained YOLOv8 classification model
    model = YOLO(str(PROJECT_ROOT / "webcam" / "models" / "yolov8n-cls.pt"))

    # Training dengan path relatif ke root proyek
    results = model.train(
        data=str(PROJECT_ROOT / "webcam" / "datasets" / "fer2013"),
        epochs=20,
        imgsz=224,
        batch=32,
        patience=10,
        device='cpu',
        project='runs/emotion',
        name='fer2013_baseline',
        workers=0               # tambahan: matikan multiprocessing
    )

    print("Training selesai!")
    print(f"Best model disimpan di: {results.save_dir}/weights/best.pt")