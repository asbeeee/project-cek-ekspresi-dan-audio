from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if __name__ == '__main__':
    # Load best model hasil training
    model = YOLO(str(PROJECT_ROOT / "webcam" / "models" / "fer2013_baseline-2" / "weights" / "best.pt"))

    # Evaluasi di test set
    metrics = model.val(
        data=str(PROJECT_ROOT / "webcam" / "datasets" / "fer2013"),
        split='test',
        imgsz=224,
        batch=32,
        device='cpu',
        workers=0
    )
    
    print(f"\n=== Hasil Test Set ===")
    print(f"Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"Top-5 Accuracy: {metrics.top5:.4f}")