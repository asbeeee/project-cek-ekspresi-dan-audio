from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if __name__ == '__main__':
    # Load model
    model = YOLO(str(PROJECT_ROOT / "webcam" / "models" / "fer2013_baseline-2" / "weights" / "best.pt"))
    
    # Nama kelas emosi (urutan sesuai folder training)
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # Buka webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Inferensi
        results = model.predict(frame, imgsz=224, verbose=False)
        
        # Ambil hasil
        probs = results[0].probs
        top_idx = probs.top1
        top_conf = probs.top1conf.item()
        
        # Tampilkan label
        label = f"{emotions[top_idx]}: {top_conf:.2f}"
        cv2.putText(frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        # Tampilkan top-3 di bawah
        top5_idx = probs.top5
        top5_conf = probs.top5conf.tolist()
        for i in range(3):
            text = f"{emotions[top5_idx[i]]}: {top5_conf[i]:.2f}"
            cv2.putText(frame, text, (20, 80 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('Emotion Recognition (FER2013)', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()