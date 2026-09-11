"""
fusion_realtime.py  (v2 - dengan deteksi wajah + bounding box)
=============================================================
Real-time Multimodal Emotion Recognition:
Webcam (Haar face detect + YOLO classify) + Mikrofon (CNN-MFCC)
-> Decision-Level Fusion (Persamaan 2.1)
-> Compound Emotion Detection (Persamaan 2.2)

Tugas Akhir - Hasbi Huda Maulaya (140910220045)

Cara pakai:
    pip install sounddevice
    python fusion_realtime.py

Kontrol:
    q = keluar | a/z = alpha +/- | t/g = tau +/-

PERUBAHAN v2:
- Tambah Haar Cascade face detector -> gambar bounding box wajah
- Classifier YOLO hanya jalan pada crop wajah (bukan seluruh frame)
  -> memperbaiki masalah "stuck di netral"
- Kalau tidak ada wajah terdeteksi, tampilkan peringatan
"""

import time
import threading
from pathlib import Path

import numpy as np
import cv2
import librosa
import torch
import torch.nn as nn
import sounddevice as sd
from ultralytics import YOLO

# ===================================================================
# KONFIGURASI
# ===================================================================
# fusion/<this file> -> project root is parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VISUAL_MODEL_PATH = str(PROJECT_ROOT / "webcam" / "models" / "fer2013_baseline-2" / "weights" / "best.pt")
AUDIO_MODEL_PATH  = str(PROJECT_ROOT / "audio" / "models" / "best_audio_cnn.pt")

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTIONS_ID = {
    'angry': 'Marah', 'disgust': 'Jijik', 'fear': 'Takut',
    'happy': 'Senang', 'neutral': 'Netral', 'sad': 'Sedih',
    'surprise': 'Terkejut'
}

ALPHA = 0.6
TAU = 0.20
TAU_CONF = 0.40

SR = 16000
DURATION = 3
N_MFCC = 40
AUDIO_INFER_INTERVAL = 1.0

DEVICE = 'cpu'
FACE_PAD = 0.20


class AudioCNN(nn.Module):
    def __init__(self, n_classes=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)


class AudioWorker:
    def __init__(self, model):
        self.model = model
        self.buffer = np.zeros(SR * DURATION, dtype=np.float32)
        self.lock = threading.Lock()
        self.p_audio = np.ones(len(EMOTIONS)) / len(EMOTIONS)
        self.audio_level = 0.0
        self.running = True

    def _callback(self, indata, frames, t, status):
        mono = indata[:, 0].astype(np.float32)
        with self.lock:
            self.buffer = np.roll(self.buffer, -len(mono))
            self.buffer[-len(mono):] = mono
            self.audio_level = float(np.abs(mono).mean())

    def _infer_loop(self):
        while self.running:
            time.sleep(AUDIO_INFER_INTERVAL)
            with self.lock:
                y = self.buffer.copy()
            mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
            # mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-8)
            x = torch.tensor(mfcc.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(self.model(x), dim=1).cpu().numpy()[0]
            self.p_audio = probs

    def start(self):
        self.stream = sd.InputStream(
            samplerate=SR, channels=1, dtype='float32',
            callback=self._callback, blocksize=int(SR * 0.1)
        )
        self.stream.start()
        self.thread = threading.Thread(target=self._infer_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.stream.stop()
        self.stream.close()


def decision_level_fusion(p_visual, p_audio, alpha):
    return alpha * p_visual + (1.0 - alpha) * p_audio


def compound_emotion_decision(p_final, tau, tau_conf):
    idx = np.argsort(p_final)[::-1]
    t1, t2 = idx[0], idx[1]
    p1, p2 = p_final[t1], p_final[t2]
    gap = p1 - p2

    if p1 < tau_conf:
        return {'label_id': 'Netral', 'type': 'TIDAK YAKIN',
                'c1': t1, 'c2': t2, 'p1': p1, 'p2': p2, 'gap': gap}
    elif gap >= tau:
        return {'label_id': EMOTIONS_ID[EMOTIONS[t1]], 'type': 'TUNGGAL',
                'c1': t1, 'c2': t2, 'p1': p1, 'p2': p2, 'gap': gap}
    else:
        lbl = f"{EMOTIONS_ID[EMOTIONS[t1]]}-{EMOTIONS_ID[EMOTIONS[t2]]}"
        return {'label_id': lbl, 'type': 'MAJEMUK',
                'c1': t1, 'c2': t2, 'p1': p1, 'p2': p2, 'gap': gap}


def get_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("Haar cascade gagal dimuat.")
    return detector


def detect_largest_face(detector, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def crop_face(frame, box, pad=FACE_PAD):
    x, y, w, h = box
    px, py = int(w * pad), int(h * pad)
    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(frame.shape[1], x + w + px)
    y2 = min(frame.shape[0], y + h + py)
    return frame[y1:y2, x1:x2]


TYPE_COLOR = {
    'TUNGGAL': (0, 220, 0),
    'MAJEMUK': (0, 165, 255),
    'TIDAK YAKIN': (160, 160, 160)
}

def draw_overlay(frame, p_visual, p_audio, p_final, result, alpha, tau,
                 fps, audio_level, face_box):
    h, w = frame.shape[:2]
    color = TYPE_COLOR.get(result['type'], (255, 255, 255))

    if face_box is not None:
        x, y, fw, fh = face_box
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
        cv2.rectangle(frame, (x, y - 28), (x + fw, y), color, -1)
        cv2.putText(frame, result['label_id'], (x + 4, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    else:
        cv2.putText(frame, "WAJAH TIDAK TERDETEKSI", (w // 2 - 180, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (330, h), (25, 25, 25), -1)
    frame = cv2.addWeighted(panel, 0.55, frame, 0.45, 0)

    cv2.putText(frame, result['label_id'], (12, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    cv2.putText(frame, f"[{result['type']}]  gap={result['gap']:.3f}", (12, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    y0 = 105
    cv2.putText(frame, f"{'Kelas':<9}{'Vis':>6}{'Aud':>6}{'Fus':>6}", (12, y0 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    for i, emo in enumerate(EMOTIONS):
        y = y0 + i * 34
        is_top1 = (i == result['c1'])
        txt_color = color if is_top1 else (230, 230, 230)
        cv2.putText(frame, f"{EMOTIONS_ID[emo]:<9}{p_visual[i]:>6.2f}{p_audio[i]:>6.2f}{p_final[i]:>6.2f}",
                    (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, txt_color, 1)
        bar_len = int(p_final[i] * 180)
        cv2.rectangle(frame, (12, y + 6), (12 + bar_len, y + 14),
                      color if is_top1 else (110, 110, 110), -1)

    yb = y0 + 7 * 34 + 16
    cv2.putText(frame, f"alpha={alpha:.2f}  tau={tau:.2f}  conf>={TAU_CONF:.2f}",
                (12, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 255), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}   mic: {'#' * min(int(audio_level * 400), 15):<15}",
                (12, yb + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 255, 180), 1)
    cv2.putText(frame, "q=keluar  a/z=alpha  t/g=tau",
                (12, yb + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    return frame


def main():
    global ALPHA, TAU

    print("[1/4] Load model visual (YOLO)...")
    visual_model = YOLO(VISUAL_MODEL_PATH)

    print("[2/4] Load face detector (Haar Cascade)...")
    face_detector = get_face_detector()

    print("[3/4] Load model audio (CNN-MFCC)...")
    audio_model = AudioCNN(len(EMOTIONS)).to(DEVICE)
    audio_model.load_state_dict(torch.load(AUDIO_MODEL_PATH, map_location=DEVICE))
    audio_model.eval()

    print("[4/4] Start mikrofon + webcam...")
    worker = AudioWorker(audio_model)
    worker.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Webcam tidak bisa dibuka.")
        worker.stop()
        return

    prev_t = time.time()
    fps = 0.0
    last_p_visual = np.ones(len(EMOTIONS)) / len(EMOTIONS)

    print("Berjalan! Posisikan wajah ke kamera & bicara ke mikrofon.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            face_box = detect_largest_face(face_detector, frame)

            if face_box is not None:
                face_img = crop_face(frame, face_box)
                if face_img.size > 0:
                    results = visual_model.predict(face_img, imgsz=224, verbose=False)
                    last_p_visual = results[0].probs.data.cpu().numpy()
            p_visual = last_p_visual

            p_audio = worker.p_audio

            p_final = decision_level_fusion(p_visual, p_audio, ALPHA)
            result = compound_emotion_decision(p_final, TAU, TAU_CONF)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
            prev_t = now

            frame = draw_overlay(frame, p_visual, p_audio, p_final,
                                 result, ALPHA, TAU, fps, worker.audio_level, face_box)
            cv2.imshow('Multimodal Emotion Fusion (Realtime)', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                ALPHA = min(1.0, ALPHA + 0.05)
            elif key == ord('z'):
                ALPHA = max(0.0, ALPHA - 0.05)
            elif key == ord('t'):
                TAU = min(0.9, TAU + 0.05)
            elif key == ord('g'):
                TAU = max(0.0, TAU - 0.05)
    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("Selesai.")


if __name__ == '__main__':
    main()