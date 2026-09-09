
Real-time multimodal emotion recognition:
YOLO (wajah, FER2013) + CNN-MFCC (audio, RAVDESS + CREMA-D) → decision-level fusion.
Termasuk respons robot AiNex Hiwonder (lambaian tangan pada ekspresi *happy*).


## Struktur

```
.
├── fusion_webcam.py           # entry point: fusion realtime + kontrol robot
├── requirements.txt
├── audio/
│   ├── scripts/               # organize, split, train audio CNN
│   └── models/                # best_audio_cnn.pt
├── webcam/
│   ├── scripts/               # training YOLO + tes webcam
│   ├── models/                # yolo pretrained + fer2013_baseline-2/
│   └── results/               # confusion matrix, val runs
├── fusion/                    # skrip fusion versi lama (test)
└── Servers/                   # requirements untuk laptop server
```

Semua path di dalam skrip memakai `Path(__file__)`, jadi tidak ada
hardcoded drive `E:\` - repo bisa di-clone ke laptop mana pun.

## Datasets (tidak ikut di repo)

Terlalu besar untuk GitHub. Download manual dan taruh di:

- `audio/datasets/ravdess/AudioWAV/` - <https://zenodo.org/record/1188976>
- `audio/datasets/crema-d/AudioWAV/` - <https://github.com/CheyneyComputerScience/CREMA-D>
- `webcam/datasets/fer2013/` - Kaggle FER2013
- `webcam/datasets/FERPlus/` - <https://github.com/microsoft/FERPlus>

Setelah dataset audio ada, jalankan:
```bash
python audio/scripts/organize_audio.py
python audio/scripts/split_audio.py
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate         # PowerShell
pip install -r requirements.txt
```

## Menjalankan

Webcam laptop + mikrofon:
```bash
python fusion_webcam.py
```

Kamera robot AiNex lewat LAN, tanpa audio, dengan respons lambaian:
```bash
python fusion_webcam.py \
  --camera_url "http://192.168.50.2:8080/stream?topic=/camera/image_raw&type=ros_compressed" \
  --no_speech \
  --wave_url http://192.168.50.2:5000/wave
```

Tanpa `--wave_url`, sistem cetak `[ROBOT] WAVE!` ke konsol (dry-run).
Cooldown lambaian default 60 detik, ubah dengan `--wave_cooldown DETIK`.

## Training ulang

Audio CNN:
```bash
python audio/scripts/train_audio_cnn.py
```

YOLO klasifikasi wajah:
```bash
python webcam/scripts/training_yolo_.py
```
