"""
common.py - definisi yang dipakai bersama oleh skrip-skrip di repo ini.

Sebelum ada berkas ini, hal-hal berikut ditulis ulang di beberapa tempat:

    EMOTIONS        di 5 berkas
    AudioCNN        di 3 berkas, arsitekturnya harus persis sama
    SR/DURATION/N_MFCC  di 3 berkas

Yang paling berbahaya adalah AudioCNN. Arsitekturnya harus identik antara
skrip yang melatih dan skrip yang memuat bobotnya; kalau salah satu diubah
sendirian, load_state_dict langsung gagal atau - lebih buruk - berhasil tapi
modelnya salah. Sekarang definisinya cuma ada di sini.

Impor torch sengaja ditaruh DI DALAM fungsi, bukan di atas, supaya skrip yang
cuma butuh daftar EMOTIONS (merge_datasets.py, kdefTrain.py, recheck.py) tidak
ikut menunggu torch dimuat.

Skrip di dalam subfolder perlu menambahkan root proyek ke sys.path dulu:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from common import EMOTIONS
"""

# ===================================================================
# KELAS EMOSI
# ===================================================================
# Urutannya menentukan indeks kelas di seluruh proyek dan HARUS cocok dengan
# urutan folder saat training (ultralytics mengurutkan nama folder secara
# alfabetis). Jangan diubah tanpa melatih ulang semua model.
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

EMOTIONS_ID = {
    'angry': 'Marah', 'disgust': 'Jijik', 'fear': 'Takut',
    'happy': 'Senang', 'neutral': 'Netral', 'sad': 'Sedih',
    'surprise': 'Terkejut'
}


# ===================================================================
# PARAMETER AUDIO
# ===================================================================
SR = 16000        # sampling rate, sesuai proposal
DURATION = 3      # detik; dipotong atau dipad ke panjang ini
N_MFCC = 40       # jumlah koefisien MFCC


def extract_mfcc(y, sr=SR):
    """MFCC dari sinyal audio, bentuknya selalu (N_MFCC, 94).

    `y` adalah array 1 dimensi. Panjangnya dipotong atau dipad dulu ke
    SR * DURATION supaya bentuk keluarannya konsisten.
    """
    import numpy as np
    import librosa

    target = SR * DURATION
    y = np.pad(y, (0, target - len(y))) if len(y) < target else y[:target]
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).astype(np.float32)


def load_wav_mfcc(path):
    """Baca berkas WAV lalu kembalikan MFCC-nya."""
    import librosa
    y, _ = librosa.load(path, sr=SR)
    return extract_mfcc(y)


# ===================================================================
# MODEL AUDIO
# ===================================================================
def make_audio_cnn(n_classes=len(EMOTIONS)):
    """CNN-MFCC untuk klasifikasi emosi dari suara.

    Dibungkus fungsi supaya torch hanya diimpor kalau memang dipakai.
    Bobot tersimpan memakai prefix 'net.' karena aslinya modul ini punya
    atribut .net berisi Sequential - lihat load_audio_cnn().
    """
    import torch.nn as nn

    class AudioCNN(nn.Module):
        def __init__(self, n_classes):
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

    return AudioCNN(n_classes)


def load_audio_cnn(path, device='cpu', n_classes=len(EMOTIONS)):
    """Buat model lalu muat bobotnya dari `path`. Kembalikan model mode eval."""
    import torch

    model = make_audio_cnn(n_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()
