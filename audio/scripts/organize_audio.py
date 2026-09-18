"""
organize_audio.py - kumpulkan empat dataset suara ke satu folder raw/ per kelas.

Sumbernya punya konvensi nama yang berbeda-beda, jadi tiap dataset punya
fungsi sendiri. Semua hasilnya mendarat di:

    audio/datasets/audio_emotion/raw/<kelas>/<prefix>_<nama asli>.wav

Prefix (ravdess_, crema_, savee_, tess_) dipertahankan supaya asal tiap
berkas masih bisa ditelusuri setelah digabung, dan supaya nama yang kebetulan
sama antar dataset tidak saling menimpa.

Kenapa SAVEE dan TESS ditambahkan: CREMA-D tidak punya kelas surprise sama
sekali, jadi sebelumnya kelas itu cuma dapat 192 berkas dari RAVDESS sementara
kelas lain 1375-1463. TESS menyumbang 400 (pleasant surprise) dan SAVEE 60,
jadi ketimpangannya turun dari 7.6x ke sekitar 2.9x.

Skrip ini aman dijalankan ulang: berkas yang sudah ada ditimpa dengan isi yang
sama, bukan digandakan, karena nama tujuannya deterministik.

Jalankan lalu lanjutkan dengan split_audio.py:
    python audio/scripts/organize_audio.py
    python audio/scripts/split_audio.py
"""
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS = PROJECT_ROOT / "audio" / "datasets"
RAVDESS_DIR = DATASETS / "ravdess" / "AudioWAV"
CREMA_DIR = DATASETS / "crema-d" / "AudioWAV"
SAVEE_DIR = DATASETS / "savee" / "AudioWAV"
TESS_DIR = DATASETS / "tess" / "AudioWAV"
OUTPUT_DIR = DATASETS / "audio_emotion"

# Kode emosi RAVDESS, posisi ke-3 pada nama berkas 03-01-05-01-02-01-12.wav
ravdess_map = {
    "01": "neutral", "02": "neutral",  # calm digabung ke neutral (opsional)
    "03": "happy", "04": "sad",
    "05": "angry", "06": "fear",
    "07": "disgust", "08": "surprise"
}

# Kode emosi CREMA-D, potongan ke-3 pada 1001_DFA_ANG_XX.wav
crema_map = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fear",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad"
}

# Kode emosi SAVEE, huruf sebelum angka pada DC_sa01.wav.
# Perhatikan 'sa' dan 'su' dua huruf sementara sisanya satu huruf - karena itu
# angkanya dibuang dulu lalu dicocokkan utuh, bukan dengan startswith() yang
# akan mengira 'sa01' itu kelas 'a'.
savee_map = {
    "a": "angry", "d": "disgust", "f": "fear", "h": "happy",
    "n": "neutral", "sa": "sad", "su": "surprise"
}

# Kode emosi TESS, potongan terakhir pada OAF_back_angry.wav.
# 'ps' = pleasant surprise; TESS memang tidak punya surprise yang netral atau
# negatif, jadi kelas surprise proyek ini dari TESS semuanya versi menyenangkan.
tess_map = {
    "angry": "angry", "disgust": "disgust", "fear": "fear", "happy": "happy",
    "neutral": "neutral", "sad": "sad", "ps": "surprise"
}


def salin(sumber, label, prefix):
    """Salin satu berkas ke raw/<label>/ dengan nama berprefix."""
    dest = OUTPUT_DIR / "raw" / label
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(sumber, dest / f"{prefix}_{sumber.name}")


def lewati_kalau_kosong(folder, nama):
    """True kalau foldernya tidak ada - datasetnya belum diunduh."""
    if not folder.is_dir():
        print(f"{nama}: folder tidak ada ({folder}), dilewati")
        return True
    return False


def organize_ravdess():
    if lewati_kalau_kosong(RAVDESS_DIR, "RAVDESS"):
        return
    count = 0
    for f in RAVDESS_DIR.rglob("*.wav"):
        parts = f.stem.split("-")
        if len(parts) < 3:
            continue
        label = ravdess_map.get(parts[2])
        if label is None:
            continue
        salin(f, label, "ravdess")
        count += 1
    print(f"RAVDESS: {count} file diproses")


def organize_crema():
    if lewati_kalau_kosong(CREMA_DIR, "CREMA-D"):
        return
    count = 0
    for f in CREMA_DIR.glob("*.wav"):
        parts = f.stem.split("_")
        if len(parts) < 3:
            continue
        label = crema_map.get(parts[2])
        if label is None:
            continue
        salin(f, label, "crema")
        count += 1
    print(f"CREMA-D: {count} file diproses")


def organize_savee():
    """SAVEE: DC_sa01.wav -> penutur DC, kode 'sa', pengulangan 01.

    Cuma 4 penutur, semuanya laki-laki penutur asli bahasa Inggris.
    Tiap penutur: 15 berkas per emosi, kecuali neutral yang 30.
    """
    if lewati_kalau_kosong(SAVEE_DIR, "SAVEE"):
        return
    count = 0
    for f in SAVEE_DIR.glob("*.wav"):
        parts = f.stem.split("_")
        if len(parts) < 2:
            continue
        kode = parts[1].rstrip("0123456789")
        label = savee_map.get(kode)
        if label is None:
            continue
        salin(f, label, "savee")
        count += 1
    print(f"SAVEE: {count} file diproses")


def organize_tess():
    """TESS: OAF_back_ps.wav -> penutur OAF, kata 'back', emosi 'ps'.

    Cuma 2 penutur perempuan (OAF umur 64, YAF umur 26), masing-masing
    mengucapkan 200 kata yang sama untuk tiap emosi. Lihat catatan soal
    kebocoran penutur di DATASETS.txt.
    """
    if lewati_kalau_kosong(TESS_DIR, "TESS"):
        return
    count = 0
    for f in TESS_DIR.glob("*.wav"):
        parts = f.stem.split("_")
        if len(parts) < 3:
            continue
        label = tess_map.get(parts[-1].lower())
        if label is None:
            continue
        salin(f, label, "tess")
        count += 1
    print(f"TESS: {count} file diproses")


if __name__ == '__main__':
    organize_ravdess()
    organize_crema()
    organize_savee()
    organize_tess()

    print("\n=== Ringkasan per kelas emosi ===")
    total = 0
    for emo_folder in sorted((OUTPUT_DIR / "raw").iterdir()):
        if not emo_folder.is_dir():
            continue
        berkas = list(emo_folder.glob("*.wav"))
        total += len(berkas)
        # Pecah per dataset asal supaya kelihatan mana yang menyumbang apa.
        asal = {}
        for b in berkas:
            asal[b.name.split("_")[0]] = asal.get(b.name.split("_")[0], 0) + 1
        rincian = ", ".join(f"{k} {v}" for k, v in sorted(asal.items()))
        print(f"{emo_folder.name:9s}: {len(berkas):5d} file  ({rincian})")
    print(f"{'TOTAL':9s}: {total:5d} file")
    print("\nLanjutkan dengan: python audio/scripts/split_audio.py")
