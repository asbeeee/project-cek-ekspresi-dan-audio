"""
kdefTrain.py - susun ulang dataset KDEF/AKDEF jadi struktur siap latih YOLO.

Struktur asli KDEF dikelompokkan per subjek, bukan per emosi:

    webcam/datasets/KDEF_and_AKDEF/KDEF/AF01/AF01AFS.JPG
                            ^^^^ ^^ ^
                            |    |  +-- sudut  : FL HL S HR FR
                            |    +------ emosi  : AF AN DI HA NE SA SU
                            +----------- subjek : [sesi A/B][gender F/M][nomor 01-35]

YOLO klasifikasi butuh <split>/<kelas>/gambar.jpg, sama seperti fer2013. Skrip
ini mengubah yang atas jadi yang bawah:

    webcam/datasets/kdef/train/fear/AF01AFS.jpg
    webcam/datasets/kdef/val/happy/BM07HAS.jpg
    ...

DUA HAL PENTING
---------------
1. Split dilakukan per ORANG, bukan per gambar. KDEF memotret 70 orang dalam
   dua sesi (A dan B), jadi folder AF01 dan BF01 adalah orang yang SAMA. Kalau
   split-nya acak per gambar, wajah orang yang sama muncul di train dan test
   sekaligus - akurasinya jadi tinggi palsu karena model cuma menghafal orang,
   bukan belajar ekspresi. Di sini 70 orang itu yang dibagi, dan semua foto
   satu orang selalu jatuh di split yang sama.

2. Wajah dipotong dengan Haar cascade + padding 20%, persis seperti yang
   dilakukan fusion_webcam.py sebelum memanggil YOLO. Gambar KDEF aslinya
   562x762 dengan banyak latar dan leher; kalau tidak dipotong, data latih
   tidak mirip dengan yang dilihat model saat inferensi. Deteksi Haar tidak
   selalu berhasil (tampak samping hampir selalu gagal), jadi ada kotak
   cadangan tetap yang dikalibrasi dari median deteksi yang berhasil.

Contoh:
    python webcam/scripts/kdefTrain.py                       # susun ulang, default
    python webcam/scripts/kdefTrain.py --angles S            # hanya tampak depan
    python webcam/scripts/kdefTrain.py --angles S HL HR FL FR
    python webcam/scripts/kdefTrain.py --no_crop --imgsz 0   # salin apa adanya
    python webcam/scripts/kdefTrain.py --train --epochs 20   # susun lalu latih
"""
import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DEFAULT = PROJECT_ROOT / "webcam" / "datasets" / "KDEF_and_AKDEF"
DST_DEFAULT = PROJECT_ROOT / "webcam" / "datasets" / "kdef"

# Kode emosi KDEF -> nama kelas yang dipakai proyek ini.
EMOSI = {
    'AF': 'fear',
    'AN': 'angry',
    'DI': 'disgust',
    'HA': 'happy',
    'NE': 'neutral',
    'SA': 'sad',
    'SU': 'surprise',
}

SUDUT = ['FL', 'HL', 'S', 'HR', 'FR']   # full left, half left, straight, half right, full right

# [sesi][gender][nomor][emosi][sudut], mis. AF01AFS = sesi A, perempuan 01, afraid, straight
POLA_KDEF = re.compile(
    r'^(?P<sesi>[AB])(?P<gender>[FM])(?P<nomor>\d{2})'
    r'(?P<emosi>AF|AN|DI|HA|NE|SA|SU)(?P<sudut>FL|HL|S|HR|FR)$', re.I)

# AKDEF (wajah rata-rata): [gender][emosi][sudut], mis. FAFS
POLA_AKDEF = re.compile(
    r'^(?P<gender>[FM])(?P<emosi>AF|AN|DI|HA|NE|SA|SU)(?P<sudut>FL|HL|S|HR|FR)$', re.I)

# Berkas yang namanya salah di rilis resmi KDEF. Nama benarnya disimpulkan
# dari berkas yang hilang di folder yang sama: AF31 kurang SA-HL dan punya satu
# berkas asing 'AF31V'; AM31 kurang SU-HR dan punya 'AM31H'. Verifikasi sendiri
# dengan membuka gambarnya kalau mau yakin.
ERRATA = {
    'AF31V': 'AF31SAHL',
    'AM31H': 'AM31SUHR',
}

# AKDEF juga punya satu nama pincang: 'FAHR' seharusnya 'FAFHR' (afraid, half right).
ERRATA_AKDEF = {
    'FAHR': 'FAFHR',
}

# Kotak wajah cadangan per sudut (x, y, w, h relatif terhadap ukuran gambar),
# dipakai kalau Haar gagal atau hasilnya ditolak. Angka S/HL/HR adalah median
# dari deteksi Haar yang wajar pada dataset ini (n=980/743/820), jadi pas untuk
# framing KDEF yang seragam. FL/FR hampir tidak pernah terdeteksi Haar, jadi
# kotaknya dilebarkan manual - potongan tampak samping memang kurang rapi.
KOTAK_CADANGAN = {
    'S':  (0.135, 0.291, 0.726, 0.535),
    'HL': (0.039, 0.337, 0.575, 0.424),
    'HR': (0.363, 0.333, 0.594, 0.438),
    'FL': (0.000, 0.300, 0.700, 0.500),
    'FR': (0.300, 0.300, 0.700, 0.500),
}

# Batas kotak Haar yang masih masuk akal untuk framing KDEF. Sekitar 1.7%
# deteksi adalah positif palsu - kotak kecil di leher, bahu, atau latar - dan
# kalau dipakai hasil potongannya bukan wajah sama sekali. Batas ini diambil
# dari persentil 1-99 deteksi yang benar.
BATAS_KOTAK = {'w': (0.40, 0.90), 'y': (0.20, 0.50), 'x': (-0.05, 0.55)}

PAD_WAJAH = 0.20   # sama dengan FACE_PAD di fusion_webcam.py


# ===================================================================
# PEMBACAAN DATASET
# ===================================================================
def baca_kdef(src, sudut_dipakai):
    """Kembalikan dict subjek -> list (path, kelas, stem, sudut).

    `stem` adalah nama berkas yang sudah dikoreksi lewat ERRATA, bukan nama
    aslinya, supaya berkas keluaran namanya benar.

    Sesi A dan B sengaja digabung ke satu subjek karena orangnya sama.
    """
    root = src / "KDEF"
    if not root.is_dir():
        raise SystemExit(f"ERROR: folder KDEF tidak ada: {root}")

    per_subjek = defaultdict(list)
    dilewati, diperbaiki = [], []

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        for f in sorted(folder.iterdir()):
            if f.suffix.upper() not in ('.JPG', '.JPEG'):
                continue
            stem = f.stem.upper()
            if stem in ERRATA:
                diperbaiki.append((stem, ERRATA[stem]))
                stem = ERRATA[stem]
            m = POLA_KDEF.match(stem)
            if not m:
                dilewati.append(f"{folder.name}/{f.name}")
                continue
            d = m.groupdict()
            if d['sudut'].upper() not in sudut_dipakai:
                continue
            subjek = (d['gender'] + d['nomor']).upper()   # F01, M23, ...
            per_subjek[subjek].append((f, EMOSI[d['emosi'].upper()], stem,
                                       d['sudut'].upper()))

    if diperbaiki:
        print("[errata] nama berkas KDEF yang diperbaiki:")
        for lama, baru in diperbaiki:
            print(f"           {lama}.JPG -> dibaca sebagai {baru}")
    if dilewati:
        print(f"[peringatan] {len(dilewati)} berkas dilewati karena namanya "
              f"tidak dikenali:")
        for nama in dilewati[:10]:
            print(f"           {nama}")

    return per_subjek


def baca_akdef(src, sudut_dipakai):
    """Kembalikan list (path, kelas, stem, sudut) untuk wajah rata-rata AKDEF."""
    root = src / "AKDEF"
    if not root.is_dir():
        print(f"[peringatan] folder AKDEF tidak ada: {root}")
        return []

    hasil = []
    for f in sorted(root.iterdir()):
        if f.suffix.upper() not in ('.JPG', '.JPEG'):
            continue
        stem = ERRATA_AKDEF.get(f.stem.upper(), f.stem.upper())
        m = POLA_AKDEF.match(stem)
        if not m:
            print(f"[peringatan] AKDEF/{f.name} dilewati, nama tidak dikenali.")
            continue
        d = m.groupdict()
        if d['sudut'].upper() not in sudut_dipakai:
            continue
        hasil.append((f, EMOSI[d['emosi'].upper()], stem, d['sudut'].upper()))
    return hasil


# ===================================================================
# PEMBAGIAN SPLIT (per orang)
# ===================================================================
def bagi_subjek(subjek, ratio, seed):
    """Bagi daftar subjek jadi train/val/test, seimbang antar gender."""
    rng = np.random.default_rng(seed)
    hasil = {'train': [], 'val': [], 'test': []}

    # Dipisah per gender dulu supaya proporsi laki/perempuan mirip di tiap split.
    for gender in ('F', 'M'):
        orang = sorted(s for s in subjek if s.startswith(gender))
        rng.shuffle(orang)
        n = len(orang)
        n_train = int(round(n * ratio[0]))
        n_val = int(round(n * ratio[1]))
        # Sisanya ke test, supaya tidak ada subjek yang hilang karena pembulatan.
        hasil['train'] += orang[:n_train]
        hasil['val'] += orang[n_train:n_train + n_val]
        hasil['test'] += orang[n_train + n_val:]

    return hasil


# ===================================================================
# PEMROSESAN GAMBAR
# ===================================================================
def buat_detektor():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    det = cv2.CascadeClassifier(path)
    if det.empty():
        raise SystemExit("ERROR: Haar cascade gagal dimuat.")
    return det


def kotak_masuk_akal(x, y, w, lebar, tinggi):
    """True kalau kotak Haar sesuai framing KDEF yang seragam."""
    rx, ry, rw = x / lebar, y / tinggi, w / lebar
    return (BATAS_KOTAK['w'][0] <= rw <= BATAS_KOTAK['w'][1]
            and BATAS_KOTAK['y'][0] <= ry <= BATAS_KOTAK['y'][1]
            and BATAS_KOTAK['x'][0] <= rx <= BATAS_KOTAK['x'][1])


def potong_wajah(img, detektor, sudut):
    """Potong wajah + padding.

    Kembalikan (gambar, status) dengan status 'haar', 'ditolak', atau 'gagal'.
    """
    tinggi, lebar = img.shape[:2]
    abu = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    wajah = detektor.detectMultiScale(abu, scaleFactor=1.1, minNeighbors=5,
                                      minSize=(80, 80))
    x = y = w = h = None
    status = 'gagal'
    if len(wajah):
        bx, by, bw, bh = max(wajah, key=lambda b: b[2] * b[3])
        if kotak_masuk_akal(bx, by, bw, lebar, tinggi):
            x, y, w, h = bx, by, bw, bh
            status = 'haar'
        else:
            status = 'ditolak'

    if x is None:
        rx, ry, rw, rh = KOTAK_CADANGAN.get(sudut, KOTAK_CADANGAN['S'])
        x, y, w, h = int(rx * lebar), int(ry * tinggi), int(rw * lebar), int(rh * tinggi)

    px, py = int(w * PAD_WAJAH), int(h * PAD_WAJAH)
    x1, y1 = max(0, x - px), max(0, y - py)
    x2, y2 = min(lebar, x + w + px), min(tinggi, y + h + py)
    return img[y1:y2, x1:x2], status


def proses_dan_simpan(path_src, path_dst, detektor, imgsz, grayscale, sudut):
    """Baca, potong, ubah ukuran, simpan.

    Kembalikan status: 'haar', 'ditolak', 'gagal', 'tanpa-crop', atau None
    kalau berkasnya tidak terpakai (tidak terbaca / kosong).
    """
    img = cv2.imread(str(path_src))
    if img is None:
        return None
    # Beberapa berkas KDEF asli isinya hitam polos - berkas pengganti berukuran
    # persis 12104 byte dengan mean 0.61. Kalau ikut dilatih, model belajar dari
    # gambar kosong. Gambar normal mean-nya di atas 55, jadi ambang 10 aman.
    if float(img.mean()) < 10.0:
        return None

    status = 'tanpa-crop'
    if detektor is not None:
        img, status = potong_wajah(img, detektor, sudut)

    if imgsz:
        interp = cv2.INTER_AREA if img.shape[0] > imgsz else cv2.INTER_LINEAR
        img = cv2.resize(img, (imgsz, imgsz), interpolation=interp)

    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(str(path_dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return status


# ===================================================================
# PENYUSUNAN ULANG
# ===================================================================
def susun(args):
    src = Path(args.src)
    dst = Path(args.dst)
    sudut_dipakai = {s.upper() for s in args.angles}

    print(f"[sumber] {src}")
    print(f"[tujuan] {dst}")
    print(f"[sudut ] {', '.join(sorted(sudut_dipakai))}")

    per_subjek = baca_kdef(src, sudut_dipakai)
    n_gambar = sum(len(v) for v in per_subjek.values())
    print(f"[baca  ] {n_gambar} gambar dari {len(per_subjek)} orang "
          f"(sesi A dan B digabung karena orangnya sama)")
    if not n_gambar:
        raise SystemExit("ERROR: tidak ada gambar yang cocok dengan filter sudut.")

    pembagian = bagi_subjek(list(per_subjek), args.ratio, args.seed)
    print(f"[split ] orang -> train {len(pembagian['train'])}, "
          f"val {len(pembagian['val'])}, test {len(pembagian['test'])}  "
          f"(seed={args.seed})")

    akdef = baca_akdef(src, sudut_dipakai) if args.include_akdef else []
    if akdef:
        print(f"[akdef ] {len(akdef)} wajah rata-rata ditambahkan ke train saja. "
              f"Wajah ini dirata-ratakan dari SEMUA subjek, termasuk yang ada di "
              f"val/test, jadi tidak boleh masuk ke sana.")

    if args.dry_run:
        print("\n[dry-run] tidak ada berkas yang ditulis.")
        rangkum_rencana(per_subjek, pembagian, akdef)
        return dst

    if dst.exists():
        if not args.force:
            raise SystemExit(
                f"ERROR: {dst} sudah ada. Hapus dulu, atau jalankan dengan --force "
                f"untuk menimpanya.")
        print(f"[hapus ] membersihkan {dst}")
        shutil.rmtree(dst)

    for split in ('train', 'val', 'test'):
        for kelas in sorted(set(EMOSI.values())):
            (dst / split / kelas).mkdir(parents=True, exist_ok=True)

    detektor = None if args.no_crop else buat_detektor()
    total = n_gambar + len(akdef)
    ditulis = 0
    status_hitung = defaultdict(int)
    dibuang = []
    hitung = defaultdict(lambda: defaultdict(int))

    tugas = []
    for split, daftar_orang in pembagian.items():
        for orang in daftar_orang:
            for path, kelas, stem, sudut in per_subjek[orang]:
                tugas.append((split, kelas, path, stem, sudut))
    for path, kelas, stem, sudut in akdef:
        tugas.append(('train', kelas, path, stem, sudut))

    for split, kelas, path, stem, sudut in tugas:
        tujuan = dst / split / kelas / (stem + ".jpg")
        status = proses_dan_simpan(path, tujuan, detektor, args.imgsz,
                                   args.grayscale, sudut)
        if status is None:
            dibuang.append(stem)
            continue
        status_hitung[status] += 1
        hitung[split][kelas] += 1
        ditulis += 1
        if ditulis % 200 == 0 or ditulis == total:
            print(f"  {ditulis}/{total} gambar", end='\r', flush=True)
    print()

    if dibuang:
        print(f"[buang ] {len(dibuang)} berkas dilewati karena isinya hitam polos "
              f"di rilis KDEF aslinya:")
        for nama in dibuang:
            print(f"           {nama}.JPG")

    if detektor is not None:
        cadangan = status_hitung['ditolak'] + status_hitung['gagal']
        print(f"[crop  ] Haar berhasil {status_hitung['haar']}, "
              f"gagal {status_hitung['gagal']}, "
              f"ditolak karena kotaknya tidak masuk akal {status_hitung['ditolak']}.")
        if cadangan:
            print(f"         {cadangan} gambar ({100.0 * cadangan / ditulis:.1f}%) "
                  f"memakai kotak cadangan per sudut.")

    cetak_tabel(hitung)
    return dst


def rangkum_rencana(per_subjek, pembagian, akdef):
    """Untuk --dry_run: berapa gambar per split tanpa menulis apa pun."""
    hitung = defaultdict(lambda: defaultdict(int))
    for split, daftar_orang in pembagian.items():
        for orang in daftar_orang:
            for _, kelas, _stem, _sudut in per_subjek[orang]:
                hitung[split][kelas] += 1
    for _, kelas, _stem, _sudut in akdef:
        hitung['train'][kelas] += 1
    cetak_tabel(hitung)


def cetak_tabel(hitung):
    kelas_semua = sorted(set(EMOSI.values()))
    splits = ['train', 'val', 'test']
    w = max(9, max(len(k) for k in kelas_semua))

    print()
    print(f"{'kelas':<{w}}" + "".join(f"  {s:>7}" for s in splits) + f"  {'TOTAL':>8}")
    print("-" * (w + 37))
    for kelas in kelas_semua:
        baris = [hitung[s][kelas] for s in splits]
        print(f"{kelas:<{w}}" + "".join(f"  {n:>7}" for n in baris)
              + f"  {sum(baris):>8}")
    print("-" * (w + 37))
    total_split = [sum(hitung[s].values()) for s in splits]
    print(f"{'TOTAL':<{w}}" + "".join(f"  {n:>7}" for n in total_split)
          + f"  {sum(total_split):>8}")


# ===================================================================
# LATIH
# ===================================================================
def latih(dst, args):
    from ultralytics import YOLO

    bobot_awal = PROJECT_ROOT / "webcam" / "models" / "yolov8n-cls.pt"
    if not bobot_awal.exists():
        print(f"[latih ] {bobot_awal.name} tidak ada, ultralytics akan mengunduhnya.")
        bobot_awal = "yolov8n-cls.pt"

    model = YOLO(str(bobot_awal))
    hasil = model.train(
        data=str(dst),
        epochs=args.epochs,
        imgsz=args.imgsz or 224,
        batch=32,
        patience=10,
        device='cpu',
        project='runs/emotion',
        name='kdef_baseline',
        workers=0,
    )
    print("\nTraining selesai!")
    print(f"Best model: {hasil.save_dir}/weights/best.pt")


def main():
    p = argparse.ArgumentParser(
        description="Susun ulang KDEF/AKDEF jadi struktur <split>/<kelas>/ "
                    "untuk melatih YOLO klasifikasi.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Contoh:\n"
               "  python webcam/scripts/kdefTrain.py --dry_run\n"
               "  python webcam/scripts/kdefTrain.py --angles S HL HR\n"
               "  python webcam/scripts/kdefTrain.py --train --epochs 20")
    p.add_argument('--src', default=str(SRC_DEFAULT),
                   help="Folder KDEF_and_AKDEF.")
    p.add_argument('--dst', default=str(DST_DEFAULT),
                   help="Folder tujuan hasil susun ulang.")
    p.add_argument('--angles', nargs='+', default=['S', 'HL', 'HR'],
                   choices=SUDUT + [s.lower() for s in SUDUT], metavar='SUDUT',
                   help="Sudut yang dipakai: FL HL S HR FR "
                        "(default: S HL HR, yaitu depan dan setengah samping).")
    p.add_argument('--ratio', nargs=3, type=float, default=[0.70, 0.15, 0.15],
                   metavar=('TRAIN', 'VAL', 'TEST'),
                   help="Proporsi pembagian ORANG (default: 0.70 0.15 0.15).")
    p.add_argument('--seed', type=int, default=42,
                   help="Seed pembagian subjek (default: 42).")
    p.add_argument('--imgsz', type=int, default=224,
                   help="Ukuran keluaran persegi; 0 = biarkan ukuran asli "
                        "(default: 224).")
    p.add_argument('--no_crop', '--no-crop', dest='no_crop', action='store_true',
                   help="Jangan potong wajah, salin bingkai penuh.")
    p.add_argument('--grayscale', action='store_true',
                   help="Simpan sebagai grayscale, mirip FER2013.")
    p.add_argument('--include_akdef', '--include-akdef', dest='include_akdef',
                   action='store_true',
                   help="Ikutkan 70 wajah rata-rata AKDEF ke split train.")
    p.add_argument('--force', action='store_true',
                   help="Timpa folder tujuan kalau sudah ada.")
    p.add_argument('--dry_run', '--dry-run', dest='dry_run', action='store_true',
                   help="Tampilkan rencana pembagian tanpa menulis berkas.")
    p.add_argument('--train', action='store_true',
                   help="Langsung latih YOLO setelah dataset tersusun.")
    p.add_argument('--epochs', type=int, default=20,
                   help="Jumlah epoch kalau --train dipakai (default: 20).")
    args = p.parse_args()

    if abs(sum(args.ratio) - 1.0) > 1e-6:
        raise SystemExit(f"ERROR: --ratio harus berjumlah 1.0, sekarang {sum(args.ratio)}")

    dst = susun(args)

    if args.dry_run:
        return
    if args.train:
        latih(dst, args)
    else:
        print(f"\nDataset siap di: {dst}")
        print("Latih dengan:")
        print(f"  python webcam/scripts/kdefTrain.py --train --epochs 20")
        print("atau arahkan skrip training yang sudah ada ke folder itu.")


if __name__ == '__main__':
    sys.exit(main())
