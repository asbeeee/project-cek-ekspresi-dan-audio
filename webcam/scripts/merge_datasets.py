"""
merge_datasets.py - gabungkan FER2013 dan KDEF jadi satu dataset latih.

Menghasilkan webcam/datasets/combined/<split>/<kelas>/ yang berisi kedua
sumber, siap dipakai YOLO klasifikasi.

DUA MASALAH YANG DITANGANI
--------------------------
1. Jumlahnya timpang jauh. FER2013 punya 28709 gambar latih, KDEF cuma 2013 -
   perbandingan 14.3 : 1. Kalau digabung apa adanya, KDEF cuma 6.6% dari data
   dan praktis tidak terasa. Karena itu KDEF diulang beberapa kali
   (--kdef_repeat, default 4) sehingga porsinya naik ke sekitar 22%. Ultralytics
   mengaugmentasi ulang tiap epoch, jadi salinan yang sama tidak dilihat model
   secara identik.

2. Domainnya beda jauh. FER2013 itu 48x48 grayscale hasil jepretan liar; KDEF
   224x224 berwarna, studio, berpose. Kalau warnanya dibiarkan, model bisa
   ambil jalan pintas: "berwarna berarti KDEF, abu-abu berarti FER2013", lalu
   belajar dua aturan terpisah alih-alih satu yang umum. Karena itu semuanya
   dijadikan grayscale (--color untuk membatalkan). Beda resolusi masih ada,
   tapi jauh lebih tidak kentara daripada beda warna.

PENTING SOAL EVALUASI
---------------------
Split test digabung di sini hanya supaya strukturnya lengkap. Untuk laporan,
ukur di test set masing-masing dataset secara TERPISAH, karena angka gabungan
menyembunyikan pertukaran antar-domain:

    python results_calculation.py face --visual_model runs/.../best.pt
    python results_calculation.py face --visual_model runs/.../best.pt \\
        --split test --no_cache

Contoh:
    python webcam/scripts/merge_datasets.py --dry_run
    python webcam/scripts/merge_datasets.py
    python webcam/scripts/merge_datasets.py --kdef_repeat 6 --cap 3000
    python webcam/scripts/merge_datasets.py --train --epochs 20
"""
import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from common import EMOTIONS

FER_DIR = PROJECT_ROOT / "webcam" / "datasets" / "fer2013"
KDEF_DIR = PROJECT_ROOT / "webcam" / "datasets" / "kdef"
DST_DEFAULT = PROJECT_ROOT / "webcam" / "datasets" / "combined"

SPLITS = ['train', 'val', 'test']
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}


def kumpulkan(root, split):
    """dict kelas -> list path, untuk satu split."""
    hasil = {}
    for kelas in EMOTIONS:
        folder = root / split / kelas
        if not folder.is_dir():
            hasil[kelas] = []
            continue
        hasil[kelas] = sorted(f for f in folder.iterdir()
                              if f.is_file() and f.suffix.lower() in IMG_EXT)
    return hasil


def periksa_sumber():
    for nama, root in [('fer2013', FER_DIR), ('kdef', KDEF_DIR)]:
        if not root.is_dir():
            raise SystemExit(
                f"ERROR: dataset '{nama}' tidak ada di {root}\n"
                f"       KDEF dibuat dengan: python webcam/scripts/kdefTrain.py")


def rencana(args):
    """Susun daftar tugas salin: (split, kelas, path_sumber, nama_tujuan)."""
    rng = random.Random(args.seed)
    tugas = []
    hitung = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for split in SPLITS:
        fer = kumpulkan(FER_DIR, split)
        kdef = kumpulkan(KDEF_DIR, split)

        for kelas in EMOTIONS:
            berkas_fer = list(fer[kelas])
            # Batasi kelas mayoritas FER2013 supaya tidak terlalu timpang.
            # Hanya di train; val dan test dibiarkan utuh agar tetap bisa
            # dibandingkan dengan hasil sebelumnya.
            if split == 'train' and args.cap and len(berkas_fer) > args.cap:
                rng.shuffle(berkas_fer)
                berkas_fer = berkas_fer[:args.cap]

            for f in berkas_fer:
                tugas.append((split, kelas, f, f"fer_{f.stem}.jpg", 'fer2013'))
                hitung[split][kelas]['fer2013'] += 1

            # KDEF diulang hanya di train. Mengulang di val/test cuma membuat
            # bobot metrik menyimpang tanpa menambah informasi.
            ulang = args.kdef_repeat if split == 'train' else 1
            for f in kdef[kelas]:
                for i in range(ulang):
                    akhiran = "" if i == 0 else f"_r{i + 1}"
                    tugas.append((split, kelas, f, f"kdef_{f.stem}{akhiran}.jpg",
                                  'kdef'))
                    hitung[split][kelas]['kdef'] += 1

    return tugas, hitung


def salin(tugas, dst, grayscale):
    """Jalankan daftar tugas. FER2013 disalin apa adanya karena sudah grayscale."""
    total = len(tugas)
    selesai = 0
    gagal = []
    # Satu berkas KDEF bisa dipakai berkali-kali; hasil konversinya di-cache
    # supaya tidak dibaca dan diubah ulang tiap pengulangan.
    cache = {}

    for split, kelas, sumber, nama, asal in tugas:
        tujuan = dst / split / kelas / nama
        try:
            if asal == 'fer2013' or not grayscale:
                shutil.copy2(sumber, tujuan)
            else:
                if sumber not in cache:
                    img = cv2.imread(str(sumber))
                    if img is None:
                        gagal.append(sumber)
                        continue
                    abu = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # Ditulis 3 kanal supaya formatnya seragam dengan FER2013.
                    cache[sumber] = cv2.cvtColor(abu, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(str(tujuan), cache[sumber],
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
        except OSError as e:
            gagal.append(f"{sumber}: {e}")
            continue

        selesai += 1
        if selesai % 1000 == 0 or selesai == total:
            print(f"  {selesai}/{total} berkas", end='\r', flush=True)
    print()

    if gagal:
        print(f"[peringatan] {len(gagal)} berkas gagal disalin:")
        for g in gagal[:10]:
            print(f"           {g}")
    return selesai


def cetak_tabel(hitung):
    w = max(9, max(len(k) for k in EMOTIONS))
    for split in SPLITS:
        total_fer = sum(hitung[split][k]['fer2013'] for k in EMOTIONS)
        total_kdef = sum(hitung[split][k]['kdef'] for k in EMOTIONS)
        if not (total_fer + total_kdef):
            continue
        print()
        print(f"--- {split} ---")
        print(f"{'kelas':<{w}}  {'fer2013':>9}  {'kdef':>9}  {'TOTAL':>9}  {'%kdef':>7}")
        print("-" * (w + 42))
        for kelas in EMOTIONS:
            f = hitung[split][kelas]['fer2013']
            k = hitung[split][kelas]['kdef']
            pct = 100.0 * k / (f + k) if (f + k) else 0.0
            print(f"{kelas:<{w}}  {f:>9}  {k:>9}  {f + k:>9}  {pct:>6.1f}%")
        print("-" * (w + 42))
        pct = 100.0 * total_kdef / (total_fer + total_kdef)
        print(f"{'TOTAL':<{w}}  {total_fer:>9}  {total_kdef:>9}  "
              f"{total_fer + total_kdef:>9}  {pct:>6.1f}%")


def pilih_device(pilihan):
    """'auto' -> pakai GPU kalau ada. Skrip lama di repo ini mengunci 'cpu',
    padahal baseline FER2013 butuh 2.4 jam di CPU untuk 20 epoch."""
    if pilihan != 'auto':
        return pilihan
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[device ] GPU terdeteksi: {torch.cuda.get_device_name(0)}")
            return '0'
    except ImportError:
        pass
    print("[device ] tidak ada GPU, pakai CPU (jauh lebih lambat).")
    return 'cpu'


def latih(dst, args):
    from ultralytics import YOLO

    bobot = PROJECT_ROOT / "webcam" / "models" / "yolov8n-cls.pt"
    model = YOLO(str(bobot) if bobot.exists() else "yolov8n-cls.pt")
    hasil = model.train(
        data=str(dst),
        epochs=args.epochs,
        imgsz=224,
        batch=args.batch,
        patience=10,
        device=pilih_device(args.device),
        project='runs/emotion',
        name='merged_baseline',
        workers=0,
    )
    print("\nTraining selesai!")
    print(f"Best model: {hasil.save_dir}/weights/best.pt")
    print("\nUltralytics menulis ke runs/ yang ada di .gitignore. Salin model")
    print("yang mau dipakai ke webcam/models/ supaya ikut tersimpan di git:")
    print(f"  cp -r {hasil.save_dir} webcam/models/merged_baseline")


def main():
    p = argparse.ArgumentParser(
        description="Gabungkan FER2013 dan KDEF jadi satu dataset latih.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Contoh:\n"
               "  python webcam/scripts/merge_datasets.py --dry_run\n"
               "  python webcam/scripts/merge_datasets.py --kdef_repeat 6 --cap 3000")
    p.add_argument('--dst', default=str(DST_DEFAULT), help="Folder tujuan.")
    p.add_argument('--kdef_repeat', '--kdef-repeat', dest='kdef_repeat',
                   type=int, default=4, metavar='N',
                   help="Berapa kali KDEF diulang di split train (default: 4). "
                        "1 = tanpa pengulangan.")
    p.add_argument('--cap', type=int, default=0, metavar='N',
                   help="Batas maksimal gambar FER2013 per kelas di train, "
                        "untuk meredam ketimpangan kelas (default: 0 = tanpa batas).")
    p.add_argument('--color', action='store_true',
                   help="Pertahankan warna KDEF. Tidak disarankan: warna jadi "
                        "petunjuk asal dataset yang bisa dijadikan jalan pintas "
                        "oleh model.")
    p.add_argument('--seed', type=int, default=42,
                   help="Seed untuk pemilihan sampel saat --cap (default: 42).")
    p.add_argument('--force', action='store_true',
                   help="Timpa folder tujuan kalau sudah ada.")
    p.add_argument('--dry_run', '--dry-run', dest='dry_run', action='store_true',
                   help="Tampilkan rencana tanpa menyalin berkas.")
    p.add_argument('--train', action='store_true',
                   help="Langsung latih YOLO setelah dataset tergabung.")
    p.add_argument('--epochs', type=int, default=20,
                   help="Jumlah epoch kalau --train dipakai (default: 20).")
    p.add_argument('--device', default='auto',
                   help="Perangkat latih: 'auto' (GPU kalau ada), 'cpu', "
                        "atau indeks GPU seperti '0' (default: auto).")
    p.add_argument('--batch', type=int, default=32,
                   help="Ukuran batch (default: 32; naikkan kalau pakai GPU).")
    args = p.parse_args()

    periksa_sumber()
    dst = Path(args.dst)

    print(f"[sumber ] {FER_DIR}")
    print(f"[sumber ] {KDEF_DIR}")
    print(f"[tujuan ] {dst}")
    print(f"[opsi   ] kdef_repeat={args.kdef_repeat}, cap={args.cap or 'tidak ada'}, "
          f"warna={'ya' if args.color else 'tidak (grayscale)'}")

    if args.color:
        print("[warning] --color aktif: KDEF tetap berwarna sementara FER2013 "
              "grayscale.\n           Model bisa menebak asal dataset dari "
              "warnanya dan belajar dua aturan terpisah.")

    tugas, hitung = rencana(args)
    cetak_tabel(hitung)

    if args.dry_run:
        print(f"\n[dry-run] {len(tugas)} berkas akan ditulis. Tidak ada yang disalin.")
        return

    if dst.exists():
        if not args.force:
            raise SystemExit(f"\nERROR: {dst} sudah ada. Jalankan dengan --force "
                             f"untuk menimpanya.")
        print(f"\n[hapus  ] membersihkan {dst}")
        shutil.rmtree(dst)

    for split in SPLITS:
        for kelas in EMOTIONS:
            (dst / split / kelas).mkdir(parents=True, exist_ok=True)

    print()
    ditulis = salin(tugas, dst, grayscale=not args.color)
    print(f"\n[selesai] {ditulis} berkas ditulis ke {dst}")

    if args.train:
        latih(dst, args)
    else:
        print("\nLatih dengan:")
        print("  python webcam/scripts/merge_datasets.py --train --epochs 20")
        print("\nSetelah selesai, ukur di kedua test set secara terpisah:")
        print("  python results_calculation.py face --no_cache \\")
        print("      --visual_model webcam/models/merged_baseline/weights/best.pt")


if __name__ == '__main__':
    sys.exit(main())
