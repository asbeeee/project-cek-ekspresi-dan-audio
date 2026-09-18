"""
merge_datasets.py - gabungkan beberapa dataset wajah jadi satu dataset latih.

Menghasilkan webcam/datasets/combined/<split>/<kelas>/ yang berisi semua
sumbernya, siap dipakai YOLO klasifikasi.

SUMBER YANG TERSEDIA
--------------------
    fer2013plus  33500 gambar  48x48 abu-abu   label FER+ (10 anotator)
    expw         37254 gambar  224x224 abu-abu wajah dipotong dari foto liar
    kdef          2936 gambar  224x224 warna   studio, berpose
    fer2013      35887 gambar  48x48 abu-abu   label asli, 1 anotator

Bawaannya fer2013plus + expw + kdef. fer2013 masih terdaftar supaya baseline
lama bisa dibuat ulang, TAPI jangan dipakai bersama fer2013plus: keduanya
gambar yang persis sama dengan label berbeda, jadi menggabungnya berarti
memberi model dua jawaban yang bertabrakan untuk satu wajah. Skrip ini
menolak kombinasi itu.

DUA MASALAH YANG DITANGANI
--------------------------
1. Jumlahnya timpang jauh. KDEF cuma 2013 gambar latih lawan puluhan ribu
   dari sumber lain, jadi porsinya tenggelam. Karena itu KDEF diulang
   beberapa kali (--repeat kdef=4, bawaannya 4). Ultralytics mengaugmentasi
   ulang tiap epoch, jadi salinan yang sama tidak dilihat model secara
   identik.

2. Domainnya beda jauh. KDEF 224x224 berwarna dan berpose; FER+ 48x48 abu-abu
   hasil jepretan liar. Kalau warnanya dibiarkan, model bisa ambil jalan
   pintas: "berwarna berarti KDEF, abu-abu berarti yang lain", lalu belajar
   beberapa aturan terpisah alih-alih satu yang umum. Karena itu semuanya
   dijadikan abu-abu (--color untuk membatalkan). Beda resolusi masih ada;
   --imgsz bisa menyeragamkannya kalau memang mau.

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
    python webcam/scripts/merge_datasets.py --sources fer2013plus,kdef
    python webcam/scripts/merge_datasets.py --repeat kdef=6 --cap 5000
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

DATASETS = PROJECT_ROOT / "webcam" / "datasets"
DST_DEFAULT = DATASETS / "combined"

# Sumber yang dikenal. 'ulang' adalah pengulangan bawaan di split train.
SUMBER = {
    'fer2013plus': {'dir': DATASETS / "fer2013plus", 'ulang': 1},
    'expw': {'dir': DATASETS / "expw", 'ulang': 1},
    'kdef': {'dir': DATASETS / "kdef", 'ulang': 4},
    'fer2013': {'dir': DATASETS / "fer2013", 'ulang': 1},
}
SUMBER_BAWAAN = ['fer2013plus', 'expw', 'kdef']

# Pasangan yang tidak boleh dipakai bersama: gambarnya sama, labelnya beda.
BENTROK = [('fer2013', 'fer2013plus')]

CARA_BUAT = {
    'fer2013plus': "python webcam/scripts/ferplus_build.py",
    'expw': "python webcam/scripts/expw_build.py",
    'kdef': "python webcam/scripts/kdefTrain.py",
    'fer2013': "(unduh FER2013 dari Kaggle)",
}

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


def periksa_sumber(nama_sumber):
    for nama in nama_sumber:
        root = SUMBER[nama]['dir']
        if not root.is_dir():
            raise SystemExit(
                f"ERROR: dataset '{nama}' tidak ada di {root}\n"
                f"       Buat dulu dengan: {CARA_BUAT[nama]}")

    for a, b in BENTROK:
        if a in nama_sumber and b in nama_sumber:
            raise SystemExit(
                f"ERROR: '{a}' dan '{b}' tidak boleh dipakai bersama.\n"
                f"       Keduanya gambar yang SAMA dengan label berbeda, jadi "
                f"model akan\n"
                f"       menerima dua jawaban yang bertabrakan untuk satu "
                f"wajah.\n"
                f"       Pilih salah satu di --sources.")


def sudah_abu_abu(root, contoh=20):
    """Tebak apakah sumbernya sudah grayscale, dengan mencicipi beberapa berkas.

    Gunanya supaya sumber yang memang sudah abu-abu (FER+, ExpW) bisa disalin
    apa adanya tanpa dibaca-tulis ulang - jauh lebih cepat untuk puluhan ribu
    berkas. Kalau ada satu saja yang berwarna, seluruh sumbernya diperlakukan
    sebagai berwarna, jadi tebakan yang meleset bikin lambat, bukan bikin salah.
    """
    dilihat = 0
    for split in SPLITS:
        for kelas in EMOTIONS:
            folder = root / split / kelas
            if not folder.is_dir():
                continue
            for f in sorted(folder.iterdir())[:3]:
                if f.suffix.lower() not in IMG_EXT:
                    continue
                img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                if img.ndim == 3 and img.shape[2] >= 3:
                    return False
                dilihat += 1
                if dilihat >= contoh:
                    return True
    return dilihat > 0


def rencana(args, nama_sumber, ulang):
    """Susun daftar tugas salin: (split, kelas, sumber, nama_tujuan, asal)."""
    rng = random.Random(args.seed)
    tugas = []
    hitung = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for split in SPLITS:
        for nama in nama_sumber:
            berkas = kumpulkan(SUMBER[nama]['dir'], split)

            for kelas in EMOTIONS:
                daftar = list(berkas[kelas])

                # Batasi kelas mayoritas supaya tidak terlalu timpang. Hanya
                # di train; val dan test dibiarkan utuh supaya metriknya tetap
                # mencerminkan sebaran aslinya.
                if split == 'train' and args.cap and len(daftar) > args.cap:
                    rng.shuffle(daftar)
                    daftar = daftar[:args.cap]

                # Pengulangan hanya di train. Mengulang di val/test cuma
                # membuat bobot metrik menyimpang tanpa menambah informasi.
                n_ulang = ulang[nama] if split == 'train' else 1
                for f in daftar:
                    for i in range(n_ulang):
                        akhiran = "" if i == 0 else f"_r{i + 1}"
                        tugas.append((split, kelas, f,
                                      f"{nama}_{f.stem}{akhiran}.jpg", nama))
                        hitung[split][kelas][nama] += 1

    return tugas, hitung


def salin(tugas, dst, grayscale, imgsz, abu_asli):
    """Jalankan daftar tugas.

    Sumber yang sudah abu-abu disalin apa adanya - kecuali kalau --imgsz minta
    diubah ukurannya.
    """
    total = len(tugas)
    selesai = 0
    gagal = []
    # Satu berkas KDEF bisa dipakai berkali-kali; hasil konversinya di-cache
    # supaya tidak dibaca dan diubah ulang tiap pengulangan.
    cache = {}

    for split, kelas, sumber, nama, asal in tugas:
        tujuan = dst / split / kelas / nama
        try:
            apa_adanya = (abu_asli.get(asal, False) or not grayscale) and not imgsz
            if apa_adanya:
                shutil.copy2(sumber, tujuan)
            else:
                if sumber not in cache:
                    img = cv2.imread(str(sumber))
                    if img is None:
                        gagal.append(sumber)
                        continue
                    if grayscale:
                        abu = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        # Ditulis 3 kanal supaya formatnya seragam.
                        img = cv2.cvtColor(abu, cv2.COLOR_GRAY2BGR)
                    if imgsz and img.shape[0] != imgsz:
                        interp = (cv2.INTER_AREA if img.shape[0] > imgsz
                                  else cv2.INTER_CUBIC)
                        img = cv2.resize(img, (imgsz, imgsz),
                                         interpolation=interp)
                    cache[sumber] = img
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


def cetak_tabel(hitung, nama_sumber):
    w = max(9, max(len(k) for k in EMOTIONS))
    lebar_kol = max(9, max(len(n) for n in nama_sumber))
    garis = "-" * (w + (lebar_kol + 2) * (len(nama_sumber) + 1) + 9)

    for split in SPLITS:
        total_split = sum(hitung[split][k][n]
                          for k in EMOTIONS for n in nama_sumber)
        if not total_split:
            continue
        print()
        print(f"--- {split} ---")
        judul = f"{'kelas':<{w}}"
        for n in nama_sumber:
            judul += f"  {n:>{lebar_kol}}"
        judul += f"  {'TOTAL':>{lebar_kol}}"
        print(judul)
        print(garis)

        for kelas in EMOTIONS:
            n_per = [hitung[split][kelas][n] for n in nama_sumber]
            baris = f"{kelas:<{w}}"
            for v in n_per:
                baris += f"  {v:>{lebar_kol}}"
            baris += f"  {sum(n_per):>{lebar_kol}}"
            print(baris)

        print(garis)
        total_per = [sum(hitung[split][k][n] for k in EMOTIONS)
                     for n in nama_sumber]
        baris = f"{'TOTAL':<{w}}"
        for v in total_per:
            baris += f"  {v:>{lebar_kol}}"
        baris += f"  {sum(total_per):>{lebar_kol}}"
        print(baris)

        # Porsi tiap sumber, angka yang paling sering ditanyakan waktu
        # menimbang --repeat.
        baris = f"{'%':<{w}}"
        for v in total_per:
            baris += f"  {100.0 * v / total_split:>{lebar_kol - 1}.1f}%"
        print(baris)

        # Ketimpangan antar kelas - alasan utama --cap ada.
        per_kelas = [sum(hitung[split][k][n] for n in nama_sumber)
                     for k in EMOTIONS]
        if min(per_kelas) > 0:
            i_min = per_kelas.index(min(per_kelas))
            i_max = per_kelas.index(max(per_kelas))
            print(f"    ketimpangan {max(per_kelas) / min(per_kelas):.1f}x  "
                  f"({EMOTIONS[i_max]} {max(per_kelas)} lawan "
                  f"{EMOTIONS[i_min]} {min(per_kelas)})")


def pilih_device(pilihan):
    """'auto' -> '0' kalau ada GPU. Nilainya gaya ultralytics, bukan torch."""
    if pilihan != 'auto':
        return pilihan
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[device] GPU terdeteksi: {torch.cuda.get_device_name(0)}")
            return 0
    except ImportError:
        pass
    print("[device] tidak ada GPU, pakai CPU (jauh lebih lambat).")
    return 'cpu'


def latih(dst, args):
    from ultralytics import YOLO

    device = pilih_device(args.device)
    # Harus sama dengan training_yolo_.py. Sebelumnya di sini yolo11n
    # sementara di sana yolov8n, jadi dua jalur latih menghasilkan arsitektur
    # berbeda dan hasilnya tidak bisa dibandingkan langsung.
    bobot_awal = PROJECT_ROOT / "webcam" / "models" / "yolov8n-cls.pt"
    model = YOLO(str(bobot_awal) if bobot_awal.exists() else 'yolov8n-cls.pt')
    hasil = model.train(
        data=str(dst),
        epochs=args.epochs,
        imgsz=224,
        batch=args.batch,
        device=device,
        project='runs/combined',
        name='gabungan',
    )
    print("\nTraining selesai.")
    print("Ultralytics menaruh hasilnya di runs/classify/runs/combined/gabungan/,")
    print("bukan persis di project= yang dioper. Salin best.pt ke")
    print("webcam/models/ kalau modelnya mau dipakai.")
    return hasil


def urai_repeat(daftar, nama_sumber):
    """Ubah ['kdef=6', 'expw=2'] jadi dict pengulangan per sumber."""
    ulang = {n: SUMBER[n]['ulang'] for n in nama_sumber}
    for item in daftar or []:
        if '=' not in item:
            raise SystemExit(f"ERROR: --repeat harus berbentuk nama=angka, "
                             f"dapatnya '{item}'")
        nama, _, nilai = item.partition('=')
        if nama not in SUMBER:
            raise SystemExit(f"ERROR: sumber '{nama}' di --repeat tidak "
                             f"dikenal. Pilihannya: {', '.join(SUMBER)}")
        if nama not in nama_sumber:
            raise SystemExit(f"ERROR: --repeat menyebut '{nama}' yang tidak "
                             f"ada di --sources.")
        try:
            n = int(nilai)
        except ValueError:
            raise SystemExit(f"ERROR: pengulangan '{nilai}' bukan angka.")
        if n < 1:
            raise SystemExit(f"ERROR: pengulangan harus >= 1, dapatnya {n}.")
        ulang[nama] = n
    return ulang


def main():
    p = argparse.ArgumentParser(
        description="Gabungkan beberapa dataset wajah jadi satu folder latih.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Sumber yang dikenal: " + ", ".join(SUMBER) + "\n"
               "fer2013 dan fer2013plus tidak boleh bersamaan - gambarnya "
               "sama, labelnya beda.")
    p.add_argument('--sources', '--sumber', dest='sources',
                   default=",".join(SUMBER_BAWAAN),
                   help="Daftar sumber dipisah koma (default: "
                        + ",".join(SUMBER_BAWAAN) + ").")
    p.add_argument('--dst', default=str(DST_DEFAULT), help="Folder tujuan.")
    p.add_argument('--repeat', action='append', metavar='NAMA=N',
                   help="Pengulangan satu sumber di split train, misalnya "
                        "'kdef=6'. Bisa diulang untuk beberapa sumber. "
                        "Bawaannya kdef=4, sisanya 1.")
    p.add_argument('--kdef_repeat', '--kdef-repeat', dest='kdef_repeat',
                   type=int, default=None, metavar='N',
                   help="Jalan pintas lama untuk --repeat kdef=N.")
    p.add_argument('--cap', type=int, default=0, metavar='N',
                   help="Batas gambar per kelas per sumber di split train. "
                        "0 = tanpa batas (default).")
    p.add_argument('--color', action='store_true',
                   help="Jangan ubah ke abu-abu. Lihat penjelasan masalah "
                        "nomor 2 di atas berkas ini sebelum memakainya.")
    p.add_argument('--imgsz', type=int, default=0, metavar='N',
                   help="Seragamkan ukuran semua gambar ke N x N. 0 = biarkan "
                        "apa adanya (default). Sumbernya campur 48x48 dan "
                        "224x224; ultralytics toh mengubah ukurannya saat "
                        "memuat, jadi ini cuma perlu kalau mau memastikan "
                        "resolusi tidak jadi petunjuk asal dataset.")
    p.add_argument('--seed', type=int, default=42,
                   help="Seed pengacakan untuk --cap (default: 42).")
    p.add_argument('--force', action='store_true',
                   help="Timpa folder tujuan kalau sudah ada.")
    p.add_argument('--dry_run', '--dry-run', dest='dry_run',
                   action='store_true',
                   help="Cuma tampilkan tabelnya, tidak menyalin apa pun.")
    p.add_argument('--train', action='store_true',
                   help="Langsung latih setelah penggabungan selesai.")
    p.add_argument('--epochs', type=int, default=20,
                   help="Epoch, kalau --train dipakai (default: 20).")
    p.add_argument('--device', default='auto',
                   help="'auto' (GPU kalau ada), 'cpu', atau '0'.")
    p.add_argument('--batch', type=int, default=32,
                   help="Batch, kalau --train dipakai (default: 32).")
    args = p.parse_args()

    nama_sumber = [s.strip() for s in args.sources.split(',') if s.strip()]
    if not nama_sumber:
        raise SystemExit("ERROR: --sources kosong.")
    for n in nama_sumber:
        if n not in SUMBER:
            raise SystemExit(f"ERROR: sumber '{n}' tidak dikenal. "
                             f"Pilihannya: {', '.join(SUMBER)}")
    if len(set(nama_sumber)) != len(nama_sumber):
        raise SystemExit("ERROR: ada sumber yang disebut dua kali di --sources.")

    periksa_sumber(nama_sumber)

    if args.kdef_repeat is not None:
        args.repeat = (args.repeat or []) + [f"kdef={args.kdef_repeat}"]
    ulang = urai_repeat(args.repeat, nama_sumber)

    dst = Path(args.dst)
    print(f"[sumber ] {', '.join(nama_sumber)}")
    print(f"[ulang  ] " + ", ".join(f"{n} x{ulang[n]}" for n in nama_sumber))
    print(f"[warna  ] {'dibiarkan' if args.color else 'diubah ke abu-abu'}"
          + (f", diseragamkan ke {args.imgsz}x{args.imgsz}" if args.imgsz
             else ""))

    tugas, hitung = rencana(args, nama_sumber, ulang)
    cetak_tabel(hitung, nama_sumber)
    print(f"\nTotal berkas yang akan ditulis: {len(tugas)}")

    if args.dry_run:
        print("\n--dry_run: tidak ada berkas yang disalin.")
        return

    if dst.exists():
        if not args.force:
            raise SystemExit(f"\nERROR: {dst} sudah ada. Pakai --force untuk "
                             f"menimpanya.")
        shutil.rmtree(dst)
        print(f"\n[bersih ] {dst} dihapus")
    for split in SPLITS:
        for kelas in EMOTIONS:
            (dst / split / kelas).mkdir(parents=True, exist_ok=True)

    abu_asli = {}
    if not args.color:
        for n in nama_sumber:
            abu_asli[n] = sudah_abu_abu(SUMBER[n]['dir'])
            if abu_asli[n]:
                print(f"[cek    ] {n} sudah abu-abu, disalin apa adanya")

    print(f"\n[salin  ] {len(tugas)} berkas ke {dst}")
    salin(tugas, dst, not args.color, args.imgsz, abu_asli)

    print(f"\nSelesai. Latih dengan:")
    print(f"  python webcam/scripts/training_yolo_.py --yes --data {dst} "
          f"--name gabungan")

    if args.train:
        latih(dst, args)


if __name__ == '__main__':
    main()
