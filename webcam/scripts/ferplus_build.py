"""
ferplus_build.py - bangun dataset wajah FER+ dari FER2013 + label FERPlus.

APA ITU FERPlus
---------------
FERPlus BUKAN kumpulan gambar baru. Isinya cuma label ULANG untuk gambar
FER2013 yang sudah ada. Tiap gambar FER2013 dinilai 10 orang, dan hasilnya
disimpan di fer2013new.csv sebagai jumlah suara per emosi. Label FER2013 asli
cuma dari satu anotator dan terkenal berisik; FER+ memperbaiki itu.

Karena yang disimpan cuma label, berkas gambarnya harus diambil dari FER2013.
Dua berkas ini dicocokkan BERDASARKAN URUTAN BARIS - baris ke-N di
fer2013new.csv adalah label untuk baris ke-N di fer2013.csv. Keduanya sama-sama
35887 baris; skrip ini berhenti kalau ternyata tidak sama.

KENAPA GAMBARNYA DIBUAT ULANG DARI CSV
--------------------------------------
webcam/datasets/fer2013/ sudah berisi JPG, tapi nama berkasnya
(Training_10118481.jpg) adalah ID acak dari unggahan Kaggle, bukan nomor baris.
Jadi tidak ada cara memasangkannya dengan fer2013new.csv. Piksel aslinya ada di
fer2013.csv, jadi lebih aman menggambar ulang dari sana: nomor barisnya jelas,
dan namanya mengikuti penamaan FERPlus sendiri (fer0000000.png) supaya bisa
ditelusuri balik.

CARA LABELNYA DITENTUKAN
------------------------
Suara terbanyak menang. Yang dibuang:
    - 'unknown' atau 'NF' (not a face) menang   -> bukan wajah / tidak jelas
    - 'contempt' menang                          -> proyek ini cuma 7 kelas
    - seri di posisi teratas                     -> anotatornya sendiri tidak sepakat
    - tidak ada suara sama sekali

PERINGATAN SOAL KELAS DISGUST
-----------------------------
FER+ membuat disgust jauh LEBIH SEDIKIT, bukan lebih banyak: 436 gambar latih
di FER2013 asli menyusut jadi 175. Sebagian besar yang dulu dilabeli disgust
ternyata dinilai emosi lain oleh 10 anotator. Jadi FER+ menukar label yang
lebih bersih dengan kelas disgust yang makin langka. Itu salah satu alasan
ExpW ditambahkan - lihat webcam/scripts/expw_build.py.

PEMBAGIAN SPLIT
---------------
Memakai kolom Usage bawaan FER2013, bukan pengacakan sendiri:
    Training    -> train
    PublicTest  -> val
    PrivateTest -> test
Ini pembagian resmi yang dipakai makalah-makalah FER+, jadi angkanya bisa
dibandingkan dengan literatur.

Contoh:
    python webcam/scripts/ferplus_build.py --dry_run
    python webcam/scripts/ferplus_build.py
    python webcam/scripts/ferplus_build.py --keep_ties
"""
import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from common import EMOTIONS

FER_CSV = PROJECT_ROOT / "webcam" / "datasets" / "fer2013" / "fer2013.csv"
FERPLUS_CSV = PROJECT_ROOT / "webcam" / "datasets" / "FERPlus" / "fer2013new.csv"
DST_DEFAULT = PROJECT_ROOT / "webcam" / "datasets" / "fer2013plus"

# Urutan kolom suara di fer2013new.csv, mulai kolom ke-3.
KOLOM_FERPLUS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger',
                 'disgust', 'fear', 'contempt', 'unknown', 'NF']

# Nama FERPlus -> nama kelas proyek ini. contempt/unknown/NF sengaja tidak ada
# di sini; gambar yang menang di salah satu itu dibuang.
KE_KELAS = {
    'neutral': 'neutral', 'happiness': 'happy', 'surprise': 'surprise',
    'sadness': 'sad', 'anger': 'angry', 'disgust': 'disgust', 'fear': 'fear',
}

USAGE_KE_SPLIT = {'Training': 'train', 'PublicTest': 'val',
                  'PrivateTest': 'test'}
SPLITS = ['train', 'val', 'test']


def baca_label(path, keep_ties):
    """Kembalikan (daftar label per baris, statistik pembuangan).

    Label bernilai None kalau barisnya dibuang.
    """
    with open(path, newline='', encoding='utf-8') as f:
        baris = list(csv.reader(f))
    if not baris:
        raise SystemExit(f"ERROR: {path} kosong")

    hasil = []
    stat = Counter()
    for r in baris[1:]:
        suara = [int(v) for v in r[2:2 + len(KOLOM_FERPLUS)]]
        total = sum(suara)
        if total == 0:
            stat['tanpa suara'] += 1
            hasil.append(None)
            continue

        puncak = max(suara)
        menang = [KOLOM_FERPLUS[i] for i, v in enumerate(suara) if v == puncak]
        if len(menang) > 1 and not keep_ties:
            stat['seri di posisi teratas'] += 1
            hasil.append(None)
            continue

        nama = menang[0]
        if nama not in KE_KELAS:
            stat[f"menang '{nama}'"] += 1
            hasil.append(None)
            continue

        stat['dipakai'] += 1
        hasil.append(KE_KELAS[nama])
    return hasil, stat


def baca_piksel(path):
    """Kembalikan (daftar (usage, array 48x48 uint8))."""
    with open(path, newline='', encoding='utf-8') as f:
        pembaca = csv.DictReader(f)
        data = []
        for i, r in enumerate(pembaca):
            piksel = np.array(r['pixels'].split(), dtype=np.uint8)
            if piksel.size != 48 * 48:
                raise SystemExit(f"ERROR: baris {i} di {path} punya "
                                 f"{piksel.size} piksel, harusnya {48 * 48}")
            data.append((r['Usage'], piksel.reshape(48, 48)))
    return data


def main():
    p = argparse.ArgumentParser(
        description="Bangun dataset FER+ dari fer2013.csv + fer2013new.csv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Keluarannya <split>/<kelas>/fer<nomor baris>.png, 48x48 "
               "grayscale.\nNomor barisnya sengaja dipertahankan supaya tiap "
               "gambar bisa\nditelusuri balik ke baris fer2013new.csv-nya.")
    p.add_argument('--fer_csv', default=str(FER_CSV),
                   help="fer2013.csv, sumber pikselnya.")
    p.add_argument('--ferplus_csv', default=str(FERPLUS_CSV),
                   help="fer2013new.csv, sumber labelnya.")
    p.add_argument('--dst', default=str(DST_DEFAULT), help="Folder tujuan.")
    p.add_argument('--keep_ties', '--keep-ties', dest='keep_ties',
                   action='store_true',
                   help="Simpan gambar yang suaranya seri di posisi teratas, "
                        "diambil kelas pertama menurut urutan kolom FERPlus. "
                        "Bawaannya dibuang - kalau 10 anotator saja tidak "
                        "sepakat, labelnya memang tidak bisa dipercaya.")
    p.add_argument('--force', action='store_true',
                   help="Timpa folder tujuan kalau sudah ada.")
    p.add_argument('--dry_run', '--dry-run', dest='dry_run',
                   action='store_true',
                   help="Cuma hitung dan tampilkan tabelnya, tidak menulis "
                        "berkas apa pun.")
    args = p.parse_args()

    fer_csv = Path(args.fer_csv)
    fp_csv = Path(args.ferplus_csv)
    dst = Path(args.dst)
    for path, nama in [(fer_csv, 'fer2013.csv'), (fp_csv, 'fer2013new.csv')]:
        if not path.is_file():
            raise SystemExit(f"ERROR: {nama} tidak ada di {path}")

    # Pagar pengaman. Windows tidak membedakan huruf besar-kecil di nama
    # folder, jadi 'webcam/datasets/ferplus' menunjuk ke folder yang SAMA
    # dengan 'webcam/datasets/FERPlus' - repo FERPlus tempat fer2013new.csv
    # berada. Tanpa cek ini, --force akan menghapus berkas sumbernya sendiri.
    # Karena itu juga folder keluarannya dinamai fer2013plus, bukan ferplus.
    dst_nyata = dst.resolve()
    for sumber in (fp_csv, fer_csv):
        if sumber.resolve().is_relative_to(dst_nyata):
            raise SystemExit(
                f"ERROR: folder tujuan {dst} berisi berkas sumber "
                f"{sumber.name}.\n"
                f"       Menulis ke situ - apalagi dengan --force - akan "
                f"menghapus sumbernya.\n"
                f"       Pakai --dst lain, misalnya "
                f"webcam/datasets/fer2013plus.")

    print(f"[baca   ] label  : {fp_csv}")
    label, stat = baca_label(fp_csv, args.keep_ties)
    print(f"[baca   ] piksel : {fer_csv}")
    piksel = baca_piksel(fer_csv)

    if len(label) != len(piksel):
        raise SystemExit(
            f"ERROR: jumlah barisnya beda - fer2013new.csv {len(label)} baris, "
            f"fer2013.csv {len(piksel)} baris.\n"
            f"       Keduanya dipasangkan menurut urutan baris, jadi ini harus "
            f"sama persis.\n"
            f"       Kemungkinan salah satu berkasnya versi lain.")
    print(f"[cek    ] {len(label)} baris, urutannya cocok")

    # Hitung dulu, tulis belakangan - supaya --dry_run bisa menampilkan tabel
    # yang persis sama dengan yang akan ditulis.
    tugas = []
    hitung = {s: Counter() for s in SPLITS}
    for i, (kelas, (usage, img)) in enumerate(zip(label, piksel)):
        if kelas is None:
            continue
        split = USAGE_KE_SPLIT.get(usage)
        if split is None:
            stat[f"usage tak dikenal '{usage}'"] += 1
            continue
        tugas.append((split, kelas, i, img))
        hitung[split][kelas] += 1

    print("\n=== Baris yang dibuang ===")
    for k, v in sorted(stat.items(), key=lambda kv: -kv[1]):
        if k != 'dipakai':
            print(f"  {k:28s}: {v:6d}")
    print(f"  {'dipakai':28s}: {stat['dipakai']:6d}")

    print("\n=== Jumlah per kelas ===")
    print(f"{'kelas':10s} {'train':>8s} {'val':>8s} {'test':>8s} {'total':>8s}")
    for kelas in EMOTIONS:
        n = [hitung[s][kelas] for s in SPLITS]
        print(f"{kelas:10s} {n[0]:8d} {n[1]:8d} {n[2]:8d} {sum(n):8d}")
    tot = [sum(hitung[s].values()) for s in SPLITS]
    print(f"{'TOTAL':10s} {tot[0]:8d} {tot[1]:8d} {tot[2]:8d} {sum(tot):8d}")

    kecil = min((hitung['train'][k], k) for k in EMOTIONS)
    print(f"\nKelas train paling sedikit: {kecil[1]} ({kecil[0]} gambar). "
          f"Ketimpangan {max(hitung['train'].values()) / max(kecil[0], 1):.1f}x.")

    if args.dry_run:
        print("\n--dry_run: tidak ada berkas yang ditulis.")
        return

    if dst.exists():
        if not args.force:
            raise SystemExit(f"ERROR: {dst} sudah ada. Pakai --force untuk "
                             f"menimpanya.")
        shutil.rmtree(dst)
        print(f"\n[bersih ] {dst} dihapus")

    for s in SPLITS:
        for kelas in EMOTIONS:
            (dst / s / kelas).mkdir(parents=True, exist_ok=True)

    print(f"[tulis  ] {len(tugas)} gambar ke {dst}")
    for n, (split, kelas, i, img) in enumerate(tugas, 1):
        cv2.imwrite(str(dst / split / kelas / f"fer{i:07d}.png"), img)
        if n % 5000 == 0:
            print(f"          {n}/{len(tugas)}")

    print("\nSelesai. Lanjutkan dengan salah satu:")
    print("  python webcam/scripts/merge_datasets.py --dry_run")
    print(f"  python webcam/scripts/training_yolo_.py --yes --data {dst} "
          f"--name fer2013plus_baseline")


if __name__ == '__main__':
    main()
