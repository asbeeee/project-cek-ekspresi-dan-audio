"""
ckplus_build.py - siapkan CK+ sebagai dataset UJI yang bersih.

KENAPA CK+ DIPAKAI SEBAGAI TEST SET, BUKAN DATA LATIH
-----------------------------------------------------
Isinya cuma 327 urutan (981 gambar). Dibanding webcam/datasets/combined yang
punya 60950 gambar latih, menambahkannya ke data latih praktis tidak terasa
kecuali diulang banyak - dan mengulang 327 urutan yang sama berkali-kali lebih
mungkin bikin hafal daripada bikin pintar.

Nilainya justru sebagai DOMAIN UJI KETIGA yang belum pernah dilihat model mana
pun: wajah studio berkualitas tinggi, ekspresi disengaja, label dari FACS
coding - jadi labelnya jauh lebih bisa dipercaya daripada ExpW. Persis peran
yang dipegang KDEF, tapi dari sumber yang berbeda.

Gunanya konkret: membandingkan dua model yang dilatih dengan dan tanpa ExpW
pada data yang sama sekali tidak dipakai keduanya.

DUA VERSI CK+ ADA DI FOLDER INI, YANG DIPAKAI CK+48
---------------------------------------------------
    CK+48/           981 PNG 48x48, 7 folder emosi, TANPA neutral
    ckextended.csv   920 baris 48x48, ADA neutral (593), tanpa ID subjek

Yang dipakai CK+48. Alasannya satu dan menentukan: nama berkasnya memuat ID
SUBJEK (S010_004_00000017.png -> subjek S010, urutan 004, frame 17), sementara
CSV-nya tidak memuat apa-apa soal subjek.

Tanpa ID subjek, pembagian train/test cuma bisa diacak per gambar - dan itu
persis kesalahan yang bikin angka surprise di modalitas audio kelihatan 0.8990
padahal aslinya 0.3958 (lihat DATASETS.txt bagian B6). Di CK+ akibatnya lebih
parah lagi: tiap urutan diwakili TEPAT 3 FRAME BERURUTAN yang nyaris identik.
Kalau diacak per gambar, frame 17 bisa di train sementara frame 18 dari urutan
yang sama ada di test. Itu bukan pengujian, itu mencocokkan gambar yang sama.

Karena itu pembagiannya di sini selalu per SUBJEK (118 subjek tersedia), tidak
pernah per gambar.

Harga yang dibayar: CK+48 tidak punya kelas neutral sama sekali. Jadi hasil
ujinya cuma mencakup 6 kelas. Kalau neutral wajib ada, ambil dari
ckextended.csv - tapi sadari baris-baris itu tidak punya ID subjek, jadi
neutral-nya tidak bisa dijamin bebas dari kebocoran subjek.

Kelas contempt dibuang: proyek ini cuma punya 7 kelas dan contempt bukan salah
satunya, sama seperti perlakuan di ferplus_build.py.

Contoh:
    python webcam/scripts/ckplus_build.py --dry_run
    python webcam/scripts/ckplus_build.py                  # semua jadi test/
    python webcam/scripts/ckplus_build.py --mode split     # 70/15/15 per subjek
"""
import argparse
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from common import EMOTIONS

SRC_DEFAULT = PROJECT_ROOT / "webcam" / "datasets" / "ckplus_raw" / "CK+48"
DST_DEFAULT = PROJECT_ROOT / "webcam" / "datasets" / "ckplus"

# Nama folder CK+48 -> kelas proyek ini. contempt sengaja tidak ada.
KE_KELAS = {
    'anger': 'angry', 'disgust': 'disgust', 'fear': 'fear',
    'happy': 'happy', 'sadness': 'sad', 'surprise': 'surprise',
}

SPLITS = ['train', 'val', 'test']
RASIO = {'train': 0.70, 'val': 0.15, 'test': 0.15}


def baca(src):
    """Kembalikan (daftar (path, kelas, subjek, urutan), statistik)."""
    hasil = []
    stat = Counter()
    for folder in sorted(src.iterdir()):
        if not folder.is_dir():
            continue
        kelas = KE_KELAS.get(folder.name.lower())
        if kelas is None:
            stat[f"folder '{folder.name}' dilewati"] += len(
                list(folder.glob('*.png')))
            continue
        for f in sorted(folder.glob('*.png')):
            bagian = f.stem.split('_')
            if len(bagian) < 3:
                stat['nama berkas tak terbaca'] += 1
                continue
            hasil.append((f, kelas, bagian[0], f"{bagian[0]}_{bagian[1]}"))
            stat['dipakai'] += 1
    return hasil, stat


def bagi_per_subjek(data, seed):
    """Bagi 70/15/15 per SUBJEK. Kembalikan dict subjek -> split."""
    subjek = sorted({s for _, _, s, _ in data})
    random.Random(seed).shuffle(subjek)
    n = len(subjek)
    n_val = max(1, round(n * RASIO['val']))
    n_test = max(1, round(n * RASIO['test']))
    n_train = n - n_val - n_test

    tugas = {}
    for s in subjek[:n_train]:
        tugas[s] = 'train'
    for s in subjek[n_train:n_train + n_val]:
        tugas[s] = 'val'
    for s in subjek[n_train + n_val:]:
        tugas[s] = 'test'
    return tugas


def main():
    p = argparse.ArgumentParser(
        description="Siapkan CK+ sebagai domain uji bersih.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Pembagian SELALU per subjek, tidak pernah per gambar - tiap "
               "urutan\npunya 3 frame berurutan yang nyaris identik.")
    p.add_argument('--src', default=str(SRC_DEFAULT),
                   help="Folder CK+48 berisi subfolder per emosi.")
    p.add_argument('--dst', default=str(DST_DEFAULT), help="Folder tujuan.")
    p.add_argument('--mode', choices=['test', 'split'], default='test',
                   help="'test' (bawaan): semua masuk test/, dipakai sebagai "
                        "domain uji murni. 'split': 70/15/15 per subjek, kalau "
                        "memang mau ikut dilatih.")
    p.add_argument('--imgsz', type=int, default=0, metavar='N',
                   help="Ubah ukuran ke NxN. 0 = biarkan 48x48 apa adanya "
                        "(default), sama seperti FER+.")
    p.add_argument('--seed', type=int, default=42,
                   help="Seed pengacakan subjek (default: 42).")
    p.add_argument('--force', action='store_true',
                   help="Timpa folder tujuan kalau sudah ada.")
    p.add_argument('--dry_run', '--dry-run', dest='dry_run',
                   action='store_true',
                   help="Cuma tampilkan tabelnya, tidak menulis apa pun.")
    args = p.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        raise SystemExit(
            f"ERROR: {src} tidak ada.\n"
            f"       Folder CK+48 diharapkan berisi subfolder anger/, "
            f"disgust/, dst.")

    data, stat = baca(src)
    if not data:
        raise SystemExit(f"ERROR: tidak ada PNG yang terbaca di {src}")

    urutan = {u for _, _, _, u in data}
    subjek = {s for _, _, s, _ in data}
    print(f"[baca   ] {stat['dipakai']} gambar, {len(urutan)} urutan, "
          f"{len(subjek)} subjek")
    for k, v in sorted(stat.items()):
        if k != 'dipakai':
            print(f"          {k}: {v}")

    if args.mode == 'test':
        tugas = {s: 'test' for s in subjek}
    else:
        tugas = bagi_per_subjek(data, args.seed)

    hitung = {s: Counter() for s in SPLITS}
    subj_split = defaultdict(set)
    for _, kelas, s, _ in data:
        hitung[tugas[s]][kelas] += 1
        subj_split[tugas[s]].add(s)

    print(f"\n=== Jumlah gambar per kelas (mode: {args.mode}) ===")
    print(f"{'kelas':10s} {'train':>7s} {'val':>6s} {'test':>6s} {'total':>7s}")
    for kelas in EMOTIONS:
        n = [hitung[s][kelas] for s in SPLITS]
        tanda = "   <- tidak ada di CK+48" if sum(n) == 0 else ""
        print(f"{kelas:10s} {n[0]:7d} {n[1]:6d} {n[2]:6d} {sum(n):7d}{tanda}")
    tot = [sum(hitung[s].values()) for s in SPLITS]
    print(f"{'TOTAL':10s} {tot[0]:7d} {tot[1]:6d} {tot[2]:6d} {sum(tot):7d}")
    print("Subjek per split: " + ", ".join(
        f"{s} {len(subj_split[s])}" for s in SPLITS if subj_split[s]))

    if args.mode == 'split':
        for a in SPLITS:
            for b in SPLITS:
                if a < b and subj_split[a] & subj_split[b]:
                    raise SystemExit(
                        f"ERROR: subjek bocor antara {a} dan {b}. "
                        f"Ini bug, laporkan.")
        print("Irisan subjek antar split: 0 (sudah dicek)")

    if args.dry_run:
        print("\n--dry_run: tidak ada berkas yang ditulis.")
        return

    if dst.exists():
        if not args.force:
            raise SystemExit(f"\nERROR: {dst} sudah ada. Pakai --force untuk "
                             f"menimpanya.")
        shutil.rmtree(dst)
        print(f"\n[bersih ] {dst} dihapus")

    ditulis = 0
    for f, kelas, s, _ in data:
        folder = dst / tugas[s] / kelas
        folder.mkdir(parents=True, exist_ok=True)
        tujuan = folder / f"ckplus_{f.stem}.png"
        if args.imgsz:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            interp = (cv2.INTER_AREA if img.shape[0] > args.imgsz
                      else cv2.INTER_CUBIC)
            cv2.imwrite(str(tujuan),
                        cv2.resize(img, (args.imgsz, args.imgsz),
                                   interpolation=interp))
        else:
            shutil.copy(f, tujuan)
        ditulis += 1

    print(f"\n[selesai] {ditulis} gambar ditulis ke {dst}")
    print("\nUkur model di domain ini dengan:")
    print(f"  python results_calculation.py face --data {dst} "
          f"--split test --tag ckplus --no_cache")


if __name__ == '__main__':
    main()
