"""
expw_build.py - potong wajah ExpW jadi dataset <split>/<kelas>/.

APA ITU ExpW
------------
Expression in-the-Wild: foto-foto hasil pencarian web, jadi wajahnya tidak
berpose, pencahayaannya sembarang, dan sering ada lebih dari satu orang dalam
satu foto. Gambarnya TIDAK terpotong - yang diberikan adalah foto utuh di
origin/ plus label.lst berisi kotak wajah dan emosinya:

    nama_berkas  id_wajah  atas  kiri  kanan  bawah  keyakinan  emosi

Jadi wajahnya harus dipotong sendiri, itu tugas skrip ini. Kode emosinya:
    0 angry   1 disgust   2 fear   3 happy   4 sad   5 surprise   6 neutral

TIGA HAL YANG PERLU DIWASPADAI
------------------------------
1. SATU FOTO BISA PUNYA BANYAK WAJAH - sampai 6 atau lebih. Kalau pembagian
   train/val/test diacak per WAJAH, dua wajah dari foto yang sama bisa jatuh
   di train dan di test sekaligus: latar, pencahayaan, bahkan orangnya sama.
   Itu kebocoran. Karena itu yang diacak di sini adalah FOTO-nya, lalu semua
   wajah dalam satu foto ikut ke split yang sama.

2. TIDAK SEMUA FOTO PUNYA LABEL. origin/ berisi 106962 berkas, tapi label.lst
   cuma menyebut 68096 di antaranya. Sisanya tidak terpakai - dilewati, bukan
   dianggap error.

3. DISTRIBUSINYA SANGAT TIMPANG. Dari 91793 wajah: neutral 34883 dan happy
   30537 (71% berdua), sementara fear cuma 1088. Kalau dimasukkan apa adanya,
   ExpW akan menenggelamkan dataset lain dan modelnya belajar menebak
   neutral/happy. Karena itu ada --cap.

CARA --cap BEKERJA
------------------
--cap N membatasi jumlah per kelas DI SPLIT TRAIN. Val dan test ikut
menyusut dengan proporsi yang sama, jadi perbandingan 70/15/15 tetap terjaga:

    train : N
    val   : N * 0.15/0.70  (jadi --cap 5000 -> 1071)
    test  : sama dengan val

Kelas yang jumlahnya di bawah batas dipakai seluruhnya - fear (1088 wajah)
tidak akan pernah kena batas. Pemilihannya acak dengan seed tetap, dan
dilakukan per FOTO juga, bukan per wajah, supaya aturan nomor 1 tidak batal.

Efek sampingnya: hasilnya bisa sedikit MELEBIHI batas. Foto terakhir yang
masuk dibawa utuh beserta semua wajahnya, jadi --cap 5000 bisa berakhir di
5002. Selisih beberapa gambar begini tidak perlu dikejar - memotong foto di
tengah malah menghidupkan lagi masalah kebocoran di nomor 1.

Contoh:
    python webcam/scripts/expw_build.py --dry_run
    python webcam/scripts/expw_build.py --cap 5000
    python webcam/scripts/expw_build.py --cap 0 --min_conf 40   # semuanya
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

SRC_DEFAULT = PROJECT_ROOT / "webcam" / "datasets" / "expw_raw"
DST_DEFAULT = PROJECT_ROOT / "webcam" / "datasets" / "expw"

# Indeks 0-6 di label.lst, urutannya dari readme.txt bawaan ExpW.
KODE_KE_KELAS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise',
                 'neutral']

SPLITS = ['train', 'val', 'test']
RASIO = {'train': 0.70, 'val': 0.15, 'test': 0.15}


def baca_label(path):
    """Kembalikan (dict foto -> list wajah, statistik baris yang dibuang).

    Tiap wajah berupa (atas, kiri, kanan, bawah, kelas).
    """
    per_foto = defaultdict(list)
    stat = Counter()
    with open(path, encoding='utf-8') as f:
        for baris in f:
            bagian = baris.split()
            if len(bagian) != 8:
                stat['format barisnya salah'] += 1
                continue
            nama = bagian[0]
            try:
                atas, kiri, kanan, bawah = (int(v) for v in bagian[2:6])
                keyakinan = float(bagian[6])
                kode = int(bagian[7])
            except ValueError:
                stat['angkanya tidak terbaca'] += 1
                continue
            if not 0 <= kode < len(KODE_KE_KELAS):
                stat['kode emosi di luar 0-6'] += 1
                continue
            per_foto[nama].append(
                (atas, kiri, kanan, bawah, keyakinan, KODE_KE_KELAS[kode]))
            stat['terbaca'] += 1
    return per_foto, stat


def potong(img, atas, kiri, kanan, bawah, pad):
    """Potong kotak wajah + padding, dijaga tetap di dalam gambar.

    Kembalikan None kalau kotaknya tidak masuk akal - ada beberapa kotak di
    label.lst yang lebar atau tingginya nol setelah dipotong ke batas gambar.
    """
    tinggi, lebar = img.shape[:2]
    h, w = bawah - atas, kanan - kiri
    if h <= 0 or w <= 0:
        return None

    dy, dx = int(h * pad), int(w * pad)
    y0, y1 = max(0, atas - dy), min(tinggi, bawah + dy)
    x0, x1 = max(0, kiri - dx), min(lebar, kanan + dx)
    if y1 - y0 < 2 or x1 - x0 < 2:
        return None
    return img[y0:y1, x0:x1]


def bagi_split(nama_foto, seed):
    """Acak daftar foto lalu bagi 70/15/15. Kembalikan dict foto -> split."""
    urut = sorted(nama_foto)          # sorted() dulu supaya tidak bergantung
    random.Random(seed).shuffle(urut)  # urutan sistem berkas
    n = len(urut)
    n_train = int(n * RASIO['train'])
    n_val = int(n * RASIO['val'])
    hasil = {}
    for f in urut[:n_train]:
        hasil[f] = 'train'
    for f in urut[n_train:n_train + n_val]:
        hasil[f] = 'val'
    for f in urut[n_train + n_val:]:
        hasil[f] = 'test'
    return hasil


def terapkan_cap(tugas, cap, seed):
    """Batasi jumlah per (split, kelas), tapi buang per FOTO bukan per wajah.

    Kalau satu foto dibuang, semua wajahnya ikut terbuang. Itu memang
    disengaja: membuang sebagian wajah dari sebuah foto tidak menambah apa-apa
    dan cuma membuat isi dataset lebih sulit dijelaskan.
    """
    if cap <= 0:
        return tugas

    batas = {s: max(1, round(cap * RASIO[s] / RASIO['train'])) for s in SPLITS}
    print(f"[cap    ] batas per kelas: "
          + ", ".join(f"{s} {batas[s]}" for s in SPLITS))

    # Kelompokkan dulu: (split, kelas) -> foto -> jumlah wajah
    per_kunci = defaultdict(lambda: defaultdict(int))
    for split, kelas, foto, *_ in tugas:
        per_kunci[(split, kelas)][foto] += 1

    foto_dipakai = defaultdict(set)   # (split, kelas) -> himpunan foto
    for kunci, foto_map in per_kunci.items():
        daftar = sorted(foto_map)
        random.Random(seed).shuffle(daftar)
        sisa = batas[kunci[0]]
        for foto in daftar:
            if sisa <= 0:
                break
            foto_dipakai[kunci].add(foto)
            sisa -= foto_map[foto]

    return [t for t in tugas if t[2] in foto_dipakai[(t[0], t[1])]]


def main():
    p = argparse.ArgumentParser(
        description="Potong wajah ExpW jadi webcam/datasets/expw/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Pembagian train/val/test dikelompokkan per FOTO, jadi dua "
               "wajah\ndari foto yang sama tidak pernah terpisah antar split.")
    p.add_argument('--src', default=str(SRC_DEFAULT),
                   help="Folder ExpW mentah, berisi label.lst dan origin/.")
    p.add_argument('--dst', default=str(DST_DEFAULT), help="Folder tujuan.")
    p.add_argument('--cap', type=int, default=5000, metavar='N',
                   help="Batas wajah per kelas di split train; val dan test "
                        "ikut proporsional. 0 = tanpa batas (default: 5000).")
    p.add_argument('--min_conf', '--min-conf', dest='min_conf', type=float,
                   default=0.0, metavar='X',
                   help="Buang kotak yang skor keyakinan detektornya di bawah "
                        "X. Sebarannya di dataset ini: p25 30.6, p50 49.8, "
                        "p75 68.9, maksimum 142.4. Bawaannya 0 alias dipakai "
                        "semua, karena kotaknya toh sudah dilabeli manusia.")
    p.add_argument('--pad', type=float, default=0.2, metavar='F',
                   help="Padding di sekeliling kotak wajah, sebagai pecahan "
                        "dari ukuran kotak. 0.2 menyamai kdefTrain.py dan "
                        "deteksi Haar di fusion_webcam.py (default: 0.2).")
    p.add_argument('--imgsz', type=int, default=224,
                   help="Sisi keluaran, piksel (default: 224).")
    p.add_argument('--color', action='store_true',
                   help="Simpan berwarna. Bawaannya grayscale, menyamai "
                        "perlakuan merge_datasets.py supaya model tidak bisa "
                        "menebak asal dataset dari warnanya.")
    p.add_argument('--seed', type=int, default=42,
                   help="Seed pengacakan split dan pemilihan cap (default: 42).")
    p.add_argument('--force', action='store_true',
                   help="Timpa folder tujuan kalau sudah ada.")
    p.add_argument('--dry_run', '--dry-run', dest='dry_run',
                   action='store_true',
                   help="Cuma hitung dan tampilkan tabelnya, tidak memotong "
                        "atau menulis berkas apa pun.")
    args = p.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    label_lst, origin = src / "label.lst", src / "origin"
    if not label_lst.is_file():
        raise SystemExit(f"ERROR: label.lst tidak ada di {label_lst}")
    if not origin.is_dir():
        raise SystemExit(f"ERROR: folder origin/ tidak ada di {origin}")

    print(f"[baca   ] {label_lst}")
    per_foto, stat = baca_label(label_lst)
    print(f"[baca   ] {stat['terbaca']} wajah di {len(per_foto)} foto")
    for k, v in sorted(stat.items()):
        if k != 'terbaca':
            print(f"          dibuang, {k}: {v}")

    # Saring keyakinan sebelum split, supaya jumlah yang dibagi sudah bersih.
    dibuang_conf = 0
    if args.min_conf > 0:
        for nama in list(per_foto):
            simpan = [w for w in per_foto[nama] if w[4] >= args.min_conf]
            dibuang_conf += len(per_foto[nama]) - len(simpan)
            if simpan:
                per_foto[nama] = simpan
            else:
                del per_foto[nama]
        print(f"[saring ] {dibuang_conf} wajah di bawah keyakinan "
              f"{args.min_conf}, sisa {len(per_foto)} foto")

    split_foto = bagi_split(per_foto.keys(), args.seed)

    tugas = []
    for nama, wajah in per_foto.items():
        split = split_foto[nama]
        for i, (atas, kiri, kanan, bawah, _, kelas) in enumerate(wajah):
            tugas.append((split, kelas, nama, i, atas, kiri, kanan, bawah))

    sebelum = len(tugas)
    tugas = terapkan_cap(tugas, args.cap, args.seed)

    hitung = {s: Counter() for s in SPLITS}
    for split, kelas, *_ in tugas:
        hitung[split][kelas] += 1

    print(f"\n=== Jumlah wajah per kelas ===")
    print(f"{'kelas':10s} {'train':>8s} {'val':>8s} {'test':>8s} {'total':>8s}")
    for kelas in EMOTIONS:
        n = [hitung[s][kelas] for s in SPLITS]
        print(f"{kelas:10s} {n[0]:8d} {n[1]:8d} {n[2]:8d} {sum(n):8d}")
    tot = [sum(hitung[s].values()) for s in SPLITS]
    print(f"{'TOTAL':10s} {tot[0]:8d} {tot[1]:8d} {tot[2]:8d} {sum(tot):8d}")
    if args.cap > 0:
        print(f"\n{sebelum - len(tugas)} wajah dilewati karena --cap "
              f"{args.cap}.")

    if args.dry_run:
        print("\n--dry_run: tidak ada gambar yang dipotong atau ditulis.")
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

    # Urutkan menurut nama foto supaya tiap foto cuma dibaca sekali dari disk,
    # walau punya beberapa wajah.
    tugas.sort(key=lambda t: t[2])
    print(f"[potong ] {len(tugas)} wajah dari {len({t[2] for t in tugas})} foto")

    gagal = Counter()
    ditulis = 0
    foto_kini, img = None, None
    for split, kelas, nama, i, atas, kiri, kanan, bawah in tugas:
        if nama != foto_kini:
            img = cv2.imread(str(origin / nama))
            foto_kini = nama
            if img is None:
                gagal['gambarnya tidak terbaca'] += 1
        if img is None:
            continue

        crop = potong(img, atas, kiri, kanan, bawah, args.pad)
        if crop is None:
            gagal['kotaknya tidak masuk akal'] += 1
            continue

        # INTER_AREA untuk mengecilkan, INTER_CUBIC untuk membesarkan.
        interp = (cv2.INTER_AREA if crop.shape[0] > args.imgsz
                  else cv2.INTER_CUBIC)
        crop = cv2.resize(crop, (args.imgsz, args.imgsz), interpolation=interp)
        if not args.color:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        keluar = dst / split / kelas / f"{Path(nama).stem}_{i}.jpg"
        cv2.imwrite(str(keluar), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        ditulis += 1
        if ditulis % 5000 == 0:
            print(f"          {ditulis}/{len(tugas)}")

    print(f"\n[selesai] {ditulis} gambar ditulis ke {dst}")
    for k, v in sorted(gagal.items()):
        print(f"          gagal, {k}: {v}")

    print("\nLanjutkan dengan:")
    print("  python webcam/scripts/merge_datasets.py --dry_run")


if __name__ == '__main__':
    main()
