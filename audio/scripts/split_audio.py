"""
split_audio.py - pecah audio_emotion/raw/ jadi train/val/test 70/15/15.

DUA CARA MEMBAGI
----------------
Bawaannya membagi PER BERKAS: semua berkas diacak lalu dibagi 70/15/15.
Akibatnya penutur yang sama muncul di train dan di test sekaligus, jadi
angka yang keluar adalah akurasi SPEAKER-DEPENDENT - model boleh mengenali
suara orangnya, bukan cuma emosinya.

Dengan --by_penutur, yang dibagi adalah PENUTUR-nya, bukan berkasnya. Semua
berkas milik satu orang masuk ke split yang sama, jadi penutur di test benar-
benar belum pernah terdengar saat latih. Ini yang SPEAKER-INDEPENDENT, dan
inilah yang menggambarkan keadaan sebenarnya waktu robot dipakai: orang yang
diajak bicara hampir selalu orang baru.

Angkanya akan LEBIH RENDAH. Itu bukan kemunduran - yang sebelumnya terlalu
tinggi. Laporkan keduanya berdampingan, jelaskan bedanya apa.

    python audio/scripts/split_audio.py
    python audio/scripts/split_audio.py --by_penutur \\
        --dst audio/datasets/audio_emotion_spk

Keluarkan yang speaker-independent ke folder LAIN (--dst) supaya dua-duanya
ada di disk dan dua model bisa dilatih untuk dibandingkan.

SOAL DATASET YANG PENUTURNYA SEDIKIT
------------------------------------
    CREMA-D  91 penutur, 82 berkas/penutur
    RAVDESS  24 penutur, 60 berkas/penutur
    SAVEE     4 penutur, 120 berkas/penutur
    TESS      2 penutur, 1400 berkas/penutur   <- tidak bisa dibagi

Dataset dengan penutur kurang dari --min_penutur (bawaan 3) TIDAK BISA dipakai
menguji speaker-independence: kalau cuma ada 2 orang, menaruh satu di test
berarti test-nya diwakili satu suara saja. TESS karena itu masuk train saja.
Konsekuensinya kelas surprise di test menyusut banyak, karena 400 dari 652
berkas surprise berasal dari TESS. Itu harus disebut di laporan, bukan
disembunyikan - lihat ringkasan yang dicetak skrip ini di akhir.

PEMBERSIHAN SPLIT LAMA
----------------------
Folder train/, val/, dan test/ dihapus dulu sebelum diisi ulang. Pembagiannya
acak dengan seed tetap, jadi begitu isi raw/ berubah, daftar berkasnya berubah
dan pembagiannya ikut berbeda. Tanpa penghapusan, berkas dari pembagian lama
tertinggal di sana dan satu berkas bisa ada di train sekaligus di test.
Pakai --keep kalau memang sengaja mau menumpuk (jarang; biasanya keliru).
"""
import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "audio" / "datasets" / "audio_emotion" / "raw"
DST_DEFAULT = PROJECT_ROOT / "audio" / "datasets" / "audio_emotion"
SPLITS = ("train", "val", "test")
RASIO = {"train": 0.70, "val": 0.15, "test": 0.15}

# Satu berkas TESS salah nama di rilis aslinya: OA_bite_neutral.wav, kurang
# huruf F. Kalau dibiarkan, 'OA' terbaca sebagai penutur ketiga yang cuma punya
# satu berkas, dan penutur satu-berkas itu bisa mendarat di test sendirian.
# Pola yang sama dipakai kdefTrain.py untuk berkas KDEF yang salah nama.
ERRATA_PENUTUR = {('tess', 'OA'): 'OAF'}


def penutur(path):
    """Kembalikan (dataset, id penutur) dari nama berkas di raw/.

    Nama berkas di raw/ berbentuk <asal>_<nama asli>, dibuat organize_audio.py.
    Tiap dataset menaruh id penuturnya di tempat yang berbeda:

        ravdess_03-01-05-01-02-01-12.wav  -> ruas ke-7 -> aktor 12
        crema_1001_DFA_ANG_XX.wav         -> ruas ke-1 -> 1001
        savee_DC_a01.wav                  -> ruas ke-1 -> DC
        tess_OAF_back_angry.wav           -> ruas ke-1 -> OAF
    """
    asal, _, sisa = path.stem.partition('_')
    if asal == 'ravdess':
        ruas = sisa.split('-')
        pid = ruas[6] if len(ruas) > 6 else '?'
    elif asal in ('crema', 'savee', 'tess'):
        pid = sisa.split('_')[0]
    else:
        # Dataset baru yang belum dikenal - diperlakukan satu penutur per
        # berkas supaya tidak diam-diam menganggapnya satu orang.
        return asal, path.stem
    return asal, ERRATA_PENUTUR.get((asal, pid), pid)


def bagi_per_berkas(berkas_per_kelas, seed):
    """Acak per berkas. Kembalikan dict split -> list berkas."""
    rng = random.Random(seed)
    hasil = {s: [] for s in SPLITS}
    for kelas in sorted(berkas_per_kelas):
        files = sorted(berkas_per_kelas[kelas])
        rng.shuffle(files)
        n = len(files)
        n_train = int(n * RASIO['train'])
        n_val = int(n * RASIO['val'])
        hasil['train'] += files[:n_train]
        hasil['val'] += files[n_train:n_train + n_val]
        hasil['test'] += files[n_train + n_val:]
    return hasil


def bagi_per_penutur(semua, seed, min_penutur):
    """Acak per PENUTUR, dikelompokkan per dataset asal.

    Dikelompokkan per dataset supaya tiap split kebagian dari semua sumber;
    kalau penutur diacak menjadi satu kolam, CREMA-D yang punya 91 penutur
    bisa memborong seluruh val dan test.
    """
    rng = random.Random(seed)
    per_dataset = defaultdict(set)
    for f in semua:
        asal, pid = penutur(f)
        per_dataset[asal].add(pid)

    tugas = {}          # (asal, pid) -> split
    ringkasan = {}      # asal -> dict split -> jumlah penutur
    train_saja = []
    for asal in sorted(per_dataset):
        daftar = sorted(per_dataset[asal])
        rng.shuffle(daftar)
        n = len(daftar)

        if n < min_penutur:
            # Tidak cukup orang untuk menguji speaker-independence.
            for pid in daftar:
                tugas[(asal, pid)] = 'train'
            ringkasan[asal] = {'train': n, 'val': 0, 'test': 0}
            train_saja.append((asal, n))
            continue

        # Pastikan val dan test masing-masing dapat minimal satu penutur;
        # int(n * 0.15) membulatkan 4 penutur jadi 0 val dan 0 test.
        n_val = max(1, round(n * RASIO['val']))
        n_test = max(1, round(n * RASIO['test']))
        n_train = n - n_val - n_test

        for pid in daftar[:n_train]:
            tugas[(asal, pid)] = 'train'
        for pid in daftar[n_train:n_train + n_val]:
            tugas[(asal, pid)] = 'val'
        for pid in daftar[n_train + n_val:]:
            tugas[(asal, pid)] = 'test'
        ringkasan[asal] = {'train': n_train, 'val': n_val, 'test': n_test}

    hasil = {s: [] for s in SPLITS}
    for f in semua:
        hasil[tugas[penutur(f)]].append(f)
    return hasil, ringkasan, train_saja


def main():
    p = argparse.ArgumentParser(
        description="Pecah raw/ jadi train/val/test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="--by_penutur menghasilkan angka yang lebih rendah tapi lebih "
               "jujur:\npenutur di test belum pernah terdengar saat latih, "
               "sama seperti\norang asing yang diajak bicara robotnya.")
    p.add_argument('--src', default=str(SRC), help="Folder raw/.")
    p.add_argument('--dst', default=str(DST_DEFAULT),
                   help="Folder tujuan. Pakai folder BERBEDA untuk hasil "
                        "--by_penutur supaya dua-duanya ada di disk.")
    p.add_argument('--by_penutur', '--by-penutur', dest='by_penutur',
                   action='store_true',
                   help="Bagi per PENUTUR, bukan per berkas. Hasilnya "
                        "speaker-independent.")
    p.add_argument('--min_penutur', '--min-penutur', dest='min_penutur',
                   type=int, default=3, metavar='N',
                   help="Dataset dengan penutur kurang dari N masuk train "
                        "saja - terlalu sedikit untuk menguji "
                        "speaker-independence (default: 3).")
    p.add_argument('--seed', type=int, default=42,
                   help="Seed pengacakan (default: 42).")
    p.add_argument('--keep', action='store_true',
                   help="Jangan hapus split lama dulu. Berisiko bocor antar "
                        "split - lihat penjelasan di atas berkas ini.")
    args = p.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"ERROR: {src} tidak ada.\n"
                         f"       Jalankan dulu audio/scripts/organize_audio.py")

    berkas_per_kelas = {}
    semua = []
    for folder in sorted(src.iterdir()):
        if not folder.is_dir():
            continue
        files = sorted(folder.glob("*.wav"))
        berkas_per_kelas[folder.name] = files
        semua += files
    if not semua:
        raise SystemExit(f"ERROR: tidak ada berkas .wav di {src}")

    cara = 'per penutur (speaker-independent)' if args.by_penutur \
        else 'per berkas (speaker-dependent)'
    print(f"[cara   ] {cara}")
    print(f"[sumber ] {len(semua)} berkas di {src}")

    ringkasan, train_saja = None, []
    if args.by_penutur:
        hasil, ringkasan, train_saja = bagi_per_penutur(
            semua, args.seed, args.min_penutur)
    else:
        hasil = bagi_per_berkas(berkas_per_kelas, args.seed)

    if ringkasan:
        print("\n=== Penutur per split ===")
        print(f"{'dataset':10s} {'total':>6s} {'train':>6s} {'val':>5s} {'test':>5s}")
        for asal in sorted(ringkasan):
            r = ringkasan[asal]
            total = r['train'] + r['val'] + r['test']
            print(f"{asal:10s} {total:6d} {r['train']:6d} {r['val']:5d} "
                  f"{r['test']:5d}")

    # Tabel jumlah berkas per kelas - ditampilkan sebelum menyalin supaya
    # kalau ada kelas yang kosong di suatu split, kelihatan sebelum menunggu
    # ribuan berkas selesai disalin.
    kelas_split = {s: Counter() for s in SPLITS}
    for s in SPLITS:
        for f in hasil[s]:
            kelas_split[s][f.parent.name] += 1

    print("\n=== Berkas per kelas ===")
    print(f"{'kelas':10s} {'train':>7s} {'val':>6s} {'test':>6s} {'total':>7s}")
    kosong = []
    for kelas in sorted(berkas_per_kelas):
        n = [kelas_split[s][kelas] for s in SPLITS]
        print(f"{kelas:10s} {n[0]:7d} {n[1]:6d} {n[2]:6d} {sum(n):7d}")
        for s, v in zip(SPLITS, n):
            if v == 0:
                kosong.append(f"{kelas}/{s}")
    tot = [len(hasil[s]) for s in SPLITS]
    print(f"{'TOTAL':10s} {tot[0]:7d} {tot[1]:6d} {tot[2]:6d} {sum(tot):7d}")

    for asal, n in train_saja:
        print(f"\nCATATAN: {asal} cuma punya {n} penutur, jadi masuk TRAIN "
              f"saja.\n"
              f"         Dengan penutur sesedikit itu, menaruhnya di test "
              f"berarti test-nya\n"
              f"         diwakili satu-dua suara saja. Sebutkan ini di "
              f"laporan.")
    if kosong:
        print(f"\nPERINGATAN: kelas kosong di split tertentu: "
              f"{', '.join(kosong)}")

    if not args.keep:
        for s in SPLITS:
            folder = dst / s
            if folder.exists():
                shutil.rmtree(folder)
                print(f"\n[bersih ] {folder} dihapus")

    print(f"\n[salin  ] ke {dst}")
    for s in SPLITS:
        for f in hasil[s]:
            tujuan = dst / s / f.parent.name
            tujuan.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, tujuan / f.name)

    print("Split selesai!")
    if args.by_penutur:
        print("\nLatih model speaker-independent-nya dengan:")
        print(f"  python audio/scripts/train_audio_cnn.py --yes --data {dst} \\")
        print(f"      --out audio/models/best_audio_cnn_spk.pt")


if __name__ == '__main__':
    main()
