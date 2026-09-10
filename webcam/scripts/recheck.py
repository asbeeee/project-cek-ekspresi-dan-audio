"""
recheck.py - hitung jumlah sampel per kelas di dataset.

Menampilkan tabel: kelas sebagai baris, split sebagai kolom, plus total dan
persentase distribusi kelas. Berguna untuk mengecek apakah split train/val/test
sudah benar dan seberapa timpang jumlah sampel antar-emosi.

Contoh:
    python webcam/scripts/recheck.py              # fer2013 (default)
    python webcam/scripts/recheck.py audio        # audio_emotion
    python webcam/scripts/recheck.py all          # dua-duanya
    python webcam/scripts/recheck.py --dir path/ke/folder --ext .png
"""
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset yang dikenal: nama -> (folder, ekstensi file yang dihitung)
DATASETS = {
    "fer2013": (PROJECT_ROOT / "webcam" / "datasets" / "fer2013",
                {".jpg", ".jpeg", ".png", ".bmp"}),
    "audio": (PROJECT_ROOT / "audio" / "datasets" / "audio_emotion",
              {".wav"}),
}

# Urutan split yang diutamakan; split lain (mis. test_original) menyusul.
SPLIT_ORDER = ["raw", "train", "val", "test"]


def count_files(folder, exts):
    """Jumlah file di `folder` yang ekstensinya cocok. Subfolder diabaikan.

    `exts` kosong berarti semua file dihitung.
    """
    if not folder.is_dir():
        return 0
    return sum(1 for f in folder.iterdir()
               if f.is_file() and (not exts or f.suffix.lower() in exts))


def find_splits(root):
    """Daftar subfolder split, split yang umum didahulukan."""
    splits = [d.name for d in root.iterdir() if d.is_dir()]
    known = [s for s in SPLIT_ORDER if s in splits]
    extra = sorted(s for s in splits if s not in SPLIT_ORDER)
    return known + extra


def find_classes(root, splits):
    """Gabungan nama kelas dari semua split, urut abjad."""
    classes = set()
    for split in splits:
        classes.update(d.name for d in (root / split).iterdir() if d.is_dir())
    return sorted(classes)


def report(name, root, exts):
    print(f"\n{'=' * 72}")
    print(f"{name}  ->  {root}")
    print("=" * 72)

    if not root.is_dir():
        print("  [LEWAT] Folder tidak ada. Dataset belum di-download?")
        return

    splits = find_splits(root)
    if not splits:
        print("  [LEWAT] Tidak ada subfolder split di dalamnya.")
        return

    classes = find_classes(root, splits)
    if not classes:
        print("  [LEWAT] Tidak ada subfolder kelas di dalam split.")
        return

    # counts[kelas][split] = jumlah file
    counts = {c: {s: count_files(root / s / c, exts) for s in splits}
              for c in classes}
    totals = {s: sum(counts[c][s] for c in classes) for s in splits}
    grand = sum(totals.values())

    w_cls = max(8, max(len(c) for c in classes))
    w_col = [max(7, len(s)) for s in splits]

    header = f"{'kelas':<{w_cls}}" + "".join(
        f"  {s:>{w}}" for s, w in zip(splits, w_col)) + f"  {'TOTAL':>8}  {'%':>6}"
    print(header)
    print("-" * len(header))

    for c in classes:
        baris = sum(counts[c].values())
        pct = 100.0 * baris / grand if grand else 0.0
        print(f"{c:<{w_cls}}" + "".join(
            f"  {counts[c][s]:>{w}}" for s, w in zip(splits, w_col))
            + f"  {baris:>8}  {pct:>5.1f}%")

    print("-" * len(header))
    print(f"{'TOTAL':<{w_cls}}" + "".join(
        f"  {totals[s]:>{w}}" for s, w in zip(splits, w_col))
        + f"  {grand:>8}  {100.0 if grand else 0.0:>5.1f}%")

    if 'raw' in splits:
        print("\nCatatan: 'raw' adalah sumber sebelum split, isinya disalin "
              "ke train/val/test,\n         jadi kolom TOTAL menghitung ganda.")

    # Rasio ketimpangan: kelas terbanyak vs kelas tersedikit.
    per_kelas = [sum(counts[c].values()) for c in classes]
    if min(per_kelas) > 0:
        print(f"\nKetimpangan kelas: {max(per_kelas) / min(per_kelas):.2f}x "
              f"(terbanyak {max(per_kelas)}, tersedikit {min(per_kelas)})")
    else:
        kosong = [c for c in classes if sum(counts[c].values()) == 0]
        print(f"\n[PERINGATAN] Kelas tanpa sampel: {', '.join(kosong)}")


def main():
    p = argparse.ArgumentParser(
        description="Hitung jumlah sampel per kelas dan per split di dataset.")
    p.add_argument('dataset', nargs='?', default='fer2013',
                   choices=list(DATASETS) + ['all'],
                   help="Dataset yang dihitung (default: fer2013).")
    p.add_argument('--dir', dest='dir', default=None, metavar='PATH',
                   help="Hitung folder lain, bukan dataset bawaan. "
                        "Strukturnya harus <PATH>/<split>/<kelas>/file.")
    p.add_argument('--ext', nargs='+', default=None, metavar='.EXT',
                   help="Ekstensi yang dihitung, mis. --ext .png .jpg "
                        "(default: ikut dataset, atau semua file untuk --dir).")
    args = p.parse_args()

    exts = None
    if args.ext:
        exts = {e.lower() if e.startswith('.') else '.' + e.lower()
                for e in args.ext}

    if args.dir:
        root = Path(args.dir).expanduser().resolve()
        report(root.name, root, exts or set())
        return

    nama = list(DATASETS) if args.dataset == 'all' else [args.dataset]
    for n in nama:
        root, default_exts = DATASETS[n]
        report(n, root, exts or default_exts)


if __name__ == '__main__':
    main()
