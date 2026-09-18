"""
split_audio.py - pecah audio_emotion/raw/ jadi train/val/test 70/15/15.

Pembagian acak per kelas dengan seed 42, jadi hasilnya sama tiap kali
dijalankan SELAMA isi raw/ tidak berubah. Begitu ada dataset baru masuk,
daftar berkasnya berubah dan pembagiannya otomatis berbeda - itu memang harus
begitu, tapi artinya split lama TIDAK boleh dibiarkan menumpuk.

Karena itu folder train/, val/, dan test/ dihapus dulu sebelum diisi ulang.
Tanpa itu, berkas dari pembagian sebelumnya akan tertinggal di sana, dan satu
berkas yang dulu di test bisa sekarang juga muncul di train - bocor, dan angka
akurasinya jadi terlalu bagus. Pakai --keep kalau memang mau menambah ke split
yang sudah ada (jarang; biasanya keliru).

    python audio/scripts/split_audio.py
    python audio/scripts/split_audio.py --seed 7
"""
import argparse
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "audio" / "datasets" / "audio_emotion" / "raw"
DST = PROJECT_ROOT / "audio" / "datasets" / "audio_emotion"
SPLITS = ("train", "val", "test")


def main():
    p = argparse.ArgumentParser(description="Pecah raw/ jadi train/val/test.")
    p.add_argument('--seed', type=int, default=42,
                   help="Seed pengacakan (default: 42).")
    p.add_argument('--keep', action='store_true',
                   help="Jangan hapus split lama dulu. Berisiko bocor antar "
                        "split - lihat penjelasan di atas berkas ini.")
    args = p.parse_args()

    if not SRC.is_dir():
        raise SystemExit(f"ERROR: {SRC} tidak ada.\n"
                         f"       Jalankan dulu audio/scripts/organize_audio.py")

    if not args.keep:
        for s in SPLITS:
            folder = DST / s
            if folder.exists():
                shutil.rmtree(folder)
                print(f"[bersih] {folder} dihapus")

    random.seed(args.seed)
    jumlah = {s: 0 for s in SPLITS}

    for emo_folder in sorted(SRC.iterdir()):
        if not emo_folder.is_dir():
            continue

        files = sorted(emo_folder.glob("*.wav"))   # sorted() dulu supaya
        random.shuffle(files)                      # urutannya tidak bergantung
                                                   # sistem berkas
        n = len(files)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        bagian = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:]
        }

        for nama, berkas in bagian.items():
            dest = DST / nama / emo_folder.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in berkas:
                shutil.copy(f, dest / f.name)
            jumlah[nama] += len(berkas)

        print(f"{emo_folder.name:9s}: {n_train:5d} train, {n_val:4d} val, "
              f"{n - n_train - n_val:4d} test")

    print(f"\nTotal    : {jumlah['train']} train, {jumlah['val']} val, "
          f"{jumlah['test']} test")
    print("Split selesai!")


if __name__ == '__main__':
    main()
