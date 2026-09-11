"""
train_audio_cnn.py - latih CNN-MFCC untuk emosi dari suara.

Sebelumnya skrip ini langsung melatih begitu dijalankan, dan EPOCHS / BATCH /
DEVICE harus diubah dengan mengedit isi berkasnya. Sekarang semuanya lewat
argumen, sama seperti webcam/scripts/training_yolo_.py.

PERLU DIINGAT: tanpa --yes skrip ini cuma mencetak rencananya lalu berhenti.
Kalau merasa "kok tidak jalan", kemungkinan besar --yes-nya yang kelupaan.

Contoh:
    python audio/scripts/train_audio_cnn.py          # cetak rencana, berhenti
    python audio/scripts/train_audio_cnn.py --help   # bantuan, aman
    python audio/scripts/train_audio_cnn.py --yes    # benar-benar melatih
    python audio/scripts/train_audio_cnn.py --yes --epochs 50 --batch 64 --workers 4

Arsitektur modelnya ada di common.py, bukan di sini. Itu disengaja: bentuknya
harus sama persis dengan yang dipakai fusion_webcam.py dan results_calculation.py
waktu memuat bobot hasil latihan ini.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# audio/scripts/<berkas ini> -> root proyek ada di parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from common import EMOTIONS, make_audio_cnn, load_wav_mfcc

DATA_DEFAULT = PROJECT_ROOT / "audio" / "datasets" / "audio_emotion"
OUT_DEFAULT = PROJECT_ROOT / "audio" / "models" / "best_audio_cnn.pt"


def pilih_device(pilihan):
    """'auto' -> pakai GPU kalau ada.

    Nilainya 'cuda'/'cpu' untuk torch, bukan '0' seperti di ultralytics.
    Versi lama skrip ini mengunci 'cpu' padahal mesin ini punya GPU.
    """
    if pilihan != 'auto':
        return pilihan
    if torch.cuda.is_available():
        print(f"[device] GPU terdeteksi: {torch.cuda.get_device_name(0)}")
        return 'cuda'
    print("[device] tidak ada GPU, pakai CPU (jauh lebih lambat).")
    return 'cpu'


class AudioDataset(Dataset):
    """Berkas WAV per kelas; MFCC dihitung saat berkas diambil, bukan di muka."""

    def __init__(self, data_dir, split):
        self.samples = []
        for idx, emo in enumerate(EMOTIONS):
            folder = Path(data_dir) / split / emo
            if folder.is_dir():
                for f in sorted(folder.glob("*.wav")):
                    self.samples.append((f, idx))
        if not self.samples:
            raise SystemExit(f"ERROR: tidak ada berkas .wav di "
                             f"{Path(data_dir) / split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        mfcc = load_wav_mfcc(path)
        return torch.tensor(mfcc).unsqueeze(0), label   # (1, 40, 94)


def evaluate(model, loader, device):
    model.eval()
    benar, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            benar += (pred == y).sum().item()
            total += y.size(0)
    return benar / total if total else 0.0


def latih(args, device):
    train_ds = AudioDataset(args.data, 'train')
    val_ds = AudioDataset(args.data, 'val')
    print(f"[data   ] train {len(train_ds)} sampel, val {len(val_ds)} sampel")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch,
                        num_workers=args.workers)

    model = make_audio_cnn(len(EMOTIONS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    acc_terbaik = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc = evaluate(model, val_dl, device)
        tanda = ""
        if val_acc > acc_terbaik:
            acc_terbaik = val_acc
            torch.save(model.state_dict(), out)
            tanda = "  <- disimpan"
        print(f"Epoch {epoch}/{args.epochs} | "
              f"loss: {total_loss / len(train_dl):.4f} | "
              f"val_acc: {val_acc:.4f}{tanda}")

    print(f"\nTraining selesai! Val accuracy terbaik: {acc_terbaik:.4f}")
    print(f"Model disimpan di: {out}")
    print("\nUkur di test set dengan:")
    print("  python results_calculation.py audio --no_cache")


def main():
    p = argparse.ArgumentParser(
        description="Latih CNN-MFCC emosi suara. Butuh --yes untuk mulai.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Bobot ditimpa tiap kali val accuracy membaik, jadi berkas\n"
               "keluarannya selalu model terbaik, bukan epoch terakhir.")
    p.add_argument('--yes', '-y', action='store_true',
                   help="Konfirmasi mulai training. Tanpa ini skrip cuma "
                        "menampilkan rencana lalu berhenti.")
    p.add_argument('--data', default=str(DATA_DEFAULT),
                   help="Folder dataset <split>/<kelas>/ (default: audio_emotion).")
    p.add_argument('--out', default=str(OUT_DEFAULT),
                   help="Path penyimpanan bobot terbaik.")
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3,
                   help="Learning rate Adam (default: 0.001).")
    p.add_argument('--workers', type=int, default=0,
                   help="Worker pemuat data. MFCC dihitung di CPU saat memuat, "
                        "jadi ini bisa membantu. Diukur di mesin ini untuk "
                        "1024 sampel: workers 0 = 12.8s, 4 = 10.4s, 8 = 18.1s. "
                        "Windows memakai spawn, jadi terlalu banyak worker "
                        "malah rugi - 4 titik terbaiknya (default: 0).")
    p.add_argument('--device', default='auto',
                   help="'auto' (GPU kalau ada), 'cpu', atau 'cuda'.")
    args = p.parse_args()

    data = Path(args.data)
    if not data.is_dir():
        raise SystemExit(f"ERROR: dataset tidak ada: {data}\n"
                         f"       Jalankan dulu audio/scripts/organize_audio.py "
                         f"lalu split_audio.py")

    device = pilih_device(args.device)
    print(f"[rencana] data    : {data}")
    print(f"[rencana] epochs  : {args.epochs}, batch {args.batch}, "
          f"lr {args.lr}, workers {args.workers}")
    print(f"[rencana] device  : {device}")
    print(f"[rencana] keluaran: {args.out}")

    if not args.yes:
        # Tampilkan perintah lengkapnya supaya tinggal disalin, tidak perlu
        # mengetik ulang argumen yang barusan dipakai.
        perintah = " ".join(["python audio/scripts/train_audio_cnn.py", "--yes"]
                            + sys.argv[1:])
        print()
        print("Training TIDAK dijalankan - skrip ini butuh --yes.")
        print("Itu disengaja, supaya training berjam-jam tidak mulai karena")
        print("skrip terlanjur dijalankan tanpa sadar.")
        print()
        print("Salin perintah ini untuk benar-benar mulai:")
        print("  " + perintah)
        return

    latih(args, device)


if __name__ == '__main__':
    main()
