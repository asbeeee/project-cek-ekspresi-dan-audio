"""
training_yolo_.py - latih YOLO klasifikasi wajah di FER2013.

Sebelumnya skrip ini langsung melatih begitu dijalankan, tanpa argumen apa pun.
Akibatnya salah ketik sedikit saja - termasuk `--help` - langsung memulai
training 20 epoch yang makan berjam-jam di CPU. Sekarang ada argparse, jadi
`--help` benar-benar cuma menampilkan bantuan, dan perlu `--yes` untuk mulai.

PERLU DIINGAT: tanpa --yes skrip ini cuma mencetak rencananya lalu berhenti.
Kalau merasa "kok tidak jalan", kemungkinan besar --yes-nya yang kelupaan.

Contoh:
    python webcam/scripts/training_yolo_.py          # cetak rencana, berhenti
    python webcam/scripts/training_yolo_.py --help   # bantuan, aman
    python webcam/scripts/training_yolo_.py --yes    # benar-benar melatih
    python webcam/scripts/training_yolo_.py --yes --epochs 30 --batch 64
    python webcam/scripts/training_yolo_.py --yes --data webcam/datasets/combined
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DEFAULT = PROJECT_ROOT / "webcam" / "datasets" / "fer2013"
BOBOT_AWAL = PROJECT_ROOT / "webcam" / "models" / "yolov8n-cls.pt"


def pilih_device(pilihan):
    """'auto' -> pakai GPU kalau ada.

    Versi lama mengunci 'cpu', dan baseline FER2013 karena itu butuh 2.4 jam
    untuk 20 epoch padahal mesin ini punya GPU.
    """
    if pilihan != 'auto':
        return pilihan
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[device] GPU terdeteksi: {torch.cuda.get_device_name(0)}")
            return '0'
    except ImportError:
        pass
    print("[device] tidak ada GPU, pakai CPU (jauh lebih lambat).")
    return 'cpu'


def main():
    p = argparse.ArgumentParser(
        description="Latih YOLO klasifikasi wajah. Butuh --yes untuk mulai.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Hasilnya mendarat di runs/classify/runs/emotion/<name>/ dan\n"
               "folder runs/ ada di .gitignore. Salin model yang mau dipakai\n"
               "ke webcam/models/ supaya ikut tersimpan di git.")
    p.add_argument('--yes', '-y', action='store_true',
                   help="Konfirmasi mulai training. Tanpa ini skrip cuma "
                        "menampilkan rencana lalu berhenti.")
    p.add_argument('--data', default=str(DATA_DEFAULT),
                   help="Folder dataset <split>/<kelas>/ (default: fer2013).")
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--imgsz', type=int, default=224)
    p.add_argument('--workers', type=int, default=0,
                   help="Jumlah worker pemuat data. 0 membuat GPU banyak "
                        "menganggur; naikkan kalau memakai GPU (default: 0).")
    p.add_argument('--device', default='auto',
                   help="'auto' (GPU kalau ada), 'cpu', atau indeks GPU '0'.")
    p.add_argument('--name', default='fer2013_baseline',
                   help="Nama run (default: fer2013_baseline).")
    args = p.parse_args()

    data = Path(args.data)
    if not data.is_dir():
        raise SystemExit(f"ERROR: dataset tidak ada: {data}")

    device = pilih_device(args.device)
    print(f"[rencana] data    : {data}")
    print(f"[rencana] epochs  : {args.epochs}, batch {args.batch}, "
          f"imgsz {args.imgsz}, workers {args.workers}")
    print(f"[rencana] device  : {device}")
    print(f"[rencana] name    : {args.name}")

    if not args.yes:
        # Tampilkan perintah lengkapnya supaya tinggal disalin, tidak perlu
        # mengetik ulang argumen yang barusan dipakai.
        perintah = " ".join(["python webcam/scripts/training_yolo_.py", "--yes"]
                            + sys.argv[1:])
        print()
        print("Training TIDAK dijalankan - skrip ini butuh --yes.")
        print("Itu disengaja: versi lama langsung melatih begitu dijalankan,")
        print("jadi salah ketik sedikit pun memulai training berjam-jam.")
        print()
        print("Salin perintah ini untuk benar-benar mulai:")
        print("  " + perintah)
        return

    from ultralytics import YOLO

    model = YOLO(str(BOBOT_AWAL) if BOBOT_AWAL.exists() else "yolov8n-cls.pt")
    hasil = model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=10,
        device=device,
        project='runs/emotion',
        name=args.name,
        workers=args.workers,
    )

    print("\nTraining selesai!")
    print(f"Best model: {hasil.save_dir}/weights/best.pt")
    print(f"Salin ke webcam/models/ kalau mau dipakai:")
    print(f"  cp -r {hasil.save_dir} webcam/models/{args.name}")


if __name__ == '__main__':
    main()
