"""
results_calculation.py - hitung accuracy, precision, recall, dan F1-score.

Mengevaluasi tiga hal:
  face    : YOLO klasifikasi wajah pada split test FER2013
  audio   : CNN-MFCC pada split test audio_emotion
  fusion  : decision-level fusion (alpha * p_visual + (1-alpha) * p_audio)

Metrik dihitung langsung dari confusion matrix (tanpa scikit-learn), memakai
definisi standar:

    precision_k = TP_k / (TP_k + FP_k)      -> kolom ke-k confusion matrix
    recall_k    = TP_k / (TP_k + FN_k)      -> baris ke-k confusion matrix
    f1_k        = 2 * P_k * R_k / (P_k + R_k)
    accuracy    = sum(TP) / total sampel

Rata-rata dilaporkan dua versi: macro (tiap kelas berbobot sama) dan weighted
(berbobot jumlah sampel). Untuk dataset timpang seperti FER2013, macro-F1 yang
paling jujur - lihat webcam/scripts/recheck.py untuk distribusi kelasnya.

Contoh:
    python results_calculation.py face
    python results_calculation.py audio
    python results_calculation.py fusion --alpha 0.6 --sweep
    python results_calculation.py all --plot

Hasil inferensi di-cache ke results/cache/*.npz, jadi menjalankan ulang
(misal untuk menyapu nilai alpha) tidak perlu inferensi ulang. Pakai
--no_cache untuk memaksa hitung ulang.
"""
import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

from common import EMOTIONS, load_audio_cnn, load_wav_mfcc

PROJECT_ROOT = Path(__file__).resolve().parent

FER_DIR = PROJECT_ROOT / "webcam" / "datasets" / "fer2013"
AUDIO_DIR = PROJECT_ROOT / "audio" / "datasets" / "audio_emotion"
VISUAL_MODEL = PROJECT_ROOT / "webcam" / "models" / "fer2013_baseline-2" / "weights" / "best.pt"
AUDIO_MODEL = PROJECT_ROOT / "audio" / "models" / "best_audio_cnn.pt"
OUT_DIR = PROJECT_ROOT / "results"
CACHE_DIR = OUT_DIR / "cache"

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}

# Parameter fusion, default sama dengan fusion_webcam.py.
ALPHA = 0.6
TAU = 0.20
TAU_CONF = 0.40

DEVICE = 'cpu'


# ===================================================================
# METRIK
# ===================================================================
def confusion_matrix(y_true, y_pred, n_classes):
    """cm[i][j] = jumlah sampel berlabel asli i yang diprediksi j."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def _safe_div(a, b):
    """Pembagian elemen-per-elemen, 0 kalau penyebutnya 0."""
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b > 0)


def metrics_from_cm(cm):
    """Precision, recall, F1 per kelas plus accuracy dan rata-ratanya."""
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)    # TP + FN, jumlah label asli
    predicted = cm.sum(axis=0).astype(np.float64)  # TP + FP, jumlah label prediksi

    precision = _safe_div(tp, predicted)
    recall = _safe_div(tp, support)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)

    total = support.sum()
    accuracy = tp.sum() / total if total else 0.0
    bobot = support / total if total else np.zeros_like(support)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support.astype(np.int64),
        'accuracy': accuracy,
        'macro': (precision.mean(), recall.mean(), f1.mean()),
        'weighted': (float((precision * bobot).sum()),
                     float((recall * bobot).sum()),
                     float((f1 * bobot).sum())),
        'total': int(total),
    }


def print_report(cm, classes, judul):
    """Cetak tabel metrik per kelas, gaya classification_report."""
    m = metrics_from_cm(cm)
    # 12 = panjang 'weighted avg', supaya kolom ringkasan tetap lurus.
    w = max(12, max(len(c) for c in classes))

    print()
    print("=" * 72)
    print(judul)
    print("=" * 72)
    print(f"{'kelas':<{w}}  {'precision':>9}  {'recall':>9}  {'f1-score':>9}  {'support':>8}")
    print("-" * 72)
    for i, c in enumerate(classes):
        print(f"{c:<{w}}  {m['precision'][i]:>9.4f}  {m['recall'][i]:>9.4f}  "
              f"{m['f1'][i]:>9.4f}  {m['support'][i]:>8d}")
    print("-" * 72)
    print(f"{'accuracy':<{w}}  {'':>9}  {'':>9}  {m['accuracy']:>9.4f}  {m['total']:>8d}")
    print(f"{'macro avg':<{w}}  {m['macro'][0]:>9.4f}  {m['macro'][1]:>9.4f}  "
          f"{m['macro'][2]:>9.4f}  {m['total']:>8d}")
    print(f"{'weighted avg':<{w}}  {m['weighted'][0]:>9.4f}  {m['weighted'][1]:>9.4f}  "
          f"{m['weighted'][2]:>9.4f}  {m['total']:>8d}")
    return m


def print_confusion(cm, classes):
    """Cetak confusion matrix. Baris = label asli, kolom = prediksi."""
    pendek = [c[:4] for c in classes]
    w = max(9, max(len(c) for c in classes))
    lebar_kolom = max(5, max(len(p) for p in pendek))

    print()
    print("Confusion matrix (baris = label asli, kolom = prediksi):")
    print(f"{'':<{w}}" + "".join(f"  {p:>{lebar_kolom}}" for p in pendek))
    for i, c in enumerate(classes):
        print(f"{c:<{w}}" + "".join(f"  {cm[i][j]:>{lebar_kolom}d}"
                                    for j in range(len(classes))))


def simpan_csv(cm, classes, nama):
    """Tulis metrik per kelas dan confusion matrix ke CSV, untuk lampiran TA."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m = metrics_from_cm(cm)

    path_metrik = OUT_DIR / f"metrics_{nama}.csv"
    with open(path_metrik, 'w', newline='', encoding='utf-8') as f:
        wtr = csv.writer(f)
        wtr.writerow(['kelas', 'precision', 'recall', 'f1_score', 'support'])
        for i, c in enumerate(classes):
            wtr.writerow([c, f"{m['precision'][i]:.6f}", f"{m['recall'][i]:.6f}",
                          f"{m['f1'][i]:.6f}", int(m['support'][i])])
        wtr.writerow(['accuracy', '', '', f"{m['accuracy']:.6f}", m['total']])
        wtr.writerow(['macro avg'] + [f"{v:.6f}" for v in m['macro']] + [m['total']])
        wtr.writerow(['weighted avg'] + [f"{v:.6f}" for v in m['weighted']] + [m['total']])

    path_cm = OUT_DIR / f"confusion_{nama}.csv"
    with open(path_cm, 'w', newline='', encoding='utf-8') as f:
        wtr = csv.writer(f)
        wtr.writerow(['asli\\prediksi'] + list(classes))
        for i, c in enumerate(classes):
            wtr.writerow([c] + [int(v) for v in cm[i]])

    print(f"\n[simpan] {path_metrik}")
    print(f"[simpan] {path_cm}")


def plot_confusion(cm, classes, nama):
    """Simpan confusion matrix ternormalisasi sebagai PNG."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib tidak terpasang, dilewati.")
        return

    baris = cm.sum(axis=1, keepdims=True)
    cmn = _safe_div(cm.astype(np.float64), baris.astype(np.float64))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha='right')
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label asli")
    ax.set_title(f"Confusion matrix ternormalisasi - {nama}")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cmn[i][j]:.2f}", ha='center', va='center',
                    fontsize=8, color='white' if cmn[i][j] > 0.5 else 'black')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"confusion_{nama}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[simpan] {path}")


# ===================================================================
# PENGUMPULAN SAMPEL
# ===================================================================
def kumpulkan_sampel(root, split, exts, prefix=None):
    """Daftar (path, label_idx) untuk semua kelas di EMOTIONS.

    `prefix` menyaring berdasarkan awalan nama berkas, dipakai untuk memisahkan
    sumber di dataset gabungan (mis. 'fer_' atau 'kdef_' di combined/).
    """
    folder_split = root / split
    if not folder_split.is_dir():
        raise SystemExit(f"ERROR: folder tidak ada: {folder_split}\n"
                         f"       Dataset belum di-download atau split salah.")
    sampel = []
    for idx, emo in enumerate(EMOTIONS):
        folder = folder_split / emo
        if not folder.is_dir():
            print(f"[peringatan] kelas '{emo}' tidak ada di {folder_split}")
            continue
        for f in sorted(folder.iterdir()):
            if not (f.is_file() and f.suffix.lower() in exts):
                continue
            if prefix and not f.name.startswith(prefix):
                continue
            sampel.append((f, idx))
    if not sampel:
        raise SystemExit(f"ERROR: tidak ada file yang cocok di {folder_split}")
    return sampel


def kunci_cache(args, nama):
    """Sidik jari pendek dari model + dataset + prefix.

    Tanpa ini, hasil inferensi model lama akan dipakai ulang diam-diam saat
    mengevaluasi model atau dataset yang berbeda - angkanya jadi salah tanpa
    peringatan apa pun.
    """
    if nama == 'visual':
        bahan = (str(Path(args.visual_model).resolve()), str(args.data),
                 str(args.prefix), str(args.imgsz))
    else:
        bahan = (str(Path(args.audio_model).resolve()),)
    return hashlib.sha1("|".join(bahan).encode()).hexdigest()[:8]


def cache_path(nama, split, kunci=None):
    akhiran = f"_{kunci}" if kunci else ""
    return CACHE_DIR / f"{nama}_{split}{akhiran}.npz"


def muat_cache(nama, split, kunci):
    p = cache_path(nama, split, kunci)
    if not p.exists():
        return None
    d = np.load(p)
    print(f"[cache] pakai hasil inferensi tersimpan: {p.name}")
    return d['probs'], d['labels']


def simpan_cache(nama, split, kunci, probs, labels):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path(nama, split, kunci), probs=probs, labels=labels)
    print(f"[cache] disimpan: {cache_path(nama, split, kunci).name}")


def _progress(i, n, t0, satuan):
    if i % 200 == 0 or i == n:
        lewat = time.time() - t0
        sisa = (lewat / i * (n - i)) if i else 0.0
        print(f"  {i}/{n} {satuan}  ({lewat:.0f}s berlalu, ~{sisa:.0f}s lagi)",
              end='\r', flush=True)


# ===================================================================
# INFERENSI VISUAL (YOLO)
# ===================================================================
def infer_visual(split, model_path, imgsz=224, batch=64, data=None, prefix=None):
    """Kembalikan (probs [N,7] urut EMOTIONS, labels [N])."""
    from ultralytics import YOLO

    if not Path(model_path).exists():
        raise SystemExit(f"ERROR: bobot YOLO tidak ada: {model_path}")

    root = Path(data) if data else FER_DIR
    sampel = kumpulkan_sampel(root, split, IMG_EXT, prefix)
    print(f"[visual] {len(sampel)} gambar dari {root / split}"
          + (f" (prefix '{prefix}')" if prefix else ""))
    print(f"[visual] model: {model_path}")

    model = YOLO(str(model_path))
    # Urutan kelas model belum tentu sama dengan EMOTIONS - petakan eksplisit.
    nama_model = {v: k for k, v in model.names.items()}
    hilang = [e for e in EMOTIONS if e not in nama_model]
    if hilang:
        raise SystemExit(f"ERROR: model tidak mengenal kelas {hilang}. "
                         f"Kelas model: {list(model.names.values())}")
    kolom = [nama_model[e] for e in EMOTIONS]

    probs = np.zeros((len(sampel), len(EMOTIONS)), dtype=np.float32)
    labels = np.array([lbl for _, lbl in sampel], dtype=np.int64)

    t0 = time.time()
    for mulai in range(0, len(sampel), batch):
        potong = sampel[mulai:mulai + batch]
        hasil = model.predict([str(p) for p, _ in potong], imgsz=imgsz,
                              device=DEVICE, verbose=False)
        for k, r in enumerate(hasil):
            probs[mulai + k] = r.probs.data.cpu().numpy()[kolom]
        _progress(min(mulai + batch, len(sampel)), len(sampel), t0, "gambar")
    print()
    return probs, labels


# ===================================================================
# INFERENSI AUDIO (CNN-MFCC)
# ===================================================================
def infer_audio(split, model_path, batch=32):
    """Kembalikan (probs [N,7] urut EMOTIONS, labels [N])."""
    import torch

    if not Path(model_path).exists():
        raise SystemExit(f"ERROR: bobot audio tidak ada: {model_path}")

    model = load_audio_cnn(model_path, DEVICE)

    sampel = kumpulkan_sampel(AUDIO_DIR, split, {'.wav'})
    print(f"[audio] {len(sampel)} file dari {AUDIO_DIR / split}")

    probs = np.zeros((len(sampel), len(EMOTIONS)), dtype=np.float32)
    labels = np.array([lbl for _, lbl in sampel], dtype=np.int64)

    t0 = time.time()
    for mulai in range(0, len(sampel), batch):
        potong = sampel[mulai:mulai + batch]
        fitur = [load_wav_mfcc(path) for path, _ in potong]
        x = torch.tensor(np.stack(fitur)).unsqueeze(1).to(DEVICE)
        with torch.no_grad():
            probs[mulai:mulai + len(potong)] = torch.softmax(model(x), dim=1).cpu().numpy()
        _progress(min(mulai + batch, len(sampel)), len(sampel), t0, "file")
    print()
    return probs, labels


def ambil_probs(nama, split, args):
    """Ambil probabilitas dari cache, atau jalankan inferensi kalau belum ada."""
    kunci = kunci_cache(args, nama)
    if not args.no_cache:
        cached = muat_cache(nama, split, kunci)
        if cached is not None:
            return cached
    if nama == 'visual':
        probs, labels = infer_visual(split, args.visual_model, args.imgsz,
                                     data=args.data, prefix=args.prefix)
    else:
        probs, labels = infer_audio(split, args.audio_model)
    simpan_cache(nama, split, kunci, probs, labels)
    return probs, labels


# ===================================================================
# FUSION
# ===================================================================
def pasangkan(labels_v, labels_a, seed=42):
    """Pasangkan sampel visual dan audio yang berlabel sama.

    FER2013 dan RAVDESS/CREMA-D bukan dataset berpasangan - tidak ada subjek
    yang sama merekam wajah dan suara sekaligus. Jadi pasangan dibuat sintetis:
    untuk tiap kelas, sampel visual dan audio berlabel sama dijodohkan acak.
    Jumlah pasangan per kelas = max(n_visual, n_audio); sisi yang lebih sedikit
    dipakai berulang. Ini menguji aturan fusion-nya, BUKAN korelasi asli antara
    ekspresi wajah dan nada suara pada orang yang sama - sebutkan batasan ini
    di laporan.
    """
    rng = np.random.default_rng(seed)
    idx_v, idx_a = [], []
    for kelas in range(len(EMOTIONS)):
        v = np.flatnonzero(labels_v == kelas)
        a = np.flatnonzero(labels_a == kelas)
        if len(v) == 0 or len(a) == 0:
            print(f"[peringatan] kelas '{EMOTIONS[kelas]}' dilewati "
                  f"(visual={len(v)}, audio={len(a)})")
            continue
        n = max(len(v), len(a))
        idx_v.append(rng.permutation(np.resize(v, n)))
        idx_a.append(rng.permutation(np.resize(a, n)))
    return np.concatenate(idx_v), np.concatenate(idx_a)


def evaluasi_fusion(p_visual, p_audio, labels, alpha):
    p_final = alpha * p_visual + (1.0 - alpha) * p_audio
    pred = p_final.argmax(axis=1)
    return confusion_matrix(labels, pred, len(EMOTIONS)), p_final


def sapu_alpha(p_visual, p_audio, labels):
    """Tabel macro-F1 dan accuracy untuk berbagai alpha."""
    print()
    print("Sapuan alpha (alpha=1.0 berarti visual saja, 0.0 audio saja):")
    print(f"{'alpha':>7}  {'accuracy':>9}  {'macro-P':>9}  {'macro-R':>9}  {'macro-F1':>9}")
    print("-" * 52)
    terbaik = (None, -1.0)
    for alpha in np.round(np.arange(0.0, 1.01, 0.1), 2):
        cm, _ = evaluasi_fusion(p_visual, p_audio, labels, float(alpha))
        m = metrics_from_cm(cm)
        if m['macro'][2] > terbaik[1]:
            terbaik = (float(alpha), m['macro'][2])
        print(f"{alpha:>7.2f}  {m['accuracy']:>9.4f}  {m['macro'][0]:>9.4f}  "
              f"{m['macro'][1]:>9.4f}  {m['macro'][2]:>9.4f}")
    print("-" * 52)
    print(f"macro-F1 tertinggi pada alpha = {terbaik[0]:.2f} ({terbaik[1]:.4f})")


def analisis_ambang(p_final, labels, tau, tau_conf):
    """Berapa banyak keputusan jadi TUNGGAL / MAJEMUK / TIDAK YAKIN.

    Metrik di atas memakai argmax biasa supaya bisa dibandingkan dengan
    penelitian lain. Sistem aslinya (fusion_webcam.py) tidak selalu memilih
    argmax: kalau confidence rendah dia bilang TIDAK YAKIN, kalau dua kelas
    teratas berdekatan dia bilang emosi majemuk. Bagian ini mengukur dampaknya.
    """
    urut = np.argsort(p_final, axis=1)[:, ::-1]
    t1, t2 = urut[:, 0], urut[:, 1]
    p1 = np.take_along_axis(p_final, t1[:, None], axis=1).ravel()
    p2 = np.take_along_axis(p_final, t2[:, None], axis=1).ravel()
    gap = p1 - p2

    # Netral tidak boleh jadi bagian emosi majemuk, sama seperti aturan
    # NO_COMPOUND di fusion_webcam.py. Pasangan yang menyertakan netral
    # dijatuhkan ke TUNGGAL walaupun selisihnya di bawah tau.
    i_netral = EMOTIONS.index('neutral')
    ada_netral = (t1 == i_netral) | (t2 == i_netral)

    tidak_yakin = p1 < tau_conf
    majemuk = (~tidak_yakin) & (gap < tau) & (~ada_netral)
    tunggal = (~tidak_yakin) & (~majemuk)
    n = len(labels)

    print()
    print(f"Dampak ambang keputusan (tau={tau:.2f}, tau_conf={tau_conf:.2f}):")
    print(f"  TUNGGAL      : {tunggal.sum():>6d}  ({100.0 * tunggal.sum() / n:5.1f}%)"
          f"  akurasi di subset ini: "
          f"{(t1[tunggal] == labels[tunggal]).mean() if tunggal.any() else 0.0:.4f}")
    print(f"  MAJEMUK      : {majemuk.sum():>6d}  ({100.0 * majemuk.sum() / n:5.1f}%)"
          f"  label asli ada di top-2: "
          f"{((t1[majemuk] == labels[majemuk]) | (t2[majemuk] == labels[majemuk])).mean() if majemuk.any() else 0.0:.4f}")
    print(f"  TIDAK YAKIN  : {tidak_yakin.sum():>6d}  ({100.0 * tidak_yakin.sum() / n:5.1f}%)"
          f"  (dilaporkan sebagai Netral oleh sistem)")


# ===================================================================
# MODE
# ===================================================================
def mode_face(args):
    probs, labels = ambil_probs('visual', args.split, args)
    cm = confusion_matrix(labels, probs.argmax(axis=1), len(EMOTIONS))
    sumber = Path(args.data).name if args.data else 'fer2013'
    if args.prefix:
        sumber += f":{args.prefix.rstrip('_')}"
    nama = f"face_{args.tag}_{args.split}" if args.tag else f"face_{args.split}"
    print_report(cm, EMOTIONS,
                 f"WAJAH - YOLO klasifikasi, split '{args.split}' {sumber}")
    print_confusion(cm, EMOTIONS)
    simpan_csv(cm, EMOTIONS, nama)
    if args.plot:
        plot_confusion(cm, EMOTIONS, nama)
    return cm


def mode_audio(args):
    probs, labels = ambil_probs('audio', args.split, args)
    cm = confusion_matrix(labels, probs.argmax(axis=1), len(EMOTIONS))
    print_report(cm, EMOTIONS, f"AUDIO - CNN-MFCC, split '{args.split}' audio_emotion")
    print_confusion(cm, EMOTIONS)
    simpan_csv(cm, EMOTIONS, f"audio_{args.split}")
    if args.plot:
        plot_confusion(cm, EMOTIONS, f"audio_{args.split}")
    return cm


def mode_fusion(args):
    pv, lv = ambil_probs('visual', args.split, args)
    pa, la = ambil_probs('audio', args.split, args)

    iv, ia = pasangkan(lv, la, seed=args.seed)
    p_visual, p_audio, labels = pv[iv], pa[ia], lv[iv]
    print(f"\n[fusion] {len(labels)} pasangan sintetis "
          f"(visual {len(lv)} sampel, audio {len(la)} sampel, seed={args.seed})")

    cm, p_final = evaluasi_fusion(p_visual, p_audio, labels, args.alpha)
    print_report(cm, EMOTIONS,
                 f"FUSION - alpha={args.alpha:.2f}, split '{args.split}'")
    print_confusion(cm, EMOTIONS)
    analisis_ambang(p_final, labels, args.tau, args.tau_conf)
    simpan_csv(cm, EMOTIONS, f"fusion_a{args.alpha:.2f}_{args.split}")
    if args.plot:
        plot_confusion(cm, EMOTIONS, f"fusion_a{args.alpha:.2f}_{args.split}")
    if args.sweep:
        sapu_alpha(p_visual, p_audio, labels)
    return cm


def main():
    p = argparse.ArgumentParser(
        description="Hitung accuracy, precision, recall, dan F1-score untuk "
                    "model wajah, audio, dan fusion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Contoh:\n"
               "  python results_calculation.py face\n"
               "  python results_calculation.py fusion --alpha 0.6 --sweep\n"
               "  python results_calculation.py all --plot")
    p.add_argument('mode', nargs='?', default='all',
                   choices=['face', 'audio', 'fusion', 'all'],
                   help="Bagian yang dievaluasi (default: all).")
    p.add_argument('--split', default='test',
                   help="Split dataset yang dipakai (default: test).")
    p.add_argument('--alpha', type=float, default=ALPHA,
                   help=f"Bobot modalitas visual pada fusion (default: {ALPHA}).")
    p.add_argument('--tau', type=float, default=TAU,
                   help=f"Ambang selisih top-1 vs top-2 (default: {TAU}).")
    p.add_argument('--tau_conf', '--tau-conf', dest='tau_conf', type=float,
                   default=TAU_CONF,
                   help=f"Ambang confidence minimum (default: {TAU_CONF}).")
    p.add_argument('--sweep', action='store_true',
                   help="Tampilkan tabel metrik untuk alpha 0.0 sampai 1.0.")
    p.add_argument('--seed', type=int, default=42,
                   help="Seed penjodohan sampel fusion (default: 42).")
    p.add_argument('--imgsz', type=int, default=224,
                   help="Ukuran input YOLO, samakan dengan saat training (default: 224).")
    p.add_argument('--visual_model', '--visual-model', dest='visual_model',
                   default=str(VISUAL_MODEL), help="Path bobot YOLO.")
    p.add_argument('--audio_model', '--audio-model', dest='audio_model',
                   default=str(AUDIO_MODEL), help="Path bobot CNN audio.")
    p.add_argument('--data', default=None, metavar='DIR',
                   help="Folder dataset visual, strukturnya <split>/<kelas>/. "
                        "Default: webcam/datasets/fer2013.")
    p.add_argument('--prefix', default=None, metavar='AWALAN',
                   help="Hanya hitung berkas yang namanya diawali ini, mis. "
                        "'kdef_' untuk memisahkan sumber di dataset combined/.")
    p.add_argument('--tag', default=None, metavar='NAMA',
                   help="Akhiran nama berkas hasil di results/, supaya "
                        "perbandingan antar-model tidak saling menimpa.")
    p.add_argument('--plot', action='store_true',
                   help="Simpan confusion matrix sebagai PNG.")
    p.add_argument('--no_cache', '--no-cache', dest='no_cache', action='store_true',
                   help="Abaikan cache, jalankan inferensi ulang.")
    args = p.parse_args()

    if args.mode in ('face', 'all'):
        mode_face(args)
    if args.mode in ('audio', 'all'):
        mode_audio(args)
    if args.mode in ('fusion', 'all'):
        mode_fusion(args)


if __name__ == '__main__':
    sys.exit(main())
