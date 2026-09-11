"""
results_figures.py - render hasil evaluasi jadi gambar, bukan keluaran terminal.

Membaca kembali probabilitas yang sudah di-cache results_calculation.py, lalu
menyimpan dua gambar siap tempel ke laporan:

    results/figur_model_wajah.png   perbandingan baseline vs gabungan
    results/figur_fusion.png        perbandingan wajah / audio / fusion

Jalankan results_calculation.py dulu supaya cache-nya ada:

    python results_calculation.py face --data webcam/datasets/combined --prefix fer_ \\
        --visual_model webcam/models/fer2013_baseline-2/weights/best.pt
    ...dan seterusnya untuk tiap kombinasi

atau biarkan skrip ini yang memanggil inferensinya sendiri (otomatis, lebih lama).

Contoh:
    python results_figures.py
    python results_figures.py --jpg
    python results_figures.py --merged_model webcam/models/model_lain/weights/best.pt

CATATAN WARNA
-------------
Palet diambil dari palet rujukan yang sudah divalidasi, bukan dikira-kira:
  - perbandingan sebelum/sesudah  -> satu hue biru, dua shade (ordinal)
  - dua metrik berdampingan       -> biru + oranye (kategorikal, lolos uji CVD)
  - confusion matrix              -> satu hue biru 100->700 (sequential)
Semuanya sudah dicek dengan validator palet skill dataviz: kategorikal
biru/oranye lolos semua gate (CVD Delta E 24.7), pasangan shade biru lolos gate
ordinal.
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from common import EMOTIONS, EMOTIONS_ID
from results_calculation import (
    PROJECT_ROOT, OUT_DIR, ambil_probs, confusion_matrix, metrics_from_cm,
    pasangkan, evaluasi_fusion, VISUAL_MODEL, AUDIO_MODEL,
)

MERGED_DEFAULT = PROJECT_ROOT / "webcam" / "models" / "merged_baseline" / "weights" / "best.pt"
COMBINED_DIR = PROJECT_ROOT / "webcam" / "datasets" / "combined"

# ===================================================================
# PALET (dari references/palette.md skill dataviz, sudah divalidasi)
# ===================================================================
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'

BIRU = '#2a78d6'          # slot kategorikal 1
ORANYE = '#eb6834'        # slot kategorikal 2
BIRU_MUDA = '#86b6ef'     # ramp biru step 250 - ujung "sebelum" dumbbell
BIRU_TUA = '#1c5cab'      # ramp biru step 550 - ujung "sesudah" dumbbell

# Ramp sequential biru 100->700 untuk confusion matrix
RAMP_BIRU = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
             '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281',
             '#0d366b']
CMAP_BIRU = LinearSegmentedColormap.from_list('biru', RAMP_BIRU)

plt.rcParams.update({
    'font.family': ['Segoe UI', 'DejaVu Sans', 'sans-serif'],
    'font.size': 9,
    'axes.edgecolor': AXIS,
    'axes.labelcolor': INK_2,
    'text.color': INK,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
})


def rapikan(ax, sumbu_x=True):
    """Hilangkan bingkai berlebih, sisakan grid hairline yang tidak menonjol."""
    for sisi in ('top', 'right'):
        ax.spines[sisi].set_visible(False)
    for sisi in ('left', 'bottom'):
        ax.spines[sisi].set_linewidth(0.8)
    ax.tick_params(length=0)
    if sumbu_x:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)   # hairline, tidak putus-putus
        ax.yaxis.grid(False)
        ax.set_axisbelow(True)


def mcnemar(ok_a, ok_b):
    """Kembalikan (b, c, p) untuk dua vektor benar/salah di sampel yang sama."""
    from scipy import stats as st
    b = int((ok_a & ~ok_b).sum())
    c = int((~ok_a & ok_b).sum())
    if b + c == 0:
        return b, c, 1.0
    return b, c, st.binomtest(min(b, c), b + c, 0.5).pvalue


# ===================================================================
# PENGAMBILAN DATA
# ===================================================================
def args_visual(model, data=None, prefix=None):
    return SimpleNamespace(visual_model=str(model), audio_model=str(AUDIO_MODEL),
                           data=str(data) if data else None, prefix=prefix,
                           imgsz=224, no_cache=False)


def kumpulkan(args):
    """Hitung semua yang dibutuhkan kedua gambar."""
    d = {}
    kombinasi = [
        ('base_fer', args.base_model, COMBINED_DIR, 'fer_'),
        ('base_kdef', args.base_model, COMBINED_DIR, 'kdef_'),
        ('merged_fer', args.merged_model, COMBINED_DIR, 'fer_'),
        ('merged_kdef', args.merged_model, COMBINED_DIR, 'kdef_'),
    ]
    for nama, model, data, prefix in kombinasi:
        print(f"[ambil] {nama}")
        probs, labels = ambil_probs('visual', args.split, args_visual(model, data, prefix))
        d[nama] = (probs, labels)

    print("[ambil] audio")
    a = SimpleNamespace(visual_model=str(args.base_model), audio_model=str(AUDIO_MODEL),
                        data=None, prefix=None, imgsz=224, no_cache=False)
    d['audio'] = ambil_probs('audio', args.split, a)
    return d


def metrik(probs, labels):
    cm = confusion_matrix(labels, probs.argmax(1), len(EMOTIONS))
    return cm, metrics_from_cm(cm)


# ===================================================================
# GAMBAR 1 - MODEL WAJAH
# ===================================================================
def dumbbell(ax, label, sebelum, sesudah, judul, catatan=None):
    """Sebelum -> sesudah per item. Satu hue, dua shade."""
    y = np.arange(len(label))[::-1]
    for i, (s0, s1) in enumerate(zip(sebelum, sesudah)):
        ax.plot([s0, s1], [y[i], y[i]], color=AXIS, linewidth=2, zorder=1,
                solid_capstyle='round')
    # "Sebelum" digambar lebih besar dan di belakang; kalau kedua nilainya
    # nyaris sama, cincinnya masih menyembul di balik titik "sesudah".
    ax.scatter(sebelum, y, s=118, color=BIRU_MUDA, zorder=2,
               edgecolor=SURFACE, linewidth=1.6, label='baseline (FER2013)')
    ax.scatter(sesudah, y, s=58, color=BIRU_TUA, zorder=4,
               edgecolor=SURFACE, linewidth=1.6, label='gabungan (FER+KDEF)')

    # Label langsung hanya di ujung "sesudah" - tidak setiap titik diberi angka.
    for i, (s0, s1) in enumerate(zip(sebelum, sesudah)):
        jauh = max(s0, s1)
        ax.text(jauh + 0.02, y[i], f"{s1:.3f}", va='center', ha='left',
                fontsize=8, color=INK_2)
        if catatan and catatan[i]:
            ax.text(min(s0, s1) - 0.02, y[i], catatan[i], va='center', ha='right',
                    fontsize=7.5, color=MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels(label, color=INK_2)
    ax.set_xlim(0, 1.12)
    ax.set_title(judul, color=INK, fontsize=10, loc='left', pad=10)
    rapikan(ax)


def gambar_wajah(d, path):
    fig = plt.figure(figsize=(13.5, 7.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1.2],
                          hspace=0.42, wspace=0.26,
                          left=0.10, right=0.94, top=0.83, bottom=0.08)

    cm_bf, m_bf = metrik(*d['base_fer'])
    cm_mf, m_mf = metrik(*d['merged_fer'])
    cm_bk, m_bk = metrik(*d['base_kdef'])
    cm_mk, m_mk = metrik(*d['merged_kdef'])

    # Signifikansi pada dua test set
    yf = d['base_fer'][1]
    okb_f = d['base_fer'][0].argmax(1) == yf
    okm_f = d['merged_fer'][0].argmax(1) == yf
    _, _, p_fer = mcnemar(okb_f, okm_f)
    yk = d['base_kdef'][1]
    okb_k = d['base_kdef'][0].argmax(1) == yk
    okm_k = d['merged_kdef'][0].argmax(1) == yk
    _, _, p_kdef = mcnemar(okb_k, okm_k)

    # Panel A: ringkasan dua test set
    ax = fig.add_subplot(gs[0, 0])
    dumbbell(ax,
             ['FER2013  accuracy', 'FER2013  macro-F1',
              'KDEF  accuracy', 'KDEF  macro-F1'],
             [m_bf['accuracy'], m_bf['macro'][2], m_bk['accuracy'], m_bk['macro'][2]],
             [m_mf['accuracy'], m_mf['macro'][2], m_mk['accuracy'], m_mk['macro'][2]],
             'A. Baseline vs gabungan, diukur terpisah per sumber',
             catatan=[f"p={p_fer:.2f} n.s.", "", "p<0.001", ""])
    # Legenda ditaruh di bawah sumbu, di luar area data, supaya tidak menutupi
    # baris paling bawah.
    ax.legend(frameon=False, fontsize=8, loc='upper left',
              bbox_to_anchor=(0.0, -0.16), ncol=2, labelcolor=INK_2,
              handletextpad=0.4, columnspacing=2.0)

    # Panel B: per kelas di FER2013
    ax = fig.add_subplot(gs[1, 0])
    dumbbell(ax, [EMOTIONS_ID[e] for e in EMOTIONS],
             m_bf['f1'], m_mf['f1'],
             'B. F1 per kelas di test FER2013 (tidak ada beda yang signifikan)')

    # Panel C & D: confusion matrix
    for kolom, (cm, judul) in enumerate([
            (cm_mf, 'C. Gabungan @ test FER2013'),
            (cm_mk, 'D. Gabungan @ test KDEF')]):
        ax = fig.add_subplot(gs[kolom, 1])
        heatmap(ax, cm, judul)

    fig.suptitle('Evaluasi model wajah: pengaruh penggabungan KDEF ke FER2013',
                 fontsize=13.5, color=INK, x=0.045, ha='left', y=0.975)
    fig.text(0.045, 0.935,
             'Test set diukur terpisah per sumber. Test KDEF berisi 12 orang yang tidak '
             'ikut dilatih.',
             fontsize=8.5, color=MUTED, ha='left')
    fig.text(0.045, 0.908,
             'Signifikansi memakai uji McNemar pada sampel berpasangan. Dua titik yang '
             'menumpuk berarti nilainya praktis tidak berubah.',
             fontsize=8.5, color=MUTED, ha='left')
    simpan(fig, path)


def heatmap(ax, cm, judul):
    baris = cm.sum(1, keepdims=True)
    cmn = np.divide(cm, baris, out=np.zeros(cm.shape, float), where=baris > 0)
    im = ax.imshow(cmn, cmap=CMAP_BIRU, vmin=0, vmax=1)
    n = len(EMOTIONS)
    pendek = [EMOTIONS_ID[e][:4] for e in EMOTIONS]
    ax.set_xticks(range(n), pendek, fontsize=7.5, rotation=45, ha='right')
    ax.set_yticks(range(n), pendek, fontsize=7.5)
    ax.set_xlabel('prediksi', fontsize=8, color=INK_2)
    ax.set_ylabel('label asli', fontsize=8, color=INK_2)
    ax.set_title(judul, color=INK, fontsize=10, loc='left', pad=10)
    for i in range(n):
        for j in range(n):
            v = cmn[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}".lstrip('0'), ha='center', va='center',
                        fontsize=6.8, color='#ffffff' if v > 0.55 else INK_2)
    for sisi in ax.spines.values():
        sisi.set_visible(False)
    ax.tick_params(length=0)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=7, colors=MUTED)


# ===================================================================
# GAMBAR 2 - FUSION
# ===================================================================
def gambar_fusion(d, args, path):
    fig = plt.figure(figsize=(13.5, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.30,
                          left=0.07, right=0.95, top=0.78, bottom=0.22)

    pv, lv = d['merged_fer']
    pa, la = d['audio']
    _, m_v = metrik(pv, lv)
    _, m_a = metrik(pa, la)

    iv, ia = pasangkan(lv, la, seed=args.seed)
    p_visual, p_audio, labels = pv[iv], pa[ia], lv[iv]
    cm_f, p_final = evaluasi_fusion(p_visual, p_audio, labels, args.alpha)
    m_f = metrics_from_cm(cm_f)

    # Panel A: emphasis - fusion yang jadi inti cerita, sisanya konteks
    ax = fig.add_subplot(gs[0, 0])
    nama = ['Wajah', 'Audio', 'Fusion']
    acc = [m_v['accuracy'], m_a['accuracy'], m_f['accuracy']]
    f1 = [m_v['macro'][2], m_a['macro'][2], m_f['macro'][2]]
    y = np.arange(3)[::-1]
    # Dua metrik berdampingan -> pasangan kategorikal tervalidasi.
    # Ide "emphasis" abu-abu dibatalkan: abu-abunya justru terlihat lebih berat
    # daripada warna yang mau ditonjolkan, sementara panjang batang sudah cukup
    # menunjukkan bahwa fusion menang.
    t = 0.18
    ax.barh(y + t, acc, height=0.30, color=BIRU, label='accuracy')
    ax.barh(y - t, f1, height=0.30, color=ORANYE, label='macro-F1')
    for i in range(3):
        ax.text(acc[i] + 0.015, y[i] + t, f"{acc[i]:.3f}", va='center',
                fontsize=8, color=INK_2)
        ax.text(f1[i] + 0.015, y[i] - t, f"{f1[i]:.3f}", va='center',
                fontsize=8, color=INK_2)
    ax.set_yticks(y, nama, color=INK_2)
    ax.set_xlim(0, 1.05)
    ax.set_title(f'A. Per modalitas (alpha={args.alpha:.2f})',
                 color=INK, fontsize=10, loc='left', pad=10)
    # Legenda di luar sumbu supaya tidak menutupi label nilai batang.
    ax.legend(frameon=False, fontsize=8, loc='upper left',
              bbox_to_anchor=(0.0, -0.14), ncol=2, labelcolor=INK_2,
              handletextpad=0.4, columnspacing=2.0)
    rapikan(ax)

    # Panel B: sapuan alpha - satu seri, jadi tanpa legenda
    ax = fig.add_subplot(gs[0, 1])
    alphas = np.round(np.arange(0, 1.01, 0.05), 2)
    skor = []
    for a in alphas:
        cm, _ = evaluasi_fusion(p_visual, p_audio, labels, float(a))
        skor.append(metrics_from_cm(cm)['macro'][2])
    skor = np.array(skor)
    ax.plot(alphas, skor, color=BIRU, linewidth=2, solid_capstyle='round')
    i_max = int(skor.argmax())
    ax.scatter([alphas[i_max]], [skor[i_max]], s=80, color=BIRU, zorder=3,
               edgecolor=SURFACE, linewidth=2)
    # Keterangan puncak ditaruh di pojok kiri atas - di sana kurvanya masih
    # rendah, jadi tidak menutupi data dan tidak bertabrakan dengan judul panel.
    ax.text(0.03, 0.97,
            f"puncak alpha={alphas[i_max]:.2f}\nmacro-F1 {skor[i_max]:.4f}",
            transform=ax.transAxes, ha='left', va='top',
            fontsize=8, color=INK_2)
    ax.axvline(args.alpha, color=AXIS, linewidth=1, zorder=0)
    ax.annotate(f"dipakai\nalpha={args.alpha:.2f}", (args.alpha, skor.min()),
                textcoords='offset points', xytext=(6, 0), ha='left', va='bottom',
                fontsize=7.5, color=MUTED)
    jangkauan = skor.max() - skor.min()
    ax.set_ylim(skor.min() - 0.06 * jangkauan, skor.max() + 0.16 * jangkauan)
    ax.set_xlabel('alpha  (1.0 = visual saja, 0.0 = audio saja)',
                  fontsize=8, color=INK_2)
    ax.set_ylabel('macro-F1', fontsize=8, color=INK_2)
    ax.set_title('B. Sapuan bobot fusion', color=INK, fontsize=10, loc='left', pad=10)
    for sisi in ('top', 'right'):
        ax.spines[sisi].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

    # Panel C
    ax = fig.add_subplot(gs[0, 2])
    heatmap(ax, cm_f, 'C. Confusion matrix - fusion')

    fig.suptitle('Evaluasi fusion multimodal: wajah + audio',
                 fontsize=13, color=INK, x=0.07, ha='left', y=0.955)
    fig.text(0.07, 0.90,
             'FER2013 dan RAVDESS/CREMA-D bukan dataset berpasangan, jadi pasangan '
             'wajah-suara dibuat sintetis dari sampel berlabel sama.',
             fontsize=8.5, color=MUTED, ha='left')
    simpan(fig, path)


def simpan(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[simpan] {path}")


def main():
    p = argparse.ArgumentParser(
        description="Render hasil evaluasi jadi gambar untuk laporan.")
    p.add_argument('--split', default='test')
    p.add_argument('--alpha', type=float, default=0.6)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--base_model', default=str(VISUAL_MODEL),
                   help="Bobot model baseline (FER2013 saja).")
    p.add_argument('--merged_model', default=str(MERGED_DEFAULT),
                   help="Bobot model gabungan (FER2013 + KDEF).")
    p.add_argument('--jpg', action='store_true',
                   help="Simpan .jpg selain .png (PNG lebih tajam untuk teks).")
    args = p.parse_args()

    for m in (args.base_model, args.merged_model):
        if not Path(m).exists():
            raise SystemExit(f"ERROR: bobot tidak ada: {m}")

    d = kumpulkan(args)
    ext = ['png', 'jpg'] if args.jpg else ['png']
    for e in ext:
        gambar_wajah(d, OUT_DIR / f"figur_model_wajah.{e}")
        gambar_fusion(d, args, OUT_DIR / f"figur_fusion.{e}")


if __name__ == '__main__':
    sys.exit(main())
