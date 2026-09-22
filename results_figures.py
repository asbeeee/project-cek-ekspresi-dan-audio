"""
results_figures.py - render hasil evaluasi jadi gambar, bukan keluaran terminal.

Membaca kembali probabilitas yang sudah di-cache results_calculation.py, lalu
menyimpan dua gambar siap tempel ke laporan:

    results/figur_model_wajah.png   perbandingan baseline vs gabungan
    results/figur_fusion.png        perbandingan wajah / audio / fusion
    results/figur_sesi_18sep.png    sebelum vs sesudah sesi 18 September 2026
    results/figur_audio_penutur.png speaker-dependent vs speaker-independent

Figur ketiga dibaca dari CSV di results/, bukan dari cache probabilitas, jadi
'--sesi' bisa dipakai untuk merendernya sendiri tanpa inferensi ulang. CSV-nya
dibuat dengan:

    M=webcam/models/gabungan_ferplus_expw/weights/best.pt
    L=webcam/models/merged_baseline/weights/best.pt
    D=webcam/datasets/combined
    python results_calculation.py face --visual_model $M --data $D --tag gabungan_all
    python results_calculation.py face --visual_model $L --data $D --tag lama_all
    # lalu ulangi dengan --prefix fer2013plus_ / expw_ / kdef_ dan --tag yang sesuai

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

LAMA_DEFAULT = PROJECT_ROOT / "webcam" / "models" / "merged_baseline" / "weights" / "best.pt"
BARU_DEFAULT = PROJECT_ROOT / "webcam" / "models" / "gabungan_fer2013" / "weights" / "best.pt"
COMBINED_DIR = PROJECT_ROOT / "webcam" / "datasets" / "combined"

# Prefix nama berkas di combined/, dipasang merge_datasets.py dari nama
# sumbernya. Dulu 'fer_' dan 'kdef_'; sejak merge_datasets.py bisa menerima
# sumber apa pun, prefiksnya jadi nama sumber itu sendiri - 'fer_' tidak ada
# lagi dan skrip ini berhenti dengan "tidak ada file yang cocok".
PREFIX_A, NAMA_A = 'fer2013_', 'FER2013'
PREFIX_B, NAMA_B = 'kdef_', 'KDEF'

NAMA_MODEL_LAMA = 'model lama (FER2013 + KDEF)'
NAMA_MODEL_BARU = 'model baru (FER2013 + ExpW + KDEF)'

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
        ('base_fer', args.base_model, COMBINED_DIR, PREFIX_A),
        ('base_kdef', args.base_model, COMBINED_DIR, PREFIX_B),
        ('merged_fer', args.merged_model, COMBINED_DIR, PREFIX_A),
        ('merged_kdef', args.merged_model, COMBINED_DIR, PREFIX_B),
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
def dumbbell(ax, label, sebelum, sesudah, judul, catatan=None,
             nama_sebelum='baseline (FER2013)',
             nama_sesudah='gabungan (FER+KDEF)', xmax=1.12, turun=False):
    """Sebelum -> sesudah per item. Satu hue, dua shade.

    `turun=True` untuk perbandingan yang nilainya MENURUN (mis. speaker-
    dependent -> speaker-independent). Tanpa itu, angkanya ditaruh di ujung
    kanan batang - padahal ujung kanan adalah titik "sebelum" kalau nilainya
    turun, jadi angkanya kelihatan menempel di titik yang salah.
    """
    y = np.arange(len(label))[::-1]
    for i, (s0, s1) in enumerate(zip(sebelum, sesudah)):
        ax.plot([s0, s1], [y[i], y[i]], color=AXIS, linewidth=2, zorder=1,
                solid_capstyle='round')
    # "Sebelum" digambar lebih besar dan di belakang; kalau kedua nilainya
    # nyaris sama, cincinnya masih menyembul di balik titik "sesudah".
    ax.scatter(sebelum, y, s=118, color=BIRU_MUDA, zorder=2,
               edgecolor=SURFACE, linewidth=1.6, label=nama_sebelum)
    ax.scatter(sesudah, y, s=58, color=BIRU_TUA, zorder=4,
               edgecolor=SURFACE, linewidth=1.6, label=nama_sesudah)

    # Label langsung hanya di ujung "sesudah" - tidak setiap titik diberi angka.
    # Angkanya menempel di titik "sesudah", di sisi yang menjauh dari titik
    # "sebelum", jadi selalu jelas angka itu milik titik yang mana.
    for i, (s0, s1) in enumerate(zip(sebelum, sesudah)):
        if turun:
            ax.text(s1 - 0.02, y[i], f"{s1:.3f}", va='center', ha='right',
                    fontsize=8, color=INK_2)
        else:
            ax.text(max(s0, s1) + 0.02, y[i], f"{s1:.3f}", va='center',
                    ha='left', fontsize=8, color=INK_2)
        if catatan and catatan[i]:
            # Catatan selalu di sisi luar titik "sebelum", supaya tidak pernah
            # bertabrakan dengan angka di atas.
            if turun:
                ax.text(s0 + 0.02, y[i], catatan[i], va='center', ha='left',
                        fontsize=7.5, color=MUTED)
            else:
                ax.text(min(s0, s1) - 0.02, y[i], catatan[i], va='center',
                        ha='right', fontsize=7.5, color=MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels(label, color=INK_2)
    ax.set_xlim(0, xmax)
    ax.set_title(judul, color=INK, fontsize=10, loc='left', pad=10)
    rapikan(ax)


def tanda_p(p):
    """Tulis p-value apa adanya, jangan dipatok di teks judul.

    Versi sebelumnya menulis "tidak ada beda yang signifikan" langsung di
    judul panel. Itu benar untuk perbandingan yang dulu, tapi salah begitu
    modelnya diganti - dan tidak ada yang mengingatkan.
    """
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}" + ("" if p < 0.05 else " n.s.")


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
             [f'{NAMA_A}  accuracy', f'{NAMA_A}  macro-F1',
              f'{NAMA_B}  accuracy', f'{NAMA_B}  macro-F1'],
             [m_bf['accuracy'], m_bf['macro'][2], m_bk['accuracy'], m_bk['macro'][2]],
             [m_mf['accuracy'], m_mf['macro'][2], m_mk['accuracy'], m_mk['macro'][2]],
             'A. Model lama vs baru, diukur terpisah per sumber',
             catatan=[tanda_p(p_fer), "", tanda_p(p_kdef), ""],
             nama_sebelum=NAMA_MODEL_LAMA, nama_sesudah=NAMA_MODEL_BARU)
    # Legenda ditaruh di bawah sumbu, di luar area data, supaya tidak menutupi
    # baris paling bawah.
    ax.legend(frameon=False, fontsize=8, loc='upper left',
              bbox_to_anchor=(0.0, -0.16), ncol=2, labelcolor=INK_2,
              handletextpad=0.4, columnspacing=2.0)

    # Panel B: per kelas di FER2013
    ax = fig.add_subplot(gs[1, 0])
    dumbbell(ax, [EMOTIONS_ID[e] for e in EMOTIONS],
             m_bf['f1'], m_mf['f1'],
             f'B. F1 per kelas di test {NAMA_A} ({tanda_p(p_fer)})',
             nama_sebelum=NAMA_MODEL_LAMA, nama_sesudah=NAMA_MODEL_BARU)

    # Panel C & D: confusion matrix
    for kolom, (cm, judul) in enumerate([
            (cm_mf, f'C. Model baru @ test {NAMA_A}'),
            (cm_mk, f'D. Model baru @ test {NAMA_B}')]):
        ax = fig.add_subplot(gs[kolom, 1])
        heatmap(ax, cm, judul)

    fig.suptitle('Evaluasi model wajah: model lama vs model baru',
                 fontsize=13.5, color=INK, x=0.045, ha='left', y=0.975)
    fig.text(0.045, 0.935,
             f'Test set diukur terpisah per sumber ({NAMA_A} dan {NAMA_B}), '
             f'pada data yang sama untuk kedua model. Test {NAMA_B} berisi 12 '
             f'orang yang tidak ikut dilatih.',
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


# ===================================================================
# GAMBAR 3 - PERBANDINGAN SEBELUM / SESUDAH SESI 18 SEPTEMBER 2026
# ===================================================================
def baca_metrik(nama):
    """Baca results/metrics_<nama>.csv jadi dict.

    Kembalikan {'kelas': {f1, precision, recall, support}, 'accuracy': x,
    'macro_f1': y}. Berkasnya dibuat results_calculation.py lewat --tag.
    """
    import csv

    path = OUT_DIR / f"metrics_{nama}.csv"
    if not path.is_file():
        raise SystemExit(
            f"ERROR: {path} tidak ada.\n"
            f"       Jalankan dulu results_calculation.py dengan --tag yang "
            f"sesuai;\n"
            f"       perintah lengkapnya ada di docstring berkas ini.")

    hasil = {'kelas': {}}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            nama_baris = r['kelas']
            if nama_baris == 'accuracy':
                hasil['accuracy'] = float(r['f1_score'])
            elif nama_baris == 'macro avg':
                hasil['macro_f1'] = float(r['f1_score'])
            elif nama_baris == 'weighted avg':
                continue
            else:
                hasil['kelas'][nama_baris] = {
                    'f1': float(r['f1_score']),
                    'support': int(r['support']),
                }
    return hasil


def gambar_sesi(path):
    """Tiga panel: wajah per domain, wajah per kelas, audio per kelas.

    Semuanya dumbbell karena tugas datanya sama - satu nilai sebelum dan satu
    nilai sesudah untuk tiap item. Satu hue biru dua shade (ordinal), bukan
    dua warna kategorikal: "lama" dan "baru" itu berurutan, bukan dua hal
    yang setara.
    """
    domain = [
        ('gabungan (semua)', 'face_lama_all_test', 'face_gabungan_all_test'),
        ('FER+', 'face_lama_fer2013plus_test', 'face_gabungan_fer2013plus_test'),
        ('ExpW', 'face_lama_expw_test', 'face_gabungan_expw_test'),
        ('KDEF', 'face_lama_kdef_test', 'face_gabungan_kdef_test'),
    ]
    d_lama = [baca_metrik(a) for _, a, _ in domain]
    d_baru = [baca_metrik(b) for _, _, b in domain]

    a_lama = baca_metrik('audio_lama_test')
    a_baru = baca_metrik('audio_test')

    fig = plt.figure(figsize=(14.5, 7.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0],
                          left=0.10, right=0.985, top=0.745, bottom=0.135,
                          wspace=0.52)

    # --- Panel A: wajah, akurasi per domain uji -------------------------
    # Jumlah sampel ditaruh di label sumbu, bukan sebagai catatan di samping
    # titik: di versi sebelumnya catatan itu bertabrakan dengan titiknya
    # sendiri waktu nilainya kecil (ExpW) maupun besar (KDEF).
    ax = fig.add_subplot(gs[0, 0])
    label = [f"{n}\nn={sum(k['support'] for k in x['kelas'].values())}"
             for (n, _, _), x in zip(domain, d_baru)]
    sebelum = [x['accuracy'] for x in d_lama]
    sesudah = [x['accuracy'] for x in d_baru]
    dumbbell(ax, label, sebelum, sesudah,
             'A. Model wajah - akurasi per domain uji',
             nama_sebelum='model lama (FER2013 + KDEF)',
             nama_sesudah='model baru (FER+ + ExpW + KDEF)',
             xmax=1.12)
    ax.set_xlabel('accuracy', fontsize=8, color=INK_2)

    # --- Panel B: wajah, F1 per kelas di test gabungan ------------------
    ax = fig.add_subplot(gs[0, 1])
    kelas_urut = sorted(EMOTIONS,
                        key=lambda k: d_baru[0]['kelas'][k]['f1'], reverse=True)
    sebelum = [d_lama[0]['kelas'][k]['f1'] for k in kelas_urut]
    sesudah = [d_baru[0]['kelas'][k]['f1'] for k in kelas_urut]
    dumbbell(ax, [EMOTIONS_ID[k] for k in kelas_urut], sebelum, sesudah,
             'B. Model wajah - F1 per kelas, test gabungan',
             nama_sebelum='model lama (FER2013 + KDEF)',
             nama_sesudah='model baru (FER+ + ExpW + KDEF)',
             xmax=1.02)
    ax.set_xlabel('F1-score', fontsize=8, color=INK_2)

    # --- Panel C: audio, F1 per kelas -----------------------------------
    # Dibuat per kelas, bukan cuma accuracy + macro F1: dua baris saja
    # menyisakan panel yang hampir kosong, dan yang menarik dari sesi ini
    # justru ada di tingkat kelas (surprise).
    ax = fig.add_subplot(gs[0, 2])
    kelas_audio = sorted(EMOTIONS,
                         key=lambda k: a_baru['kelas'][k]['f1'], reverse=True)
    dumbbell(ax, [EMOTIONS_ID[k] for k in kelas_audio],
             [a_lama['kelas'][k]['f1'] for k in kelas_audio],
             [a_baru['kelas'][k]['f1'] for k in kelas_audio],
             'C. Model audio - F1 per kelas',
             nama_sebelum='sebelum (RAVDESS + CREMA-D)',
             nama_sesudah='sesudah (+ SAVEE + TESS)',
             xmax=1.02)
    ax.set_xlabel('F1-score', fontsize=8, color=INK_2)

    # Dua pasang legenda - panel A/B dan panel C memakai seri yang berbeda.
    h1, l1 = fig.axes[0].get_legend_handles_labels()
    fig.legend(h1, l1, loc='lower left', frameon=False,
               bbox_to_anchor=(0.10, 0.012), ncol=2, labelcolor=INK_2,
               fontsize=8.5, handletextpad=0.4, columnspacing=1.4)
    h3, l3 = fig.axes[2].get_legend_handles_labels()
    fig.legend(h3, l3, loc='lower left', frameon=False,
               bbox_to_anchor=(0.695, 0.012), ncol=1, labelcolor=INK_2,
               fontsize=8.5, handletextpad=0.4)

    fig.suptitle('Hasil training sesi 18 September 2026: sebelum vs sesudah',
                 fontsize=13.5, color=INK, x=0.10, ha='left', y=0.960)
    fig.text(0.10, 0.900,
             'Panel A dan B: kedua model diuji pada test set yang SAMA '
             '(combined, 9381 gambar), jadi selisihnya murni beda model - '
             'kecuali baris ExpW,',
             fontsize=8.5, color=MUTED, ha='left')
    fig.text(0.10, 0.866,
             'yang memang diharapkan melonjak karena model lama tidak pernah '
             'melihat ExpW. Baris FER+ dan KDEF yang lebih informatif.',
             fontsize=8.5, color=MUTED, ha='left')
    fig.text(0.10, 0.826,
             f"Panel C: test set audionya BERBEDA (1337 lalu 1829 sampel) "
             f"karena datasetnya bertambah. Accuracy "
             f"{a_lama['accuracy']:.3f} -> {a_baru['accuracy']:.3f}, macro F1 "
             f"{a_lama['macro_f1']:.3f} -> {a_baru['macro_f1']:.3f}.",
             fontsize=8.5, color=MUTED, ha='left')
    simpan(fig, path)


# ===================================================================
# GAMBAR 4 - SPEAKER-DEPENDENT vs SPEAKER-INDEPENDENT (AUDIO)
# ===================================================================
def baca_confusion(nama):
    """Baca results/confusion_<nama>.csv jadi array (7,7)."""
    import csv

    path = OUT_DIR / f"confusion_{nama}.csv"
    if not path.is_file():
        raise SystemExit(f"ERROR: {path} tidak ada.")
    with open(path, newline='', encoding='utf-8') as f:
        baris = list(csv.reader(f))
    return np.array([[int(v) for v in r[1:]] for r in baris[1:]], dtype=np.int64)


def gambar_penutur(path):
    """Apa yang terjadi kalau penutur di test belum pernah terdengar saat latih.

    Ini bukan perbandingan "model lama vs model baru" - keduanya model yang
    sama arsitekturnya, dilatih dengan cara yang sama. Yang berbeda cuma CARA
    MEMBAGI datanya. Angka yang turun di sini bukan kemunduran; yang di kiri
    memang terlalu tinggi karena penutur yang sama ada di train dan di test.
    """
    dep = baca_metrik('audio_test')
    ind = baca_metrik('audio_spk_test')
    cm = baca_confusion('audio_spk_test')

    fig = plt.figure(figsize=(13.2, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0],
                          left=0.095, right=0.975, top=0.715, bottom=0.135,
                          wspace=0.34)

    # --- Kiri: F1 per kelas, turun semua ---------------------------------
    ax = fig.add_subplot(gs[0, 0])
    urut = sorted(EMOTIONS, key=lambda k: dep['kelas'][k]['f1'], reverse=True)
    label = [f"{EMOTIONS_ID[k]}\nn={ind['kelas'][k]['support']}" for k in urut]
    dumbbell(ax, label,
             [dep['kelas'][k]['f1'] for k in urut],
             [ind['kelas'][k]['f1'] for k in urut],
             'A. F1 per kelas - penutur dikenal vs penutur asing',
             nama_sebelum='speaker-dependent (bagi per berkas)',
             nama_sesudah='speaker-independent (bagi per penutur)',
             xmax=1.08, turun=True)
    ax.set_xlabel('F1-score', fontsize=8, color=INK_2)

    # --- Kanan: salahnya ke mana -----------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    heatmap(ax, cm, 'B. Confusion matrix - speaker-independent')

    h, l = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower left', frameon=False,
               bbox_to_anchor=(0.095, 0.012), ncol=2, labelcolor=INK_2,
               fontsize=8.5, handletextpad=0.4, columnspacing=1.6)

    fig.suptitle('Audio: apa yang terjadi kalau penuturnya orang asing',
                 fontsize=13.5, color=INK, x=0.095, ha='left', y=0.955)
    fig.text(0.095, 0.888,
             f"Arsitektur dan cara latihnya SAMA. Yang berbeda cuma cara "
             f"membagi data. Accuracy {dep['accuracy']:.4f} -> "
             f"{ind['accuracy']:.4f}, macro F1 {dep['macro_f1']:.4f} -> "
             f"{ind['macro_f1']:.4f}.",
             fontsize=8.5, color=MUTED, ha='left')
    fig.text(0.095, 0.852,
             'Angka yang turun ini bukan kemunduran - yang di kiri memang '
             'terlalu tinggi, karena penutur yang sama ada di train dan di '
             'test sekaligus.',
             fontsize=8.5, color=MUTED, ha='left')
    fig.text(0.095, 0.800,
             'surprise jatuh paling jauh (0.899 -> 0.396). Itu bukan '
             'kebetulan: 400 dari 652 berkas surprise berasal dari TESS, yang '
             'cuma punya 2 penutur dan',
             fontsize=8.5, color=MUTED, ha='left')
    fig.text(0.095, 0.764,
             'mengucapkan 200 kata yang sama untuk tiap emosi. Nilai 0.899 itu '
             'sebagian besar hafalan suara, bukan pengenalan emosi. '
             'Support-nya juga tinggal 47.',
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
    p.add_argument('--base_model', default=str(LAMA_DEFAULT),
                   help="Bobot model PEMBANDING (yang lama).")
    p.add_argument('--merged_model', default=str(BARU_DEFAULT),
                   help="Bobot model yang dinilai (yang baru).")
    p.add_argument('--jpg', action='store_true',
                   help="Simpan .jpg selain .png (PNG lebih tajam untuk teks).")
    p.add_argument('--penutur', action='store_true',
                   help="Cuma render figur speaker-dependent vs "
                        "speaker-independent untuk model audio. Dibaca dari "
                        "CSV di results/, jadi tidak menjalankan inferensi "
                        "ulang.")
    p.add_argument('--sesi', action='store_true',
                   help="Cuma render figur perbandingan sebelum/sesudah sesi "
                        "18 September 2026. Dibaca dari CSV di results/, jadi "
                        "tidak menjalankan inferensi ulang - jauh lebih cepat.")
    args = p.parse_args()

    ext = ['png', 'jpg'] if args.jpg else ['png']

    if args.sesi:
        for e in ext:
            gambar_sesi(OUT_DIR / f"figur_sesi_18sep.{e}")
        return

    if args.penutur:
        for e in ext:
            gambar_penutur(OUT_DIR / f"figur_audio_penutur.{e}")
        return

    for m in (args.base_model, args.merged_model):
        if not Path(m).exists():
            raise SystemExit(f"ERROR: bobot tidak ada: {m}")

    d = kumpulkan(args)
    for e in ext:
        gambar_wajah(d, OUT_DIR / f"figur_model_wajah.{e}")
        gambar_fusion(d, args, OUT_DIR / f"figur_fusion.{e}")
        gambar_sesi(OUT_DIR / f"figur_sesi_18sep.{e}")
        gambar_penutur(OUT_DIR / f"figur_audio_penutur.{e}")


if __name__ == '__main__':
    sys.exit(main())
