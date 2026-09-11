
Real-time multimodal emotion recognition:
YOLO (wajah, FER2013) + CNN-MFCC (audio, RAVDESS + CREMA-D) → decision-level fusion.
Termasuk respons robot AiNex Hiwonder (lambaian tangan pada ekspresi *happy*).

> **Respons lambaian sedang DIMATIKAN.** Kodenya masih lengkap, tapi pemicunya
> dikomentari supaya sistem hanya mengenali ekspresi tanpa menggerakkan robot.
> Cara menghidupkannya ada di bagian [Respons robot: sisi robot](#respons-robot-sisi-robot).


## Struktur

```
.
├── fusion_webcam.py           # entry point: fusion realtime + kontrol robot
├── common.py                  # EMOTIONS, arsitektur AudioCNN, parameter MFCC
├── results_calculation.py     # accuracy / precision / recall / F1
├── requirements.txt
├── DATASETS.txt               # rincian dataset yang dipakai
├── CHANGELOG.txt              # catatan perubahan
├── audio/
│   ├── scripts/               # organize, split, train audio CNN
│   ├── models/                # best_audio_cnn.pt
│   └── datasets/              # ravdess, crema-d, audio_emotion (tidak di git)
├── webcam/
│   ├── scripts/               # kdefTrain, merge_datasets, recheck, training
│   ├── models/                # yolo pretrained + fer2013_baseline-2/
│   │                          #   + merged_baseline/
│   ├── results/               # confusion matrix, val runs
│   └── datasets/              # fer2013, kdef, combined (tidak di git)
├── results/                   # metrics_*.csv dan confusion_*.csv
├── Servers/                   # ainex_wave_server.py + requirements laptop
└── archive/                   # skrip lama (tidak di git, lihat README di dalamnya)
```

Semua path di dalam skrip memakai `Path(__file__)`, jadi tidak ada
hardcoded drive `E:\` - repo bisa di-clone ke laptop mana pun.

Definisi yang dipakai bersama - daftar `EMOTIONS`, arsitektur `AudioCNN`, dan
parameter MFCC - ada di `common.py`. Arsitektur AudioCNN khususnya harus sama
persis antara skrip yang melatih dan yang memuat bobot, jadi definisinya
sengaja cuma ada di satu tempat. Skrip di dalam subfolder menambahkan root ke
`sys.path` dulu sebelum mengimpornya.

## Datasets (tidak ikut di repo)

Terlalu besar untuk GitHub. Download manual dan taruh di:

- `audio/datasets/ravdess/AudioWAV/` - <https://zenodo.org/record/1188976>
- `audio/datasets/crema-d/AudioWAV/` - <https://github.com/CheyneyComputerScience/CREMA-D>
- `webcam/datasets/fer2013/` - Kaggle FER2013
- `webcam/datasets/FERPlus/` - <https://github.com/microsoft/FERPlus>

Setelah dataset audio ada, jalankan:
```bash
python audio/scripts/organize_audio.py
python audio/scripts/split_audio.py
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate         # PowerShell
pip install -r requirements.txt
```

## Menjalankan

Webcam laptop + mikrofon:
```bash
python fusion_webcam.py
```

Kamera robot AiNex lewat LAN, tanpa audio, dengan respons lambaian:
```bash
python fusion_webcam.py \
  --camera_url "http://192.168.50.2:8080/stream?topic=/camera/image_raw&type=ros_compressed" \
  --no_speech \
  --wave_url http://192.168.50.2:5000/wave
```

Tanpa `--wave_url`, sistem cetak `[ROBOT] WAVE!` ke konsol (dry-run).
Cooldown lambaian default 60 detik, ubah dengan `--wave_cooldown DETIK`.

Catatan: selama respons lambaian dimatikan, `--wave_url` dan `--wave_token`
tetap diterima tapi tidak berpengaruh - tidak ada request yang dikirim.

## Respons robot: sisi robot

### Status: DIMATIKAN

Pemicu lambaian di `fusion_webcam.py` sedang dikomentari, jadi sistem tidak
mengirim apa pun ke robot. Saat dijalankan, konsol mencetak:

```
[robot] Respons robot DIMATIKAN di kode (WAVE_ENABLED = False di bagian KONFIGURASI).
```

Untuk menghidupkannya lagi perlu **dua langkah** di `fusion_webcam.py`:

1. ubah `WAVE_ENABLED = False` jadi `True` di bagian KONFIGURASI paling atas
2. hapus tanda `#` pada blok `RESPONS ROBOT DIMATIKAN SEMENTARA` di dalam loop
   utama (tiga baris yang memanggil `robot.trigger_wave()`)

Kalau cuma salah satu yang dikerjakan, lambaian tetap tidak jalan. Indikator
`ROBOT:` di overlay juga ikut disembunyikan selama `WAVE_ENABLED` masih `False`,
supaya tampilannya tidak menyesatkan.

Sisa dokumentasi di bawah ini berlaku setelah lambaian dihidupkan kembali.

`--wave_url` menunjuk ke server kecil yang harus jalan **di robot**. Server itu
ada di `Servers/ainex_wave_server.py`. Salin ke Raspberry Pi robot, lalu:

```bash
# image Pi 5 (ROS jalan di dalam Docker)
docker exec -it -u ubuntu -w /home/ubuntu <container_id> /bin/bash
python3 ainex_wave_server.py --action wave

# image Pi 4 (ROS1 Noetic langsung di host)
python3 ainex_wave_server.py --action wave
```

`--action` adalah nama action group tanpa `.d6a`. Lihat daftarnya di
`~/software/ainex_controller/ActionGroups/`, atau buat sendiri lewat
ActionGroupEditor di PC software Hiwonder. Server memanggil
`ainex_kinematics.motion_manager.MotionManager.run_action(<action>)`.

### Soal "token"

AiNex **tidak** butuh token untuk menggerakkan servo atau menjalankan action
group. API key yang disebut di dokumentasi Hiwonder (`llm_api_key` /
`vllm_api_key` di `/home/ubuntu/large_models/config.py`) hanya dipakai fitur
*AI Large Model* - chat LLM, speech recognition, text-to-speech - dan tidak ada
hubungannya dengan respons ekspresi di TA ini.

Kalau endpoint `/wave` tetap mau dikunci, pakai shared secret buatan sendiri:

```bash
# di robot
python3 ainex_wave_server.py --action wave --token RAHASIA123

# di laptop
python fusion_webcam.py --camera_url ... \
                        --wave_url http://192.168.50.2:5000/wave \
                        --wave_token RAHASIA123
```

## Dataset KDEF (opsional, untuk latih ulang model wajah)

KDEF aslinya dikelompokkan per subjek (`KDEF/AF01/AF01AFS.JPG`), bukan per
emosi. Susun ulang jadi struktur YOLO dengan:

```bash
python webcam/scripts/kdefTrain.py --dry_run   # lihat rencana pembagian
python webcam/scripts/kdefTrain.py             # tulis ke webcam/datasets/kdef/
python webcam/scripts/kdefTrain.py --train --epochs 20
```

Hasilnya 2936 gambar, 7 kelas, seimbang (~420 per kelas), sudah dipotong wajah
dengan Haar + padding 20% seperti yang dilakukan `fusion_webcam.py` saat
inferensi.

Yang perlu diketahui:

- **Split per orang, bukan per gambar.** KDEF memotret 70 orang dalam dua sesi,
  jadi folder `AF01` dan `BF01` adalah orang yang sama. Semua foto satu orang
  selalu masuk ke split yang sama, supaya angka test tidak tinggi palsu karena
  model menghafal wajah. Pembagian juga seimbang gender (24/24, 5/5, 6/6).
- **Sudut** default `S HL HR` (depan dan setengah samping). Tampak samping penuh
  (`FL FR`) bisa ditambah dengan `--angles S HL HR FL FR`, tapi Haar hampir
  selalu gagal di sana sehingga potongannya memakai kotak cadangan.
- **AKDEF** (70 wajah rata-rata) tidak diikutkan secara default. Wajah itu
  dirata-ratakan dari semua subjek termasuk yang ada di val/test, jadi kalau
  dipakai (`--include_akdef`) hanya masuk ke train.
- Empat berkas KDEF asli isinya hitam polos dan otomatis dibuang; dua berkas
  bernama salah (`AF31V`, `AM31H`) diperbaiki lewat tabel errata di skrip.

## Gambar hasil untuk laporan

```bash
python results_figures.py          # PNG
python results_figures.py --jpg    # PNG + JPG
```

Menghasilkan dua gambar di `results/`:

- `figur_model_wajah.png` - baseline vs gabungan di kedua test set, F1 per kelas,
  dan dua confusion matrix. Perbedaan yang tidak signifikan ditandai `n.s.`
  (uji McNemar pada sampel berpasangan).
- `figur_fusion.png` - perbandingan wajah / audio / fusion, sapuan bobot alpha,
  dan confusion matrix fusion.

Membaca probabilitas dari `results/cache/`, jadi kalau cache-nya sudah ada,
gambarnya jadi dalam hitungan detik tanpa inferensi ulang.

## Gabungan FER2013 + KDEF

```bash
python webcam/scripts/merge_datasets.py --dry_run
python webcam/scripts/merge_datasets.py
python webcam/scripts/merge_datasets.py --train --epochs 20
```

Menghasilkan `webcam/datasets/combined/` berisi 44862 gambar dari kedua sumber.

Dua hal yang ditangani skrip ini:

- **Timpang 14.3 : 1.** FER2013 punya 28709 gambar latih, KDEF cuma 2013. Kalau
  digabung apa adanya KDEF cuma 6.6% dari data. KDEF diulang 4x di split train
  (`--kdef_repeat`) sehingga porsinya jadi 21.9%. Efek terbesarnya di kelas
  `disgust`: 436 -> 1580 gambar latih.
- **Domain beda.** FER2013 48x48 grayscale liar, KDEF 224x224 berwarna studio.
  Semua dijadikan grayscale supaya model tidak memakai warna sebagai jalan
  pintas menebak asal dataset. Pakai `--color` kalau ingin mempertahankan warna.

Ketimpangan kelas turun dari 16.36x (FER2013 saja) jadi 5.64x. Turunkan lagi
dengan `--cap N` untuk membatasi kelas mayoritas FER2013.

Ukur hasilnya di test set masing-masing dataset **secara terpisah**, bukan di
test gabungan - angka gabungan menyembunyikan pertukaran antar-domain.

### Hasil (20 epoch, RTX 4060, ~50 menit)

Diukur di test set masing-masing sumber secara terpisah, keduanya grayscale.
Test KDEF berisi 12 orang yang tidak pernah dilihat saat latih, jadi angkanya
person-independent.

| model                | FER2013 acc | FER2013 macro-F1 | KDEF acc | KDEF macro-F1 |
|----------------------|------------:|-----------------:|---------:|--------------:|
| baseline (FER2013)   |      0.6811 |           0.6547 |   0.5238 |        0.4630 |
| gabungan (FER+KDEF)  |      0.6731 |           0.6564 |   0.9306 |        0.9296 |

Di domain lamanya praktis impas (accuracy -0.8pp, macro-F1 +0.2pp), sementara di
domain barunya naik drastis (+40.7pp accuracy). F1 `disgust` di test FER2013
naik dari 0.6000 ke 0.6731 - kelas yang memang paling kekurangan data dan paling
banyak dibantu KDEF.

### Catatan GPU

Skrip lama di repo ini (`training_yolo_.py`, `train_audio_cnn.py`,
`results_calculation.py`, `fusion_webcam.py`) mengunci `device='cpu'`. Baseline
FER2013 karena itu butuh 2.4 jam untuk 20 epoch. `merge_datasets.py` memakai
`--device auto` yang otomatis memilih GPU kalau ada.

## Evaluasi (accuracy, precision, recall, F1)

```bash
python results_calculation.py face             # YOLO wajah, split test FER2013
python results_calculation.py audio            # CNN-MFCC, split test audio_emotion
python results_calculation.py fusion --sweep   # fusion + tabel metrik per nilai alpha
python results_calculation.py all --plot       # ketiganya, plus PNG confusion matrix
```

Metrik dihitung langsung dari confusion matrix (precision per kolom, recall per
baris), dilaporkan per kelas plus macro avg dan weighted avg. Hasil disimpan ke
`results/metrics_*.csv` dan `results/confusion_*.csv` untuk lampiran laporan.

Probabilitas hasil inferensi di-cache di `results/cache/`, jadi mengulang
`fusion --sweep` dengan alpha berbeda tidak perlu inferensi ulang. Paksa hitung
ulang dengan `--no_cache`.

Catatan: FER2013 dan RAVDESS/CREMA-D bukan dataset berpasangan, jadi mode
`fusion` menjodohkan sampel wajah dan suara berlabel sama secara acak
(`--seed`). Itu menguji aturan fusion-nya, bukan korelasi asli wajah-suara pada
orang yang sama - batasan ini perlu disebut di laporan.

## Training ulang

### YOLO klasifikasi wajah - BUTUH `--yes`

```bash
python webcam/scripts/training_yolo_.py --yes
```

**Tanpa `--yes`, skrip cuma mencetak rencananya lalu berhenti.** Ini disengaja:
versi lama langsung melatih begitu dijalankan tanpa argumen apa pun, sehingga
salah ketik sedikit - `--help` sekalipun - langsung memulai training 20 epoch
yang makan berjam-jam. Sekarang perlu konfirmasi eksplisit.

Jadi kalau menjalankannya dan keluarannya seperti ini, tidak ada yang rusak,
memang tinggal menambahkan `--yes`:

```
[rencana] data    : .../webcam/datasets/fer2013
[rencana] epochs  : 20, batch 32, imgsz 224, workers 0
[rencana] device  : 0

Training TIDAK dijalankan. Tambahkan --yes kalau memang mau mulai.
```

Argumen yang sering dipakai:

```bash
# latih di dataset gabungan, pakai GPU, pemuat data diperbanyak
python webcam/scripts/training_yolo_.py --yes   --data webcam/datasets/combined --name merged_baseline   --epochs 20 --batch 64 --workers 8

python webcam/scripts/training_yolo_.py --help   # aman, cuma bantuan
```

`--device` default `auto`, jadi GPU dipakai otomatis kalau ada. `--workers 0`
(default lama) membuat GPU banyak menganggur; naikkan kalau memakai GPU.

Hasil training mendarat di `runs/classify/runs/emotion/<name>/` yang ada di
`.gitignore`. Salin model yang mau dipakai ke `webcam/models/` supaya ikut
tersimpan di git.

### Audio CNN - BUTUH `--yes` juga

```bash
python audio/scripts/train_audio_cnn.py --yes
```

Sama seperti skrip YOLO: tanpa `--yes` cuma mencetak rencana lalu berhenti,
dan pesannya menampilkan perintah lengkap yang tinggal disalin.

```bash
python audio/scripts/train_audio_cnn.py --yes   --epochs 50 --batch 64 --workers 4
```

`--device` default `auto` (dulu terkunci `'cpu'`). Soal `--workers`: MFCC
dihitung di CPU tiap kali sampel dimuat, jadi worker tambahan memang membantu,
tapi Windows memakai spawn sehingga terlalu banyak worker malah rugi. Diukur
di mesin ini untuk 1024 sampel:

| workers | waktu |
|---------|-------|
| 0       | 12.8s |
| 4       | 10.4s |
| 8       | 18.1s |

Bobot ditimpa tiap kali val accuracy membaik, jadi `best_audio_cnn.pt` selalu
berisi model terbaik - bukan epoch terakhir. Latih ulang akan MENIMPA model
yang sekarang dipakai; simpan dulu salinannya, atau arahkan ke berkas lain
dengan `--out`.
