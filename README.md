
Real-time multimodal emotion recognition:
YOLO (wajah, FER+ + ExpW + KDEF) + CNN-MFCC (audio, RAVDESS + CREMA-D + SAVEE + TESS) → decision-level fusion.
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
│   └── datasets/              # ravdess, crema-d, savee, tess, audio_emotion (tidak di git)
├── webcam/
│   ├── scripts/               # ferplus_build, expw_build, kdefTrain, merge_datasets, recheck, training
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
- `audio/datasets/savee/AudioWAV/` - <http://kahlan.eps.surrey.ac.uk/savee/> (isi folder `ALL/`)
- `audio/datasets/tess/AudioWAV/` - <https://tspace.library.utoronto.ca/handle/1807/24487>
- `webcam/datasets/fer2013/` - Kaggle FER2013 (yang dipakai sekarang tinggal `fer2013.csv`-nya)
- `webcam/datasets/FERPlus/` - <https://github.com/microsoft/FERPlus>
- `webcam/datasets/expw_raw/` - ExpW, <http://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html> (isinya `label.lst` + `origin/`)

SAVEE dan TESS diratakan jadi satu folder `AudioWAV/` seperti dua dataset
lainnya - label dibaca dari nama berkas, bukan dari nama folder. Untuk TESS itu
berarti 14 folder emosinya digabung; jangan ikut menyalin folder bersarang di
dalam unduhannya, isinya salinan persis dan akan terhitung dua kali.

Setelah dataset audio ada, jalankan:
```bash
python audio/scripts/organize_audio.py
python audio/scripts/split_audio.py
```

Untuk angka **speaker-independent** - penutur di test belum pernah terdengar
saat latih - pakai `--by_penutur` dan keluarkan ke folder lain supaya dua-duanya
ada di disk:

```bash
python audio/scripts/split_audio.py --by_penutur \
    --dst audio/datasets/audio_emotion_spk
```

TESS cuma punya 2 penutur, jadi masuk train saja - dengan dua orang, menaruh
salah satunya di test tidak mengukur apa pun. Akibatnya kelas `surprise` di
test menyusut dari 289 jadi 47 berkas. Rinciannya di `DATASETS.txt` bagian B6.

`organize_audio.py` melewati dataset yang foldernya belum ada, jadi aman
dijalankan walau baru punya sebagian. `split_audio.py` MENGHAPUS `train/`,
`val/`, dan `test/` lama sebelum mengisi ulang - harus begitu, karena begitu
ada dataset baru masuk pembagiannya berubah, dan berkas sisa pembagian lama
bisa membuat satu berkas ada di train sekaligus di test.

Hasilnya 12162 berkas: 8512 train, 1821 val, 1829 test. Rincian per kelas dan
per dataset ada di `DATASETS.txt`.

Untuk dataset wajah, jalankan tiga skrip ini berurutan:
```bash
python webcam/scripts/ferplus_build.py     # FER2013 + label FER+ -> fer2013plus/
python webcam/scripts/expw_build.py        # potong wajah ExpW    -> expw/
python webcam/scripts/kdefTrain.py         # potong wajah KDEF    -> kdef/
python webcam/scripts/merge_datasets.py    # gabungkan            -> combined/
```

**FERPlus bukan gambar baru** - isinya label ulang untuk gambar FER2013 yang
sama, hasil voting 10 anotator, dipasangkan menurut urutan baris CSV.
`ferplus_build.py` menggambar ulang pikselnya dari `fer2013.csv` karena nama
berkas JPG di `fer2013/` adalah ID acak Kaggle, bukan nomor baris.

Perlu diketahui: FER+ membuat kelas `disgust` MENYUSUT dari 436 ke 175 gambar
latih. ExpW yang menambalnya kembali ke 4114. Jangan memakai `fer2013` dan
`fer2013plus` bersamaan - gambarnya sama, labelnya beda, dan
`merge_datasets.py` menolak kombinasi itu.

Hasil gabungannya 79729 berkas: 60950 train, 9398 val, 9381 test.

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

Menghasilkan tiga gambar di `results/`:

- `figur_model_wajah.png` - baseline vs gabungan di kedua test set, F1 per kelas,
  dan dua confusion matrix. Perbedaan yang tidak signifikan ditandai `n.s.`
  (uji McNemar pada sampel berpasangan).
- `figur_fusion.png` - perbandingan wajah / audio / fusion, sapuan bobot alpha,
  dan confusion matrix fusion.
- `figur_sesi_18sep.png` - sebelum vs sesudah penambahan FER+, ExpW, SAVEE, dan
  TESS. Dibaca dari CSV di `results/`, bukan dari cache probabilitas, jadi bisa
  dirender sendiri tanpa inferensi ulang:

  ```bash
  python results_figures.py --sesi --jpg
  ```

Dua gambar pertama membaca probabilitas dari `results/cache/`, jadi kalau
cache-nya sudah ada, gambarnya jadi dalam hitungan detik tanpa inferensi ulang.

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

### Hasil lama (FER2013 + KDEF, 20 epoch, RTX 4060, ~50 menit)

Diukur di test set masing-masing sumber secara terpisah, keduanya grayscale.
Test KDEF berisi 12 orang yang tidak pernah dilihat saat latih, jadi angkanya
person-independent.

| model                | FER2013 acc | FER2013 macro-F1 | KDEF acc | KDEF macro-F1 |
|----------------------|------------:|-----------------:|---------:|--------------:|
| baseline (FER2013)   |      0.6811 |           0.6547 |   0.5238 |        0.4630 |
| gabungan (FER+KDEF)  |      0.6731 |           0.6564 |   0.9306 |        0.9296 |

Di domain lamanya praktis impas (accuracy -0.8pp, macro-F1 +0.2pp), sementara di
domain barunya naik drastis (+40.7pp accuracy).

### Hasil sekarang (FER+ + ExpW + KDEF, 20 epoch, RTX 4060, ~31 menit)

Model: `webcam/models/gabungan_ferplus_expw/weights/best.pt`
(yolov8n-cls, batch 64, workers 8, val top-1 terbaik 0.6964 di epoch 19).

KEDUA model diuji di test set yang SAMA (`webcam/datasets/combined`, 9381
gambar), jadi selisihnya murni beda model - bukan beda data uji.

| test set          |     n | acc lama | acc baru | macro-F1 lama | macro-F1 baru |
|-------------------|------:|---------:|---------:|--------------:|--------------:|
| gabungan (semua)  |  9381 |   0.4741 |   0.6976 |        0.4306 |        0.6293 |
| FER+              |  3350 |   0.6773 |   0.8546 |        0.5934 |        0.7359 |
| ExpW              |  5527 |   0.3094 |   0.5792 |        0.2885 |        0.4900 |
| KDEF              |   504 |   0.9306 |   0.9524 |        0.9296 |        0.9524 |

Baris ExpW memang diharapkan melonjak - model lama tidak pernah melihat ExpW
sama sekali. Yang lebih informatif adalah **FER+ naik 17.7pp** (label yang lebih
bersih memang lebih mudah dipelajari) dan **KDEF tetap naik 2.2pp** walau
porsinya di data latih turun dari 21.9% ke 13.2%.

F1 per kelas di test gabungan, kelas yang paling banyak berubah:

| kelas    | F1 lama | F1 baru |
|----------|--------:|--------:|
| takut    |  0.1991 |  0.5043 |
| marah    |  0.4223 |  0.6607 |
| netral   |  0.4872 |  0.7097 |
| jijik    |  0.2006 |  0.3287 |

`jijik` tetap yang paling lemah (0.3287). Itu wajar: FER+ cuma menyisakan 175
gambar latih untuk kelas ini, dan tambalan dari ExpW justru kelas yang labelnya
paling berisik.

### Audio dan fusion

Test set audionya BERBEDA sebelum dan sesudah (1337 lalu 1829 sampel) karena
datasetnya sendiri bertambah, jadi angkanya tidak sebanding satu lawan satu.

| model  | acc sebelum | acc sesudah | macro-F1 sebelum | macro-F1 sesudah |
|--------|------------:|------------:|-----------------:|-----------------:|
| audio  |      0.5939 |      0.6419 |           0.6096 |           0.6632 |
| fusion |      0.7734 |      0.8140 |           0.7570 |           0.7689 |

Fusion tetap di atas kedua modalitas tunggalnya, yang memang jadi alasan
pendekatan ini dipakai. Bobot `ALPHA` ditentukan dengan sapuan, bukan ditebak:

```bash
python results_calculation.py fusion --data webcam/datasets/combined --sweep
```

| alpha | accuracy | macro-F1 |            |
|------:|---------:|---------:|------------|
|  0.00 |   0.6700 |   0.6286 | audio saja |
|  0.40 |   0.8297 |   0.7898 |            |
|  **0.50** | **0.8384** | **0.7961** | **dipakai sekarang** |
|  0.60 |   0.8140 |   0.7689 | nilai lama |
|  1.00 |   0.6976 |   0.6293 | visual saja |

Fusion di alpha 0.50 unggul ~14pp dari modalitas tunggal terbaiknya, diukur di
data yang sama. Ulangi sapuannya tiap kali salah satu model dilatih ulang -
nilai terbaiknya bergantung pada seberapa bagus tiap model relatif terhadap
yang lain.

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

`merge_datasets.py` sekarang menerima sumber apa pun lewat `--sources`:

```bash
python webcam/scripts/merge_datasets.py --dry_run              # lihat tabelnya dulu
python webcam/scripts/merge_datasets.py --sources fer2013plus,kdef
python webcam/scripts/merge_datasets.py --repeat kdef=6 --cap 5000
python webcam/scripts/merge_datasets.py --sources fer2013,kdef --force   # baseline lama
```

| sumber | jumlah | bentuk |
|--------|--------|--------|
| `fer2013plus` | 33500 | 48x48 abu-abu, label FER+ |
| `expw` | 37254 | 224x224 abu-abu, wajah dipotong dari foto liar |
| `kdef` | 2936 | 224x224 warna, studio |
| `fer2013` | 35887 | 48x48 abu-abu, label asli (jangan digabung dengan `fer2013plus`) |

### Audio CNN - BUTUH `--yes` juga

```bash
python audio/scripts/train_audio_cnn.py --yes
```

Sama seperti skrip YOLO: tanpa `--yes` cuma mencetak rencana lalu berhenti,
dan pesannya menampilkan perintah lengkap yang tinggal disalin.

```bash
python audio/scripts/train_audio_cnn.py --yes   --epochs 50 --batch 64 --workers 4
python audio/scripts/train_audio_cnn.py --yes   --balance --select macro
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

Tiap epoch melaporkan dua angka: `acc` (akurasi biasa) dan `macro` (rata-rata
akurasi per kelas). Keduanya dipisah karena kelas `surprise` cuma punya 456
sampel train lawan 1346 kelas lain - CREMA-D tidak punya kelas itu sama sekali.
Kalau `macro` jauh di bawah `acc`, kelas kecil yang dikorbankan. Dua opsi untuk
itu, dua-duanya mati secara bawaan supaya hasilnya masih bisa dibandingkan
dengan latihan-latihan sebelumnya:

| opsi | efek |
|------|------|
| `--balance` | kelas kecil diberi bobot lebih besar di `CrossEntropyLoss` |
| `--select macro` | checkpoint terbaik dipilih dari `macro`, bukan `acc` |

Bobot ditimpa tiap kali skor val membaik, jadi `best_audio_cnn.pt` selalu
berisi model terbaik - bukan epoch terakhir. Latih ulang akan MENIMPA model
yang sekarang dipakai; simpan dulu salinannya, atau arahkan ke berkas lain
dengan `--out`.
