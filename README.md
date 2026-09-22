
Real-time multimodal emotion recognition:
YOLO (wajah, FER2013 + ExpW + KDEF) + CNN-MFCC (audio, RAVDESS + CREMA-D + SAVEE + TESS) → decision-level fusion.
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
├── Servers/                   # requirements laptop (wave server sudah dihapus)
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

Latih dan ukur modelnya terpisah dari yang speaker-dependent:

```bash
python audio/scripts/train_audio_cnn.py --yes \
    --data audio/datasets/audio_emotion_spk \
    --out audio/models/best_audio_cnn_spk.pt --workers 4

python results_calculation.py audio \
    --audio_model audio/models/best_audio_cnn_spk.pt \
    --audio_data audio/datasets/audio_emotion_spk --tag spk
```

Hasilnya (gambar: `results/figur_audio_penutur.png`):

| pembagian | accuracy | macro F1 |
|-----------|---------:|---------:|
| speaker-dependent | 0.6419 | 0.6632 |
| speaker-independent | **0.4778** | **0.4387** |

Sapuan alpha untuk fusion dengan model audio ini menaruh nilai terbaik di
**0.60**, bergeser dari 0.50 - wajar, karena modalitas audionya lebih lemah
jadi porsinya berkurang. Fusion tetap unggul dari kedua modalitas tunggalnya
(0.7619 lawan visual-saja 0.6976 dan audio-saja 0.5039).

`ALPHA` di kode tetap **0.50**, karena itu nilai yang benar untuk
`best_audio_cnn.pt` yang dipakai `fusion_webcam.py`. Kalau pindah ke
`best_audio_cnn_spk.pt`, ganti ALPHA jadi 0.60 berbarengan.

Model mana yang dipakai di sistem akhir? Perbandingan yang adil tidak mungkin
dibuat - menjalankan model speaker-dependent di test speaker-independent tidak
sah, karena penutur di test itu ikut melatihnya. Saran: **pakai**
`best_audio_cnn.pt` (dilatih dengan semua data) di sistem akhir, tapi
**laporkan** 0.4778 / 0.4387 sebagai perkiraan jujur untuk penutur asing.

Turun 16.4pp accuracy dan 22.5pp macro-F1. Yang paling jatuh adalah `surprise`,
dari 0.8990 ke 0.3958 - 400 dari 652 berkasnya berasal dari TESS, jadi angka
lamanya sebagian besar hafalan suara. Laporkan **kedua** angka berdampingan;
yang speaker-independent itu yang menggambarkan robot bertemu orang asing.

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

**Label FER+ sempat dipakai, lalu dibatalkan 22 September 2026.** Di atas
kertas FER+ lebih baik (10 anotator lawan 1), tapi diukur di test set berlabel
pihak ketiga ternyata kalah - terutama karena model FER+ terlalu sering
menjawab `neutral`. Rinciannya di `DATASETS.txt` bagian A4.

`ferplus_build.py` tetap ada dan tetap jalan kalau percobaannya mau diulang.
Jangan memakai `fer2013` dan `fer2013plus` bersamaan - gambarnya sama, labelnya
beda pada 34.3% kasus, dan `merge_datasets.py` menolak kombinasi itu.

Hasil gabungannya 82116 berkas: 62881 train, 9613 val, 9622 test.

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

## Respons robot: DIHAPUS

Pemicu lambaian tangan saat ekspresi *happy* sudah **dihapus** dari repo ini
pada 18 September 2026 - bukan sekadar dimatikan seperti sebelumnya.

Alasannya lingkup, bukan teknis: bagian respons robot sedang dibicarakan dengan
pemilik proyek pendamping yang juga punya responsnya sendiri, jadi keduanya
tidak perlu saling menabrak.

Yang ikut hilang:

- kelas `RobotController` dan konstanta `WAVE_ENABLED` di `fusion_webcam.py`
- indikator `ROBOT:` di overlay
- argumen `--wave_url`, `--wave_method`, `--wave_cooldown`, `--wave_timeout`,
  `--wave_token`, `--no_wave`
- `Servers/ainex_wave_server.py` (jembatan HTTP di sisi robot)

**Yang TIDAK hilang: dukungan kamera robot.** `--camera_url`, pembaca MJPEG,
`--rotate` / `--flip`, dan diagnosa koneksi semuanya masih ada. Membaca gambar
*dari* robot beda urusan dengan mengirim gerakan *ke* robot.

Semuanya masih tersimpan di riwayat git kalau nanti mau dipakai lagi:

```bash
git log --oneline -- Servers/ainex_wave_server.py
git show <commit>:Servers/ainex_wave_server.py > Servers/ainex_wave_server.py
```

Commit terakhir yang masih memuatnya bisa dicari dengan perintah pertama di
atas.

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

### Hasil sekarang (FER2013 + ExpW + KDEF, 20 epoch, RTX 4060, ~34 menit)

Model: `webcam/models/gabungan_fer2013/weights/best.pt`
(yolov8n-cls, batch 64, workers 4, val top-1 terbaik 0.6274 di epoch 14).

Diukur terpisah per domain, plus CK+ sebagai domain uji luar yang tidak dipakai
melatih model mana pun:

| domain uji | n | accuracy | macro-F1 |
|------------|--:|---------:|---------:|
| gabungan (semua) | 9622 | 0.6356 | 0.5913 |
| FER2013 | 3591 | 0.6901 | 0.6683 |
| ExpW | 5527 | 0.5721 | 0.4836 |
| KDEF | 504 | 0.9444 | 0.9445 |
| **CK+** | 927 | **0.7638** | **0.6136** |

Angka gabungan (0.6356) ditarik turun oleh ExpW, yang isinya 57% dari test set
dan labelnya paling berisik. Yang lebih menggambarkan pemakaian sebenarnya -
wajah pada jarak percakapan - adalah KDEF dan CK+.

**Jangan bandingkan angka ini dengan hasil FER+ sebelumnya secara langsung.**
Test set-nya beda (9622 lawan 9381) DAN sistem labelnya beda; FER2013 dan FER+
tidak sepakat pada 34.3% gambar. Perbandingan yang sah cuma di CK+, KDEF, dan
ExpW - tabel lengkapnya di `DATASETS.txt` bagian A4.

### Audio dan fusion

Test set audionya BERBEDA sebelum dan sesudah (1337 lalu 1829 sampel) karena
datasetnya sendiri bertambah, jadi angkanya tidak sebanding satu lawan satu.

| model  | acc sebelum | acc sesudah | macro-F1 sebelum | macro-F1 sesudah |
|--------|------------:|------------:|-----------------:|-----------------:|
| audio  |      0.5939 |      0.6419 |           0.6096 |           0.6632 |

Fusion dengan model wajah yang sekarang (22 September 2026): **acc 0.8020,
macro-F1 0.7736** di 9622 sampel test gabungan.

Fusion tetap di atas kedua modalitas tunggalnya, yang memang jadi alasan
pendekatan ini dipakai. Bobot `ALPHA` ditentukan dengan sapuan, bukan ditebak:

```bash
python results_calculation.py fusion --data webcam/datasets/combined --sweep
```

| alpha | accuracy | macro-F1 |            |
|------:|---------:|---------:|------------|
|  0.00 |   0.6633 |   0.6404 | audio saja |
|  0.40 |   0.7937 |   0.7668 |            |
|  **0.50** | **0.8020** | **0.7736** | **dipakai sekarang** |
|  0.60 |   0.7748 |   0.7444 |            |
|  1.00 |   0.6356 |   0.5913 | visual saja |

Fusion di alpha 0.50 unggul ~14pp dari modalitas tunggal terbaiknya, diukur di
data yang sama. Alpha optimalnya tetap 0.50 walau model wajahnya diganti -
sudah disapu ulang, bukan diasumsikan. Ulangi sapuannya tiap kali salah satu model dilatih ulang -
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
| `fer2013` | 35887 | 48x48 abu-abu, label asli - **dipakai sekarang** |
| `expw` | 37254 | 224x224 abu-abu, wajah dipotong dari foto liar |
| `kdef` | 2936 | 224x224 warna, studio |
| `fer2013plus` | 33500 | 48x48 abu-abu, label FER+ (dibatalkan, lihat DATASETS.txt A4) |

Selain itu ada `webcam/datasets/ckplus/` (927 gambar) yang dipakai **hanya
sebagai domain uji**, tidak pernah untuk latih - dibuat dengan
`webcam/scripts/ckplus_build.py`.

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
