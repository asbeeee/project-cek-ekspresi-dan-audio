# CLAUDE.md

Instruksi untuk Claude Code di repo ini.

## Jangan jalankan training

**Jangan pernah menjalankan training sendiri.** Siapkan skrip dan datasetnya,
lalu berikan perintahnya ke user untuk dijalankan sendiri.

Alasannya: kalau Claude yang menjalankan (apalagi di background), user tidak
bisa melihat progress bar ultralytics/torch secara langsung - output-nya
terkunci di proses milik Claude. Training bisa makan puluhan menit sampai
berjam-jam, jadi user perlu bisa memantaunya sendiri di terminalnya.

Ini berlaku untuk semua yang berjalan lama dan menghasilkan progres bertahap:
`model.train(...)`, `merge_datasets.py --train`, `kdefTrain.py --train`,
`training_yolo_.py`, `train_audio_cnn.py`.

Yang boleh dijalankan sendiri: penyusunan/penyalinan dataset, evaluasi,
inferensi, dan skrip cek - semuanya selesai dalam hitungan menit.

## Konteks proyek

- Mesin ini punya **NVIDIA RTX 4060 (8.6 GB)** dan torch build CUDA. Skrip lama
  mengunci `device='cpu'` - baseline FER2013 karena itu butuh 2.4 jam untuk 20
  epoch. Pakai `--device auto` / `device=0` kalau menulis skrip training baru.
- Ultralytics menyimpan hasil di `runs/classify/<project>/<name>/`, bukan
  persis di `project=` yang dioper. Jadi `project='runs/emotion'` mendarat di
  `runs/classify/runs/emotion/<name>/`. Folder `runs/` ada di `.gitignore`;
  model yang benar-benar dipakai disalin ke `webcam/models/<nama>/`.
- Definisi yang dipakai bersama (EMOTIONS, EMOTIONS_ID, arsitektur AudioCNN,
  parameter MFCC) ada di `common.py` di root. Jangan menulis ulang di skrip
  lain - arsitektur AudioCNN khususnya harus sama persis antara skrip yang
  melatih dan skrip yang memuat bobot. Skrip di subfolder perlu menambahkan
  root ke sys.path dulu, lihat `audio/scripts/train_audio_cnn.py`.
- Skrip lama yang sudah digantikan ada di `archive/` (gitignored), penjelasan
  tiap berkas di `archive/README.txt`.
- Dataset besar tidak ikut di git (lihat `.gitignore`). Semua path di skrip
  memakai `Path(__file__)`, jangan hardcode drive `E:\`.
- Komentar dan output skrip di repo ini memakai bahasa Indonesia.
