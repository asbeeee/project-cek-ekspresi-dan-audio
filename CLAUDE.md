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
  `runs/classify/runs/emotion/<name>/`.
- Dataset besar tidak ikut di git (lihat `.gitignore`). Semua path di skrip
  memakai `Path(__file__)`, jangan hardcode drive `E:\`.
- Komentar dan output skrip di repo ini memakai bahasa Indonesia.
