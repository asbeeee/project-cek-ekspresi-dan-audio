"""
fusion_webcam.py  (v3 - dukungan kamera robot AiNex / Hiwonder lewat jaringan)
=============================================================================
Real-time Multimodal Emotion Recognition:
Kamera (Haar face detect + YOLO classify) + Mikrofon (CNN-MFCC)
-> Decision-Level Fusion (Persamaan 2.1)
-> Compound Emotion Detection (Persamaan 2.2)

Tugas Akhir - Hasbi Huda Maulaya (140910220045)

Cara pakai
----------
1) Webcam laptop, dengan audio (perilaku lama):
       python fusion_webcam.py

2) Kamera robot AiNex Hiwonder lewat kabel LAN, tanpa audio:
       python fusion_webcam.py --camera_url "http://192.168.50.2:8080/stream?topic=/camera/image_raw&type=ros_compressed" --no_speech

   Di PowerShell URL WAJIB diapit tanda kutip ganda, karena ada karakter '&'.

3) Kalau gambar robot terbalik atau miring:
       ... --rotate 180
       ... --rotate 90

Kontrol
-------
    q = keluar | a/z = alpha +/- | t/g = tau +/-
    (a/z dinonaktifkan saat --no_speech, karena fusion memakai visual saja)

Input grayscale (GRAYSCALE_INPUT)
--------------------------------
Model wajah dilatih dari gambar grayscale - FER2013 memang selalu grayscale,
dan KDEF diubah jadi grayscale saat digabung - sementara kamera mengirim gambar
berwarna. Kalau dibiarkan, model dipakai di kondisi yang tidak pernah dilihatnya
saat latih. Karena itu potongan wajah diubah ke grayscale dulu sebelum masuk
model. Diukur pada 504 gambar test KDEF (12 orang yang tidak ikut dilatih),
akurasi model gabungan naik dari 0.8770 (berwarna) jadi 0.9306 (grayscale).

Yang diubah HANYA potongan wajah yang masuk model. Frame yang tampil di layar
tetap berwarna. Status mode terlihat di overlay sebagai 'input=gray'/'input=warna'.

Untuk mematikannya, ubah GRAYSCALE_INPUT jadi False di bagian KONFIGURASI,
atau jalankan dengan --no_grayscale tanpa mengubah kode.

Emosi majemuk dan 'netral'
--------------------------
Netral tidak pernah jadi bagian emosi majemuk. Netral itu ketiadaan ekspresi,
jadi gabungan seperti "Sedih-Netral" tidak punya arti - yang dimaksud sebenarnya
"agak sedih", dan itu sudah terwakili oleh label tunggalnya. Kalau salah satu
dari dua kelas teratas adalah netral, keputusannya dipaksa jadi TUNGGAL memakai
kelas teratas, walaupun selisihnya lebih kecil dari tau.

PERUBAHAN v3
------------
- Tambah --camera_url : membaca MJPEG / ROS web_video_server dari robot AiNex.
  Pembacaan dilakukan di thread terpisah supaya frame selalu yang terbaru,
  karena cv2.VideoCapture pada stream HTTP sering menumpuk latensi berdetik-detik.
  Kalau mode 'stream' gagal, otomatis jatuh ke mode 'snapshot' polling.
- Tambah --no_speech : jalan visual-only, tanpa mikrofon, tanpa model audio,
  dan tanpa butuh sounddevice / librosa terpasang. Alpha dikunci 1.0.
- Tambah --rotate / --flip / --no_flip / --width untuk menyesuaikan pose kamera
  robot. Default: frame dari --camera_url TIDAK dicermin, webcam laptop dicermin.
- Semua path model, alpha, tau, dan device bisa dioper lewat argumen.
- Ada diagnosa koneksi otomatis kalau robot tidak terjangkau.
"""

import argparse
import socket
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from common import EMOTIONS, EMOTIONS_ID, SR, DURATION, N_MFCC, load_audio_cnn

# Impor audio dibuat opsional supaya --no_speech tetap jalan
# di mesin yang tidak punya sounddevice / librosa / mikrofon.
try:
    import sounddevice as sd
except Exception as _e:
    sd = None
    _SD_ERR = _e
else:
    _SD_ERR = None

try:
    import librosa
except Exception as _e:
    librosa = None
    _LIBROSA_ERR = _e
else:
    _LIBROSA_ERR = None


# ===================================================================
# KONFIGURASI (default; bisa ditimpa lewat argumen CLI)
# ===================================================================
BASE_DIR = Path(__file__).resolve().parent
VISUAL_MODEL_PATH = str(BASE_DIR / "webcam" / "models" / "fer2013_baseline-2" / "weights" / "best.pt")
AUDIO_MODEL_PATH = str(BASE_DIR / "audio" / "models" / "best_audio_cnn.pt")

# EMOTIONS, EMOTIONS_ID, SR, DURATION, N_MFCC, dan model audio diimpor dari
# common.py supaya definisinya cuma ada di satu tempat.

ALPHA = 0.6
TAU = 0.20
TAU_CONF = 0.40

# Ubah potongan wajah ke grayscale sebelum masuk model, menyamakannya dengan
# data latih. Lihat penjelasan lengkap di docstring paling atas.
# Ganti ke False untuk mematikan, atau pakai --no_grayscale saat menjalankan.
GRAYSCALE_INPUT = True

# Kelas yang tidak boleh muncul sebagai bagian emosi majemuk. Lihat penjelasan
# di docstring paling atas.
NO_COMPOUND = ('neutral',)

# Respons robot (lambaian tangan saat 'happy') DIMATIKAN.
# Untuk menghidupkannya kembali ada DUA langkah:
#   1. ubah baris ini jadi True
#   2. hapus tanda '#' pada blok "RESPONS ROBOT DIMATIKAN SEMENTARA"
#      di dalam loop utama (cari kata WAVE_ENABLED)
# Kalau cuma salah satu, lambaian tetap tidak jalan.
WAVE_ENABLED = False

AUDIO_INFER_INTERVAL = 1.0

DEVICE = 'cpu'
FACE_PAD = 0.20


# ===================================================================
# PEKERJA AUDIO
# ===================================================================
class AudioWorker:
    def __init__(self, model):
        self.model = model
        self.buffer = np.zeros(SR * DURATION, dtype=np.float32)
        self.lock = threading.Lock()
        self.p_audio = np.ones(len(EMOTIONS)) / len(EMOTIONS)
        self.audio_level = 0.0
        self.running = True
        self.stream = None
        self.thread = None

    def _callback(self, indata, frames, t, status):
        mono = indata[:, 0].astype(np.float32)
        with self.lock:
            self.buffer = np.roll(self.buffer, -len(mono))
            self.buffer[-len(mono):] = mono
            self.audio_level = float(np.abs(mono).mean())

    def _infer_loop(self):
        while self.running:
            time.sleep(AUDIO_INFER_INTERVAL)
            with self.lock:
                y = self.buffer.copy()
            mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
            # mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-8)
            x = torch.tensor(mfcc.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(self.model(x), dim=1).cpu().numpy()[0]
            self.p_audio = probs

    def start(self):
        self.stream = sd.InputStream(
            samplerate=SR, channels=1, dtype='float32',
            callback=self._callback, blocksize=int(SR * 0.1)
        )
        self.stream.start()
        self.thread = threading.Thread(target=self._infer_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()


# ===================================================================
# ROBOT CONTROLLER (AiNex Hiwonder - respons ekspresi)
# ===================================================================
# Kirim perintah gerak (mis. lambaian tangan) ke robot ketika ekspresi
# tertentu terdeteksi. Cooldown mencegah pengulangan terus-menerus.
#
# Cara integrasi dengan robot AiNex Hiwonder:
#   Robot AiNex biasanya menjalankan ROS + rosbridge / server HTTP kecil
#   yang memicu action group (misal action group "wave" yang sudah
#   didesain lewat ActionGroupEditor). Cara paling portabel dari sisi
#   PC ini adalah HTTP request ke endpoint yang di sisi robot memanggil
#   action group tersebut.
#
#   Server kecil itu sudah disediakan di repo: Servers/ainex_wave_server.py.
#   Salin ke robot, jalankan di sana (di dalam kontainer Docker untuk image
#   Pi 5), lalu di sisi PC ini panggil:
#       python fusion_webcam.py --camera_url ... \
#              --wave_url http://192.168.50.2:5000/wave
#
#   Catatan soal "token": AiNex tidak butuh token untuk menggerakkan servo.
#   API key di dokumentasi Hiwonder (llm_api_key / vllm_api_key di
#   /home/ubuntu/large_models/config.py) hanya untuk fitur AI Large Model
#   (chat LLM + text-to-speech), bukan untuk action group. Kalau endpoint
#   /wave mau dikunci, pakai shared secret sendiri: --token di sisi robot,
#   --wave_token di sisi PC.
#
# Kalau --wave_url tidak diisi, sistem tetap jalan tapi hanya mencetak
# "[ROBOT] WAVE!" ke konsol (mode dry-run, berguna untuk pengujian).
class RobotController:
    def __init__(self, wave_url=None, method='POST', cooldown=60.0,
                 timeout=2.0, enabled=True, token=None):
        self.wave_url = wave_url
        self.method = method.upper()
        self.token = token
        self.cooldown = float(cooldown)
        self.timeout = float(timeout)
        self.enabled = enabled
        self.last_wave_ts = 0.0  # 0 = belum pernah, boleh langsung wave
        self.last_status = ""    # untuk overlay
        self._lock = threading.Lock()

    def cooldown_remaining(self, now=None):
        if self.last_wave_ts == 0.0:
            return 0.0
        now = now if now is not None else time.time()
        return max(0.0, self.cooldown - (now - self.last_wave_ts))

    def can_wave(self, now=None):
        return self.enabled and self.cooldown_remaining(now) <= 0.0

    def trigger_wave(self):
        """Panggil dari main loop. Non-blocking: HTTP dieksekusi di thread."""
        if not self.can_wave():
            return False
        with self._lock:
            if not self.can_wave():
                return False
            self.last_wave_ts = time.time()
        threading.Thread(target=self._do_wave, daemon=True).start()
        return True

    def _do_wave(self):
        if self.wave_url is None:
            print("[ROBOT] WAVE! (dry-run: --wave_url belum diset)")
            self.last_status = "dry-run"
            return
        try:
            req = Request(self.wave_url, method=self.method)
            # body kosong; kalau perlu payload, tambahkan di sini
            if self.method in ('POST', 'PUT'):
                req.data = b''
                req.add_header('Content-Type', 'application/json')
            if self.token:
                req.add_header('Authorization', 'Bearer %s' % self.token)
            with urlopen(req, timeout=self.timeout) as resp:
                print("[ROBOT] WAVE terkirim (%s %s -> HTTP %d)" %
                      (self.method, self.wave_url, resp.status))
                self.last_status = "ok"
        except Exception as e:
            print("[ROBOT] WAVE gagal: %s" % e)
            self.last_status = "gagal: %s" % type(e).__name__



class DummyAudio:
    """Pengganti AudioWorker saat --no_speech: probabilitas audio nol."""

    def __init__(self, n_classes=len(EMOTIONS)):
        self.p_audio = np.zeros(n_classes, dtype=np.float32)
        self.audio_level = 0.0

    def start(self):
        pass

    def stop(self):
        pass


# ===================================================================
# KAMERA JARINGAN (AiNex Hiwonder / ROS web_video_server / MJPEG)
# ===================================================================
class NetworkCamera:
    """
    Pembaca stream MJPEG dari robot AiNex Hiwonder.

    Kenapa tidak pakai cv2.VideoCapture(url) saja?
      - Backend FFMPEG sering gagal pada 'type=ros_compressed' dari
        web_video_server, dan kalaupun berhasil ia menumpuk buffer sehingga
        gambar tertinggal beberapa detik dari kondisi nyata.

    Kelas ini membaca stream di thread sendiri dan hanya menyimpan frame
    TERAKHIR, jadi loop utama selalu memproses gambar paling baru.

    Dua mode:
      stream   : multipart/x-mixed-replace, frame dipotong per penanda JPEG.
      snapshot : polling '/snapshot?topic=...' berulang (fallback otomatis).
    """

    CHUNK = 16384
    MAX_BUF = 8 * 1024 * 1024      # buang buffer kalau tidak ketemu JPEG utuh
    SOI = b'\xff\xd8'              # penanda awal JPEG
    EOI = b'\xff\xd9'              # penanda akhir JPEG

    def __init__(self, url, timeout=8.0, retry=True, verbose=True):
        self.url = url
        self.timeout = timeout
        self.retry = retry
        self.verbose = verbose

        self.lock = threading.Lock()
        self.frame = None
        self.frame_count = 0
        self.running = False
        self.thread = None
        self.mode = 'stream'
        self.last_error = None

    # ---------- util ----------
    @staticmethod
    def to_snapshot_url(url):
        """web_video_server juga menyediakan /snapshot dengan query yang sama."""
        if '/stream' in url:
            return url.replace('/stream', '/snapshot', 1)
        return None

    def _open(self, url):
        req = Request(url, headers={'User-Agent': 'fusion_webcam/3.0'})
        return urlopen(req, timeout=self.timeout)

    def _publish(self, jpg_bytes):
        img = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return False
        with self.lock:
            self.frame = img
            self.frame_count += 1
        return True

    # ---------- mode stream ----------
    def _run_stream(self):
        resp = self._open(self.url)
        buf = b''
        while self.running:
            chunk = resp.read(self.CHUNK)
            if not chunk:
                raise ConnectionError("stream berhenti, server menutup koneksi")
            buf += chunk

            while True:
                soi = buf.find(self.SOI)
                if soi < 0:
                    if len(buf) > self.MAX_BUF:
                        buf = b''
                    break
                eoi = buf.find(self.EOI, soi + 2)
                if eoi < 0:
                    if soi > 0:
                        buf = buf[soi:]          # buang header multipart di depan
                    if len(buf) > self.MAX_BUF:
                        buf = b''
                    break
                self._publish(buf[soi:eoi + 2])
                buf = buf[eoi + 2:]

    # ---------- mode snapshot ----------
    def _run_snapshot(self, url):
        while self.running:
            data = self._open(url).read()
            if not self._publish(data):
                raise ConnectionError("balasan /snapshot bukan JPEG yang valid")
            time.sleep(0.03)

    # ---------- loop thread pembaca ----------
    def _loop(self):
        snap_url = self.to_snapshot_url(self.url)
        while self.running:
            try:
                if self.mode == 'stream':
                    self._run_stream()
                elif snap_url:
                    self._run_snapshot(snap_url)
                else:
                    raise ConnectionError("mode snapshot tidak tersedia untuk URL ini")
            except Exception as e:
                self.last_error = e
                if not self.running:
                    return
                # Belum pernah dapat frame sama sekali, coba mode snapshot.
                if self.mode == 'stream' and self.frame_count == 0 and snap_url:
                    if self.verbose:
                        print("[kamera] mode stream gagal (%s); mencoba mode snapshot..." % e)
                    self.mode = 'snapshot'
                    continue
                if not self.retry:
                    return
                if self.verbose:
                    print("[kamera] koneksi terputus (%s); menyambung ulang 1 detik lagi..." % e)
                time.sleep(1.0)

    # ---------- API mirip cv2.VideoCapture ----------
    def start(self, wait=10.0):
        """Nyalakan thread pembaca, tunggu sampai frame pertama datang."""
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

        deadline = time.time() + wait
        while time.time() < deadline:
            with self.lock:
                if self.frame is not None:
                    return True
            if not self.thread.is_alive():
                break
            time.sleep(0.05)
        return False

    def isOpened(self):
        with self.lock:
            return self.running and self.frame is not None

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def read_with_id(self):
        """Seperti read(), tapi ikut mengembalikan nomor urut frame.

        Dipakai loop utama untuk melewati frame yang belum berganti, supaya
        Haar cascade dan YOLO tidak dijalankan berulang pada gambar yang sama.
        """
        with self.lock:
            if self.frame is None:
                return False, None, self.frame_count
            return True, self.frame.copy(), self.frame_count

    def release(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.5)


def diagnose_connection(url):
    """Cetak petunjuk kalau kamera robot tidak bisa dibuka."""
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    base = "%s://%s" % (parsed.scheme, parsed.netloc)

    print("\n--- Diagnosa koneksi kamera ---")
    if not host:
        print("  URL tidak bisa dibaca: %s" % url)
        print("-------------------------------\n")
        return

    try:
        sock = socket.create_connection((host, port), timeout=3.0)
        sock.close()
        print("  [OK]    Port %s di %s terbuka, robot terjangkau." % (port, host))
        print("  Berarti masalahnya ada di parameter URL, bukan di jaringan.")
        print("  Buka %s/ di browser untuk melihat daftar topik kamera yang tersedia," % base)
        print("  lalu samakan nilai topic= pada --camera_url dengan yang muncul di sana.")
        print("  Topik yang umum di AiNex: /camera/image_raw atau /usb_cam/image_raw")
    except OSError as e:
        print("  [GAGAL] Tidak bisa membuka %s:%s (%s)" % (host, port, e))
        print("  Periksa satu per satu:")
        print("    1. ping %s" % host)
        print("    2. Kabel LAN terpasang, dan adaptor Ethernet laptop aktif.")
        print("    3. IP laptop harus satu subnet dengan robot, contoh 192.168.50.x")
        print("       Cek dengan: ipconfig")
        print("       Kalau laptop dapat IP 169.254.x.x, set IP statis manual,")
        print("       misal 192.168.50.100 dengan netmask 255.255.255.0")
        print("    4. Node web_video_server di robot sudah jalan:")
        print("       rosrun web_video_server web_video_server")
        print("    5. Firewall Windows tidak memblokir Python.")
    print("-------------------------------\n")


def open_camera(args):
    """Kembalikan objek kamera dengan API read/release/isOpened sesuai argumen."""
    if args.camera_url:
        print("[kamera] Menyambung ke robot: %s" % args.camera_url)
        cam = NetworkCamera(args.camera_url, timeout=args.connect_timeout)
        if not cam.start(wait=args.connect_timeout):
            err = cam.last_error
            cam.release()
            print("ERROR: Tidak ada frame dari kamera robot dalam %.0f detik." % args.connect_timeout)
            if err is not None:
                print("       Penyebab terakhir: %s" % err)
            diagnose_connection(args.camera_url)
            return None
        _, probe = cam.read()
        h, w = probe.shape[:2]
        print("[kamera] Tersambung (mode %s), resolusi %dx%d." % (cam.mode, w, h))
        return cam

    print("[kamera] Membuka webcam lokal index %d..." % args.camera)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: Webcam index %d tidak bisa dibuka." % args.camera)
        print("       Coba index lain (--camera 1), atau pakai kamera robot dengan --camera_url")
        return None
    return cap


def transform_frame(frame, flip, rotate, width):
    if rotate == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotate == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if flip:
        frame = cv2.flip(frame, 1)
    if width and frame.shape[1] != width:
        scale = width / float(frame.shape[1])
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        frame = cv2.resize(frame, (width, int(round(frame.shape[0] * scale))),
                           interpolation=interp)
    return frame


# ===================================================================
# FUSION
# ===================================================================
def decision_level_fusion(p_visual, p_audio, alpha):
    return alpha * p_visual + (1.0 - alpha) * p_audio


def compound_emotion_decision(p_final, tau, tau_conf):
    idx = np.argsort(p_final)[::-1]
    t1, t2 = idx[0], idx[1]
    p1, p2 = p_final[t1], p_final[t2]
    gap = p1 - p2

    if p1 < tau_conf:
        return {'label_id': 'Netral', 'type': 'TIDAK YAKIN',
                'c1': t1, 'c2': t2, 'p1': p1, 'p2': p2, 'gap': gap}
    elif gap >= tau:
        return {'label_id': EMOTIONS_ID[EMOTIONS[t1]], 'type': 'TUNGGAL',
                'c1': t1, 'c2': t2, 'p1': p1, 'p2': p2, 'gap': gap}
    elif EMOTIONS[t1] in NO_COMPOUND or EMOTIONS[t2] in NO_COMPOUND:
        # Netral bukan ekspresi yang bisa dicampur. "Sedih-Netral" sebenarnya
        # berarti "agak sedih", dan itu sudah diwakili label tunggalnya.
        return {'label_id': EMOTIONS_ID[EMOTIONS[t1]], 'type': 'TUNGGAL',
                'c1': t1, 'c2': t2, 'p1': p1, 'p2': p2, 'gap': gap}
    else:
        lbl = "%s-%s" % (EMOTIONS_ID[EMOTIONS[t1]], EMOTIONS_ID[EMOTIONS[t2]])
        return {'label_id': lbl, 'type': 'MAJEMUK',
                'c1': t1, 'c2': t2, 'p1': p1, 'p2': p2, 'gap': gap}


# ===================================================================
# DETEKSI WAJAH
# ===================================================================
def get_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("Haar cascade gagal dimuat.")
    return detector


def detect_largest_face(detector, frame, min_size=80):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                      minSize=(min_size, min_size))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def crop_face(frame, box, pad=FACE_PAD):
    x, y, w, h = box
    px, py = int(w * pad), int(h * pad)
    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(frame.shape[1], x + w + px)
    y2 = min(frame.shape[0], y + h + py)
    return frame[y1:y2, x1:x2]


# ===================================================================
# TAMPILAN
# ===================================================================
TYPE_COLOR = {
    'TUNGGAL': (0, 220, 0),
    'MAJEMUK': (0, 165, 255),
    'TIDAK YAKIN': (160, 160, 160)
}


def draw_overlay(frame, p_visual, p_audio, p_final, result, alpha, tau,
                 fps, audio_level, face_box, speech_on=True, source_label="",
                 robot=None):
    h, w = frame.shape[:2]
    color = TYPE_COLOR.get(result['type'], (255, 255, 255))

    if face_box is not None:
        x, y, fw, fh = face_box
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
        cv2.rectangle(frame, (x, y - 28), (x + fw, y), color, -1)
        cv2.putText(frame, result['label_id'], (x + 4, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    else:
        cv2.putText(frame, "WAJAH TIDAK TERDETEKSI", (max(10, w // 2 - 180), 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (330, h), (25, 25, 25), -1)
    frame = cv2.addWeighted(panel, 0.55, frame, 0.45, 0)

    cv2.putText(frame, result['label_id'], (12, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    cv2.putText(frame, "[%s]  gap=%.3f" % (result['type'], result['gap']), (12, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    y0 = 105
    cv2.putText(frame, "%-9s%6s%6s%6s" % ('Kelas', 'Vis', 'Aud', 'Fus'), (12, y0 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    for i, emo in enumerate(EMOTIONS):
        y = y0 + i * 34
        is_top1 = (i == result['c1'])
        txt_color = color if is_top1 else (230, 230, 230)
        aud_txt = ("%6.2f" % p_audio[i]) if speech_on else "%6s" % "--"
        line = "%-9s%6.2f%s%6.2f" % (EMOTIONS_ID[emo], p_visual[i], aud_txt, p_final[i])
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, txt_color, 1)
        bar_len = int(p_final[i] * 180)
        cv2.rectangle(frame, (12, y + 6), (12 + bar_len, y + 14),
                      color if is_top1 else (110, 110, 110), -1)

    yb = y0 + 7 * 34 + 16
    cv2.putText(frame, "alpha=%.2f  tau=%.2f  conf>=%.2f  input=%s"
                % (alpha, tau, TAU_CONF, "gray" if GRAYSCALE_INPUT else "warna"),
                (12, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 255), 1)

    if speech_on:
        mic_txt = "mic: %-15s" % ('#' * min(int(audio_level * 400), 15))
    else:
        mic_txt = "mic: OFF (--no_speech)"
    cv2.putText(frame, "FPS: %.1f   %s" % (fps, mic_txt),
                (12, yb + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 255, 180), 1)

    if source_label:
        cv2.putText(frame, source_label, (12, yb + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 200, 255), 1)

    keys = "q=keluar  t/g=tau" if not speech_on else "q=keluar  a/z=alpha  t/g=tau"
    cv2.putText(frame, keys, (12, yb + 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # Status robot (kanan atas, tidak menutup panel kiri)
    if robot is not None and robot.enabled:
        remaining = robot.cooldown_remaining()
        if remaining <= 0.0:
            rbt_txt = "ROBOT: siap lambai"
            rbt_color = (0, 220, 0)
        else:
            rbt_txt = "ROBOT: cooldown %.0fs" % remaining
            rbt_color = (0, 165, 255)
        (tw, th_) = cv2.getTextSize(rbt_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        cv2.rectangle(frame, (w - tw - 20, 12), (w - 8, 12 + th_ + 12),
                      (25, 25, 25), -1)
        cv2.putText(frame, rbt_txt, (w - tw - 14, 12 + th_ + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, rbt_color, 2)

    return frame


# ===================================================================
# CLI
# ===================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="fusion_webcam.py",
        description="Multimodal emotion recognition realtime, dari webcam laptop atau kamera robot AiNex Hiwonder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Contoh kamera robot AiNex lewat LAN:\n"
            "  python fusion_webcam.py "
            "--camera_url \"http://192.168.50.2:8080/stream?topic=/camera/image_raw&type=ros_compressed\" "
            "--no_speech\n"
        ),
    )

    g_src = p.add_argument_group("Sumber gambar")
    g_src.add_argument('--camera_url', '--camera-url', dest='camera_url', default=None,
                       metavar='URL',
                       help="URL stream MJPEG / ROS web_video_server dari robot. "
                            "Di PowerShell apit dengan tanda kutip ganda karena ada karakter &.")
    g_src.add_argument('--camera', dest='camera', type=int, default=0, metavar='N',
                       help="Index webcam lokal, dipakai kalau --camera_url tidak diisi (default: 0).")
    g_src.add_argument('--rotate', type=int, choices=[0, 90, 180, 270], default=0,
                       help="Putar frame searah jarum jam, untuk kamera robot yang terpasang miring.")
    g_src.add_argument('--flip', dest='flip', action='store_true', default=None,
                       help="Paksa cermin horizontal.")
    g_src.add_argument('--no_flip', '--no-flip', dest='flip', action='store_false',
                       help="Paksa tanpa cermin. Default sudah tanpa cermin untuk --camera_url.")
    g_src.add_argument('--width', type=int, default=0, metavar='PX',
                       help="Skala frame ke lebar tertentu sebelum diproses (0 = biarkan asli).")
    g_src.add_argument('--connect_timeout', '--connect-timeout', dest='connect_timeout',
                       type=float, default=10.0, metavar='DETIK',
                       help="Batas tunggu frame pertama dari kamera robot (default: 10).")
    g_src.add_argument('--min_face', '--min-face', dest='min_face', type=int, default=80,
                       metavar='PX',
                       help="Ukuran wajah minimum untuk Haar cascade (default: 80). "
                            "Turunkan kalau resolusi kamera robot kecil.")

    g_aud = p.add_argument_group("Audio")
    g_aud.add_argument('--no_speech', '--no-speech', dest='no_speech', action='store_true',
                       help="Jalankan visual-only: tanpa mikrofon, tanpa model audio, alpha dikunci 1.0.")

    g_fus = p.add_argument_group("Parameter fusion")
    g_fus.add_argument('--alpha', type=float, default=ALPHA,
                       help="Bobot modalitas visual pada Persamaan 2.1 (default: %.2f)." % ALPHA)
    g_fus.add_argument('--tau', type=float, default=TAU,
                       help="Ambang gap untuk emosi majemuk, Persamaan 2.2 (default: %.2f)." % TAU)
    g_fus.add_argument('--tau_conf', '--tau-conf', dest='tau_conf', type=float, default=TAU_CONF,
                       help="Ambang keyakinan minimum (default: %.2f)." % TAU_CONF)

    g_mdl = p.add_argument_group("Model")
    # default=None supaya konstanta GRAYSCALE_INPUT di atas yang menentukan
    # kalau tidak ada argumen yang dipakai.
    g_mdl.add_argument('--grayscale', dest='grayscale', action='store_true',
                       default=None,
                       help="Paksa potongan wajah jadi grayscale sebelum masuk "
                            "model (default: %s)." % ("aktif" if GRAYSCALE_INPUT
                                                      else "nonaktif"))
    g_mdl.add_argument('--no_grayscale', '--no-grayscale', dest='grayscale',
                       action='store_false',
                       help="Kirim potongan wajah apa adanya (berwarna). "
                            "Tampilan di layar tidak terpengaruh keduanya.")
    g_mdl.add_argument('--visual_model', '--visual-model', dest='visual_model',
                       default=VISUAL_MODEL_PATH, help="Path bobot YOLO klasifikasi wajah.")
    g_mdl.add_argument('--audio_model', '--audio-model', dest='audio_model',
                       default=AUDIO_MODEL_PATH, help="Path bobot CNN-MFCC audio.")
    g_mdl.add_argument('--device', default=DEVICE, help="cpu atau cuda (default: cpu).")
    g_mdl.add_argument('--imgsz', type=int, default=224, help="Ukuran input YOLO (default: 224).")

    g_rbt = p.add_argument_group("Respons robot (AiNex Hiwonder)")
    g_rbt.add_argument('--wave_url', '--wave-url', dest='wave_url', default=None,
                       metavar='URL',
                       help="Endpoint HTTP di robot yang memicu action group 'wave'. "
                            "Kalau kosong, sistem hanya mencetak '[ROBOT] WAVE!' (dry-run).")
    g_rbt.add_argument('--wave_method', '--wave-method', dest='wave_method',
                       choices=['GET', 'POST'], default='POST',
                       help="Metode HTTP untuk --wave_url (default: POST).")
    g_rbt.add_argument('--wave_cooldown', '--wave-cooldown', dest='wave_cooldown',
                       type=float, default=60.0, metavar='DETIK',
                       help="Jeda minimal antar-lambaian (default: 60 detik).")
    g_rbt.add_argument('--wave_timeout', '--wave-timeout', dest='wave_timeout',
                       type=float, default=2.0, metavar='DETIK',
                       help="Timeout request HTTP wave (default: 2 detik).")
    g_rbt.add_argument('--wave_token', '--wave-token', dest='wave_token',
                       default=None, metavar='TOKEN',
                       help="Shared secret yang dikirim sebagai header "
                            "'Authorization: Bearer TOKEN'. Isi kalau server di "
                            "robot (Servers/ainex_wave_server.py) dijalankan "
                            "dengan --token.")
    g_rbt.add_argument('--no_wave', '--no-wave', dest='no_wave', action='store_true',
                       help="Matikan respons robot sepenuhnya.")

    args = p.parse_args(argv)

    # Kamera robot tidak dicermin; webcam laptop dicermin, sama seperti versi lama.
    if args.flip is None:
        args.flip = args.camera_url is None

    return args


# ===================================================================
# MAIN
# ===================================================================
def main(argv=None):
    global ALPHA, TAU, TAU_CONF, DEVICE, GRAYSCALE_INPUT

    args = parse_args(argv)
    ALPHA = 1.0 if args.no_speech else args.alpha
    if args.grayscale is not None:
        GRAYSCALE_INPUT = args.grayscale
    TAU = args.tau
    TAU_CONF = args.tau_conf
    DEVICE = args.device

    total = 3 if args.no_speech else 4

    print("[1/%d] Load model visual (YOLO)..." % total)
    visual_model = YOLO(args.visual_model)

    print("[2/%d] Load face detector (Haar Cascade)..." % total)
    face_detector = get_face_detector()

    if args.no_speech:
        print("[audio] --no_speech aktif: mikrofon dan model audio dilewati, alpha dikunci 1.00.")
        worker = DummyAudio()
    else:
        if sd is None:
            print("ERROR: modul sounddevice tidak tersedia (%s)." % _SD_ERR)
            print("       Pasang dengan: pip install sounddevice")
            print("       Atau jalankan dengan --no_speech")
            return 1
        if librosa is None:
            print("ERROR: modul librosa tidak tersedia (%s)." % _LIBROSA_ERR)
            print("       Pasang dengan: pip install librosa")
            print("       Atau jalankan dengan --no_speech")
            return 1
        print("[3/%d] Load model audio (CNN-MFCC)..." % total)
        audio_model = load_audio_cnn(args.audio_model, DEVICE)
        worker = AudioWorker(audio_model)

    print("[%d/%d] Siapkan sumber gambar..." % (total, total))
    cap = open_camera(args)
    if cap is None:
        return 1

    robot = RobotController(
        wave_url=args.wave_url,
        method=args.wave_method,
        cooldown=args.wave_cooldown,
        timeout=args.wave_timeout,
        enabled=not args.no_wave and WAVE_ENABLED,
        token=args.wave_token,
    )
    if not WAVE_ENABLED:
        print("[robot] Respons robot DIMATIKAN di kode "
              "(WAVE_ENABLED = False di bagian KONFIGURASI).")
    elif args.no_wave:
        print("[robot] Respons robot DIMATIKAN (--no_wave).")
    elif args.wave_url is None:
        print("[robot] Mode dry-run: setiap 'happy' hanya cetak '[ROBOT] WAVE!' "
              "(cooldown %.0fs). Isi --wave_url untuk kirim ke robot." % args.wave_cooldown)
    else:
        print("[robot] Wave -> %s %s (cooldown %.0fs)." %
              (args.wave_method, args.wave_url, args.wave_cooldown))

    try:
        worker.start()
    except Exception as e:
        print("ERROR: mikrofon gagal dibuka (%s). Jalankan ulang dengan --no_speech." % e)
        cap.release()
        return 1

    if args.camera_url:
        source_label = "robot: %s" % urlparse(args.camera_url).netloc
    else:
        source_label = "webcam lokal #%d" % args.camera

    prev_t = time.time()
    fps = 0.0
    last_p_visual = np.ones(len(EMOTIONS)) / len(EMOTIONS)
    empty_reads = 0
    last_frame_id = -1
    streaming = hasattr(cap, 'read_with_id')

    def handle_key(key):
        """Tangani tombol. Kembalikan False kalau diminta keluar."""
        global ALPHA, TAU
        if key == ord('q'):
            return False
        elif key == ord('a') and not args.no_speech:
            ALPHA = min(1.0, ALPHA + 0.05)
        elif key == ord('z') and not args.no_speech:
            ALPHA = max(0.0, ALPHA - 0.05)
        elif key == ord('t'):
            TAU = min(0.9, TAU + 0.05)
        elif key == ord('g'):
            TAU = max(0.0, TAU - 0.05)
        return True

    if args.no_speech:
        print("Berjalan! Posisikan wajah ke kamera robot.\n")
    else:
        print("Berjalan! Posisikan wajah ke kamera & bicara ke mikrofon.\n")

    try:
        while True:
            if streaming:
                ret, frame, frame_id = cap.read_with_id()
                # Stream robot (sekitar 20-25 fps) lebih lambat dari loop ini.
                # Lewati frame yang belum berganti supaya Haar cascade dan YOLO
                # tidak dijalankan berulang kali pada gambar yang sama persis.
                if ret and frame_id == last_frame_id:
                    if not handle_key(cv2.waitKey(5) & 0xFF):
                        break
                    continue
                last_frame_id = frame_id
            else:
                ret, frame = cap.read()

            if not ret or frame is None:
                empty_reads += 1
                # Kamera jaringan boleh kosong sesaat saat menyambung ulang.
                if args.camera_url and empty_reads < 250:
                    if not handle_key(cv2.waitKey(20) & 0xFF):
                        break
                    continue
                print("Sumber gambar berhenti mengirim frame.")
                break
            empty_reads = 0

            frame = transform_frame(frame, args.flip, args.rotate, args.width)

            face_box = detect_largest_face(face_detector, frame, args.min_face)

            if face_box is not None:
                face_img = crop_face(frame, face_box)
                if face_img.size > 0:
                    if GRAYSCALE_INPUT:
                        abu = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                        face_img = cv2.cvtColor(abu, cv2.COLOR_GRAY2BGR)
                    results = visual_model.predict(face_img, imgsz=args.imgsz, verbose=False)
                    last_p_visual = results[0].probs.data.cpu().numpy()
            p_visual = last_p_visual

            p_audio = worker.p_audio

            p_final = decision_level_fusion(p_visual, p_audio, ALPHA)
            result = compound_emotion_decision(p_final, TAU, TAU_CONF)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
            prev_t = now

            # ----------------------------------------------------------------
            # RESPONS ROBOT DIMATIKAN SEMENTARA
            # ----------------------------------------------------------------
            # Untuk menghidupkan lagi: hapus tanda '#' pada tiga baris di bawah,
            # DAN ubah WAVE_ENABLED jadi True di bagian KONFIGURASI paling atas.
            #
            # Perilakunya: lambai kalau ekspresi TUNGGAL & top-1 = happy.
            # Kondisi TUNGGAL memastikan tidak memicu pada ekspresi majemuk
            # (misal happy-surprise) atau saat model tidak yakin.
            #
            # if (result['type'] == 'TUNGGAL'
            #         and EMOTIONS[result['c1']] == 'happy'):
            #     robot.trigger_wave()

            frame = draw_overlay(frame, p_visual, p_audio, p_final,
                                 result, ALPHA, TAU, fps, worker.audio_level, face_box,
                                 speech_on=not args.no_speech, source_label=source_label,
                                 robot=robot)
            cv2.imshow('Multimodal Emotion Fusion (Realtime)', frame)

            if not handle_key(cv2.waitKey(1) & 0xFF):
                break
    except KeyboardInterrupt:
        print("\nDihentikan pengguna.")
    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("Selesai.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
