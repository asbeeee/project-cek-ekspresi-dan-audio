#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ainex_wave_server.py - jembatan HTTP di sisi robot AiNex Hiwonder.

File ini TIDAK dijalankan di laptop. Salin ke Raspberry Pi robot, jalankan di
sana, lalu dari laptop panggil:

    python fusion_webcam.py --camera_url ... --wave_url http://<IP_ROBOT>:5000/wave

Di robot (Pi 5 / image Docker):
    docker exec -it -u ubuntu -w /home/ubuntu <container_id> /bin/bash
    python3 ainex_wave_server.py --action wave

Di robot (Pi 4 / image ROS1 non-Docker):
    python3 ainex_wave_server.py --action wave

Catatan "token":
  Robot AiNex TIDAK butuh token apa pun untuk menggerakkan servo atau
  menjalankan action group. Token/API key di dokumentasi Hiwonder hanya untuk
  fitur AI Large Model (chat LLM + TTS), disimpan di
  /home/ubuntu/large_models/config.py sebagai llm_api_key / vllm_api_key.
  Kalau tetap ingin endpoint ini terkunci, jalankan dengan --token RAHASIA
  lalu di laptop pakai --wave_token RAHASIA. Itu shared secret buatan sendiri,
  bukan token dari Hiwonder.
"""
import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --------------------------------------------------------------------------
# Backend gerak: coba beberapa API Hiwonder, pakai yang tersedia di image.
# --------------------------------------------------------------------------
_LOCK = threading.Lock()
_BUSY = False


def make_runner(action_name):
    """Kembalikan fungsi run() yang menjalankan action group `action_name`."""

    # 1) AiNex ROS (Pi 5 / Pi 4 image resmi) - paling umum.
    try:
        from ainex_kinematics.motion_manager import MotionManager
        mm = MotionManager()

        def run():
            mm.run_action(action_name)

        print("[backend] ainex_kinematics.motion_manager.MotionManager")
        return run
    except Exception as e:
        print("[backend] MotionManager tidak tersedia: %s" % e)

    # 2) SDK Hiwonder generik (TonyPi/SpiderPi style), kadang ikut terpasang.
    try:
        import hiwonder.ActionGroupControl as AGC

        def run():
            AGC.runActionGroup(action_name)

        print("[backend] hiwonder.ActionGroupControl")
        return run
    except Exception as e:
        print("[backend] ActionGroupControl tidak tersedia: %s" % e)

    # 3) Fallback: panggil pemutar action group lewat subprocess.
    import subprocess

    def run():
        subprocess.run(['python3', '-c',
                        'from ainex_kinematics.motion_manager import MotionManager;'
                        'MotionManager().run_action("%s")' % action_name],
                       check=True)

    print("[backend] subprocess fallback")
    return run


def make_handler(runner, token, action_name):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def _reply(self, code, msg):
            body = ('{"status": "%s"}' % msg).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self):
            if not token:
                return True
            hdr = self.headers.get('Authorization', '')
            return hdr == 'Bearer %s' % token

        def _handle(self):
            global _BUSY
            if self.path.split('?')[0] not in ('/wave', '/'):
                self._reply(404, 'not found')
                return
            if not self._authorized():
                self._reply(401, 'unauthorized')
                return
            # Buang body kalau ada, supaya koneksi keep-alive tidak macet.
            n = int(self.headers.get('Content-Length') or 0)
            if n:
                self.rfile.read(n)

            with _LOCK:
                if _BUSY:
                    self._reply(429, 'busy')
                    return
                _BUSY = True

            self._reply(200, 'ok')  # balas dulu, gerak belakangan

            def worker():
                global _BUSY
                try:
                    print("[robot] menjalankan action '%s'" % action_name)
                    runner()
                    print("[robot] selesai")
                except Exception as e:
                    print("[robot] GAGAL: %s" % e)
                finally:
                    with _LOCK:
                        _BUSY = False

            threading.Thread(target=worker, daemon=True).start()

        do_GET = _handle
        do_POST = _handle

        def log_message(self, fmt, *a):
            pass  # senyapkan log bawaan

    return Handler


def main():
    p = argparse.ArgumentParser(
        description="Server HTTP kecil di robot AiNex: POST /wave -> jalankan action group.")
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=5000)
    p.add_argument('--action', default='wave',
                   help="Nama action group tanpa ekstensi .d6a (default: wave). "
                        "Daftar action ada di ~/software/ainex_controller/ActionGroups/")
    p.add_argument('--token', default=None,
                   help="Shared secret opsional. Kalau diisi, request wajib "
                        "membawa header 'Authorization: Bearer <token>'.")
    args = p.parse_args()

    runner = make_runner(args.action)
    handler = make_handler(runner, args.token, args.action)
    srv = ThreadingHTTPServer((args.host, args.port), handler)
    print("[server] siap di http://%s:%d/wave (action='%s', token=%s)"
          % (args.host, args.port, args.action, 'ya' if args.token else 'tidak'))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] berhenti")


if __name__ == '__main__':
    main()
