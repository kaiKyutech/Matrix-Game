import argparse
import base64
import os
import queue
import threading
import time
from pathlib import Path

import cv2

from flask import Flask, jsonify, render_template_string, request, send_file, send_from_directory
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO
from inference_streaming import InteractiveGameInference
from utils.action_provider import SocketIOActionProvider
from utils.misc import set_seed


HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Matrix-Game 2.0 Web Controller</title>
  <style>
    body { margin: 0; font-family: system-ui, sans-serif; background: #101114; color: #f5f5f5; }
    main { max-width: 1120px; margin: 0 auto; padding: 24px; }
    .panel { background: #1b1d23; border: 1px solid #30333d; border-radius: 12px; padding: 16px; margin-bottom: 16px; }
    input, button { font-size: 16px; padding: 10px; border-radius: 8px; border: 1px solid #40444f; }
    input[type="text"] { width: min(720px, 100%); background: #111318; color: #f5f5f5; }
    input[type="file"] { background: #111318; color: #f5f5f5; }
    button { cursor: pointer; background: #2b6df6; color: white; }
    button.stop { background: #c43b3b; }
    canvas, img.preview { width: 100%; background: #050505; border-radius: 12px; }
    canvas { display: none; }
    img.preview { object-fit: contain; max-height: 420px; }
    kbd { background: #30333d; padding: 2px 6px; border-radius: 4px; }
    .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .hint { color: #adb3c2; line-height: 1.5; }
    .active { color: #78d98b; font-weight: 700; }
    #status { color: #9fd39f; }
    #actionState { margin-top: 12px; }
  </style>
</head>
<body>
<main>
  <h1>Matrix-Game 2.0 Web Controller</h1>
  <div class="panel">
    <p class="hint">Initial image / live generated frames:</p>
    <img id="preview" class="preview" alt="initial image preview">
    <canvas id="frameCanvas"></canvas>
  </div>
  <div class="panel">
    <p>Use <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> for movement and <kbd>I</kbd><kbd>J</kbd><kbd>K</kbd><kbd>L</kbd> for camera. Mouse drag on the page also sends camera deltas.</p>
    <p class="hint">Important: the first run may spend several minutes compiling/autotuning before the first video appears. Key and mouse inputs are queued for the next generated chunk, so this page shows what the server has received.</p>
    <div class="row">
      <input id="imgPath" type="text" value="demo_images/universal/0000.png" placeholder="Image path on the server">
      <button id="startBtn">Start</button>
      <button id="stopBtn" class="stop">Stop</button>
    </div>
    <div class="row">
      <input id="imageFile" type="file" accept="image/png,image/jpeg,image/webp">
      <button id="uploadBtn">Upload Image</button>
    </div>
    <p id="status">Idle</p>
    <div id="actionState" class="hint">Active keys: none | mouse delta: 0, 0</div>
  </div>
</main>
<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
<script>
const socket = io();
const statusEl = document.getElementById('status');
const canvasEl = document.getElementById('frameCanvas');
const canvasCtx = canvasEl.getContext('2d');
const previewEl = document.getElementById('preview');
const imgPathEl = document.getElementById('imgPath');
const imageFileEl = document.getElementById('imageFile');
const actionStateEl = document.getElementById('actionState');
const activeKeys = new Set();

function setStatus(text) { statusEl.textContent = text; }
function sendKey(key, pressed) { socket.emit('key', {key, pressed}); }
function setActionState(data) {
  const keys = data.keys && data.keys.length ? data.keys.join(' ').toUpperCase() : 'none';
  const delta = data.mouse_delta || [0, 0];
  const dx = Number(delta[0] || 0).toFixed(1);
  const dy = Number(delta[1] || 0).toFixed(1);
  actionStateEl.innerHTML = `Active keys: <span class="active">${keys}</span> | mouse delta: ${dx}, ${dy}`;
}

window.addEventListener('keydown', (event) => {
  const key = event.key.toLowerCase();
  if (!['w','a','s','d','q','i','j','k','l','u','z','c'].includes(key)) return;
  event.preventDefault();
  if (!activeKeys.has(key)) {
    activeKeys.add(key);
    sendKey(key, true);
  }
});
window.addEventListener('keyup', (event) => {
  const key = event.key.toLowerCase();
  if (!activeKeys.has(key)) return;
  event.preventDefault();
  activeKeys.delete(key);
  sendKey(key, false);
});
window.addEventListener('mousemove', (event) => {
  if (event.buttons !== 1) return;
  socket.emit('mouse_delta', {dx: event.movementX, dy: event.movementY});
});

document.getElementById('uploadBtn').onclick = async () => {
  const file = imageFileEl.files[0];
  if (!file) {
    setStatus('Choose an image file first.');
    return;
  }
  setStatus('Uploading image...');
  const formData = new FormData();
  formData.append('image', file);
  const res = await fetch('/upload', {method: 'POST', body: formData});
  const data = await res.json();
  if (!res.ok) {
    setStatus(data.message || JSON.stringify(data));
    return;
  }
  imgPathEl.value = data.path;
  previewEl.style.display = 'block';
  previewEl.src = '/preview?path=' + encodeURIComponent(data.path) + '&t=' + Date.now();
  setStatus('Uploaded image: ' + data.path);
};

document.getElementById('startBtn').onclick = async () => {
  imgPathEl.blur();
  canvasEl.style.display = 'none';
  previewEl.style.display = 'block';
  previewEl.src = '/preview?path=' + encodeURIComponent(imgPathEl.value) + '&t=' + Date.now();
  setStatus('Starting...');
  const res = await fetch('/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({img_path: imgPathEl.value})
  });
  const data = await res.json();
  setStatus(data.message || JSON.stringify(data));
};

document.getElementById('stopBtn').onclick = async () => {
  const res = await fetch('/stop', {method: 'POST'});
  const data = await res.json();
  setStatus(data.message || JSON.stringify(data));
};

socket.on('status', (data) => setStatus(data.message));
socket.on('action_state', (data) => setActionState(data));
socket.on('frame', (data) => {
  const image = new Image();
  image.onload = () => {
    if (canvasEl.width !== image.width || canvasEl.height !== image.height) {
      canvasEl.width = image.width;
      canvasEl.height = image.height;
    }
    previewEl.style.display = 'none';
    canvasEl.style.display = 'block';
    canvasCtx.drawImage(image, 0, 0);
  };
  image.src = 'data:image/jpeg;base64,' + data.jpeg;
});
</script>
</body>
</html>
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/inference_yaml/inference_universal.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--output_folder", type=str, default="outputs/")
    parser.add_argument("--max_num_output_frames", type=int, default=360)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrained_model_path", type=str, default="Matrix-Game-2.0")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--stream_fps", type=float, default=12.0)
    parser.add_argument("--jpeg_quality", type=int, default=85)
    return parser.parse_args()


class MatrixGameWebApp:
    def __init__(self, args):
        self.args = args
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode="threading")
        self.action_provider = SocketIOActionProvider(device="cuda")
        self.pipeline = None
        self.mode = None
        self.worker = None
        self.stop_event = threading.Event()
        self.current_video_url = None
        self.upload_folder = Path("uploads")
        self.frame_queue = None
        self.frame_streamer = None
        self._lock = threading.Lock()
        self._setup_routes()
        self._setup_socketio()

    def run(self):
        self._validate_startup_paths()
        set_seed(self.args.seed)
        os.makedirs(self.args.output_folder, exist_ok=True)
        self.upload_folder.mkdir(parents=True, exist_ok=True)
        self.socketio.run(self.app, host=self.args.host, port=self.args.port, allow_unsafe_werkzeug=True)

    def _validate_startup_paths(self):
        missing_paths = []
        if not Path(self.args.config_path).is_file():
            missing_paths.append(f"--config_path not found: {self.args.config_path}")
        if self.args.checkpoint_path and not Path(self.args.checkpoint_path).is_file():
            missing_paths.append(f"--checkpoint_path not found: {self.args.checkpoint_path}")
        if not Path(self.args.pretrained_model_path).is_dir():
            missing_paths.append(f"--pretrained_model_path not found: {self.args.pretrained_model_path}")
        if missing_paths:
            message = "Invalid startup path(s):\n" + "\n".join(f"  - {path}" for path in missing_paths)
            raise FileNotFoundError(message)

    def _setup_routes(self):
        @self.app.get("/")
        def index():
            return render_template_string(HTML)

        @self.app.post("/upload")
        def upload():
            uploaded = request.files.get("image")
            if uploaded is None or uploaded.filename == "":
                return jsonify({"ok": False, "message": "image file is required"}), 400
            filename = secure_filename(uploaded.filename)
            suffix = Path(filename).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                return jsonify({"ok": False, "message": "image must be .jpg, .jpeg, .png, or .webp"}), 400
            target = self.upload_folder / filename
            counter = 1
            while target.exists():
                target = self.upload_folder / f"{Path(filename).stem}_{counter}{suffix}"
                counter += 1
            uploaded.save(target)
            return jsonify({"ok": True, "path": str(target)})

        @self.app.post("/start")
        def start():
            payload = request.get_json(silent=True) or {}
            img_path = payload.get("img_path", "").strip()
            if not img_path:
                return jsonify({"ok": False, "message": "img_path is required"}), 400
            with self._lock:
                if self.worker is not None and self.worker.is_alive():
                    return jsonify({"ok": False, "message": "Generation is already running"}), 409
                self.stop_event.clear()
                self.action_provider.clear()
                self.frame_queue = queue.Queue(maxsize=240)
                self.frame_streamer = threading.Thread(target=self._stream_frames, daemon=True)
                self.frame_streamer.start()
                self.worker = threading.Thread(target=self._run_generation, args=(img_path,), daemon=True)
                self.worker.start()
            return jsonify({"ok": True, "message": f"Started: {img_path}"})

        @self.app.post("/stop")
        def stop():
            self.stop_event.set()
            self.action_provider.clear()
            return jsonify({"ok": True, "message": "Stop requested"})

        @self.app.get("/status")
        def status():
            running = self.worker is not None and self.worker.is_alive()
            return jsonify({"running": running, "video_url": self.current_video_url})

        @self.app.get("/preview")
        def preview():
            img_path = request.args.get("path", "").strip()
            if not img_path:
                return jsonify({"ok": False, "message": "path is required"}), 400
            path = Path(img_path)
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                return jsonify({"ok": False, "message": "preview path must be an image file"}), 400
            if not path.is_file():
                return jsonify({"ok": False, "message": f"image not found: {img_path}"}), 404
            return send_file(path)

        @self.app.get("/outputs/<path:filename>")
        def outputs(filename):
            return send_from_directory(self.args.output_folder, filename)

    def _setup_socketio(self):
        @self.socketio.on("key")
        def on_key(data):
            self.action_provider.set_key(data.get("key", ""), bool(data.get("pressed")))
            self._emit_action_state()

        @self.socketio.on("mouse_delta")
        def on_mouse_delta(data):
            self.action_provider.set_mouse_delta(data.get("dx", 0), data.get("dy", 0))
            self._emit_action_state()

    def _run_generation(self, img_path):
        self._emit_status("Loading models..." if self.pipeline is None else "Starting generation...")
        try:
            if self.pipeline is None:
                self.pipeline = InteractiveGameInference(self.args)
                self.mode = self.pipeline.config.pop('mode')
            self._emit_status("Generating. Use WASD/IJKL or drag mouse.")
            self.pipeline.generate_videos(
                self.mode,
                img_path=img_path,
                action_provider=self.action_provider,
                should_continue=lambda: not self.stop_event.is_set(),
                progress_callback=self._on_progress,
                frame_callback=self._on_frames,
            )
            self._emit_status("Generation finished")
        except Exception as exc:
            self._emit_status(f"Generation failed: {exc}")
        finally:
            self.stop_event.set()
            self._stop_frame_stream()

    def _on_frames(self, frames, current_start_frame, num_blocks):
        if self.frame_queue is None:
            return
        for frame in frames:
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)

    def _stop_frame_stream(self):
        if self.frame_queue is None:
            return
        try:
            self.frame_queue.put_nowait(None)
        except queue.Full:
            self.frame_queue.get_nowait()
            self.frame_queue.put_nowait(None)

    def _stream_frames(self):
        interval = 1.0 / max(self.args.stream_fps, 0.1)
        while True:
            frame = self.frame_queue.get()
            if frame is None:
                break
            success, encoded = cv2.imencode(
                ".jpg",
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), int(self.args.jpeg_quality)],
            )
            if success:
                jpeg = base64.b64encode(encoded).decode("ascii")
                self.socketio.emit("frame", {"jpeg": jpeg})
            time.sleep(interval)

    def _on_progress(self, output_path, current_start_frame, num_blocks):
        filename = Path(output_path).name
        self.current_video_url = f"/outputs/{filename}"
        self._emit_status(f"Generated through latent frame {current_start_frame}; total blocks: {num_blocks}")

    def _emit_status(self, message):
        print(message, flush=True)
        self.socketio.emit("status", {"message": message})

    def _emit_action_state(self):
        self.socketio.emit("action_state", self.action_provider.snapshot())


if __name__ == "__main__":
    MatrixGameWebApp(parse_args()).run()
