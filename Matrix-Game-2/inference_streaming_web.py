import argparse
import os
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request, send_from_directory
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
    input { width: min(720px, 100%); background: #111318; color: #f5f5f5; }
    button { cursor: pointer; background: #2b6df6; color: white; }
    button.stop { background: #c43b3b; }
    video { width: 100%; background: #050505; border-radius: 12px; }
    kbd { background: #30333d; padding: 2px 6px; border-radius: 4px; }
    .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    #status { color: #9fd39f; }
  </style>
</head>
<body>
<main>
  <h1>Matrix-Game 2.0 Web Controller</h1>
  <div class="panel">
    <p>Use <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> for movement and <kbd>I</kbd><kbd>J</kbd><kbd>K</kbd><kbd>L</kbd> for camera. Mouse drag on the page also sends camera deltas.</p>
    <div class="row">
      <input id="imgPath" value="demo_images/universal/0000.png" placeholder="Image path on the server">
      <button id="startBtn">Start</button>
      <button id="stopBtn" class="stop">Stop</button>
    </div>
    <p id="status">Idle</p>
  </div>
  <div class="panel">
    <video id="video" controls autoplay muted loop playsinline></video>
  </div>
</main>
<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
<script>
const socket = io();
const statusEl = document.getElementById('status');
const videoEl = document.getElementById('video');
const imgPathEl = document.getElementById('imgPath');
const activeKeys = new Set();

function setStatus(text) { statusEl.textContent = text; }
function sendKey(key, pressed) { socket.emit('key', {key, pressed}); }

window.addEventListener('keydown', (event) => {
  const key = event.key.toLowerCase();
  if (!'wasdqijklu zc'.includes(key)) return;
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

document.getElementById('startBtn').onclick = async () => {
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
socket.on('video', (data) => {
  videoEl.src = data.url + '?t=' + Date.now();
  videoEl.load();
  videoEl.play();
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
        self._lock = threading.Lock()
        self._setup_routes()
        self._setup_socketio()

    def run(self):
        set_seed(self.args.seed)
        os.makedirs(self.args.output_folder, exist_ok=True)
        self.socketio.run(self.app, host=self.args.host, port=self.args.port, allow_unsafe_werkzeug=True)

    def _setup_routes(self):
        @self.app.get("/")
        def index():
            return render_template_string(HTML)

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

        @self.app.get("/outputs/<path:filename>")
        def outputs(filename):
            return send_from_directory(self.args.output_folder, filename)

    def _setup_socketio(self):
        @self.socketio.on("key")
        def on_key(data):
            self.action_provider.set_key(data.get("key", ""), bool(data.get("pressed")))

        @self.socketio.on("mouse_delta")
        def on_mouse_delta(data):
            self.action_provider.set_mouse_delta(data.get("dx", 0), data.get("dy", 0))

    def _run_generation(self, img_path):
        self._emit_status("Loading models..." if self.pipeline is None else "Starting generation...")
        if self.pipeline is None:
            self.pipeline = InteractiveGameInference(self.args)
            self.mode = self.pipeline.config.pop('mode')
        self._emit_status("Generating. Use WASD/IJKL or drag mouse.")
        try:
            self.pipeline.generate_videos(
                self.mode,
                img_path=img_path,
                action_provider=self.action_provider,
                should_continue=lambda: not self.stop_event.is_set(),
                progress_callback=self._on_progress,
            )
            self._emit_status("Generation finished")
        except Exception as exc:
            self._emit_status(f"Generation failed: {exc}")
        finally:
            self.stop_event.set()

    def _on_progress(self, output_path, current_start_frame, num_blocks):
        filename = Path(output_path).name
        self.current_video_url = f"/outputs/{filename}"
        self.socketio.emit("video", {"url": self.current_video_url})
        self._emit_status(f"Generated through latent frame {current_start_frame}; total blocks: {num_blocks}")

    def _emit_status(self, message):
        print(message, flush=True)
        self.socketio.emit("status", {"message": message})


if __name__ == "__main__":
    MatrixGameWebApp(parse_args()).run()
