import threading

import torch


CAMERA_VALUE = 0.1


class CLIActionProvider:
    def get_current_action(self, mode="universal"):
        from pipeline.causal_inference import get_current_action

        return get_current_action(mode=mode)


class SocketIOActionProvider:
    def __init__(self, device="cuda", camera_value=CAMERA_VALUE):
        self.device = device
        self.camera_value = camera_value
        self._lock = threading.Lock()
        self._keys = set()
        self._mouse_delta = [0.0, 0.0]

    def set_key(self, key, pressed):
        key = key.lower()
        with self._lock:
            if pressed:
                self._keys.add(key)
            else:
                self._keys.discard(key)

    def set_mouse_delta(self, dx, dy):
        with self._lock:
            self._mouse_delta[0] += float(dx)
            self._mouse_delta[1] += float(dy)

    def clear(self):
        with self._lock:
            self._keys.clear()
            self._mouse_delta = [0.0, 0.0]

    def snapshot(self):
        with self._lock:
            return {
                "keys": sorted(self._keys),
                "mouse_delta": list(self._mouse_delta),
            }

    def get_current_action(self, mode="universal"):
        with self._lock:
            keys = set(self._keys)
            mouse_delta = self._mouse_delta
            self._mouse_delta = [0.0, 0.0]

        if mode == "universal":
            mouse = self._universal_mouse(keys, mouse_delta)
            keyboard = self._first_match(keys, {
                "w": [1, 0, 0, 0],
                "s": [0, 1, 0, 0],
                "a": [0, 0, 1, 0],
                "d": [0, 0, 0, 1],
            }, [0, 0, 0, 0])
            return {
                "mouse": torch.tensor(mouse, device=self.device),
                "keyboard": torch.tensor(keyboard, device=self.device),
            }

        if mode == "gta_drive":
            mouse = self._first_match(keys, {
                "a": [0, -self.camera_value],
                "d": [0, self.camera_value],
            }, [0, 0])
            keyboard = self._first_match(keys, {
                "w": [1, 0],
                "s": [0, 1],
            }, [0, 0])
            return {
                "mouse": torch.tensor(mouse, device=self.device),
                "keyboard": torch.tensor(keyboard, device=self.device),
            }

        if mode == "templerun":
            keyboard = self._first_match(keys, {
                "w": [0, 1, 0, 0, 0, 0, 0],
                "s": [0, 0, 1, 0, 0, 0, 0],
                "z": [0, 0, 0, 1, 0, 0, 0],
                "c": [0, 0, 0, 0, 1, 0, 0],
                "a": [0, 0, 0, 0, 0, 1, 0],
                "d": [0, 0, 0, 0, 0, 0, 1],
            }, [1, 0, 0, 0, 0, 0, 0])
            return {"keyboard": torch.tensor(keyboard, device=self.device)}

        raise ValueError(f"Unsupported mode: {mode}")

    def _universal_mouse(self, keys, mouse_delta):
        dy = self._clip(mouse_delta[1] / 200.0, -1.0, 1.0) * self.camera_value
        dx = self._clip(mouse_delta[0] / 200.0, -1.0, 1.0) * self.camera_value
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            return [dy, dx]
        return self._first_match(keys, {
            "i": [self.camera_value, 0],
            "k": [-self.camera_value, 0],
            "j": [0, -self.camera_value],
            "l": [0, self.camera_value],
        }, [0, 0])

    def _first_match(self, keys, mapping, default):
        for key, value in mapping.items():
            if key in keys:
                return value
        return default

    def _clip(self, value, lower, upper):
        return max(lower, min(upper, value))
