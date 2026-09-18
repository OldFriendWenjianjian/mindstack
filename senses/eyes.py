"""Eyes — live camera feeding the existing soup.

CameraAdapter reads N seconds of frames from a V4L2 device (or any
OpenCV-readable source, incl. RTSP URLs) and returns them as a video
signal for soup.extract(). The soup side is unchanged: motion energy,
hot-region path, JPEG keyframes for the cloud vision model.
"""
import time

import numpy as np


class CameraAdapter:
    """read() -> [(frames, {modality: video})] for the Loop."""

    def __init__(self, source=0, window_s=2.0, fps=4.0, size=256):
        self.source = source
        self.window_s = window_s
        self.fps = fps
        self.size = size

    def _open(self):
        import cv2
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"camera not reachable: {self.source}")
        return cap

    def read(self):
        import cv2
        cap = self._open()
        try:
            frames = []
            interval = 1.0 / self.fps
            t0 = time.time()
            while time.time() - t0 < self.window_s:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (self.size, self.size))
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                time.sleep(interval)
            if not frames:
                raise RuntimeError("camera delivered no frames")
            return [(frames, {"modality": "video", "want_clip": True})]
        finally:
            cap.release()


def camera_available(source=0):
    try:
        import cv2
        cap = cv2.VideoCapture(source)
        ok = cap.isOpened()
        cap.release()
        return ok
    except Exception:
        return False
