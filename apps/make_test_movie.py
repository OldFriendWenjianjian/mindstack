"""Generate a synthetic 3-scene test movie with known structure.

Scene 1 (0-4s):  calm — slow drifting circle, low motion
Scene 2 (4-8s):  action — bouncing ball + hard cut at 4s, high motion
Scene 3 (8-12s): calm again — slow vertical bars, hard cut at 8s

Ground truth: 2 hard cuts, one high-motion middle segment.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

W = H = 320
FPS = 24
DUR = 12.0


def frame(t):
    img = np.zeros((H, W, 3), np.uint8)
    if t < 4.0:                      # scene 1: drifting circle on dark gray
        img[:] = (40, 40, 40)
        x = int(60 + 40 * t)
        cv2_circle(img, (x, 160), 30, (200, 180, 160))
    elif t < 8.0:                    # scene 2: bright bg, bouncing ball
        img[:] = (180, 180, 190)
        x = int(40 + 200 * abs(np.sin(t * 3)))
        y = int(160 + 100 * abs(np.cos(t * 5)))
        cv2_circle(img, (x, y), 24, (40, 40, 220))
        cv2_rect(img, (0, 0, W, 20), (30, 30, 30))
    else:                            # scene 3: moving vertical bars
        img[:] = (60, 80, 60)
        for i in range(8):
            x0 = int((i * 40 + 80 * t) % (W + 40)) - 20
            cv2_rect(img, (x0, 0, 18, H), (120, 160, 120))
    return img


def cv2_circle(img, c, r, color):
    import cv2
    cv2.circle(img, c, r, color, -1)


def cv2_rect(img, rc, color):
    import cv2
    x, y, w, h = rc
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "test_movie.mp4"
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out, fourcc, FPS, (W, H))
    n = int(DUR * FPS)
    for i in range(n):
        vw.write(frame(i / FPS))
    vw.release()
    print(f"wrote {out} ({DUR:.0f}s, {n} frames @ {FPS} fps)")


if __name__ == "__main__":
    main()
