"""Synthetic kitten-cam test clip: still -> sudden play burst -> still.

0-3s  : empty room, static camera (should produce NO burst)
3-6s   : small dark 'kitten' blob darts around (should burst)
6-9s   : still again (no burst)
"""
import sys
import numpy as np

W = H = 256
FPS = 12


def frame(t):
    import cv2
    img = np.full((H, W, 3), (90, 85, 80), np.uint8)
    cv2.rectangle(img, (0, 200, W, 56), (70, 62, 58), -1)      # floor line
    if 3.0 <= t < 6.0:
        x = int(128 + 90 * np.sin(t * 7))
        y = int(150 + 40 * abs(np.cos(t * 9)))
        cv2.ellipse(img, (x, y), (22, 14), 0, 0, 360, (40, 35, 30), -1)
        cv2.circle(img, (x, y - 14), 9, (40, 35, 30), -1)      # head
        cv2.line(img, (x + 18, y - 4), (x + 30, y - 12), (40, 35, 30), 3)  # tail
    return img


def main():
    import cv2
    out = sys.argv[1] if len(sys.argv) > 1 else "test_cat.mp4"
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n = int(9.0 * FPS)
    for i in range(n):
        vw.write(frame(i / FPS))
    vw.release()
    print(f"wrote {out} (9s @ {FPS} fps)")


if __name__ == "__main__":
    main()
