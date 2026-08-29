#!/usr/bin/env python3
"""Generates the benchmark image set in tmp/bench.

Four image kinds, chosen because they sit at different points of the compressibility range
that a PNG codec has to cope with:

  photo     smooth gradients plus sensor-like noise; filters leave small residuals
  ui        large flat regions and hard edges; highly compressible
  noise     incompressible; the worst case for any entropy coder
  gradient  perfectly smooth; the best case for filtering
"""
import os
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tmp", "bench")

W, H = 1920, 1080


def photo(rng):
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = 128 + 100 * np.sin(x / 190.0) * np.cos(y / 130.0)
    g = 128 + 90 * np.sin((x + y) / 240.0)
    b = 128 + 110 * np.cos((x - y) / 300.0)
    img = np.stack([r, g, b], axis=-1)
    img += rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def ui(rng):
    img = np.full((H, W, 3), 246, dtype=np.uint8)
    for _ in range(300):
        x0, y0 = rng.integers(0, W - 40), rng.integers(0, H - 40)
        w, h = rng.integers(20, 400), rng.integers(10, 120)
        color = rng.integers(0, 256, 3)
        img[y0:y0 + h, x0:x0 + w] = color
    for x in range(0, W, 64):
        img[:, x:x + 2] = 30
    return img


def noise(rng):
    return rng.integers(0, 256, (H, W, 3), dtype=np.uint8)


def gradient(rng):
    y, x = np.mgrid[0:H, 0:W]
    img = np.stack([(x * 255 // W), (y * 255 // H), ((x + y) * 255 // (W + H))], axis=-1)
    return img.astype(np.uint8)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(3)
    for name, fn in [("photo", photo), ("ui", ui), ("noise", noise), ("gradient", gradient)]:
        rgb = fn(rng)
        Image.fromarray(rgb).save(os.path.join(OUT, f"{name}_rgb.png"), compress_level=6)
        rgba = np.dstack([rgb, np.full((H, W), 255, np.uint8)])
        Image.fromarray(rgba).save(os.path.join(OUT, f"{name}_rgba.png"), compress_level=6)
        gray = np.asarray(Image.fromarray(rgb).convert("L"))
        Image.fromarray(gray).save(os.path.join(OUT, f"{name}_gray.png"), compress_level=6)
        print(name, "written")


if __name__ == "__main__":
    main()
