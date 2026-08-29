#!/usr/bin/env python3
"""Generates the reference corpus used by the integration tests.

Writes zlib streams (tmp/z) and PNG images (tmp/png) produced by known-good implementations,
so the tests can check png-spark against something other than itself. The output is not
committed; run this script before `cargo test`.
"""
import os, random, zlib, struct, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gen_zlib_corpus(seed=7, count=200):
    out = os.path.join(ROOT, "tmp", "z")
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        os.remove(os.path.join(out, f))
    rng = random.Random(seed)

    fixed = {
        "empty": b"",
        "onebyte": b"\x42",
        "hello": b"Hello world! Hello world! Hello world!",
        "zeros": bytes(65536),
        "ones": b"\xff" * 70000,
        "text": b"the quick brown fox jumps over the lazy dog. " * 3000,
    }
    for name, data in fixed.items():
        for lvl in (0, 1, 6, 9):
            _emit(out, f"{name}_{lvl}", data, lvl)

    # Randomised inputs spanning the compressibility range, so every table shape and both
    # block types get exercised.
    for i in range(count):
        n = rng.choice([1, 2, 3, 7, 15, 16, 17, 31, 255, 256, 257, 1000, 40000, 200000])
        mode = i % 5
        if mode == 0:
            data = bytes(rng.getrandbits(8) for _ in range(n))
        elif mode == 1:
            data = bytes(rng.choice(b"ab") for _ in range(n))
        elif mode == 2:
            alphabet = bytes(rng.getrandbits(8) for _ in range(rng.randint(1, 6)))
            data = bytes(rng.choice(alphabet) for _ in range(n))
        elif mode == 3:
            data = bytes((j * j // 7) % 256 for j in range(n))
        else:
            base = bytes(rng.getrandbits(8) for _ in range(min(n, 64)))
            data = (base * (n // len(base) + 1))[:n]
        _emit(out, f"rand{i}", data, rng.choice([0, 1, 5, 6, 9]), rng.choice([9, 10, 12, 15]))


def _emit(out, name, data, level, window=15):
    comp = zlib.compressobj(level, zlib.DEFLATED, window)
    stream = comp.compress(data) + comp.flush()
    with open(os.path.join(out, f"{name}.z"), "wb") as f:
        f.write(stream)
    with open(os.path.join(out, f"{name}.raw"), "wb") as f:
        f.write(data)


def main():
    gen_zlib_corpus()
    print("wrote zlib corpus to tmp/z")
    print(f"wrote {gen_png_corpus()} pngs to tmp/png")


# ---------------------------------------------------------------------------------------
# PNG corpus
# ---------------------------------------------------------------------------------------

SIGNATURE = b"\x89PNG\r\n\x1a\n"
ADAM7 = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4),
         (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
SAMPLES = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def chunk(kind, payload):
    return (struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))


def row_bytes(width, bpp_bits):
    return (width * bpp_bits + 7) // 8


def paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def apply_filter(ftype, row, prev, stride):
    out = bytearray(len(row))
    for i in range(len(row)):
        a = row[i - stride] if i >= stride else 0
        b = prev[i]
        c = prev[i - stride] if i >= stride else 0
        x = row[i]
        if ftype == 0:
            out[i] = x
        elif ftype == 1:
            out[i] = (x - a) & 0xFF
        elif ftype == 2:
            out[i] = (x - b) & 0xFF
        elif ftype == 3:
            out[i] = (x - ((a + b) >> 1)) & 0xFF
        else:
            out[i] = (x - paeth(a, b, c)) & 0xFF
    return bytes(out)


def get_pixel_bits(data, rb, bits, row, x):
    """Reads one pixel's packed bits out of a row-major packed buffer."""
    if bits >= 8:
        n = bits // 8
        off = row * rb + x * n
        return data[off:off + n]
    off = row * rb
    bit = x * bits
    mask = (1 << bits) - 1
    return (data[off + bit // 8] >> (8 - bits - bit % 8)) & mask


def set_pixel_bits(buf, rb, bits, row, x, value):
    if bits >= 8:
        n = bits // 8
        off = row * rb + x * n
        buf[off:off + n] = value
        return
    off = row * rb
    bit = x * bits
    mask = (1 << bits) - 1
    shift = 8 - bits - bit % 8
    buf[off + bit // 8] = (buf[off + bit // 8] & ~(mask << shift) & 0xFF) | (value << shift)


def write_png(path, width, height, color_type, depth, data, interlace, rng, palette=None, trns=None):
    """Writes a PNG whose scanlines use a random mix of all five filter types."""
    bits = SAMPLES[color_type] * depth
    stride = max(1, bits // 8)
    raw = bytearray()

    if interlace == 0:
        passes = [(0, 0, 1, 1, width, height)]
    else:
        passes = []
        for (xs, ys, xd, yd) in ADAM7:
            pw = (width - xs + xd - 1) // xd if width > xs else 0
            ph = (height - ys + yd - 1) // yd if height > ys else 0
            passes.append((xs, ys, xd, yd, pw, ph))

    full_rb = row_bytes(width, bits)
    for (xs, ys, xd, yd, pw, ph) in passes:
        if pw == 0 or ph == 0:
            continue
        prb = row_bytes(pw, bits)
        prev = bytes(prb)
        for j in range(ph):
            line = bytearray(prb)
            for k in range(pw):
                set_pixel_bits(line, prb, bits, 0, k,
                               get_pixel_bits(data, full_rb, bits, ys + j * yd, xs + k * xd))
            ftype = rng.randrange(5)
            raw.append(ftype)
            raw += apply_filter(ftype, line, prev, stride)
            prev = bytes(line)

    ihdr = struct.pack(">IIBBBBB", width, height, depth, color_type, 0, 0, interlace)
    body = SIGNATURE + chunk(b"IHDR", ihdr)
    if palette is not None:
        body += chunk(b"PLTE", palette)
    if trns is not None:
        body += chunk(b"tRNS", trns)
    compressed = zlib.compress(bytes(raw), rng.choice([1, 6, 9]))
    # Split IDAT so the multi-chunk path gets exercised too.
    if rng.random() < 0.4 and len(compressed) > 4:
        cut = len(compressed) // 2
        body += chunk(b"IDAT", compressed[:cut]) + chunk(b"IDAT", compressed[cut:])
    else:
        body += chunk(b"IDAT", compressed)
    body += chunk(b"IEND", b"")

    with open(path, "wb") as f:
        f.write(body)
    with open(path.replace(".png", ".raw"), "wb") as f:
        f.write(data)


def gen_png_corpus(seed=11):
    out = os.path.join(ROOT, "tmp", "png")
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        os.remove(os.path.join(out, f))
    rng = random.Random(seed)

    combos = [(0, d) for d in (1, 2, 4, 8, 16)]
    combos += [(2, d) for d in (8, 16)]
    combos += [(3, d) for d in (1, 2, 4, 8)]
    combos += [(4, d) for d in (8, 16)]
    combos += [(6, d) for d in (8, 16)]

    count = 0
    for (ct, depth) in combos:
        for interlace in (0, 1):
            for (w, h) in [(1, 1), (1, 9), (9, 1), (7, 5), (32, 17), (63, 40)]:
                bits = SAMPLES[ct] * depth
                rb = row_bytes(w, bits)
                data = bytearray(rng.getrandbits(8) for _ in range(rb * h))
                palette = trns = None
                if ct == 3:
                    entries = min(1 << depth, 1 << depth)
                    palette = bytes(rng.getrandbits(8) for _ in range(entries * 3))
                    if rng.random() < 0.5:
                        trns = bytes(rng.getrandbits(8) for _ in range(entries))
                elif ct == 0 and rng.random() < 0.3:
                    trns = struct.pack(">H", rng.randrange(1 << depth))
                elif ct == 2 and rng.random() < 0.3:
                    trns = struct.pack(">HHH", *(rng.randrange(1 << depth) for _ in range(3)))
                # Clear any bits past the end of the last byte of each row so the expected
                # output is exactly what a decoder must produce.
                if bits < 8:
                    used = w * bits
                    if used % 8:
                        keep = 0xFF << (8 - used % 8) & 0xFF
                        for r in range(h):
                            data[r * rb + rb - 1] &= keep
                name = f"ct{ct}_d{depth}_i{interlace}_{w}x{h}"
                write_png(os.path.join(out, name + ".png"), w, h, ct, depth,
                          bytes(data), interlace, rng, palette, trns)
                count += 1
    return count


if __name__ == "__main__":
    main()
