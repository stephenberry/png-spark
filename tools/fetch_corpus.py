#!/usr/bin/env python3
"""Fetches real-world PNG corpora for `cargo run --release -p png-spark-bench -- corpus`.

The synthetic set from gen_bench_images.py is twelve 1920x1080 images of one shape, which
says nothing about small files, indexed colour, or the screenshot and icon data that real
PNG workloads are full of. These two corpora do, and the `png` crate measures itself
against both, so the numbers are directly comparable.

  image-png  14 files, ~14 MB, from image-rs/image-png `tests/benches`. That project picked
             them to spread across filter behaviours: photographs filtered every which way,
             an unfiltered engraving, indexed images, text screenshots, and 16px icons where
             per-file overhead is the whole cost.
  qoi        ~2800 files, 1.1 GB, the QOI benchmark suite. Textures, photographs,
             screenshots, game frames, icons and wallpapers, grouped by kind.

Both unpack into tmp/corpus/<set>/, which is gitignored. The images keep their own licences:
see tmp/corpus/image-png/README.md, and https://qoiformat.org/benchmark/ for the QOI suite.

Usage:
    python3 tools/fetch_corpus.py image-png       # small, quick
    python3 tools/fetch_corpus.py qoi             # 1.1 GB download
    python3 tools/fetch_corpus.py all
"""
import argparse
import json
import os
import shutil
import sys
import tarfile
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "tmp", "corpus")

QOI_URL = "https://qoiformat.org/benchmark/qoi_benchmark_suite.tar"
IMAGE_PNG_API = "https://api.github.com/repos/image-rs/image-png/contents/tests/benches"

# GitHub rejects requests without one, and it is polite to say who is asking.
HEADERS = {"User-Agent": "png-spark-fetch-corpus"}


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def progress(done, total):
    if not sys.stderr.isatty():
        return
    if total:
        bar = int(40 * done / total)
        sys.stderr.write(f"\r  [{'#' * bar}{'.' * (40 - bar)}] {human(done)} / {human(total)}")
    else:
        sys.stderr.write(f"\r  {human(done)}")
    sys.stderr.flush()


def fetch(url, path):
    """Downloads `url` to `path`, resuming a previous partial download where the server
    allows it. The QOI archive is a gigabyte, so a dropped connection must not mean
    starting over."""
    part = path + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0

    request = urllib.request.Request(url, headers=dict(HEADERS))
    if have:
        request.add_header("Range", f"bytes={have}-")

    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as e:
        # 416 means the partial file is already the whole thing, or is longer than the
        # resource; either way the safe move is to start again.
        if have and e.code in (416, 501):
            os.remove(part)
            return fetch(url, path)
        raise

    resuming = response.status == 206
    if have and not resuming:
        have = 0  # Server ignored the range; rewrite from the start.

    remaining = response.headers.get("Content-Length")
    total = have + int(remaining) if remaining is not None else 0

    with open(part, "ab" if resuming else "wb") as out:
        if not resuming:
            out.truncate(0)
        done = have
        progress(done, total)
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            progress(done, total)
    if sys.stderr.isatty():
        sys.stderr.write("\n")

    os.replace(part, path)


def safe_members(archive):
    """Yields the PNG files in `archive` under `images/`, rejecting anything that would
    write outside the destination. Tar entries are attacker-controlled in general, and a
    corpus is not worth trusting with a path like `../../.ssh`."""
    for member in archive:
        if not member.isfile() or not member.name.lower().endswith(".png"):
            continue
        name = member.name
        if name.startswith("images/"):
            name = name[len("images/"):]
        parts = name.split("/")
        if any(p in ("", ".", "..") for p in parts) or name.startswith("/"):
            print(f"  skipping suspicious entry {member.name!r}", file=sys.stderr)
            continue
        yield member, os.path.join(*parts)


def fetch_qoi(keep_archive):
    dest = os.path.join(CORPUS, "qoi")
    if os.path.isdir(dest) and any(os.scandir(dest)):
        print(f"qoi: already present in {dest}")
        return

    archive = os.path.join(ROOT, "tmp", "qoi_benchmark_suite.tar")
    if not os.path.exists(archive):
        print(f"qoi: downloading {QOI_URL} (1.1 GB)")
        os.makedirs(os.path.dirname(archive), exist_ok=True)
        fetch(QOI_URL, archive)

    print("qoi: extracting")
    count = 0
    with tarfile.open(archive) as tar:
        for member, relative in safe_members(tar):
            target = os.path.join(dest, relative)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                continue
            with source, open(target, "wb") as out:
                shutil.copyfileobj(source, out)
            count += 1
    print(f"qoi: {count} images in {dest}")

    if keep_archive:
        print(f"qoi: keeping {archive}")
    else:
        os.remove(archive)


def fetch_image_png():
    dest = os.path.join(CORPUS, "image-png")
    if os.path.isdir(dest) and any(f.name.endswith(".png") for f in os.scandir(dest)):
        print(f"image-png: already present in {dest}")
        return

    os.makedirs(dest, exist_ok=True)
    print(f"image-png: listing {IMAGE_PNG_API}")
    request = urllib.request.Request(IMAGE_PNG_API, headers=dict(HEADERS))
    with urllib.request.urlopen(request, timeout=60) as response:
        entries = json.load(response)

    # README.md comes along because it is the licence and provenance record for the rest.
    wanted = [e for e in entries
              if e["type"] == "file" and (e["name"].endswith(".png") or e["name"] == "README.md")]
    for entry in sorted(wanted, key=lambda e: e["name"]):
        target = os.path.join(dest, entry["name"])
        print(f"  {entry['name']} ({human(entry['size'])})")
        fetch(entry["download_url"], target)
    print(f"image-png: {sum(1 for e in wanted if e['name'].endswith('.png'))} images in {dest}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="sets:\n"
               "  image-png   14 images, ~14 MB, from image-rs/image-png tests/benches\n"
               "  qoi         ~2800 images, 1.1 GB, the QOI benchmark suite\n"
               "  all         both",
    )
    parser.add_argument("sets", nargs="+", choices=["image-png", "qoi", "all"])
    parser.add_argument("--keep-archive", action="store_true",
                        help="keep the downloaded QOI tarball instead of deleting it after extraction")
    args = parser.parse_args()

    chosen = set(args.sets)
    if "all" in chosen:
        chosen = {"image-png", "qoi"}

    os.makedirs(CORPUS, exist_ok=True)
    if "image-png" in chosen:
        fetch_image_png()
    if "qoi" in chosen:
        fetch_qoi(args.keep_archive)

    print("\nrun: cargo run --release -p png-spark-bench -- corpus")


if __name__ == "__main__":
    main()
