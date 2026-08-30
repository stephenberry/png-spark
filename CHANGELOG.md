# Changelog

## 0.2.0

- `Encoder::encode_to` writes a PNG to any `io::Write`, filtering and compressing a band at a time. Peak working memory grows with the image's width but not its height, against the whole file plus a filtered copy of every row before.
- `WriteError` carries either an encoding fault or the sink's `io::Error`. `Error` is unchanged and still `Clone + PartialEq + Eq`.
- `Deflater::zlib_start` and `zlib_push` compress a zlib stream in pieces.
- `Encoder::encode` now goes through the same banded path. Output is a fraction of a percent larger, and a large image is split across several `IDAT` chunks instead of one.

## 0.1.0

First release.

- Decodes and encodes every PNG colour type and bit depth, interlaced or not
- Its own DEFLATE, CRC-32, Adler-32 and filter code, so there are no dependencies
- Ancillary chunks carried through on both read and write
- A 512 MiB default ceiling on decompressed size, for input you did not write
- Requires Rust 1.96, edition 2024
