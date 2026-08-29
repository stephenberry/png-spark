# Changelog

## 0.1.0

First release.

- Decodes and encodes every PNG colour type and bit depth, interlaced or not
- Its own DEFLATE, CRC-32, Adler-32 and filter code, so there are no dependencies
- Ancillary chunks carried through on both read and write
- A 512 MiB default ceiling on decompressed size, for input you did not write
- Requires Rust 1.96, edition 2024
