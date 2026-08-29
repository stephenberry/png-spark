# png-spark

A fast PNG encoder and decoder for Rust, with **zero dependencies**. Complete format coverage, and a single compression setting chosen for speed.

png-spark implements the whole PNG format — every colour type, every bit depth, interlaced or not — on top of its own DEFLATE codec, its own checksums, and its own filter code. Nothing outside the standard library is involved.

```toml
[dependencies]
png-spark = "0.1"
```

```rust
// Decode
let image = png_spark::decode(&std::fs::read("input.png")?)?;
println!("{}x{} {:?}", image.width(), image.height(), image.color_type());
let rgba = image.to_rgba8()?;

// Encode
let png = png_spark::encode_rgba8(image.width(), image.height(), &rgba)?;
std::fs::write("output.png", png)?;
```

## Why

The usual Rust PNG stack is `png` + `fdeflate` + `miniz_oxide` + `flate2` + `crc32fast` + a handful of others: 15 crates, and a build that does about five times the compiler work of this one. png-spark is one crate with no dependency graph at all, and it is faster.

| | png-spark | png 0.18 + fdeflate 0.3 |
| --- | --- | --- |
| Crates compiled | 1 | 15 |
| Clean release build | 1.0 s CPU | 5.3 s CPU |

## Performance

In one line: png-spark encodes an order of magnitude faster than `png` and decodes a little faster, at the same file size on synthetic images and about a tenth larger on real ones.

Measured on an Apple M-series laptop against `png` 0.18 and `fdeflate` 0.3, over twelve 1920×1080 images spanning the compressibility range — smooth gradients, flat UI graphics, noisy photographs, and pure noise — in grey, RGB and RGBA. `cargo run --release -p png-spark-bench` reproduces all of it.

### Decoding

| | speed |
| --- | --- |
| **PNG decode** vs `png` | **5% faster** — never slower on any of the 12, and up to 23% faster |
| **zlib inflate** vs `fdeflate` | **18% faster** — faster on all 12, by 4% at worst and 2.3x at best |

Inflate wins most on compressible data, where a redesigned Adler-32 and shorter dependency chains in the Huffman decode loop matter most; on incompressible data both implementations are bound by the same serial table lookups.

`to_rgba8` and `to_rgb8` run at roughly 1.8 GP/s over the corpus below. Layouts that already match the request are a straight copy, and the rest resolve the colour type and bit depth once per scanline rather than once per pixel.

### Encoding

Whole-PNG encoding, filtering included, against `png` at its default setting:

| | speed | output size |
| --- | --- | --- |
| **PNG encode** vs `png` | **12.4x faster** | **1% smaller** |

`png` spends most of its encode time scoring filters. png-spark scores them on a sample of each row, and scores each candidate through a specialization of the loop rather than a filter argument, so the predictor is straight-line code instead of a branch on every byte examined.

Raw zlib compression, against `fdeflate` alone:

| | speed | output size |
| --- | --- | --- |
| **zlib deflate** vs `fdeflate` | **20% faster** | **19% smaller** |

Faster *and* a fifth smaller. It reaches 6–8x on incompressible input, where it recognises the data is not worth coding and stops trying, and is about 20% slower on the flat graphics `fdeflate` is specialised for.

There is no compression level to choose; see [One compression setting](#design-notes) below for why.

### On a real corpus

The twelve images above are synthetic, all the same size and shape. The QOI benchmark suite is 2848 real files — photographs, screenshots, game textures, icons and wallpapers — and `cargo run --release -p png-spark-bench -- corpus` reports it per directory ([see below](#real-world-corpora) for how to fetch it). Against `png` at its default setting, over the whole suite:

| | speed | output size |
| --- | --- | --- |
| **PNG encode** vs `png` | **17.9x faster** | **11% larger** |

Decoding is **30% faster** across the same files. Each library decodes its own re-encoded output there rather than a shared file, so it is a coarser figure than the 5% above, and it flatters png-spark: its own encoder picks `Paeth` for about two thirds of all scanline bytes, which is the filter the reverse pass below is fastest at.

Size is where the single compression setting shows its cost, and the cost is not uniform. On the suite's photographs png-spark is about **23x faster for 1–10% more bytes**. On flat graphics — icons, web screenshots, tiled textures — its files are 20–46% larger, because that is exactly the data an LZ77 match finder is good at and png-spark does not have one.

Against `png`'s `Fast` setting, which routes through `fdeflate` instead of flate2, the picture is closer: png-spark writes 3% smaller files but takes about 15% longer. `fdeflate` writes with a Huffman code fixed in advance for PNG data, where png-spark fits one to the block in front of it, and on files this size that counting pass is not fully amortised. Run `cargo run --release -p png-spark-bench -- corpus tmp/corpus/qoi fast` to see it per directory.

## What it does

- Colour types: grayscale, RGB, indexed, grayscale+alpha, RGBA
- Bit depths: 1, 2, 4, 8, 16
- Interlacing: Adam7 on read; output is always non-interlaced
- Chunks: `IHDR`, `PLTE`, `tRNS`, `IDAT`, `IEND`, plus any ancillary chunks you attach; unrecognised critical chunks are an error
- Conversion to 8-bit RGB or RGBA, resolving palettes, `tRNS`, and sub-byte depths

Not supported: APNG, and writing interlaced files.

## Reading files you did not write

`IHDR` states an image's dimensions in thirteen bytes, and a decoder needs the buffer they imply before it can read any of the compressed data, so a seventy-byte file can name a size in petabytes. The decoder carries a ceiling on that size, `DEFAULT_MAX_DECOMPRESSED_SIZE`, of 512 MiB, which is room for a 16-bit RGBA image of 8000×8000 pixels. A header over it is an error rather than an allocation, and `Decoder::max_decompressed_size` raises or removes it where the images really are that large.

An allocation the operating system refuses comes back as `Error::OutOfMemory` rather than aborting the process. `read_info` parses the header and colour chunks and stops at the image data, for a caller that wants to decide something about a file before decoding it.

## Carrying your own data in a PNG

PNG stores everything in typed chunks, and a decoder must skip any *ancillary* chunk it does not recognise. An ancillary chunk of your own is therefore a place to keep application data inside the image file: arbitrary bytes, just under 2 GiB of them, with no escaping and no encoding, which every other PNG reader ignores.

```rust
use png_spark::{Chunk, Decoder, Keep};

// Attach data to an image on the way out.
image.info.metadata.push(Chunk::new(*b"apPd", asset_id.to_vec()));
let png = png_spark::encode(&image.info, &image.data)?;

// Ask for it on the way back in.
let image = Decoder::new().keep(Keep::Only(vec![*b"apPd"])).decode(&png)?;
let asset_id = image.info.chunk(b"apPd");
```

The four type bytes are not free-form. `apPd` is lower case first, marking the chunk ancillary; lower case second, marking it private to one application and so unable to collide with a registered type; upper case third, which the specification reserves; and lower case fourth, saying an editor that does not understand the chunk may still copy it through. The encoder rejects a type that breaks those rules rather than writing a file it could not read back, and writes each chunk on whichever side of `PLTE` the specification requires. What a *registered* type means is not checked, so two `gAMA` chunks or a `hIST` without a palette are the caller's mistake to avoid; private types have no such rules to break.

Ignoring a chunk is not the same as preserving it. A tool that rewrites the file may well drop it: libpng discards unknown chunks unless asked for them, and optimisers such as oxipng and pngcrush strip ancillary chunks by default. Data that has to survive an arbitrary third-party tool does not belong here; data that has to survive your own pipeline does.

Decoding keeps nothing by default, since retaining a chunk means copying it and nothing the decoder returns depends on one. `Keep::All` takes everything the file carries. What is kept travels on `Info`, so handing a decoded image straight back to the encoder preserves its metadata. A chunk that fails its CRC is dropped rather than returned.

Payloads are written exactly as given; compress yours first if it is worth compressing.

## Design notes

**Decoding gives you the file's own format.** `decode` returns pixels exactly as the file stores them, because that is the only representation that is always correct and always free. `to_rgba8` and `to_rgb8` convert when you want a uniform layout.

**The decompressor is not a state machine.** PNG says up front how many bytes the image data expands to, so the whole stream is decoded in one call against one buffer. That removes the per-symbol state checks and output clamping a resumable decoder needs, and leaves the bit buffer and output cursor in registers.

**Checksums use the hardware.** Adler-32 has an AArch64 dot-product path that runs at roughly 50 GB/s; CRC-32 uses the AArch64 CRC instructions where present and slice-by-16 otherwise. Both fall back to portable code, and every implementation is tested against the same reference.

**Chunk CRCs are checked; the Adler-32 is not, by default.** The Adler-32 inside the compressed stream covers the same bytes the chunk CRC already covered. Checking one of them catches the same file corruption for one pass over the data instead of two. `Decoder::checks(Checks::Full)` turns both on.

**A bad CRC on an ancillary chunk drops the chunk, not the file.** Nothing a decode returns is built from a colour profile or a text comment, and files exist whose metadata was rewritten without recomputing its checksum. Losing the image over four stale bytes describing something discarded anyway helps nobody, so it is skipped, as libpng and the `png` crate both do. On a critical chunk it remains an error.

**One compression setting, deliberately.** There is no level knob. The compressor codes literals and runs of zero bytes and nothing else: no hash table, no match finder, no token buffer, just two sequential passes over each block. Zero runs are the repetition filtered image data is actually made of, since a flat region leaves a zero residual under any of the PNG filters, and finding them costs a scan rather than a random memory access per input byte.

An LZ77 match finder on top of this buys a few percent of size for several times the time — and on photographs it buys nothing at all, turning up short matches that displace literals the entropy coder was already handling well and leaving the file *larger*. That is the wrong trade for a library whose reason to exist is speed, so it is not offered. If you want the last few percent, use the `png` crate.

**Blocks reuse the previous block's Huffman code.** Only the first block of a stream is scanned twice, once to count and once to write. Every block after it is written in a single pass using the code fitted to the block before, while counting its own symbols for the next one. Adjacent bands of the same image have near-identical byte distributions, so a code one block out of date costs a fraction of a percent and halves the number of times the data is walked.

**Filters are sticky.** Consecutive scanlines filtered the same way stay similar to each other, and that similarity is what becomes a match one scanline back. A filter has to beat the previous row's choice by an eighth before the encoder will switch, which is worth a few percent on its own.

**Reverse `Paeth` runs two rows at a time.** Undoing a filter is serial along a row — pixel *x* cannot start until pixel *x-1* is known — and `Paeth` is a long enough chain of compares and selects that the loop waits far more than it works: on its own it reconstructs at around 800 MB/s where `Up` manages 28 GB/s. Two rows offset by one pixel break the wait. Row *r* pixel *x* and row *r+1* pixel *x-1* depend only on values settled before the step, so the two predictions issue together and the chain retires two pixels instead of one. It is worth 1.8x on 8-bit RGBA, and about 15% on decoding a real corpus, where `Paeth` covers 42% of scanline bytes and adjacent rows share it often enough for the pair path to cover 92% of them. Rows without a `Paeth` neighbour fall back to the one-row loop.

## Safety

The crate is safe Rust apart from four narrow places, each with the invariant that justifies it written next to it: the literal stores and the match reads and copies in the inflate loop, the bit writer's eight-byte flush, the SIMD checksum paths, and the fallible zeroed allocation the decoder sizes its buffers with. Everything else — chunk parsing, filtering, conversion — is bounds-checked. Malformed input is a tested case: the suite feeds truncated files, single-bit corruptions at every byte, and thousands of random byte strings through the decoder, and requires errors rather than panics.

## Testing

```sh
python3 tools/gen_testdata.py     # generate the reference corpus (once)
cargo test                        # 74 tests, no dependencies
cargo test -p png-spark-bench     # cross-checks against png and fdeflate
```

The corpus is produced by a reference implementation rather than by png-spark, so the tests check the format and not just self-consistency: 224 zlib streams from zlib itself at every level and window size, and 180 PNGs covering every colour type, bit depth, interlace mode and filter combination. The benchmark crate additionally round-trips png-spark's output through `fdeflate` and the `png` crate, and `fdeflate`'s output back through png-spark.

### Real-world corpora

The figures above come from a synthetic set of twelve 1920×1080 images, which says nothing about small files, indexed colour, or the screenshot and icon data real workloads are full of. Two public corpora cover that, and the `png` crate measures itself against both, so the numbers line up with the ones that project publishes:

```sh
python3 tools/fetch_corpus.py image-png   # 15 files, ~14 MB, from image-rs/image-png
python3 tools/fetch_corpus.py qoi         # ~2800 files, 1.1 GB, the QOI benchmark suite
cargo run --release -p png-spark-bench -- corpus
```

Both land in `tmp/corpus/`, which is gitignored, and keep their own licences. The `corpus` mode re-encodes and re-decodes every file, reporting compression ratio and megapixels per second for png-spark and `png` side by side, per directory. It is also a round-trip sweep over real files: each image is decoded back and compared with the pixels that went in, and png-spark's output is checked against the `png` crate's decoder as well.

## Licence

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT licence ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this crate, as defined in the Apache-2.0 licence, shall be dual licensed as above, without any additional terms or conditions.
