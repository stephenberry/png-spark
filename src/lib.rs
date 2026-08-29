//! A fast PNG encoder and decoder with no dependencies.
//!
//! png-spark reads and writes the whole PNG format — every colour type, every bit depth,
//! interlaced or not — through its own DEFLATE implementation, its own checksums, and its
//! own filter code. Nothing outside the standard library is involved, so it builds in a
//! couple of seconds and adds nothing to a dependency tree.
//!
//! # Decoding
//!
//! ```no_run
//! let bytes = std::fs::read("input.png")?;
//! let image = png_spark::decode(&bytes)?;
//!
//! println!("{}x{} {:?}", image.width(), image.height(), image.color_type());
//!
//! // `image.data` holds the pixels exactly as the file stores them. Convert when you need
//! // a uniform layout:
//! let rgba = image.to_rgba8()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Because the data is in the file's own format, the colour type is not the whole story on
//! transparency: a palette or greyscale image keeps its alpha in a `tRNS` chunk rather than
//! in its pixels. Ask [`Info::has_alpha`] rather than reading the colour type, or convert
//! with `to_rgba8`, which resolves `tRNS` for you.
//!
//! # Reading files you did not write
//!
//! `IHDR` states an image's dimensions in thirteen bytes, and a decoder needs the buffer
//! they imply before it can read any of the compressed data. A seventy-byte file can
//! therefore name a size in petabytes, so the decoder carries a ceiling on it, of
//! [`DEFAULT_MAX_DECOMPRESSED_SIZE`]; a header over
//! it is an error rather than an allocation. Raise it with
//! [`Decoder::max_decompressed_size`] where the images really are that large.
//!
//! [`read_info`] parses the header and colour chunks and stops at the image data, for a
//! caller that wants to decide something about a file before decoding it.
//!
//! # Encoding
//!
//! ```no_run
//! let width = 256;
//! let height = 256;
//! let pixels: Vec<u8> = (0..width * height * 4).map(|i| i as u8).collect();
//!
//! let png = png_spark::encode_rgba8(width as u32, height as u32, &pixels)?;
//! std::fs::write("output.png", png)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! For repeated work, reuse a [`Decoder`] or [`Encoder`]: they hold the Huffman tables and
//! scratch buffers, so a second image costs no allocation for them.
//!
//! ```no_run
//! use png_spark::{Encoder, FilterStrategy};
//!
//! let mut encoder = Encoder::new();
//! encoder.filter(FilterStrategy::Adaptive);
//! ```
//!
//! # Choosing settings
//!
//! There is no compression level. The compressor has one mode - literals and zero runs,
//! coded in a single pass - because that is the setting worth having: it reaches within a
//! few percent of a full LZ77 match finder on filtered image data for several times the
//! speed. Anyone who wants the last few percent should use the `png` crate.
//!
//! [`FilterStrategy`] decides how each scanline's filter is picked. The default scores the
//! five filters on a sample of the row; [`FilterStrategy::Adaptive`] scores all of it, for
//! a small gain on images with sharp edges.
//!
//! # Carrying your own data in a PNG
//!
//! PNG stores everything in typed chunks, and a decoder must skip any *ancillary* chunk it
//! does not recognise. An ancillary chunk of your own is therefore a place to keep
//! application data inside the image file: arbitrary bytes, up to `i32::MAX` of them, with
//! no escaping and no encoding, which every other PNG reader ignores.
//!
//! Ignoring is not the same as preserving. A tool that rewrites the file may well drop the
//! chunk: libpng discards unknown chunks unless the application asks for them, and
//! optimisers such as oxipng and pngcrush strip ancillary chunks by default. Data that must
//! survive an arbitrary third-party tool does not belong here; data that must survive your
//! own pipeline does.
//!
//! Attach chunks by putting them on the [`Info`] you encode with, and ask for them back
//! with [`Decoder::keep`]:
//!
//! ```
//! use png_spark::{Chunk, Decoder, Keep};
//!
//! let mut image = png_spark::decode(&make_png())?;
//! image.info.metadata.push(Chunk::new(*b"apPd", b"anything at all".to_vec()));
//! let png = png_spark::encode(&image.info, &image.data)?;
//!
//! let read_back = Decoder::new().keep(Keep::Only(vec![*b"apPd"])).decode(&png)?;
//! assert_eq!(read_back.info.chunk(b"apPd"), Some(&b"anything at all"[..]));
//! # fn make_png() -> Vec<u8> { png_spark::encode_rgba8(2, 2, &[0; 16]).unwrap() }
//! # Ok::<(), png_spark::Error>(())
//! ```
//!
//! The four type bytes are not free-form: `apPd` above is lower case first, marking the
//! chunk ancillary; lower case second, marking it private to one application and so unable
//! to collide with a registered type; upper case third, which the specification reserves;
//! and lower case fourth, saying an editor that does not understand it may still copy it
//! through. [`Chunk::validate`] enforces the rules, and the encoder rejects a chunk that
//! breaks them rather than writing a file it could not read back.
//!
//! What the encoder checks is the type bytes and the placement relative to `PLTE`. It does
//! not know what a *registered* type means, so writing two `gAMA` chunks, or a `hIST` for an
//! image with no palette, is the caller's mistake to avoid. Private types have no such
//! rules to break.
//!
//! Decoding keeps nothing by default, because retaining a chunk means copying it and
//! nothing the decoder returns depends on one. Metadata that is kept travels on `Info`, so
//! handing a decoded image straight back to the encoder preserves it.
//!
//! There is no compression here: a payload is written exactly as given. Compress it first
//! if it is worth compressing.
//!
//! # Layout
//!
//! The pieces are public in their own right, so the DEFLATE and checksum implementations can
//! be used on their own:
//!
//! - [`inflate`] and [`deflate`] — zlib streams, independent of PNG
//! - [`crc32`] and [`adler32`] — the two checksums, with SIMD paths where they exist
//! - [`filter`] — the five PNG scanline filters, forward and reverse
//! - [`decoder`], [`encoder`], [`common`] — the PNG layer itself

#![warn(missing_docs, missing_debug_implementations)]

pub mod adler32;
pub mod common;
pub mod crc32;
pub mod decoder;
pub mod deflate;
pub mod encoder;
pub mod error;
pub mod filter;
pub mod huffman;
pub mod inflate;
pub mod tables;
pub mod transform;

pub use common::{BitDepth, Chunk, ColorType, Info, Interlacing};
pub use decoder::{Checks, DEFAULT_MAX_DECOMPRESSED_SIZE, Decoder, Image, Keep, decode, read_info};
pub use encoder::{Encoder, FilterStrategy, encode, encode_rgb8, encode_rgba8};
pub use error::Error;
pub use filter::Filter;
