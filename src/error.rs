//! Errors produced when reading or writing PNG data.

use crate::inflate::InflateError;

/// Non-exhaustive: a decoder discovers new ways for a file to be wrong as it hardens, and
/// naming one should not cost a major version. Match with a fallback arm.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The data does not start with the eight-byte PNG signature.
    NotAPng,
    /// The file ended in the middle of a chunk.
    TruncatedChunk,
    /// A chunk's stored CRC does not match its contents.
    BadChunkCrc { chunk: [u8; 4] },
    /// The first chunk was not `IHDR`.
    MissingHeader,
    /// The file contained no `IDAT` chunk.
    MissingImageData,
    /// An indexed image had no `PLTE` chunk.
    MissingPalette,
    /// A chunk carried a length the specification does not allow.
    InvalidChunkLength { chunk: [u8; 4], length: usize },
    /// A metadata chunk's type is not one the encoder is allowed to write.
    ///
    /// See [`Chunk::validate`](crate::common::Chunk::validate) for the rules.
    InvalidChunkType { chunk: [u8; 4] },
    /// A chunk marked critical was not one this decoder understands.
    ///
    /// A critical chunk may change how the image is to be interpreted, so ignoring one
    /// risks producing the wrong pixels rather than an error.
    UnknownCriticalChunk { chunk: [u8; 4] },
    /// `IHDR` declared a zero width or height.
    EmptyImage,
    /// The image is too large to address on this platform.
    ImageTooLarge,
    /// The image's decompressed size exceeds the decoder's limit.
    ///
    /// `IHDR` states how large an image expands to before any of it has been read, so a
    /// thirteen-byte header can ask for a buffer of any size the platform can address. See
    /// [`Decoder::max_decompressed_size`](crate::Decoder::max_decompressed_size).
    SizeLimitExceeded { size: usize, limit: usize },
    /// The allocator could not provide a buffer the image needs.
    OutOfMemory { bytes: usize },
    /// `IHDR` declared a colour type that is not one of the five PNG defines.
    InvalidColorType(u8),
    /// `IHDR` declared a bit depth that is not valid for its colour type.
    InvalidBitDepth { color_type: u8, bit_depth: u8 },
    /// `IHDR` declared a compression or filter method other than the single defined one.
    UnsupportedMethod { field: &'static str, value: u8 },
    /// `IHDR` declared an interlace method other than none or Adam7.
    InvalidInterlaceMethod(u8),
    /// A scanline used a filter byte outside `0..=4`.
    InvalidFilter { row: usize },
    /// A palette index referred past the end of `PLTE`.
    PaletteIndexOutOfRange,
    /// The compressed image data could not be decoded.
    Inflate(InflateError),
    /// The pixel buffer handed to the encoder is not the size the header describes.
    WrongBufferSize { expected: usize, found: usize },
}

impl From<InflateError> for Error {
    fn from(error: InflateError) -> Self {
        Error::Inflate(error)
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::NotAPng => f.write_str("data does not begin with a PNG signature"),
            Error::TruncatedChunk => f.write_str("file ended in the middle of a chunk"),
            Error::BadChunkCrc { chunk } => {
                write!(f, "CRC mismatch in `{}` chunk", chunk_name(chunk))
            }
            Error::MissingHeader => f.write_str("file does not start with an IHDR chunk"),
            Error::MissingImageData => f.write_str("file contains no IDAT chunk"),
            Error::MissingPalette => f.write_str("indexed image has no PLTE chunk"),
            Error::InvalidChunkLength { chunk, length } => {
                write!(f, "invalid length {length} for `{}` chunk", chunk_name(chunk))
            }
            Error::InvalidChunkType { chunk } => {
                write!(f, "`{}` is not a writable ancillary chunk type", chunk_name(chunk))
            }
            Error::EmptyImage => f.write_str("image has zero width or height"),
            Error::ImageTooLarge => f.write_str("image dimensions exceed the addressable range"),
            Error::SizeLimitExceeded { size, limit } => {
                write!(f, "image expands to {size} bytes, over the {limit} byte limit")
            }
            Error::OutOfMemory { bytes } => {
                write!(f, "could not allocate {bytes} bytes for the image")
            }
            Error::InvalidColorType(value) => write!(f, "invalid colour type {value}"),
            Error::InvalidBitDepth { color_type, bit_depth } => {
                write!(f, "bit depth {bit_depth} is not valid for colour type {color_type}")
            }
            Error::UnsupportedMethod { field, value } => {
                write!(f, "unsupported {field} {value}")
            }
            Error::InvalidInterlaceMethod(value) => {
                write!(f, "invalid interlace method {value}")
            }
            Error::InvalidFilter { row } => write!(f, "invalid filter byte on row {row}"),
            Error::PaletteIndexOutOfRange => f.write_str("palette index out of range"),
            Error::Inflate(error) => write!(f, "{error}"),
            Error::UnknownCriticalChunk { chunk } => {
                write!(f, "unrecognised critical chunk `{}`", chunk_name(chunk))
            }
            Error::WrongBufferSize { expected, found } => {
                write!(f, "expected {expected} bytes of pixel data, got {found}")
            }
        }
    }
}

impl std::error::Error for Error {}

/// A chunk type, rendered for a message.
///
/// Escaped rather than transcribed: a type is meant to be four ASCII letters, but the errors
/// that name one are exactly the cases where it is not, and a raw control byte in a log line
/// is worse than useless.
fn chunk_name(chunk: &[u8; 4]) -> String {
    chunk.escape_ascii().to_string()
}
