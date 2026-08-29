//! Image description shared by the decoder and the encoder.

use crate::error::Error;

/// The eight bytes every PNG file begins with.
pub const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

/// How samples are laid out within a pixel.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum ColorType {
    /// One grey sample per pixel.
    Grayscale = 0,
    /// Three samples per pixel: red, green, blue.
    Rgb = 2,
    /// One palette index per pixel, with the colours themselves in `PLTE`.
    Indexed = 3,
    /// A grey sample and an alpha sample per pixel.
    GrayscaleAlpha = 4,
    /// Four samples per pixel: red, green, blue, alpha.
    Rgba = 6,
}

impl ColorType {
    /// Reads a colour type from its `IHDR` byte, which is the discriminant itself.
    ///
    /// PNG leaves 1, 5, 7 and everything above undefined, and those are the values that
    /// produce [`Error::InvalidColorType`].
    pub fn from_byte(byte: u8) -> Result<Self, Error> {
        match byte {
            0 => Ok(ColorType::Grayscale),
            2 => Ok(ColorType::Rgb),
            3 => Ok(ColorType::Indexed),
            4 => Ok(ColorType::GrayscaleAlpha),
            6 => Ok(ColorType::Rgba),
            other => Err(Error::InvalidColorType(other)),
        }
    }

    /// Number of samples stored per pixel. Indexed images store one index.
    #[inline]
    pub const fn samples(self) -> usize {
        match self {
            ColorType::Grayscale | ColorType::Indexed => 1,
            ColorType::GrayscaleAlpha => 2,
            ColorType::Rgb => 3,
            ColorType::Rgba => 4,
        }
    }

    /// Whether this colour type stores an alpha sample.
    #[inline]
    pub const fn has_alpha(self) -> bool {
        matches!(self, ColorType::GrayscaleAlpha | ColorType::Rgba)
    }
}

/// Bits per sample.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum BitDepth {
    /// One bit per sample; eight samples to the byte.
    One = 1,
    /// Two bits per sample; four samples to the byte.
    Two = 2,
    /// Four bits per sample; two samples to the byte.
    Four = 4,
    /// One byte per sample.
    Eight = 8,
    /// Two bytes per sample, stored most significant byte first.
    Sixteen = 16,
}

impl BitDepth {
    /// Reads a bit depth from its `IHDR` byte, which is the discriminant itself.
    ///
    /// `None` for a value PNG does not define. Whether a defined depth is legal for a
    /// particular colour type is a separate question, and not one this answers.
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            1 => Some(BitDepth::One),
            2 => Some(BitDepth::Two),
            4 => Some(BitDepth::Four),
            8 => Some(BitDepth::Eight),
            16 => Some(BitDepth::Sixteen),
            _ => None,
        }
    }

    /// Bits per sample, which is the discriminant itself.
    #[inline]
    pub const fn bits(self) -> usize {
        self as usize
    }
}

/// Whether the image is stored progressively.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Interlacing {
    /// Scanlines in order, top to bottom.
    None,
    /// Seven passes over a subsampled grid, so a partial file shows the whole image
    /// coarsely. See [`ADAM7_PASSES`].
    Adam7,
}

/// The seven Adam7 passes as `(x_start, y_start, x_step, y_step)`.
pub const ADAM7_PASSES: [(usize, usize, usize, usize); 7] = [
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
];

/// Width and height of one Adam7 pass over an image of the given size.
#[inline]
pub fn adam7_pass_size(pass: usize, width: usize, height: usize) -> (usize, usize) {
    let (x_start, y_start, x_step, y_step) = ADAM7_PASSES[pass];
    let pass_width = width.saturating_sub(x_start).div_ceil(x_step);
    let pass_height = height.saturating_sub(y_start).div_ceil(y_step);
    (pass_width, pass_height)
}

/// An ancillary chunk carried alongside the image.
///
/// PNG stores everything in typed chunks, and requires a decoder to skip any ancillary
/// chunk it does not recognise. That makes an ancillary chunk the place to put application
/// data inside a PNG: the payload is arbitrary bytes, up to `i32::MAX` of them, with no
/// escaping and no encoding, and every other PNG reader in the world ignores it.
///
/// The four type bytes carry meaning in their capitalisation, and [`Chunk::validate`]
/// enforces it:
///
/// | byte | lower case | upper case |
/// | --- | --- | --- |
/// | 1 | ancillary: an unknowing decoder skips it | critical: an unknowing decoder fails |
/// | 2 | private: yours alone | public: registered in the specification |
/// | 3 | reserved, and meaningless so far | what a conforming type uses |
/// | 4 | an editor that does not understand it may copy it through | unsafe to copy |
///
/// So a private chunk for an application uses a lower-case first byte, a lower-case second
/// byte, an upper-case third, and a lower-case fourth if an editor that rewrites the image
/// should carry it across: `b"apPd"`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Chunk {
    /// The four-byte chunk type.
    pub kind: [u8; 4],
    /// The chunk payload, stored verbatim.
    pub data: Vec<u8>,
}

impl Chunk {
    /// Pairs a four-byte chunk type with its payload.
    ///
    /// Neither is checked here: the encoder is where an unwritable type is refused, and
    /// [`Chunk::validate`] is where the rules are.
    pub fn new(kind: [u8; 4], data: Vec<u8>) -> Self {
        Self { kind, data }
    }

    /// Checks that this chunk is one the encoder can write.
    ///
    /// The type must be four ASCII letters, its first lower case, and its third upper case. A
    /// critical type is rejected because the encoder writes the critical chunks itself and a
    /// decoder must fail on a critical type it does not know, so accepting one would produce a
    /// file png-spark could not read back. A lower-case third byte is reserved by the
    /// specification and means nothing yet.
    ///
    /// `tRNS` is excluded for the same reason as the critical types: the encoder writes it
    /// itself from [`Info::transparency`], where it is checked against the colour type. A second
    /// one here would be an illegal duplicate, would silently displace the real transparency on
    /// read-back, and if its length did not suit the colour type would produce a file png-spark
    /// itself rejects.
    ///
    /// The payload is limited to `i32::MAX` bytes, the longest a PNG chunk may declare.
    ///
    /// Both sides of the library apply this, so anything the decoder hands back in
    /// [`Info::metadata`] can be handed to the encoder again.
    pub fn validate(&self) -> Result<(), Error> {
        if !writable_kind(self.kind) {
            return Err(Error::InvalidChunkType { chunk: self.kind });
        }
        if self.data.len() > i32::MAX as usize {
            return Err(Error::InvalidChunkLength { chunk: self.kind, length: self.data.len() });
        }
        Ok(())
    }
}

/// Whether `kind` names a chunk png-spark will carry as metadata.
///
/// The rules, and the reasons behind them, are documented on [`Chunk::validate`].
pub(crate) fn writable_kind(kind: [u8; 4]) -> bool {
    kind.iter().all(u8::is_ascii_alphabetic)
        && kind[0].is_ascii_lowercase()
        && kind[2].is_ascii_uppercase()
        && kind != *b"tRNS"
}

/// Everything `IHDR` and the colour chunks say about an image.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Info {
    /// Width in pixels. Zero is not a valid image.
    pub width: u32,
    /// Height in pixels. Zero is not a valid image.
    pub height: u32,
    /// How samples are laid out within a pixel.
    ///
    /// Not the whole story on transparency, since `tRNS` carries alpha for colour types
    /// that have no alpha sample. [`Info::has_alpha`] is the question to ask about an image.
    pub color_type: ColorType,
    /// Bits per sample. Not every depth is legal for every colour type.
    pub bit_depth: BitDepth,
    /// Whether the file stores the image progressively.
    ///
    /// The decoder resolves Adam7 and hands back a normal image; the encoder only writes
    /// [`Interlacing::None`].
    pub interlacing: Interlacing,
    /// `PLTE` contents: RGB triples, present for indexed images.
    pub palette: Option<Vec<u8>>,
    /// `tRNS` contents, in the layout its colour type defines.
    pub transparency: Option<Vec<u8>>,
    /// Ancillary chunks carried with the image.
    ///
    /// Empty unless the encoder is given some or the decoder is asked to keep them; see
    /// [`Chunk`] and [`Keep`](crate::decoder::Keep).
    ///
    /// The encoder writes these in order, except that it moves the few types the
    /// specification places before `PLTE` ahead of it, so a file written from a list that
    /// was in the wrong order comes back in the right one.
    pub metadata: Vec<Chunk>,
}

impl Info {
    /// Describes a non-interlaced image with no palette, transparency or metadata.
    ///
    /// The remaining fields are public, so an image that needs them sets them afterwards or
    /// builds the struct directly.
    pub fn new(width: u32, height: u32, color_type: ColorType, bit_depth: BitDepth) -> Self {
        Self {
            width,
            height,
            color_type,
            bit_depth,
            interlacing: Interlacing::None,
            palette: None,
            transparency: None,
            metadata: Vec::new(),
        }
    }

    /// The payload of the first metadata chunk of this type, if one is present.
    pub fn chunk(&self, kind: &[u8; 4]) -> Option<&[u8]> {
        self.metadata.iter().find(|chunk| &chunk.kind == kind).map(|chunk| &chunk.data[..])
    }

    /// Whether any pixel of the image can be other than fully opaque.
    ///
    /// PNG spells transparency two ways, and the colour type only tells half the story: a
    /// palette or greyscale image carries its alpha in `tRNS` instead of in the pixels, so
    /// its colour type stays [`Indexed`](ColorType::Indexed) or
    /// [`Grayscale`](ColorType::Grayscale) while the image is nonetheless transparent.
    ///
    /// Decoders that expand every image to RGBA hide that distinction. png-spark hands back
    /// the file's own format, so it does not, and a caller that reads the colour type alone
    /// silently loses the alpha of every palette image. Ask here instead.
    ///
    /// This is a question about the format, not about the pixels: it is true for an image
    /// whose alpha channel happens to be opaque throughout, and for a `tRNS` chunk none of
    /// whose entries any pixel actually uses.
    ///
    /// [`ColorType::has_alpha`] answers the narrower question of whether the pixels
    /// themselves carry an alpha sample. That is the one to ask about a layout; this is the
    /// one to ask about an image.
    #[inline]
    pub fn has_alpha(&self) -> bool {
        self.color_type.has_alpha() || self.transparency.is_some()
    }

    /// Bits occupied by one pixel.
    #[inline]
    pub const fn bits_per_pixel(&self) -> usize {
        self.color_type.samples() * self.bit_depth.bits()
    }

    /// Filter stride: bytes per pixel, rounded up to one byte for sub-byte depths.
    ///
    /// The filters operate on whole bytes, so packed formats use a stride of one.
    #[inline]
    pub const fn filter_stride(&self) -> usize {
        let bits = self.bits_per_pixel();
        if bits < 8 { 1 } else { bits / 8 }
    }

    /// Bytes in one scanline of a non-interlaced image.
    #[inline]
    pub const fn row_bytes(&self) -> usize {
        row_bytes_for(self.width as usize, self.bits_per_pixel())
    }

    /// Bytes in the fully decoded image, without filter bytes.
    #[inline]
    pub const fn output_size(&self) -> usize {
        self.row_bytes() * self.height as usize
    }

    /// Bytes the compressed stream expands to, including one filter byte per scanline.
    ///
    /// For interlaced images every non-empty pass contributes its own scanlines.
    pub fn decompressed_size(&self) -> usize {
        match self.interlacing {
            Interlacing::None => (1 + self.row_bytes()) * self.height as usize,
            Interlacing::Adam7 => {
                let bits = self.bits_per_pixel();
                (0..7)
                    .map(|pass| {
                        let (w, h) =
                            adam7_pass_size(pass, self.width as usize, self.height as usize);
                        if w == 0 || h == 0 { 0 } else { (1 + row_bytes_for(w, bits)) * h }
                    })
                    .sum()
            }
        }
    }

    /// Validates the combination of colour type and bit depth against the specification.
    pub fn validate(&self) -> Result<(), Error> {
        let depth = self.bit_depth.bits() as u8;
        let allowed = match self.color_type {
            ColorType::Grayscale => matches!(depth, 1 | 2 | 4 | 8 | 16),
            ColorType::Rgb | ColorType::GrayscaleAlpha | ColorType::Rgba => {
                matches!(depth, 8 | 16)
            }
            ColorType::Indexed => matches!(depth, 1 | 2 | 4 | 8),
        };
        if !allowed {
            return Err(Error::InvalidBitDepth {
                color_type: self.color_type as u8,
                bit_depth: depth,
            });
        }
        if self.width == 0 || self.height == 0 {
            return Err(Error::EmptyImage);
        }
        // Reject sizes whose scanline arithmetic would not fit in a `usize`.
        let pixels = (self.width as u64) * (self.height as u64);
        let bytes = pixels.saturating_mul(self.bits_per_pixel() as u64) / 8 + self.height as u64;
        if bytes > usize::MAX as u64 / 2 {
            return Err(Error::ImageTooLarge);
        }
        Ok(())
    }
}

/// Allocates `len` zeroed bytes, returning `None` rather than aborting when the allocator
/// cannot provide them.
///
/// `vec![0u8; len]` compiles to this same `alloc_zeroed` call, and for buffers of image size
/// that matters: the allocator hands back pages the operating system has already zeroed
/// instead of writing over freshly mapped memory, so the zeroing is free. What `vec!` also
/// does is abort the process on failure, which is not a library's decision to make. Calling
/// the allocator directly keeps the free zeroing and turns refusal into a value.
pub(crate) fn zeroed_vec(len: usize) -> Option<Vec<u8>> {
    if len == 0 {
        return Some(Vec::new());
    }
    // Fails only when `len` exceeds `isize::MAX`, which no allocator would satisfy either.
    let layout = std::alloc::Layout::from_size_align(len, 1).ok()?;
    // SAFETY: `len` is non-zero above, so the layout has non-zero size.
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        return None;
    }
    // SAFETY: `ptr` was just allocated by the global allocator under a layout of exactly
    // `len` bytes at align 1, which is what a `Vec<u8>` of length and capacity `len` requires
    // of its buffer, and `alloc_zeroed` initialised every one of those bytes.
    Some(unsafe { Vec::from_raw_parts(ptr, len, len) })
}

/// Bytes needed for `width` pixels of `bits_per_pixel` bits each.
// `div_ceil` says this more directly but is not const-callable until Rust 1.83.
#[allow(clippy::manual_div_ceil)]
#[inline]
pub const fn row_bytes_for(width: usize, bits_per_pixel: usize) -> usize {
    (width * bits_per_pixel + 7) / 8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeroed_vec_hands_back_an_owned_zeroed_buffer() {
        // Built from a raw allocation rather than `vec![]`, so what needs checking is that
        // the result behaves as a `Vec` in every respect: correct length, every byte
        // initialised to zero, writable, and freed under the layout it was allocated with.
        assert_eq!(zeroed_vec(0).unwrap(), Vec::<u8>::new());

        for len in [1usize, 7, 16, 1000, 65_536] {
            let mut buffer = zeroed_vec(len).unwrap();
            assert_eq!(buffer.len(), len);
            assert_eq!(buffer.capacity(), len);
            assert!(buffer.iter().all(|&byte| byte == 0), "{len} bytes were not zeroed");

            buffer[len - 1] = 0xAB;
            buffer.push(0xCD);
            assert_eq!(buffer.len(), len + 1);
        }

        // A size no `Layout` can describe is refused before the allocator is asked, and a
        // size it can describe but no allocator can satisfy is refused by the null check.
        // Both paths have to work: the first is arithmetic, the second is the whole reason
        // this function exists.
        assert!(zeroed_vec(usize::MAX).is_none());
        if !cfg!(miri) {
            // Both `black_box` calls are load-bearing, and this assertion silently passes
            // without them at any optimisation level above zero. The inner one stops the
            // size being folded through `Layout::from_size_align` at compile time; the outer
            // one stops the allocation being removed as dead, which takes the null check
            // with it and leaves the refusal looking like a success. Skipped under Miri,
            // which treats a request this large as an error of its own rather than modelling
            // the null a real allocator returns.
            use std::hint::black_box;
            assert!(black_box(zeroed_vec(black_box(isize::MAX as usize))).is_none());
        }
    }
}
