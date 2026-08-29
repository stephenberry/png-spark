//! PNG decoding.

use crate::common::{
    ADAM7_PASSES, BitDepth, Chunk, ColorType, Info, Interlacing, SIGNATURE, adam7_pass_size,
    row_bytes_for, writable_kind, zeroed_vec,
};
use crate::crc32::crc32;
use crate::error::Error;
use crate::filter::unfilter_image;
use crate::inflate::{Inflater, OUTPUT_SLACK};

/// A decoded image and the description of its pixel layout.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Image {
    /// The image's dimensions and layout, along with any chunks the decoder was asked to keep.
    pub info: Info,
    /// Pixel data in the image's own format, tightly packed, with no filter bytes and no
    /// padding between rows beyond what a sub-byte bit depth requires.
    pub data: Vec<u8>,
}

impl Image {
    /// Width in pixels.
    #[inline]
    pub fn width(&self) -> u32 {
        self.info.width
    }

    /// Height in pixels.
    #[inline]
    pub fn height(&self) -> u32 {
        self.info.height
    }

    /// How samples are laid out within a pixel of [`Image::data`].
    #[inline]
    pub fn color_type(&self) -> ColorType {
        self.info.color_type
    }

    /// Bits per sample in [`Image::data`].
    #[inline]
    pub fn bit_depth(&self) -> BitDepth {
        self.info.bit_depth
    }
}

/// Which integrity checks the decoder performs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Checks {
    /// Verify the CRC of every chunk. This is the default.
    ///
    /// A chunk that fails is an error when it is critical and dropped when it is ancillary,
    /// since nothing the decoder returns is built from an ancillary chunk.
    ///
    /// The compressed stream also carries an Adler-32 over the data it expands to, but that
    /// covers the same bytes a second time: if the `IDAT` payload is intact and the
    /// decompressor is correct, so is its output. Checking only the CRC therefore detects
    /// the same file corruption for one pass over the data instead of two.
    Crc,
    /// Verify chunk CRCs *and* the Adler-32 of the decompressed data.
    ///
    /// Worth the second pass when the decompressed data must be guarded against faults that
    /// arise after the CRC has been checked, such as memory errors.
    Full,
    /// Verify nothing, and accept whatever the file contains.
    None,
}

/// Which ancillary chunks a decode retains.
///
/// Nothing the decoder returns is built from an ancillary chunk, so by default they are
/// skipped without being copied. Retaining one costs an allocation and a copy of its
/// payload, which is why it is asked for rather than assumed.
///
/// A chunk that fails its CRC is dropped before this is consulted, so a retained chunk has
/// been verified unless [`Checks::None`] is in force.
///
/// A chunk whose type the encoder would refuse is dropped too, however it was asked for: a
/// type that is not four ASCII letters is malformed, and `tRNS` is read into
/// [`Info::transparency`] instead. Everything retained can therefore be written back out.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub enum Keep {
    /// Retain nothing. This is the default.
    #[default]
    None,
    /// Retain the ancillary chunks whose type appears in this list.
    Only(Vec<[u8; 4]>),
    /// Retain every ancillary chunk the file carries.
    ///
    /// The retained payloads are bounded by the size of the file they came from.
    All,
}

impl Keep {
    #[inline]
    fn wants(&self, kind: [u8; 4]) -> bool {
        match self {
            Keep::None => false,
            Keep::Only(kinds) => kinds.contains(&kind),
            Keep::All => true,
        }
    }
}

/// The decoder's default ceiling on [`Info::decompressed_size`], in bytes.
///
/// Half a gigabyte admits any photograph or screen capture a caller is likely to have meant
/// to decode: a 16-bit RGBA image of 8000 by 8000 pixels fits inside it. What it refuses is
/// the class of file that only exists to be refused.
pub const DEFAULT_MAX_DECOMPRESSED_SIZE: usize = 512 << 20;

/// A reusable PNG decoder.
///
/// Reusing one decoder across images keeps the Huffman decoding tables allocated, which is
/// worth roughly 20 KiB of allocation and initialisation per image.
pub struct Decoder {
    inflater: Inflater,
    checks: Checks,
    keep: Keep,
    max_decompressed_size: Option<usize>,
}

impl core::fmt::Debug for Decoder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Decoder")
            .field("checks", &self.checks)
            .field("keep", &self.keep)
            .field("max_decompressed_size", &self.max_decompressed_size)
            .finish_non_exhaustive()
    }
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder {
    /// A decoder that checks chunk CRCs, keeps no metadata, and refuses an image declaring
    /// more than [`DEFAULT_MAX_DECOMPRESSED_SIZE`] bytes.
    pub fn new() -> Self {
        Self {
            inflater: Inflater::new(),
            checks: Checks::Crc,
            keep: Keep::None,
            max_decompressed_size: Some(DEFAULT_MAX_DECOMPRESSED_SIZE),
        }
    }

    /// Selects which integrity checks to perform. See [`Checks`].
    pub fn checks(&mut self, checks: Checks) -> &mut Self {
        self.checks = checks;
        self
    }

    /// Selects which ancillary chunks to retain. See [`Keep`].
    ///
    /// Retained chunks arrive in [`Info::metadata`], in file order, and are written back out
    /// by [`Encoder`](crate::Encoder) if that `Info` is handed to it again.
    ///
    /// ```no_run
    /// use png_spark::{Decoder, Keep};
    ///
    /// let mut decoder = Decoder::new();
    /// decoder.keep(Keep::Only(vec![*b"apPd"]));
    /// let image = decoder.decode(&std::fs::read("asset.png")?)?;
    /// let payload = image.info.chunk(b"apPd");
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn keep(&mut self, keep: Keep) -> &mut Self {
        self.keep = keep;
        self
    }

    /// Sets the largest [`Info::decompressed_size`] this decoder will accept, or `None` to
    /// accept any size the platform can address.
    ///
    /// `IHDR` is thirteen bytes and states the image's dimensions, and the decoder needs a
    /// buffer of the size they imply before it can read a single compressed byte. Nothing in
    /// the file has to justify that number: a seventy-byte PNG can name a width and height
    /// whose product runs to petabytes, and a decoder that believes it will ask the allocator
    /// for petabytes. PNG is a format that arrives from elsewhere far more often than it is
    /// written locally, so an unbounded decoder is a way to take a process down from across
    /// a network, and the ceiling is [`DEFAULT_MAX_DECOMPRESSED_SIZE`] rather than absent.
    ///
    /// The limit is on the decompressed data, filter bytes included, and not on pixels: the
    /// same pixel count spans a sixty-four-fold range of buffer sizes across PNG's colour
    /// types and bit depths, from one bit a pixel to sixty-four, so a pixel count does not
    /// bound what gets allocated.
    ///
    /// It bounds the largest single buffer, not the decode's peak. An interlaced image is
    /// unfiltered pass by pass into a second buffer of its own, so Adam7 holds close to
    /// twice the limit at once; every other image holds it once.
    ///
    /// A header over the limit is [`Error::SizeLimitExceeded`], reported before anything
    /// image-sized is allocated. Raising it for a caller who really does read hundred-
    /// megapixel images is a one-liner, and passing `None` restores the unbounded behaviour
    /// outright:
    ///
    /// ```
    /// use png_spark::Decoder;
    ///
    /// let mut decoder = Decoder::new();
    /// decoder.max_decompressed_size(Some(4 << 30));
    /// ```
    ///
    /// [`read_info`](Self::read_info) is not bounded by this, because it allocates nothing
    /// that depends on the dimensions. A caller wanting to apply a policy of its own can
    /// read the header, decide, and only then decode.
    pub fn max_decompressed_size(&mut self, bytes: Option<usize>) -> &mut Self {
        self.max_decompressed_size = bytes;
        self
    }

    /// Decodes a PNG into its native pixel format.
    pub fn decode(&mut self, png: &[u8]) -> Result<Image, Error> {
        let parsed = self.parse(png)?;
        let info = parsed.info;

        self.inflater.verify_checksum(self.checks == Checks::Full);
        let request = info.decompressed_size() + OUTPUT_SLACK;
        let mut buffer = zeroed_vec(request).ok_or(Error::OutOfMemory { bytes: request })?;
        match parsed.idat {
            IdatData::Contiguous(data) => self.inflater.zlib(data, &mut buffer)?,
            IdatData::Joined(data) => self.inflater.zlib(&data, &mut buffer)?,
        };

        let data = match info.interlacing {
            Interlacing::None => {
                unfilter_image(
                    &mut buffer,
                    info.row_bytes(),
                    info.height as usize,
                    info.filter_stride(),
                )
                .map_err(|row| Error::InvalidFilter { row })?;
                buffer.truncate(info.output_size());
                buffer
            }
            Interlacing::Adam7 => deinterlace(&info, &mut buffer)?,
        };

        if info.color_type == ColorType::Indexed {
            let palette = info.palette.as_ref().ok_or(Error::MissingPalette)?;
            validate_palette_indices(&data, &info, palette.len() / 3)?;
        }

        Ok(Image { info, data })
    }

    fn parse<'a>(&self, png: &'a [u8]) -> Result<Parsed<'a>, Error> {
        let (mut info, mut chunks) = open(png, self.checks)?;

        // Checked here rather than at the allocation, so a header naming a petabyte costs
        // the thirty-three bytes already read and not a scan of whatever follows it.
        if let Some(limit) = self.max_decompressed_size {
            let size = info.decompressed_size();
            if size > limit {
                return Err(Error::SizeLimitExceeded { size, limit });
            }
        }

        // The first IDAT is remembered separately so that the common single-chunk case can
        // borrow the compressed bytes instead of copying them.
        let mut first_idat: Option<&[u8]> = None;
        let mut joined: Option<Vec<u8>> = None;

        while let Some(chunk) = chunks.next()? {
            match &chunk.kind {
                b"IEND" => break,
                b"IDAT" => match (&mut joined, first_idat) {
                    (Some(buffer), _) => buffer.extend_from_slice(chunk.data),
                    (None, Some(first)) => {
                        let mut buffer = Vec::with_capacity(first.len() * 2 + chunk.data.len());
                        buffer.extend_from_slice(first);
                        buffer.extend_from_slice(chunk.data);
                        joined = Some(buffer);
                    }
                    (None, None) => first_idat = Some(chunk.data),
                },
                _ => absorb(&mut info, &chunk, &self.keep, first_idat.is_none())?,
            }
        }

        if info.color_type == ColorType::Indexed && info.palette.is_none() {
            return Err(Error::MissingPalette);
        }

        let idat = match joined {
            Some(buffer) => IdatData::Joined(buffer),
            None => IdatData::Contiguous(first_idat.ok_or(Error::MissingImageData)?),
        };

        Ok(Parsed { info, idat })
    }

    /// Reads everything a file says about its image, without decoding any of it.
    ///
    /// Parsing stops at the first `IDAT`, so the cost is the header and whatever colour
    /// chunks precede the image data, and nothing is allocated whose size depends on the
    /// dimensions. That makes this the seam for a caller who wants to decide something about
    /// an image before committing to decoding it, whether that is a size policy stricter
    /// than [`max_decompressed_size`](Self::max_decompressed_size), a colour type it has no
    /// use for, or simply the dimensions.
    ///
    /// Everything the decode would read before the pixels is present, `PLTE` and `tRNS`
    /// included, so [`Info::has_alpha`] answers correctly here. What is absent is any
    /// ancillary chunk that trails the image data, since reaching one would mean reading
    /// the whole file.
    ///
    /// Only the chunks up to the first `IDAT` are examined, so this is the weaker check of
    /// the two in both directions. Nothing that only the pixels could contradict is caught,
    /// such as a palette an index runs past; neither is a structural fault in the tail of
    /// the file, such as an unknown critical chunk, a bad CRC, or a truncated trailer after
    /// the image data. A header this accepts can still fail to decode. A file this rejects
    /// would never have decoded.
    ///
    /// ```
    /// # let png = png_spark::encode_rgba8(2, 2, &[0; 16])?;
    /// let info = png_spark::Decoder::new().read_info(&png)?;
    /// assert_eq!((info.width, info.height), (2, 2));
    /// assert!(info.has_alpha());
    /// # Ok::<(), png_spark::Error>(())
    /// ```
    pub fn read_info(&self, png: &[u8]) -> Result<Info, Error> {
        read_header(png, self.checks, &self.keep)
    }
}

/// Reads the signature and `IHDR`, leaving the reader positioned on the chunk after them.
///
/// Every entry point starts this way, and the two that follow it differ only in what they do
/// with the rest of the file, so sharing the prologue is what keeps them from drifting apart
/// on what counts as a valid header.
fn open(png: &[u8], checks: Checks) -> Result<(Info, ChunkReader<'_>), Error> {
    if png.len() < SIGNATURE.len() || png[..SIGNATURE.len()] != SIGNATURE {
        return Err(Error::NotAPng);
    }

    let mut chunks =
        ChunkReader { data: png, pos: SIGNATURE.len(), verify: checks != Checks::None };

    let header = chunks.next()?.ok_or(Error::MissingHeader)?;
    if &header.kind != b"IHDR" {
        return Err(Error::MissingHeader);
    }
    Ok((parse_ihdr(header.data)?, chunks))
}

/// Folds one chunk that is neither `IDAT` nor `IEND` into the header being built.
///
/// `before_idat` says whether the image data has started. `PLTE` and `tRNS` that follow it
/// are dropped rather than recorded: the specification puts both ahead of the image data,
/// and honouring a late one would mean the pixels depended on an ordering this decoder does
/// not otherwise respect.
fn absorb(
    info: &mut Info,
    chunk: &RawChunk<'_>,
    keep: &Keep,
    before_idat: bool,
) -> Result<(), Error> {
    match &chunk.kind {
        b"PLTE" => {
            if chunk.data.len() > 256 * 3 || !chunk.data.len().is_multiple_of(3) {
                return Err(Error::InvalidChunkLength {
                    chunk: chunk.kind,
                    length: chunk.data.len(),
                });
            }
            if before_idat {
                info.palette = Some(chunk.data.to_vec());
            }
        }
        b"tRNS" => {
            if before_idat {
                validate_trns(info, chunk.data)?;
                info.transparency = Some(chunk.data.to_vec());
            }
        }
        kind if is_critical(*kind) => {
            // Critical chunks we do not understand may change how the image should be
            // interpreted, so decoding cannot safely continue.
            return Err(Error::UnknownCriticalChunk { chunk: *kind });
        }
        kind if keep.wants(*kind) && writable_kind(*kind) => {
            info.metadata.push(Chunk { kind: *kind, data: chunk.data.to_vec() });
        }
        _ => {}
    }
    Ok(())
}

/// Reads the header and the colour chunks ahead of the image data. See
/// [`Decoder::read_info`].
///
/// Free of the [`Decoder`] so that the standalone [`read_info`] can call it without building
/// one: a `Decoder` owns an [`Inflater`], and reading a header should not cost the twenty
/// kilobytes of Huffman tables that decoding one does.
fn read_header(png: &[u8], checks: Checks, keep: &Keep) -> Result<Info, Error> {
    let (mut info, mut chunks) = open(png, checks)?;

    while let Some(chunk) = chunks.next()? {
        match &chunk.kind {
            b"IDAT" => {
                if info.color_type == ColorType::Indexed && info.palette.is_none() {
                    return Err(Error::MissingPalette);
                }
                return Ok(info);
            }
            b"IEND" => break,
            _ => absorb(&mut info, &chunk, keep, true)?,
        }
    }

    Err(Error::MissingImageData)
}

struct Parsed<'a> {
    info: Info,
    idat: IdatData<'a>,
}

enum IdatData<'a> {
    /// A single `IDAT` chunk, borrowed straight from the input.
    Contiguous(&'a [u8]),
    /// Several `IDAT` chunks, concatenated.
    Joined(Vec<u8>),
}

/// A chunk borrowed from the input, as the reader yields it.
struct RawChunk<'a> {
    kind: [u8; 4],
    data: &'a [u8],
}

struct ChunkReader<'a> {
    data: &'a [u8],
    pos: usize,
    verify: bool,
}

impl<'a> ChunkReader<'a> {
    /// Yields the next chunk that passes its CRC.
    ///
    /// A bad CRC on a critical chunk is an error, because the image depends on it. On an
    /// ancillary chunk it is not: nothing the decoder produces depends on one, and files
    /// exist whose pixels are perfectly intact but whose colour profile or text metadata was
    /// rewritten without recomputing its checksum. Refusing those would lose an image over
    /// four stale bytes describing something that is discarded anyway, so the chunk is
    /// dropped and reading continues, as libpng and the `png` crate both do.
    ///
    /// Skipping still trusts the chunk's length field to find the next header, since the CRC
    /// does not cover it. A length corrupted along with the body resynchronises somewhere
    /// arbitrary, which then fails as a truncated or unknown critical chunk.
    fn next(&mut self) -> Result<Option<RawChunk<'a>>, Error> {
        loop {
            if self.pos == self.data.len() {
                return Ok(None);
            }
            if self.pos + 8 > self.data.len() {
                return Err(Error::TruncatedChunk);
            }

            let length = u32::from_be_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
            // The specification caps chunk lengths at 2^31 - 1.
            if length > i32::MAX as u32 {
                return Err(Error::TruncatedChunk);
            }
            let length = length as usize;

            let kind: [u8; 4] = self.data[self.pos + 4..self.pos + 8].try_into().unwrap();
            let body_start = self.pos + 8;
            let end = body_start + length + 4;
            if end > self.data.len() {
                return Err(Error::TruncatedChunk);
            }

            let data = &self.data[body_start..body_start + length];
            if self.verify {
                let stored =
                    u32::from_be_bytes(self.data[body_start + length..end].try_into().unwrap());
                if crc32(&self.data[self.pos + 4..body_start + length]) != stored {
                    if is_critical(kind) {
                        return Err(Error::BadChunkCrc { chunk: kind });
                    }
                    self.pos = end;
                    continue;
                }
            }

            self.pos = end;
            return Ok(Some(RawChunk { kind, data }));
        }
    }
}

/// A chunk is critical when the first letter of its type is upper case.
fn is_critical(kind: [u8; 4]) -> bool {
    kind[0].is_ascii_uppercase()
}

fn parse_ihdr(data: &[u8]) -> Result<Info, Error> {
    if data.len() != 13 {
        return Err(Error::InvalidChunkLength { chunk: *b"IHDR", length: data.len() });
    }

    let width = u32::from_be_bytes(data[0..4].try_into().unwrap());
    let height = u32::from_be_bytes(data[4..8].try_into().unwrap());
    let bit_depth = BitDepth::from_byte(data[8])
        .ok_or(Error::InvalidBitDepth { color_type: data[9], bit_depth: data[8] })?;
    let color_type = ColorType::from_byte(data[9])?;

    if data[10] != 0 {
        return Err(Error::UnsupportedMethod { field: "compression method", value: data[10] });
    }
    if data[11] != 0 {
        return Err(Error::UnsupportedMethod { field: "filter method", value: data[11] });
    }
    let interlacing = match data[12] {
        0 => Interlacing::None,
        1 => Interlacing::Adam7,
        other => return Err(Error::InvalidInterlaceMethod(other)),
    };

    let mut info = Info::new(width, height, color_type, bit_depth);
    info.interlacing = interlacing;
    info.validate()?;
    Ok(info)
}

/// Checks a `tRNS` chunk against the colour type it accompanies.
///
/// Shared with the encoder, which must not write a chunk it would itself reject.
pub(crate) fn validate_trns(info: &Info, data: &[u8]) -> Result<(), Error> {
    let expected = match info.color_type {
        ColorType::Grayscale => 2,
        ColorType::Rgb => 6,
        // For indexed images tRNS gives one alpha per palette entry, and may be short.
        ColorType::Indexed => {
            if data.len() > 256 {
                return Err(Error::InvalidChunkLength { chunk: *b"tRNS", length: data.len() });
            }
            return Ok(());
        }
        ColorType::GrayscaleAlpha | ColorType::Rgba => {
            return Err(Error::InvalidChunkLength { chunk: *b"tRNS", length: data.len() });
        }
    };
    if data.len() != expected {
        return Err(Error::InvalidChunkLength { chunk: *b"tRNS", length: data.len() });
    }
    Ok(())
}

/// Rejects palette indices with no matching `PLTE` entry, so later conversions can index the
/// palette without a per-pixel range check.
fn validate_palette_indices(data: &[u8], info: &Info, entries: usize) -> Result<(), Error> {
    if entries == 0 {
        return Err(Error::MissingPalette);
    }
    match info.bit_depth {
        BitDepth::Eight => {
            if data.iter().any(|&index| index as usize >= entries) {
                return Err(Error::PaletteIndexOutOfRange);
            }
        }
        depth => {
            // Sub-byte indices can only exceed the palette if the palette is smaller than
            // the depth's full range, which is rare enough to check the cheap way first.
            let max = (1usize << depth.bits()) - 1;
            if max >= entries {
                let width = info.width as usize;
                let bits = depth.bits();
                let row_bytes = info.row_bytes();
                for row in 0..info.height as usize {
                    let line = &data[row * row_bytes..(row + 1) * row_bytes];
                    for x in 0..width {
                        let bit = x * bits;
                        let index = (line[bit / 8] >> (8 - bits - bit % 8)) & max as u8;
                        if index as usize >= entries {
                            return Err(Error::PaletteIndexOutOfRange);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Unfilters each Adam7 pass in place and scatters the passes into a single image.
fn deinterlace(info: &Info, buffer: &mut [u8]) -> Result<Vec<u8>, Error> {
    let width = info.width as usize;
    let height = info.height as usize;
    let bits = info.bits_per_pixel();
    let row_bytes = info.row_bytes();
    let stride = info.filter_stride();

    let request = row_bytes * height;
    let mut image = zeroed_vec(request).ok_or(Error::OutOfMemory { bytes: request })?;
    let mut offset = 0usize;
    let mut row_counter = 0usize;

    for (pass, &(x_start, y_start, x_step, y_step)) in ADAM7_PASSES.iter().enumerate() {
        let (pass_width, pass_height) = adam7_pass_size(pass, width, height);
        if pass_width == 0 || pass_height == 0 {
            continue;
        }
        let pass_row_bytes = row_bytes_for(pass_width, bits);
        let region = &mut buffer[offset..offset + pass_height * (1 + pass_row_bytes)];

        unfilter_image(region, pass_row_bytes, pass_height, stride)
            .map_err(|row| Error::InvalidFilter { row: row_counter + row })?;

        for row in 0..pass_height {
            let source = &region[row * pass_row_bytes..(row + 1) * pass_row_bytes];
            let target_row = y_start + row * y_step;
            let target = &mut image[target_row * row_bytes..(target_row + 1) * row_bytes];
            scatter_row(source, target, pass_width, x_start, x_step, bits);
        }

        offset += pass_height * (1 + pass_row_bytes);
        row_counter += pass_height;
    }

    Ok(image)
}

/// Writes the pixels of one interlace pass row into their positions in the full row.
fn scatter_row(
    source: &[u8],
    target: &mut [u8],
    pass_width: usize,
    x_start: usize,
    x_step: usize,
    bits: usize,
) {
    if bits >= 8 {
        let pixel = bits / 8;
        for k in 0..pass_width {
            let to = (x_start + k * x_step) * pixel;
            target[to..to + pixel].copy_from_slice(&source[k * pixel..(k + 1) * pixel]);
        }
    } else {
        let mask = (1u8 << bits) - 1;
        for k in 0..pass_width {
            let from_bit = k * bits;
            let value = (source[from_bit / 8] >> (8 - bits - from_bit % 8)) & mask;
            let to_bit = (x_start + k * x_step) * bits;
            let shift = 8 - bits - to_bit % 8;
            let slot = &mut target[to_bit / 8];
            *slot = (*slot & !(mask << shift)) | (value << shift);
        }
    }
}

/// Decodes a PNG into its native pixel format.
pub fn decode(png: &[u8]) -> Result<Image, Error> {
    Decoder::new().decode(png)
}

/// Reads a PNG's header and colour chunks without decoding its pixels.
///
/// See [`Decoder::read_info`].
pub fn read_info(png: &[u8]) -> Result<Info, Error> {
    read_header(png, Checks::Crc, &Keep::None)
}
