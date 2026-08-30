//! PNG encoding.

use crate::common::{Chunk, ColorType, Info, SIGNATURE};
use crate::crc32::Crc32;
use crate::deflate::Deflater;
use crate::error::{Error, WriteError};
use crate::filter::{Filter, filter_row};

/// How the encoder picks a filter for each scanline.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum FilterStrategy {
    /// Score all five filters on every byte of the row and keep the best.
    ///
    /// The most thorough option, and the most expensive: it walks the row five times before
    /// writing it once.
    Adaptive,
    /// Score all five filters on a sample of the row and keep the best.
    ///
    /// A scanline is homogeneous enough that a fraction of its pixels ranks the filters the
    /// same way the whole row does, so this reaches essentially the same sizes as
    /// [`Self::Adaptive`] for a fraction of the work. This is the default.
    #[default]
    Sampled,
    /// Always use the given filter.
    ///
    /// Useful when the data's structure is already known: `Up` for images that vary slowly
    /// down the page, `None` for data that is already incompressible.
    Fixed(Filter),
}

/// How much better than the previous row's filter another filter must score to displace it,
/// as a right shift of its score.
///
/// Consecutive rows filtered the same way stay similar to each other, and that similarity is
/// what the compressor turns into matches one scanline apart. A row that switches filters
/// resembles nothing before it, so a switch has to earn more than the small local gain the
/// residual sum can see. One eighth is enough to stop the choice flapping between filters
/// that score within noise of each other, without holding on to a filter that has genuinely
/// stopped suiting the image.
const STICKY_SHIFT: u32 = 3;

/// How much better than the rest Paeth must score to be chosen, as a percentage of its own
/// residual sum.
///
/// The five filters do not cost the same to reverse. Paeth's predictor depends on three
/// neighbours at once and takes a dozen operations to evaluate, where `Up` is a packed add
/// and `Sub` a single carry; a decoder spends the bulk of its reconstruction time on
/// whatever share of the rows carry Paeth. A row that chooses it over `Up` because the
/// residuals came out one percent smaller has bought almost nothing and charged that cost
/// to every future reader of the file.
///
/// A tenth is enough to settle those marginal rows the other way while leaving Paeth
/// wherever it genuinely suits the image. Over a corpus of 2863 files it moves Paeth from
/// 65% of scanline bytes to 49%, costs 0.22% in compressed size, and returns 8% in decode
/// throughput; no file in that corpus grew by more than 14%, and a third of them came out
/// smaller, the marginal rows having cost more in row-to-row similarity than their
/// residuals ever saved.
const PAETH_HANDICAP_PERCENT: u64 = 10;

/// One pixel in every `SAMPLE_STEP` is scored when choosing a filter.
///
/// Whole pixels are sampled rather than individual bytes so that every channel contributes,
/// which matters because the channels of an image rarely have the same statistics.
const SAMPLE_STEP: usize = 8;

/// Bytes of filtered scanline the encoder works on at a time.
///
/// Matched to the compressor's own block, so an ordinary image's band is one block and the
/// per-block Huffman code is fitted to as much data as it would have been. A band is always a
/// whole number of rows, since the filter is chosen per row, so a single row wider than this
/// makes a larger band rather than a partial one.
///
/// Working in bands is also why the encoder needs no second image-sized buffer: a band is
/// still in cache when the compressor reads it back.
const BAND_TARGET_BYTES: usize = 256 * 1024;

/// Largest `IDAT` payload written at once.
///
/// A chunk states its length before its data, so this much compressed output has to be held
/// before any of it can be written. Splitting the image data across several `IDAT` chunks is
/// ordinary PNG, and every decoder joins them back together.
const IDAT_CHUNK_BYTES: usize = 64 * 1024;

/// A reusable PNG encoder.
pub struct Encoder {
    deflater: Deflater,
    strategy: FilterStrategy,
    /// Filter bytes and filtered scanlines for the band being compressed.
    filtered: Vec<u8>,
    /// An all-zero row, standing in for the row above the first one.
    zero_row: Vec<u8>,
    /// Compressed bytes not yet long enough to fill an `IDAT`.
    idat: Vec<u8>,
}

impl core::fmt::Debug for Encoder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Encoder")
            .field("strategy", &self.strategy)
            .field("deflater", &self.deflater)
            .finish_non_exhaustive()
    }
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Encoder {
    /// An encoder with the default filter strategy.
    ///
    /// It holds the Huffman tables and scratch buffers, so encoding a second image through
    /// the same one costs no allocation for them.
    pub fn new() -> Self {
        Self {
            deflater: Deflater::new(),
            strategy: FilterStrategy::default(),
            filtered: Vec::new(),
            zero_row: Vec::new(),
            idat: Vec::new(),
        }
    }

    /// Sets how scanline filters are chosen. See [`FilterStrategy`].
    pub fn filter(&mut self, strategy: FilterStrategy) -> &mut Self {
        self.strategy = strategy;
        self
    }

    /// Encodes `data` as a PNG appended to `output`.
    ///
    /// `data` must be `info.output_size()` bytes in the layout `info` describes: tightly
    /// packed scanlines with no filter bytes. Interlaced output is not produced; an `info`
    /// asking for it is encoded progressively-free instead, which any decoder reads
    /// identically.
    ///
    /// See [`Encoder::encode_to`] to write somewhere other than memory.
    pub fn encode(&mut self, info: &Info, data: &[u8], output: &mut Vec<u8>) -> Result<(), Error> {
        match self.encode_to(info, data, output) {
            Ok(()) => Ok(()),
            Err(WriteError::Encode(error)) => Err(error),
            // `Vec` is the one sink whose writes cannot fail: its `write_all` is an
            // `extend_from_slice`, and an allocation it cannot satisfy aborts rather than
            // returning. Nothing reaches here.
            Err(WriteError::Io(error)) => {
                unreachable!("writing a PNG into a Vec returned an io error: {error}")
            }
        }
    }

    /// Encodes `data` as a PNG written to `output`.
    ///
    /// The image is filtered and compressed a band at a time and the compressed bytes leave
    /// as they are produced, so neither the finished file nor a filtered copy of the image is
    /// ever resident. Peak working memory is a band, a scanline and a chunk: a few hundred kilobytes for ordinary images. It grows with the image's width but not with its height, against the whole encoded file plus a filtered copy of every row for
    /// [`Encoder::encode`].
    ///
    /// [`Encoder::encode`] is this with a `Vec` sink, so the two write identical bytes.
    ///
    /// Nothing is written until `info` and `data` have been checked, so a
    /// [`WriteError::Encode`] leaves `output` untouched. A [`WriteError::Io`] can leave a
    /// partial file behind, since by then the earlier chunks have gone.
    ///
    /// ```no_run
    /// # fn main() -> Result<(), png_spark::WriteError> {
    /// # let (info, pixels) = (png_spark::Info::new(1, 1, png_spark::ColorType::Rgba,
    /// #     png_spark::BitDepth::Eight), vec![0u8; 4]);
    /// let mut file = std::io::BufWriter::new(std::fs::File::create("out.png")?);
    /// png_spark::Encoder::new().encode_to(&info, &pixels, &mut file)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn encode_to<W: std::io::Write>(
        &mut self,
        info: &Info,
        data: &[u8],
        mut output: W,
    ) -> Result<(), WriteError> {
        info.validate()?;
        if data.len() != info.output_size() {
            return Err(
                Error::WrongBufferSize { expected: info.output_size(), found: data.len() }.into()
            );
        }
        if info.color_type == ColorType::Indexed && info.palette.is_none() {
            return Err(Error::MissingPalette.into());
        }
        if let Some(palette) = &info.palette
            && (palette.len() > 256 * 3 || !palette.len().is_multiple_of(3))
        {
            return Err(Error::InvalidChunkLength { chunk: *b"PLTE", length: palette.len() }.into());
        }
        if let Some(transparency) = &info.transparency {
            crate::decoder::validate_trns(info, transparency)?;
        }
        for chunk in &info.metadata {
            chunk.validate()?;
        }

        write_leading_chunks(info, &mut output)?;

        let row_bytes = info.row_bytes();
        let height = info.height as usize;
        // At least one row, however wide: a single scanline of a very large image can exceed
        // the target on its own, and a band has to be a whole number of rows because the
        // filter is chosen per row.
        let band_rows = (BAND_TARGET_BYTES / (1 + row_bytes)).max(1);

        self.zero_row.clear();
        self.zero_row.resize(row_bytes, 0);
        // Sized once for the widest band; `filter_rows` overwrites every byte it uses, so the
        // zero fill here is the only one and the per-band resize below never grows it.
        self.filtered.clear();
        self.filtered.resize(band_rows * (1 + row_bytes), 0);
        self.idat.clear();
        let mut stream = self.deflater.zlib_start(&mut self.idat);

        // The filter choice is sticky from one row to the next, and that has to survive a
        // band boundary or the bands would each restart the heuristic.
        let mut chosen = Filter::None;
        let mut first = 0usize;
        while first < height {
            let past = (first + band_rows).min(height);
            let band = self.filter_band(info, data, first, past, &mut chosen);
            self.deflater.zlib_push(
                &mut stream,
                &self.filtered[..band],
                past == height,
                &mut self.idat,
            );

            // Every full chunk goes out before anything is moved, so the band costs one
            // compaction of the short tail rather than one per chunk.
            let mut sent = 0;
            while self.idat.len() - sent >= IDAT_CHUNK_BYTES {
                write_chunk(&mut output, b"IDAT", &self.idat[sent..sent + IDAT_CHUNK_BYTES])?;
                sent += IDAT_CHUNK_BYTES;
            }
            self.idat.drain(..sent);
            first = past;
        }
        if !self.idat.is_empty() {
            write_chunk(&mut output, b"IDAT", &self.idat)?;
        }

        output.write_all(&IEND)?;
        Ok(())
    }

    /// Filters rows `first..past` into [`Encoder::filtered`], carrying the filter choice.
    /// Returns how much of [`Encoder::filtered`] the band occupies.
    fn filter_band(
        &mut self,
        info: &Info,
        data: &[u8],
        first: usize,
        past: usize,
        chosen: &mut Filter,
    ) -> usize {
        let row_bytes = info.row_bytes();
        match info.filter_stride() {
            1 => self.filter_rows::<1>(data, row_bytes, first, past, chosen),
            2 => self.filter_rows::<2>(data, row_bytes, first, past, chosen),
            3 => self.filter_rows::<3>(data, row_bytes, first, past, chosen),
            4 => self.filter_rows::<4>(data, row_bytes, first, past, chosen),
            6 => self.filter_rows::<6>(data, row_bytes, first, past, chosen),
            8 => self.filter_rows::<8>(data, row_bytes, first, past, chosen),
            _ => unreachable!("PNG pixel strides are 1, 2, 3, 4, 6 or 8 bytes"),
        }
        (past - first) * (1 + row_bytes)
    }

    fn filter_rows<const BPP: usize>(
        &mut self,
        data: &[u8],
        row_bytes: usize,
        first: usize,
        past: usize,
        chosen: &mut Filter,
    ) {
        for index in first..past {
            let row = &data[index * row_bytes..(index + 1) * row_bytes];
            let previous = if index == 0 {
                &self.zero_row[..]
            } else {
                &data[(index - 1) * row_bytes..index * row_bytes]
            };

            let filter = match self.strategy {
                FilterStrategy::Fixed(filter) => filter,
                FilterStrategy::Adaptive => choose_filter::<BPP>(previous, row, 1, *chosen),
                FilterStrategy::Sampled => {
                    choose_filter::<BPP>(previous, row, SAMPLE_STEP, *chosen)
                }
            };
            *chosen = filter;

            let base = (index - first) * (1 + row_bytes);
            self.filtered[base] = filter as u8;
            filter_row::<BPP>(
                filter,
                previous,
                row,
                &mut self.filtered[base + 1..base + 1 + row_bytes],
            );
        }
    }
}

/// Picks the filter whose residuals are smallest, scoring every `step`-th pixel.
///
/// Smaller residuals mean a byte distribution more tightly clustered around zero, which is
/// what the entropy coder downstream turns into fewer bits. This is the heuristic the PNG
/// specification itself suggests, and it holds up well in practice, with two caveats: the
/// previous row's filter is favoured by [`STICKY_SHIFT`], and Paeth is held back by
/// [`PAETH_HANDICAP_PERCENT`] to keep rows off the slowest filter to reverse when it wins
/// by a margin too small to be worth the decoder's time.
fn choose_filter<const BPP: usize>(
    previous: &[u8],
    row: &[u8],
    step: usize,
    previous_choice: Filter,
) -> Filter {
    // Scored one filter per specialization rather than through a `Filter` argument. The
    // predictor sits in the innermost loop, so a runtime filter puts a four-way branch chain
    // on every scored byte and blocks vectorization of the whole sum.
    let mut scores = [
        residual_score::<BPP, 0>(previous, row, step),
        residual_score::<BPP, 1>(previous, row, step),
        residual_score::<BPP, 2>(previous, row, step),
        residual_score::<BPP, 3>(previous, row, step),
        residual_score::<BPP, 4>(previous, row, step),
    ];

    // Charged after scoring rather than inside it: the handicap is about what the row will
    // cost to read back, which is a property of the filter and not of these residuals. A
    // score is at most 128 per byte of one scanline, so the multiplication cannot overflow.
    let paeth = &mut scores[Filter::Paeth as usize];
    *paeth += *paeth * PAETH_HANDICAP_PERCENT / 100;

    let mut best = previous_choice;
    let mut best_score = {
        let score = scores[previous_choice as usize];
        score - (score >> STICKY_SHIFT)
    };

    for filter in Filter::ALL {
        if filter == previous_choice {
            continue;
        }
        let score = scores[filter as usize];
        if score < best_score {
            best_score = score;
            best = filter;
        }
    }
    best
}

/// Sum of absolute residuals filter `FILTER` would produce over the sampled pixels.
///
/// `FILTER` is a filter's type byte, pinned by the assertion below to its position in
/// [`Filter::ALL`] so that `scores[filter as usize]` selects the matching specialization.
fn residual_score<const BPP: usize, const FILTER: u8>(
    previous: &[u8],
    row: &[u8],
    step: usize,
) -> u64 {
    // Cutting both rows to one length up front is what lets the bounds checks fold away:
    // nothing otherwise relates `previous.len()` to `row.len()`. The two are equal at every
    // call site, so this costs nothing at runtime.
    let length = row.len().min(previous.len());
    let row = &row[..length];
    let previous = &previous[..length];

    let stride = BPP * step;
    let mut total = 0u64;

    // The first pixel has no left neighbour, so it is scored on its own terms.
    for index in 0..BPP.min(length) {
        total += magnitude(residual::<FILTER>(row[index], 0, previous[index], 0));
    }

    // Whole pixels. Bounded by `index + BPP <= length` rather than by a clamped inner end,
    // so the body is `BPP` iterations of a constant and unrolls into straight-line code; a
    // runtime end leaves a loop the compiler will neither unroll nor vectorize.
    let mut index = BPP;
    while index + BPP <= length {
        for offset in 0..BPP {
            let i = index + offset;
            total +=
                magnitude(residual::<FILTER>(row[i], row[i - BPP], previous[i], previous[i - BPP]));
        }
        index += stride;
    }

    // A row need not divide into whole pixels, and the remainder can only be the last of
    // them: the stride carries `index` past `length` immediately afterwards either way.
    while index < length {
        total += magnitude(residual::<FILTER>(
            row[index],
            row[index - BPP],
            previous[index],
            previous[index - BPP],
        ));
        index += 1;
    }
    total
}

// Ties the `FILTER` parameter of `residual_score` and `residual` to the filter type bytes,
// so that indexing by `filter as usize` reaches the specialization that scored it.
const _: () = {
    let mut index = 0;
    while index < Filter::ALL.len() {
        assert!(Filter::ALL[index] as u8 == index as u8);
        index += 1;
    }
};

#[inline(always)]
fn residual<const FILTER: u8>(x: u8, left: u8, above: u8, upper_left: u8) -> u8 {
    match FILTER {
        0 => x,
        1 => x.wrapping_sub(left),
        2 => x.wrapping_sub(above),
        3 => x.wrapping_sub(((left as u16 + above as u16) >> 1) as u8),
        _ => x.wrapping_sub(crate::filter::paeth_predictor(left, above, upper_left)),
    }
}

/// Distance of a residual byte from zero, treating it as a signed value.
#[inline(always)]
fn magnitude(residual: u8) -> u64 {
    (residual as i8).unsigned_abs() as u64
}

/// Whether the specification requires this chunk to appear before `PLTE`.
fn precedes_palette(chunk: &Chunk) -> bool {
    /// The eight types the specification orders before `PLTE`. Every other ancillary type
    /// is either required after `PLTE`, like `bKGD` and `hIST`, or unconstrained.
    const BEFORE_PLTE: [[u8; 4]; 8] =
        [*b"cHRM", *b"gAMA", *b"iCCP", *b"sBIT", *b"sRGB", *b"cICP", *b"mDCv", *b"cLLi"];
    BEFORE_PLTE.contains(&chunk.kind)
}

/// Writes the signature and every chunk that precedes the image data.
fn write_leading_chunks<W: std::io::Write>(info: &Info, output: &mut W) -> Result<(), WriteError> {
    output.write_all(&SIGNATURE)?;

    let mut header = [0u8; 13];
    header[0..4].copy_from_slice(&info.width.to_be_bytes());
    header[4..8].copy_from_slice(&info.height.to_be_bytes());
    header[8] = info.bit_depth.bits() as u8;
    header[9] = info.color_type as u8;
    header[10] = 0; // compression method: deflate
    header[11] = 0; // filter method: the five adaptive filters
    header[12] = 0; // interlace method: none
    write_chunk(output, b"IHDR", &header)?;

    // Metadata is written in the order it was given, split around `PLTE`: the types listed at
    // `BEFORE_PLTE` go ahead of the palette because the specification puts them there, and
    // everything else goes after it, which is where the types that must follow `PLTE` need to
    // be and is legal for the rest.
    for chunk in info.metadata.iter().filter(|chunk| precedes_palette(chunk)) {
        write_chunk(output, &chunk.kind, &chunk.data)?;
    }

    if let Some(palette) = &info.palette {
        write_chunk(output, b"PLTE", palette)?;
    }
    if let Some(transparency) = &info.transparency {
        write_chunk(output, b"tRNS", transparency)?;
    }

    for chunk in info.metadata.iter().filter(|chunk| !precedes_palette(chunk)) {
        write_chunk(output, &chunk.kind, &chunk.data)?;
    }
    Ok(())
}

/// Writes one chunk: length, type, payload, CRC.
///
/// The CRC is accumulated over the pieces as they go out rather than taken over a contiguous
/// buffer, which is what lets the payload be handed straight to the sink instead of staged in
/// one. That matters for `IDAT`, and it means an `Info` carrying a large `iCCP` profile is not
/// copied on its way out either.
fn write_chunk<W: std::io::Write>(
    output: &mut W,
    kind: &[u8; 4],
    data: &[u8],
) -> std::io::Result<()> {
    let mut header = [0u8; 8];
    header[..4].copy_from_slice(&(data.len() as u32).to_be_bytes());
    header[4..].copy_from_slice(kind);
    output.write_all(&header)?;
    output.write_all(data)?;

    let mut checksum = Crc32::new();
    checksum.update(&header[4..]);
    checksum.update(data);
    output.write_all(&checksum.finish().to_be_bytes())
}

/// `IEND` is the same twelve bytes in every PNG ever written, CRC included.
const IEND: [u8; 12] = [0, 0, 0, 0, b'I', b'E', b'N', b'D', 0xAE, 0x42, 0x60, 0x82];

/// Encodes an image described by `info` as a PNG, with the default settings.
pub fn encode(info: &Info, data: &[u8]) -> Result<Vec<u8>, Error> {
    let mut output = Vec::with_capacity(data.len() / 4 + 1024);
    Encoder::new().encode(info, data, &mut output)?;
    Ok(output)
}

/// Encodes tightly packed 8-bit RGBA pixels as a PNG.
pub fn encode_rgba8(width: u32, height: u32, data: &[u8]) -> Result<Vec<u8>, Error> {
    encode(&Info::new(width, height, ColorType::Rgba, crate::common::BitDepth::Eight), data)
}

/// Encodes tightly packed 8-bit RGB pixels as a PNG.
pub fn encode_rgb8(width: u32, height: u32, data: &[u8]) -> Result<Vec<u8>, Error> {
    encode(&Info::new(width, height, ColorType::Rgb, crate::common::BitDepth::Eight), data)
}
