//! PNG encoding.

use crate::common::{Chunk, ColorType, Info, SIGNATURE};
use crate::crc32::crc32;
use crate::deflate::Deflater;
use crate::error::Error;
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

/// A reusable PNG encoder.
pub struct Encoder {
    deflater: Deflater,
    strategy: FilterStrategy,
    /// Filter bytes and filtered scanlines, ready to compress.
    filtered: Vec<u8>,
    /// An all-zero row, standing in for the row above the first one.
    zero_row: Vec<u8>,
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
    pub fn encode(&mut self, info: &Info, data: &[u8], output: &mut Vec<u8>) -> Result<(), Error> {
        info.validate()?;
        if data.len() != info.output_size() {
            return Err(Error::WrongBufferSize { expected: info.output_size(), found: data.len() });
        }
        if info.color_type == ColorType::Indexed && info.palette.is_none() {
            return Err(Error::MissingPalette);
        }
        if let Some(palette) = &info.palette
            && (palette.len() > 256 * 3 || !palette.len().is_multiple_of(3))
        {
            return Err(Error::InvalidChunkLength { chunk: *b"PLTE", length: palette.len() });
        }
        if let Some(transparency) = &info.transparency {
            crate::decoder::validate_trns(info, transparency)?;
        }
        for chunk in &info.metadata {
            chunk.validate()?;
        }

        output.extend_from_slice(&SIGNATURE);

        let mut header = [0u8; 13];
        header[0..4].copy_from_slice(&info.width.to_be_bytes());
        header[4..8].copy_from_slice(&info.height.to_be_bytes());
        header[8] = info.bit_depth.bits() as u8;
        header[9] = info.color_type as u8;
        header[10] = 0; // compression method: deflate
        header[11] = 0; // filter method: the five adaptive filters
        header[12] = 0; // interlace method: none
        write_chunk(output, b"IHDR", &header);

        // Metadata is written in the order it was given, split around `PLTE`: the types
        // listed at `BEFORE_PLTE` go ahead of the palette because the specification puts
        // them there, and everything else goes after it, which is where the types that must
        // follow `PLTE` need to be and is legal for the rest.
        for chunk in info.metadata.iter().filter(|chunk| precedes_palette(chunk)) {
            write_chunk(output, &chunk.kind, &chunk.data);
        }

        if let Some(palette) = &info.palette {
            write_chunk(output, b"PLTE", palette);
        }
        if let Some(transparency) = &info.transparency {
            write_chunk(output, b"tRNS", transparency);
        }

        for chunk in info.metadata.iter().filter(|chunk| !precedes_palette(chunk)) {
            write_chunk(output, &chunk.kind, &chunk.data);
        }

        self.filter_image(info, data);

        // The compressed data is written straight into the output, and the chunk's length
        // and CRC are filled in afterwards, so the image never exists as a second copy.
        let start = output.len();
        output.extend_from_slice(&[0, 0, 0, 0]);
        output.extend_from_slice(b"IDAT");
        self.deflater.zlib(&self.filtered, output);
        let length = (output.len() - start - 8) as u32;
        output[start..start + 4].copy_from_slice(&length.to_be_bytes());
        let checksum = crc32(&output[start + 4..]);
        output.extend_from_slice(&checksum.to_be_bytes());

        write_chunk(output, b"IEND", &[]);
        Ok(())
    }

    fn filter_image(&mut self, info: &Info, data: &[u8]) {
        let row_bytes = info.row_bytes();
        let height = info.height as usize;

        self.filtered.clear();
        self.filtered.resize(height * (1 + row_bytes), 0);
        self.zero_row.clear();
        self.zero_row.resize(row_bytes, 0);

        match info.filter_stride() {
            1 => self.filter_rows::<1>(data, row_bytes, height),
            2 => self.filter_rows::<2>(data, row_bytes, height),
            3 => self.filter_rows::<3>(data, row_bytes, height),
            4 => self.filter_rows::<4>(data, row_bytes, height),
            6 => self.filter_rows::<6>(data, row_bytes, height),
            8 => self.filter_rows::<8>(data, row_bytes, height),
            _ => unreachable!("PNG pixel strides are 1, 2, 3, 4, 6 or 8 bytes"),
        }
    }

    fn filter_rows<const BPP: usize>(&mut self, data: &[u8], row_bytes: usize, height: usize) {
        let mut chosen = Filter::None;
        for index in 0..height {
            let row = &data[index * row_bytes..(index + 1) * row_bytes];
            let previous = if index == 0 {
                &self.zero_row[..]
            } else {
                &data[(index - 1) * row_bytes..index * row_bytes]
            };

            let filter = match self.strategy {
                FilterStrategy::Fixed(filter) => filter,
                FilterStrategy::Adaptive => choose_filter::<BPP>(previous, row, 1, chosen),
                FilterStrategy::Sampled => choose_filter::<BPP>(previous, row, SAMPLE_STEP, chosen),
            };
            chosen = filter;

            let base = index * (1 + row_bytes);
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

fn write_chunk(output: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    output.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = output.len();
    output.extend_from_slice(kind);
    output.extend_from_slice(data);
    let checksum = crc32(&output[start..]);
    output.extend_from_slice(&checksum.to_be_bytes());
}

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
