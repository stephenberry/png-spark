//! DEFLATE decompression (RFC 1951), with the zlib framing (RFC 1950) that PNG's `IDAT`
//! stream uses.
//!
//! The decompressor is deliberately *not* a resumable state machine. PNG tells us the exact
//! size of the decompressed data up front, so the whole stream can be decoded in one call
//! against one output buffer. That removes the per-iteration state checks and output
//! clamping a streaming decoder needs, and lets the hot loop keep the bit buffer, the output
//! cursor, and the table references in registers.
//!
//! A caller who does not know the size in advance is served by
//! [`decompress_zlib_to_vec`], which pays for that choice by decoding again into a larger
//! buffer rather than resuming into one. The match window *is* the output buffer, so there
//! is no smaller piece of state that could be carried across a reallocation.

use crate::adler32::Adler32;
use crate::common::zeroed_vec;
use crate::tables::{
    CLCL_ORDER, DIST_BASE, DIST_EXTRA, FIXED_DIST_LENGTHS, FIXED_LITLEN_LENGTHS, LEN_BASE,
    LEN_EXTRA,
};

/// Extra bytes the caller must leave at the end of the output buffer.
///
/// Match copies are performed 16 bytes at a time regardless of the match length, so the last
/// copy of a block may write up to 15 bytes past the logical end of the data. Those bytes are
/// scratch and are never part of the result.
pub const OUTPUT_SLACK: usize = 16;

/// Non-exhaustive, for the reason [`Error`](crate::Error) is. Match with a fallback arm.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InflateError {
    /// The two-byte zlib header is not a valid PNG-compatible header.
    BadZlibHeader,
    /// A zlib stream requested a preset dictionary, which PNG does not permit.
    PresetDictionary,
    /// The input ended in the middle of the stream.
    UnexpectedEof,
    /// A block header used the reserved block type.
    InvalidBlockType,
    /// A stored block's length and its complement disagree.
    InvalidStoredLength,
    /// A dynamic block declared more literal/length codes than DEFLATE allows.
    InvalidHlit,
    /// A dynamic block declared more distance codes than DEFLATE allows.
    InvalidHdist,
    /// A code length repeat ran before the first length or past the last one.
    InvalidCodeLengthRepeat,
    /// The code length code lengths do not form a complete Huffman tree.
    BadCodeLengthTree,
    /// The literal/length code lengths do not form a complete Huffman tree.
    BadLiteralLengthTree,
    /// The distance code lengths do not form a valid Huffman tree.
    BadDistanceTree,
    /// A literal/length symbol outside the declared alphabet was decoded.
    InvalidLiteralLengthCode,
    /// A distance symbol outside the declared alphabet was decoded.
    InvalidDistanceCode,
    /// A match referred further back than the start of the output.
    DistanceTooFarBack,
    /// The stream decoded to more bytes than the caller said to expect.
    OutputOverflow,
    /// The stream decoded to fewer bytes than the caller said to expect.
    OutputUnderflow,
    /// The trailing Adler-32 checksum does not match the decompressed data.
    WrongChecksum,
    /// The allocator could not provide an output buffer of the requested size.
    OutOfMemory { bytes: usize },
}

impl core::fmt::Display for InflateError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let message = match self {
            Self::BadZlibHeader => "invalid zlib header",
            Self::PresetDictionary => "zlib stream requires a preset dictionary",
            Self::UnexpectedEof => "compressed stream ended unexpectedly",
            Self::InvalidBlockType => "reserved deflate block type",
            Self::InvalidStoredLength => "stored block length does not match its complement",
            Self::InvalidHlit => "too many literal/length codes",
            Self::InvalidHdist => "too many distance codes",
            Self::InvalidCodeLengthRepeat => "code length repeat out of range",
            Self::BadCodeLengthTree => "invalid code length huffman tree",
            Self::BadLiteralLengthTree => "invalid literal/length huffman tree",
            Self::BadDistanceTree => "invalid distance huffman tree",
            Self::InvalidLiteralLengthCode => "invalid literal/length code",
            Self::InvalidDistanceCode => "invalid distance code",
            Self::DistanceTooFarBack => "match distance points before the start of the output",
            Self::OutputOverflow => "compressed stream expands to more data than expected",
            Self::OutputUnderflow => "compressed stream expands to less data than expected",
            Self::WrongChecksum => "adler-32 checksum mismatch",
            Self::OutOfMemory { bytes } => {
                return write!(f, "could not allocate {bytes} bytes of output");
            }
        };
        f.write_str(message)
    }
}

impl std::error::Error for InflateError {}

// ---------------------------------------------------------------------------------------
// Decoding table layout
// ---------------------------------------------------------------------------------------
//
// Both Huffman codes are decoded through a direct-indexed primary table plus, for codes too
// long to fit, a secondary table. Each primary entry is a `u32`:
//
//   bits  0..8   number of code bits this entry consumes
//   bits  8..12  literal count (1 or 2) for literal entries, extra-bit count otherwise
//   bit   12     FLAG_INVALID      symbol is not usable in this alphabet
//   bit   13     FLAG_SECONDARY    payload is an offset into the secondary table
//   bit   14     FLAG_EXCEPTIONAL  entry needs the slow path (end of block, or secondary)
//   bit   15     FLAG_LITERAL      entry yields literal bytes directly
//   bits 16..32  payload: literal bytes, length/distance base, or secondary table offset
//
// Secondary entries are `u16`: the symbol in bits 4..16 and the total code length in bits
// 0..4.

const LITLEN_TABLE_BITS: u32 = 12;
const LITLEN_TABLE_SIZE: usize = 1 << LITLEN_TABLE_BITS;
const DIST_TABLE_BITS: u32 = 9;
const DIST_TABLE_SIZE: usize = 1 << DIST_TABLE_BITS;
const CLCL_TABLE_BITS: u32 = 7;
const CLCL_TABLE_SIZE: usize = 1 << CLCL_TABLE_BITS;

/// Mask applied to an entry when it is used directly as a shift amount.
///
/// A code length never exceeds 15, so the low six bits of an entry are exactly its length.
/// Masking with 63 rather than 255 matches what the shift instruction already does on every
/// target of interest, so the mask folds away and the table load feeds the next shift with
/// nothing in between.
const SHIFT_MASK: u32 = 0x3f;

const FLAG_INVALID: u32 = 0x1000;
const FLAG_SECONDARY: u32 = 0x2000;
const FLAG_EXCEPTIONAL: u32 = 0x4000;
const FLAG_LITERAL: u32 = 0x8000;

/// Payloads for the literal/length alphabet, indexed by symbol.
static LITLEN_ENTRIES: [u32; 288] = {
    let mut entries = [0u32; 288];
    let mut sym = 0;
    while sym < 256 {
        entries[sym] = (sym as u32) << 16 | (1 << 8) | FLAG_LITERAL;
        sym += 1;
    }
    entries[256] = FLAG_EXCEPTIONAL;
    sym += 1;
    while sym <= 285 {
        entries[sym] = (LEN_BASE[sym - 257] as u32) << 16 | (LEN_EXTRA[sym - 257] as u32) << 8;
        sym += 1;
    }
    // 286 and 287 are part of the fixed code's bit assignment but may never be emitted.
    entries[286] = FLAG_EXCEPTIONAL | FLAG_INVALID;
    entries[287] = FLAG_EXCEPTIONAL | FLAG_INVALID;
    entries
};

/// Payloads for the distance alphabet, indexed by symbol.
static DIST_ENTRIES: [u32; 32] = {
    let mut entries = [0u32; 32];
    let mut sym = 0;
    while sym < 30 {
        entries[sym] = (DIST_BASE[sym] as u32) << 16 | (DIST_EXTRA[sym] as u32) << 8 | FLAG_LITERAL;
        sym += 1;
    }
    // 30 and 31 are likewise unusable; leaving them without FLAG_LITERAL makes the decoder
    // reject them.
    entries
};

// ---------------------------------------------------------------------------------------
// Bit reader
// ---------------------------------------------------------------------------------------

/// Little-endian bit reader over a complete input buffer.
///
/// The buffer always holds at least 57 valid bits after a refill, which is more than the 48
/// bits the longest literal/length + distance pair can consume. That lets the decode loop
/// refill once per symbol pair and never check bit availability in between.
#[derive(Clone, Copy)]
struct BitReader<'a> {
    input: &'a [u8],
    /// Number of input bytes already shifted into `buf`.
    pos: usize,
    buf: u64,
    nbits: u32,
    /// Zero bits synthesized past the end of `input` that are currently inside `buf`.
    ///
    /// `padding - nbits` never decreases once the input is exhausted, so comparing the two
    /// at the end of the stream is enough to detect that a symbol read past the real data.
    padding: u32,
}

impl<'a> BitReader<'a> {
    #[inline]
    fn new(input: &'a [u8]) -> Self {
        Self { input, pos: 0, buf: 0, nbits: 0, padding: 0 }
    }

    #[inline(always)]
    fn refill(&mut self) {
        if self.pos + 8 <= self.input.len() {
            let word = u64::from_le_bytes(self.input[self.pos..self.pos + 8].try_into().unwrap());
            self.buf |= word << self.nbits;
            // Only the low `64 - nbits` bits of `word` made it in; advance by that many
            // whole bytes. Equal to `(63 - nbits) >> 3` for every `nbits` this can see, but
            // it needs no constant in a register, which matters across five inlined copies.
            debug_assert!(self.nbits < 64);
            self.pos += 7 - (self.nbits >> 3) as usize;
            self.nbits |= 56;
        } else {
            self.refill_tail();
        }
    }

    #[cold]
    fn refill_tail(&mut self) {
        while self.nbits <= 56 {
            if self.pos < self.input.len() {
                self.buf |= (self.input[self.pos] as u64) << self.nbits;
                self.pos += 1;
            } else {
                self.padding += 8;
            }
            self.nbits += 8;
        }
    }

    #[inline(always)]
    fn peek(&self, count: u32) -> u32 {
        (self.buf & ((1u64 << count) - 1)) as u32
    }

    #[inline(always)]
    fn consume(&mut self, count: u32) {
        debug_assert!(count <= self.nbits);
        self.buf >>= count;
        self.nbits -= count;
    }

    #[inline(always)]
    fn take(&mut self, count: u32) -> u32 {
        let value = self.peek(count);
        self.consume(count);
        value
    }

    /// Discards bits up to the next byte boundary.
    #[inline]
    fn align(&mut self) {
        self.consume(self.nbits % 8);
    }

    /// Byte offset of the next unread byte, or an error if decoding consumed bits that were
    /// never in the input.
    fn byte_position(&self) -> Result<usize, InflateError> {
        if self.padding > self.nbits {
            return Err(InflateError::UnexpectedEof);
        }
        debug_assert_eq!(self.nbits % 8, 0, "byte_position requires an aligned reader");
        Ok(self.pos - ((self.nbits - self.padding) / 8) as usize)
    }

    /// Restarts reading at an absolute byte offset, discarding any buffered bits.
    #[inline]
    fn seek(&mut self, byte_pos: usize) {
        self.pos = byte_pos;
        self.buf = 0;
        self.nbits = 0;
        self.padding = 0;
    }
}

// ---------------------------------------------------------------------------------------
// Table construction
// ---------------------------------------------------------------------------------------

/// Advances to the next canonical codeword in bit-reversed (stream) order.
///
/// Codewords are stored least-significant-bit-first so a table can be indexed with raw bits
/// straight from the stream. Incrementing in that order means carrying from the high end.
#[inline]
fn next_codeword(mut codeword: u16, table_size: u16) -> u16 {
    if codeword == table_size - 1 {
        return codeword;
    }
    let advance = (u16::BITS - 1) - (codeword ^ (table_size - 1)).leading_zeros();
    let bit = 1 << advance;
    codeword &= bit - 1;
    codeword |= bit;
    codeword
}

/// Builds a decoding table from canonical Huffman code lengths.
///
/// Returns `false` if the lengths do not describe a complete Huffman tree. When
/// `allow_degenerate` is set (distance codes only), an alphabet with zero or one used symbol
/// is accepted, matching what real encoders emit for streams with no matches.
///
/// `double_literal` packs two consecutive literal symbols into a single primary entry
/// whenever their combined code length still fits the table. On filtered image data, whose
/// byte distribution is dominated by a handful of short codes, this resolves a large fraction
/// of literals with one lookup instead of two.
// The loops below count in code lengths, which drive shifts and table widths as well as
// indexing the histogram; iterating the histogram instead would obscure that.
#[allow(clippy::needless_range_loop)]
fn build_table(
    lengths: &[u8],
    entries: &[u32],
    codes: &mut [u16; 288],
    primary: &mut [u32],
    secondary: &mut Vec<u16>,
    allow_degenerate: bool,
    double_literal: bool,
) -> bool {
    let mut histogram = [0usize; 16];
    for &length in lengths {
        histogram[length as usize] += 1;
    }

    let mut max_length = 15;
    while max_length > 1 && histogram[max_length] == 0 {
        max_length -= 1;
    }

    if allow_degenerate {
        if histogram[1..].iter().all(|&count| count == 0) {
            // No distances are used at all; any distance code in the stream is an error.
            primary.fill(0);
            secondary.clear();
            return true;
        }
        if max_length == 1 && histogram[1] == 1 {
            // A single one-bit code. The other half of the code space is invalid.
            let symbol = lengths.iter().position(|&l| l == 1).unwrap();
            codes[symbol] = 0;
            let entry = entries.get(symbol).copied().unwrap_or((symbol as u32) << 16) | 1;
            for pair in primary.chunks_mut(2) {
                pair[0] = entry;
                pair[1] = 0;
            }
            secondary.clear();
            return true;
        }
    }

    // Starting index of each code length within the length-sorted symbol list, plus a
    // Kraft sum check that the tree is exactly complete.
    let mut offsets = [0usize; 16];
    let mut codespace_used = 0usize;
    offsets[1] = histogram[0];
    for length in 1..max_length {
        offsets[length + 1] = offsets[length] + histogram[length];
        codespace_used = (codespace_used << 1) + histogram[length];
    }
    codespace_used = (codespace_used << 1) + histogram[max_length];
    if codespace_used != 1 << max_length {
        return false;
    }

    let mut next_index = offsets;
    let mut sorted_symbols = [0u16; 288];
    for (symbol, &length) in lengths.iter().enumerate() {
        sorted_symbols[next_index[length as usize]] = symbol as u16;
        next_index[length as usize] += 1;
    }

    let primary_bits = primary.len().trailing_zeros() as usize;
    let primary_mask = (1u16 << primary_bits) - 1;

    let mut codeword = 0u16;
    let mut cursor = histogram[0];

    // Iterate over every primary table width, not just the lengths in use: the doubling
    // step at the end of each round is what fills the table when the longest code is
    // shorter than the table.
    for length in 1..=primary_bits {
        let table_end = 1usize << length;

        for _ in 0..histogram[length] {
            let symbol = sorted_symbols[cursor] as usize;
            cursor += 1;

            primary[codeword as usize] =
                entries.get(symbol).copied().unwrap_or((symbol as u32) << 16) | length as u32;
            codes[symbol] = codeword;
            codeword = next_codeword(codeword, table_end as u16);
        }

        if double_literal {
            // Every way of splitting `length` bits into two shorter codes gives a pair of
            // literals that can be decoded together.
            for len1 in 1..length.saturating_sub(1) {
                let len2 = length - len1;
                for i1 in offsets[len1]..next_index[len1] {
                    for i2 in offsets[len2]..next_index[len2] {
                        let sym1 = sorted_symbols[i1] as usize;
                        let sym2 = sorted_symbols[i2] as usize;
                        if sym1 < 256 && sym2 < 256 {
                            let combined = codes[sym1] | (codes[sym2] << len1);
                            primary[combined as usize] = (sym1 as u32) << 16
                                | (sym2 as u32) << 24
                                | FLAG_LITERAL
                                | (2 << 8)
                                | length as u32;
                        }
                    }
                }
            }
        }

        // Codes shorter than the table width cover every index sharing their low bits;
        // doubling the filled prefix replicates them into the rest of the table.
        if length < primary_bits {
            primary.copy_within(0..table_end, table_end);
        }
    }

    secondary.clear();
    if max_length > primary_bits {
        let mut subtable_start = 0usize;
        let mut subtable_prefix = u16::MAX;

        for length in (primary_bits + 1)..=max_length {
            let subtable_size = 1usize << (length - primary_bits);

            for _ in 0..histogram[length] {
                if codeword & primary_mask != subtable_prefix {
                    subtable_prefix = codeword & primary_mask;
                    subtable_start = secondary.len();
                    primary[subtable_prefix as usize] = (subtable_start as u32) << 16
                        | FLAG_EXCEPTIONAL
                        | FLAG_SECONDARY
                        | (subtable_size as u32 - 1);
                    secondary.resize(subtable_start + subtable_size, 0);
                }

                let symbol = sorted_symbols[cursor];
                cursor += 1;
                codes[symbol as usize] = codeword;
                secondary[subtable_start + (codeword >> primary_bits) as usize] =
                    (symbol << 4) | length as u16;
                codeword = next_codeword(codeword, 1 << length);
            }

            // Longer codes sharing this prefix need the subtable to grow; the existing
            // entries are replicated so the wider index still resolves them.
            if length < max_length && codeword & primary_mask == subtable_prefix {
                secondary.extend_from_within(subtable_start..);
                let grown = secondary.len() - subtable_start;
                primary[subtable_prefix as usize] = (subtable_start as u32) << 16
                    | FLAG_EXCEPTIONAL
                    | FLAG_SECONDARY
                    | (grown as u32 - 1);
            }
        }
    }

    true
}

// ---------------------------------------------------------------------------------------
// Decompressor
// ---------------------------------------------------------------------------------------

/// A reusable DEFLATE decompressor.
///
/// Holding the decoding tables across calls avoids reallocating roughly 20 KiB per image.
pub struct Inflater {
    litlen: Box<[u32; LITLEN_TABLE_SIZE]>,
    litlen_secondary: Vec<u16>,
    dist: Box<[u32; DIST_TABLE_SIZE]>,
    dist_secondary: Vec<u16>,
    codes: [u16; 288],
    code_lengths: [u8; 320],
    verify_checksum: bool,
}

impl core::fmt::Debug for Inflater {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Inflater")
            .field("verify_checksum", &self.verify_checksum)
            .finish_non_exhaustive()
    }
}

impl Default for Inflater {
    fn default() -> Self {
        Self::new()
    }
}

impl Inflater {
    pub fn new() -> Self {
        Self {
            litlen: vec![0u32; LITLEN_TABLE_SIZE].into_boxed_slice().try_into().unwrap(),
            litlen_secondary: Vec::new(),
            dist: vec![0u32; DIST_TABLE_SIZE].into_boxed_slice().try_into().unwrap(),
            dist_secondary: Vec::new(),
            codes: [0; 288],
            code_lengths: [0; 320],
            verify_checksum: true,
        }
    }

    /// Enables or disables verification of the trailing Adler-32 checksum.
    ///
    /// PNG chunks carry their own CRC, so a caller that already verified those has checked
    /// the same bytes once; skipping the Adler-32 avoids a second pass over the output.
    pub fn verify_checksum(&mut self, verify: bool) -> &mut Self {
        self.verify_checksum = verify;
        self
    }

    /// Decompresses a zlib stream into `output`.
    ///
    /// The stream must expand to exactly `output.len() - OUTPUT_SLACK` bytes; the trailing
    /// slack is scratch space for match copies and never holds result data.
    ///
    /// PNG is the shape this is cut for: `IHDR` states the decompressed size, so a stream
    /// that stops short is a corrupt file and saying so is the useful answer. A caller who
    /// does not know the length in advance has no such expectation to check against and
    /// wants [`zlib_at_most`](Self::zlib_at_most).
    pub fn zlib(&mut self, input: &[u8], output: &mut [u8]) -> Result<usize, InflateError> {
        let written = self.zlib_at_most(input, output)?;
        if written != output.len() - OUTPUT_SLACK {
            return Err(InflateError::OutputUnderflow);
        }
        Ok(written)
    }

    /// Decompresses a zlib stream into `output`, accepting any length that fits.
    ///
    /// Identical to [`zlib`](Self::zlib) except that a stream expanding to less than
    /// `output.len() - OUTPUT_SLACK` bytes is a success rather than
    /// [`OutputUnderflow`](InflateError::OutputUnderflow); the count is returned. One that
    /// expands to more is still [`OutputOverflow`](InflateError::OutputOverflow), since
    /// there is nowhere to put the rest.
    ///
    /// Nothing beyond the returned length is meaningful: the buffer is handed to the
    /// decoder whole, and the bytes past the data are whatever match copies overwrote.
    pub fn zlib_at_most(&mut self, input: &[u8], output: &mut [u8]) -> Result<usize, InflateError> {
        assert!(output.len() >= OUTPUT_SLACK, "output buffer must include OUTPUT_SLACK");
        let limit = output.len() - OUTPUT_SLACK;

        if input.len() < 2 {
            return Err(InflateError::UnexpectedEof);
        }
        let cmf = input[0];
        let flg = input[1];
        if cmf & 0x0f != 8 || cmf >> 4 > 7 || (u16::from(cmf) * 256 + u16::from(flg)) % 31 != 0 {
            return Err(InflateError::BadZlibHeader);
        }
        if flg & 0x20 != 0 {
            return Err(InflateError::PresetDictionary);
        }

        let (written, consumed) = self.inflate(&input[2..], output, limit)?;

        let trailer = &input[2 + consumed..];
        if trailer.len() < 4 {
            return Err(InflateError::UnexpectedEof);
        }
        if self.verify_checksum {
            let expected = u32::from_be_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]);
            let mut checksum = Adler32::new();
            checksum.update(&output[..written]);
            if checksum.finish() != expected {
                return Err(InflateError::WrongChecksum);
            }
        }

        Ok(written)
    }

    /// Decodes a raw DEFLATE stream, returning the bytes written and the input bytes read.
    fn inflate(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        limit: usize,
    ) -> Result<(usize, usize), InflateError> {
        let mut reader = BitReader::new(input);
        let mut out_pos = 0usize;

        loop {
            reader.refill();
            let last_block = reader.take(1) != 0;
            let block_type = reader.take(2);

            match block_type {
                0 => {
                    reader.align();
                    let start = reader.byte_position()?;
                    if start + 4 > input.len() {
                        return Err(InflateError::UnexpectedEof);
                    }
                    let len = u16::from_le_bytes([input[start], input[start + 1]]) as usize;
                    let nlen = u16::from_le_bytes([input[start + 2], input[start + 3]]) as usize;
                    if len ^ 0xffff != nlen {
                        return Err(InflateError::InvalidStoredLength);
                    }
                    let data_start = start + 4;
                    if data_start + len > input.len() {
                        return Err(InflateError::UnexpectedEof);
                    }
                    if out_pos + len > limit {
                        return Err(InflateError::OutputOverflow);
                    }
                    output[out_pos..out_pos + len]
                        .copy_from_slice(&input[data_start..data_start + len]);
                    out_pos += len;
                    reader.seek(data_start + len);
                }
                1 => {
                    self.build_fixed_tables()?;
                    out_pos = self.decode_block(&mut reader, output, out_pos)?;
                }
                2 => {
                    self.read_dynamic_header(&mut reader)?;
                    out_pos = self.decode_block(&mut reader, output, out_pos)?;
                }
                _ => return Err(InflateError::InvalidBlockType),
            }

            if last_block {
                break;
            }
        }

        reader.align();
        let consumed = reader.byte_position()?;
        Ok((out_pos, consumed))
    }

    fn build_fixed_tables(&mut self) -> Result<(), InflateError> {
        if !build_table(
            &FIXED_LITLEN_LENGTHS,
            &LITLEN_ENTRIES,
            &mut self.codes,
            &mut self.litlen[..],
            &mut self.litlen_secondary,
            false,
            true,
        ) {
            return Err(InflateError::BadLiteralLengthTree);
        }
        if !build_table(
            &FIXED_DIST_LENGTHS,
            &DIST_ENTRIES,
            &mut self.codes,
            &mut self.dist[..],
            &mut self.dist_secondary,
            true,
            false,
        ) {
            return Err(InflateError::BadDistanceTree);
        }
        Ok(())
    }

    fn read_dynamic_header(&mut self, reader: &mut BitReader) -> Result<(), InflateError> {
        reader.refill();
        let hlit = reader.take(5) as usize + 257;
        let hdist = reader.take(5) as usize + 1;
        let hclen = reader.take(4) as usize + 4;

        if hlit > 286 {
            return Err(InflateError::InvalidHlit);
        }
        if hdist > 30 {
            return Err(InflateError::InvalidHdist);
        }

        let mut clcl_lengths = [0u8; 19];
        for &symbol in CLCL_ORDER.iter().take(hclen) {
            reader.refill();
            clcl_lengths[symbol as usize] = reader.take(3) as u8;
        }

        let mut clcl_table = [0u32; CLCL_TABLE_SIZE];
        let mut clcl_secondary = Vec::new();
        if !build_table(
            &clcl_lengths,
            &[],
            &mut self.codes,
            &mut clcl_table,
            &mut clcl_secondary,
            false,
            false,
        ) {
            return Err(InflateError::BadCodeLengthTree);
        }

        let total = hlit + hdist;
        let lengths = &mut self.code_lengths[..];
        lengths[..total].fill(0);

        let mut index = 0usize;
        while index < total {
            reader.refill();
            let entry = clcl_table[reader.peek(CLCL_TABLE_BITS) as usize];
            let code_bits = entry & 0xff;
            if code_bits == 0 {
                return Err(InflateError::BadCodeLengthTree);
            }
            reader.consume(code_bits);

            match entry >> 16 {
                symbol @ 0..=15 => {
                    lengths[index] = symbol as u8;
                    index += 1;
                }
                16 => {
                    if index == 0 {
                        return Err(InflateError::InvalidCodeLengthRepeat);
                    }
                    let repeat = 3 + reader.take(2) as usize;
                    let previous = lengths[index - 1];
                    if index + repeat > total {
                        return Err(InflateError::InvalidCodeLengthRepeat);
                    }
                    lengths[index..index + repeat].fill(previous);
                    index += repeat;
                }
                17 => {
                    let repeat = 3 + reader.take(3) as usize;
                    if index + repeat > total {
                        return Err(InflateError::InvalidCodeLengthRepeat);
                    }
                    index += repeat;
                }
                _ => {
                    let repeat = 11 + reader.take(7) as usize;
                    if index + repeat > total {
                        return Err(InflateError::InvalidCodeLengthRepeat);
                    }
                    index += repeat;
                }
            }
        }

        // The distance alphabet is always built over 32 symbols so that symbols 30 and 31,
        // which are legal in the code but never usable, are rejected by the decoder.
        let mut dist_lengths = [0u8; 32];
        dist_lengths[..hdist].copy_from_slice(&lengths[hlit..hlit + hdist]);

        if !build_table(
            &lengths[..hlit],
            &LITLEN_ENTRIES,
            &mut self.codes,
            &mut self.litlen[..],
            &mut self.litlen_secondary,
            false,
            true,
        ) {
            return Err(InflateError::BadLiteralLengthTree);
        }
        if !build_table(
            &dist_lengths,
            &DIST_ENTRIES,
            &mut self.codes,
            &mut self.dist[..],
            &mut self.dist_secondary,
            true,
            false,
        ) {
            return Err(InflateError::BadDistanceTree);
        }

        Ok(())
    }

    /// Writes the one or two literal bytes an entry carries, without re-checking bounds.
    ///
    /// # Safety
    /// `pos + 2 <= output.len()` must hold. The decode loop establishes this by refusing to
    /// enter an iteration unless `pos <= output.len() - OUTPUT_SLACK`, and by advancing `pos`
    /// by at most six over the literals it then writes speculatively.
    #[inline(always)]
    unsafe fn store_literals(output: &mut [u8], pos: usize, entry: u32) {
        debug_assert!(pos + 2 <= output.len());
        // One unaligned halfword rather than two byte stores. Written separately the pair
        // never gets merged, and the second store's address has to be materialized on its
        // own instead of folding into the first store's register offset.
        let target = output.as_mut_ptr().add(pos);
        let pair = ((entry >> 16) as u16).to_le_bytes();
        target.cast::<[u8; 2]>().write_unaligned(pair);
    }

    /// Copies sixteen bytes within `output`, from `source` to `dest`.
    ///
    /// The ranges may overlap: the read completes into a register before the write starts,
    /// which is exactly the semantics the overlapping-match loop below relies on. Written as
    /// a fixed-size read and write so it stays a register pair rather than becoming a call
    /// to `memmove`, which would spill the whole decode loop around it.
    ///
    /// # Safety
    /// `source + 16 <= output.len()` and `dest + 16 <= output.len()`.
    #[inline(always)]
    unsafe fn copy16(output: &mut [u8], source: usize, dest: usize) {
        debug_assert!(source + 16 <= output.len() && dest + 16 <= output.len());
        let chunk = output.as_ptr().add(source).cast::<[u8; 16]>().read_unaligned();
        output.as_mut_ptr().add(dest).cast::<[u8; 16]>().write_unaligned(chunk);
    }

    /// Writes sixteen bytes at `dest`.
    ///
    /// # Safety
    /// `dest + 16 <= output.len()`.
    #[inline(always)]
    unsafe fn store16(output: &mut [u8], dest: usize, value: [u8; 16]) {
        debug_assert!(dest + 16 <= output.len());
        output.as_mut_ptr().add(dest).cast::<[u8; 16]>().write_unaligned(value);
    }

    /// Decodes one compressed block, returning the new output position.
    ///
    /// The bit reader is copied into a local for the duration of the loop. Left behind a
    /// `&mut` it would have to be spilled to memory after every consume, because the
    /// compiler cannot otherwise rule out aliasing with the output buffer; as a local it
    /// stays in registers.
    fn decode_block(
        &mut self,
        reader: &mut BitReader,
        output: &mut [u8],
        start_pos: usize,
    ) -> Result<usize, InflateError> {
        // Binding the tables as fixed-size array references, rather than slices, lets the
        // masked table indices be proven in range and drops the bounds checks from the two
        // hottest loads in the loop.
        let litlen: &[u32; LITLEN_TABLE_SIZE] = &self.litlen;
        let dist_table: &[u32; DIST_TABLE_SIZE] = &self.dist;
        let litlen_secondary = &self.litlen_secondary[..];
        let dist_secondary = &self.dist_secondary[..];

        let limit = output.len() - OUTPUT_SLACK;
        let mut pos = start_pos;
        let mut r = *reader;

        let result = 'block: {
            r.refill();
            let mut entry = litlen[r.peek(LITLEN_TABLE_BITS) as usize];

            loop {
                if pos > limit {
                    break 'block Err(InflateError::OutputOverflow);
                }

                let mut bits = r.buf;
                let mut code_bits = entry & 0xff;

                if entry & FLAG_LITERAL != 0 {
                    // Literals dominate filtered image data, so speculatively resolve a
                    // short chain of them. Each entry carries its own code length, so the
                    // next table index is available without waiting for the previous store.
                    //
                    // The bits are shifted cumulatively rather than by a running sum of code
                    // lengths: that keeps the loop between one table load and the next down
                    // to a shift and a mask, which is the whole latency budget of this loop.
                    let bits1 = bits >> (entry & SHIFT_MASK);
                    let entry2 = litlen[(bits1 as u32 & 0xfff) as usize];
                    let bits2 = bits1 >> (entry2 & SHIFT_MASK);
                    let entry3 = litlen[(bits2 as u32 & 0xfff) as usize];
                    let bits3 = bits2 >> (entry3 & SHIFT_MASK);
                    let entry4 = litlen[(bits3 as u32 & 0xfff) as usize];

                    let code_bits2 = entry2 & 0xff;
                    let code_bits3 = entry3 & 0xff;

                    // SAFETY: `pos <= limit` was checked above, so `pos + 2 <=
                    // output.len() - 14`. Each store below advances `pos` by at most two and
                    // there are at most three of them, keeping every access in bounds.
                    unsafe { Self::store_literals(output, pos, entry) };
                    pos += ((entry >> 8) & 0xf) as usize;

                    if entry2 & FLAG_LITERAL != 0 {
                        unsafe { Self::store_literals(output, pos, entry2) };
                        pos += ((entry2 >> 8) & 0xf) as usize;

                        if entry3 & FLAG_LITERAL != 0 {
                            unsafe { Self::store_literals(output, pos, entry3) };
                            pos += ((entry3 >> 8) & 0xf) as usize;

                            r.consume(code_bits + code_bits2 + code_bits3);
                            r.refill();
                            entry = entry4;
                            continue;
                        }

                        r.consume(code_bits + code_bits2);
                        r.refill();
                        entry = entry3;
                        code_bits = code_bits3;
                        bits = r.buf;
                    } else {
                        r.consume(code_bits);
                        r.refill();
                        entry = entry2;
                        code_bits = code_bits2;
                        bits = r.buf;
                    }
                }

                // Whatever is left is a match length, the end-of-block marker, or a code too
                // long for the primary table.
                let (length_base, length_extra, code_bits) = if entry & FLAG_EXCEPTIONAL == 0 {
                    (entry >> 16, (entry >> 8) & 0xf, code_bits)
                } else if entry & FLAG_SECONDARY != 0 {
                    let index =
                        (entry >> 16) + ((bits >> LITLEN_TABLE_BITS) as u32 & (entry & 0xff));
                    let secondary = match litlen_secondary.get(index as usize) {
                        Some(&value) => value,
                        None => break 'block Err(InflateError::InvalidLiteralLengthCode),
                    };
                    let symbol = (secondary >> 4) as usize;
                    let secondary_bits = (secondary & 0xf) as u32;

                    match symbol {
                        0..=255 => {
                            r.consume(secondary_bits);
                            r.refill();
                            // SAFETY: `pos <= limit`, so one byte is in bounds.
                            unsafe { output.as_mut_ptr().add(pos).write(symbol as u8) };
                            pos += 1;
                            entry = litlen[r.peek(LITLEN_TABLE_BITS) as usize];
                            continue;
                        }
                        256 => {
                            r.consume(secondary_bits);
                            break 'block Ok(pos);
                        }
                        257..=285 => (
                            LEN_BASE[symbol - 257] as u32,
                            LEN_EXTRA[symbol - 257] as u32,
                            secondary_bits,
                        ),
                        _ => break 'block Err(InflateError::InvalidLiteralLengthCode),
                    }
                } else if entry & FLAG_INVALID != 0 || code_bits == 0 {
                    break 'block Err(InflateError::InvalidLiteralLengthCode);
                } else {
                    r.consume(code_bits);
                    break 'block Ok(pos);
                };

                bits >>= code_bits;
                let length = (length_base + (bits as u32 & ((1 << length_extra) - 1))) as usize;
                bits >>= length_extra;

                let dist_entry = dist_table[(bits as u32 & 0x1ff) as usize];
                let (dist_base, dist_extra, dist_bits) = if dist_entry & FLAG_LITERAL != 0 {
                    (dist_entry >> 16, (dist_entry >> 8) & 0xf, dist_entry & 0xff)
                } else if dist_entry & FLAG_SECONDARY != 0 {
                    let index = (dist_entry >> 16)
                        + ((bits >> DIST_TABLE_BITS) as u32 & (dist_entry & 0xff));
                    let secondary = match dist_secondary.get(index as usize) {
                        Some(&value) => value,
                        None => break 'block Err(InflateError::InvalidDistanceCode),
                    };
                    let symbol = (secondary >> 4) as usize;
                    if symbol >= 30 {
                        break 'block Err(InflateError::InvalidDistanceCode);
                    }
                    (
                        DIST_BASE[symbol] as u32,
                        DIST_EXTRA[symbol] as u32,
                        (secondary & 0xf) as u32,
                    )
                } else {
                    break 'block Err(InflateError::InvalidDistanceCode);
                };

                bits >>= dist_bits;
                let distance = (dist_base + (bits as u32 & ((1 << dist_extra) - 1))) as usize;

                r.consume(code_bits + length_extra + dist_bits + dist_extra);
                r.refill();
                entry = litlen[r.peek(LITLEN_TABLE_BITS) as usize];

                if distance > pos {
                    break 'block Err(InflateError::DistanceTooFarBack);
                }
                if pos + length > limit {
                    break 'block Err(InflateError::OutputOverflow);
                }

                // Every access below stays inside the buffer: `pos + length <= limit` was
                // just checked and the buffer carries `OUTPUT_SLACK` bytes beyond `limit`,
                // so writing a full 16-byte block at any offset below `pos + length` is in
                // bounds, as is reading one from the strictly earlier `source`.
                let source = pos - distance;
                if distance == 1 {
                    // Byte runs are the most common match in filtered image data.
                    // SAFETY: `source < pos <= limit` by the two checks above, so the byte
                    // is inside the buffer. Indexing keeps the check: `limit` is a wrapping
                    // subtraction and `pos + length` a wrapping add as far as the compiler
                    // can tell, so it cannot rebuild the invariant the note above states.
                    // The panic edge also spills `limit` back onto the hottest literal path.
                    let byte = unsafe { *output.get_unchecked(source) };
                    if length <= 16 {
                        // SAFETY: the slack window makes a full block write in bounds.
                        unsafe { Self::store16(output, pos, [byte; 16]) };
                    } else {
                        output[pos..pos + length].fill(byte);
                    }
                } else {
                    // Copy a fixed 16 bytes at a time. When the distance is shorter than the
                    // step, each pass extends the correctly-filled prefix by `distance`
                    // bytes, so the run resolves itself after `ceil(length / distance)`
                    // passes.
                    let step = distance.min(16);
                    let mut offset = 0;
                    loop {
                        // SAFETY: see the note above the `distance == 1` branch.
                        unsafe { Self::copy16(output, source + offset, pos + offset) };
                        offset += step;
                        if offset >= length {
                            break;
                        }
                    }
                }
                pos += length;
            }
        };

        *reader = r;
        result
    }
}

/// Decompresses a zlib stream that is known to expand to exactly `expected_len` bytes.
pub fn decompress_zlib(input: &[u8], expected_len: usize) -> Result<Vec<u8>, InflateError> {
    // `expected_len` is the caller's, and on the paths this exists for it came out of a file.
    // A length that cannot have its slack added is refused like any other unmeetable size.
    let request = expected_len
        .checked_add(OUTPUT_SLACK)
        .ok_or(InflateError::OutOfMemory { bytes: expected_len })?;
    let mut output = zeroed_vec(request).ok_or(InflateError::OutOfMemory { bytes: request })?;
    let written = Inflater::new().zlib(input, &mut output)?;
    output.truncate(written);
    Ok(output)
}

/// Decompresses a zlib stream of unknown length, growing the output buffer as needed and
/// giving up past `max_output` bytes.
///
/// Use this when the length is not recorded anywhere trustworthy. A length field read from
/// an untrusted file is not an answer to that problem, it is a restatement of it: taking one
/// at face value is an instruction to allocate whatever the file asks for, which is why
/// `max_output` is a parameter and not a default. Truncation is caught regardless, by the
/// Adler-32 the zlib framing already carries over the data.
///
/// The cost of not knowing is decoding more than once. The decompressor holds its whole
/// output buffer as the match window and so cannot resume into a larger one; a buffer that
/// fills is therefore discarded and the stream decoded again into a buffer twice the size.
/// The discarded attempts sum to just under the size of the one that succeeds, so the
/// doubling bounds the wasted work at one extra pass over the data. That is not free, and a
/// caller who *does* know the length should say so through [`decompress_zlib`] instead.
///
/// Returns [`OutputOverflow`](InflateError::OutputOverflow) if the stream expands past
/// `max_output`, having allocated no more than that. A corrupt stream reaches the same
/// answer the same way, so `max_output` bounds the work this will do on a hostile input as
/// well as the memory: budget it as two passes over `max_output` bytes, however few bytes of
/// input arrived.
pub fn decompress_zlib_to_vec(input: &[u8], max_output: usize) -> Result<Vec<u8>, InflateError> {
    // Deflate's best case is a little over 1000:1, so no first guess drawn from the input
    // length is safe; four times it settles ordinary data in a single pass, and the floor
    // keeps small streams from starting a doubling run at a handful of bytes.
    let mut capacity = input.len().saturating_mul(4).max(1024).min(max_output);

    let mut inflater = Inflater::new();
    loop {
        let request = capacity
            .checked_add(OUTPUT_SLACK)
            .ok_or(InflateError::OutOfMemory { bytes: capacity })?;
        let mut output =
            zeroed_vec(request).ok_or(InflateError::OutOfMemory { bytes: request })?;
        match inflater.zlib_at_most(input, &mut output) {
            Ok(written) => {
                output.truncate(written);
                return Ok(output);
            }
            Err(InflateError::OutputOverflow) if capacity < max_output => {
                capacity = capacity.saturating_mul(2).min(max_output);
            }
            Err(error) => return Err(error),
        }
    }
}
