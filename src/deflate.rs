//! DEFLATE compression (RFC 1951) with the zlib framing (RFC 1950) PNG requires.
//!
//! The compressor codes literals and runs of zero bytes, and nothing else. There is no hash
//! table, no match finder, and no token buffer: each block is scanned to count its symbols
//! and scanned again to write them, so the data moves through cache twice and the only
//! memory touched besides the input is a Huffman code that fits in a few kilobytes.
//!
//! That is a deliberate limit rather than a missing feature. Zero runs are the form of
//! repetition filtered image data is actually full of - a flat region leaves a residual of
//! zero under any of the PNG filters - and finding them costs a sequential scan instead of a
//! random memory access per input byte. A general LZ77 match finder buys a few percent of
//! size for several times the time, which is the wrong trade for this library; anyone who
//! wants it should reach for the `png` crate instead.

use crate::adler32::Adler32;
use crate::huffman::{Builder, canonical_codes};
use crate::tables::{CLCL_ORDER, DIST_EXTRA, LEN_BASE, LEN_EXTRA, LEN_TO_SYM};

/// Shortest and longest match this compressor will emit.
///
/// DEFLATE's own minimum is three, but a three-byte match saves nothing here: it costs a
/// length code plus a distance code where three literals would have cost three cheap ones.
const MIN_MATCH: usize = 4;
const MAX_MATCH: usize = 258;

/// Input bytes per block.
///
/// Small enough that both scans of a block hit the same cache lines, large enough that the
/// block header is a rounding error.
const RUN_BLOCK_BYTES: usize = 262144;

/// Blocks that may reuse the previous block's cost measurement before one is taken again.
const RUN_PROBE_INTERVAL: u32 = 8;

/// Shortest run of zeros worth coding as a match.
///
/// The first zero has nothing before it to copy from, so it stays a literal and the match
/// covers the rest; a match needs three bytes, so the run needs four.
const MIN_ZERO_RUN: usize = 4;

/// What [`walk_runs`] hands to its consumer.
enum Item<'a> {
    /// Bytes to code one at a time.
    Literals(&'a [u8]),
    /// A run of `length` zeros to code as a match at distance one.
    ZeroRun(usize),
}

/// Repeated `0x01` and `0x80` bytes, for the word-at-a-time zero search below.
const LOW_ONES: u64 = 0x0101_0101_0101_0101;
const HIGH_BITS: u64 = 0x8080_8080_8080_8080;

/// Marks the zero bytes of `word`, lowest set byte first.
///
/// The classic trick: subtracting one from every byte borrows out of any byte that was
/// zero, and masking against the complement keeps only bytes that were below `0x80` to
/// begin with. Borrows can also light up a later byte holding `0x01`, so only the *lowest*
/// marked byte is guaranteed to be a real zero - which is exactly the one being looked for.
#[inline(always)]
fn zero_bytes(word: u64) -> u64 {
    word.wrapping_sub(LOW_ONES) & !word & HIGH_BITS
}

/// Bytes the zero search advances when a window holds no run.
///
/// A run of four is detected only when it starts within the first five bytes of the eight
/// being examined, so stepping five at a time is what makes the search complete.
const RUN_SCAN_STRIDE: usize = 5;

/// Splits `input[start..end]` into literal spans and zero runs.
///
/// Zeros dominate filtered image data: a flat region leaves a residual of zero under any of
/// the filters, and those regions are most of what makes an image compressible at all.
/// Finding them needs no hash table and no memory beyond the input itself, which is what
/// makes this compressor fast.
///
/// The search looks for four zeros in a row rather than for a single zero. Isolated zeros
/// are everywhere in filtered image data - a constant alpha channel leaves one in every
/// pixel - and stopping at each of them turns the scan into a chain of unpredictable
/// branches. Testing for the run instead leaves a branch that is almost always not taken,
/// and so almost always predicted.
///
/// An iterator rather than a callback, so that the consuming loop stays in the caller. A
/// closure's captures reach the compiler as pointers it must assume may alias the bytes
/// being written, which costs the caller's tallies and bit writer their registers; locals
/// in the caller's own frame do not have that problem.
#[inline]
fn walk_runs(input: &[u8], start: usize, end: usize) -> Runs<'_> {
    Runs {
        // Nothing past `end` is read, and saying so up front makes `end` the length the
        // compiler checks every window against. Otherwise each window pays its loop bound
        // and a second, always-redundant check against the full input.
        input: &input[..end],
        pos: start,
        literals_from: start,
        pending: 0,
        done: false,
    }
}

/// The literal spans and zero runs of one block, in the order they must be coded.
struct Runs<'a> {
    input: &'a [u8],
    pos: usize,
    literals_from: usize,
    /// Zeros of the run just found that are still to be handed out, after the first.
    pending: usize,
    done: bool,
}

impl<'a> Iterator for Runs<'a> {
    type Item = Item<'a>;

    // Inlined into the consuming loop on purpose. Left to itself the compiler outlines
    // this, and a call per item is most of the cost on flat images, where a block is
    // thousands of short zero-run items rather than a few long literal spans.
    #[inline(always)]
    fn next(&mut self) -> Option<Item<'a>> {
        // A run longer than one match is handed out across several calls.
        if self.pending > 0 {
            let take = self.pending.min(MAX_MATCH);
            // Never leave a tail too short to code as a match of its own.
            let take = if self.pending > take && self.pending - take < MIN_MATCH {
                take - MIN_MATCH
            } else {
                take
            };
            self.pending -= take;
            return Some(Item::ZeroRun(take));
        }
        if self.done {
            return None;
        }

        let end = self.input.len();
        loop {
            while self.pos + 8 <= end {
                let word =
                    u64::from_le_bytes(self.input[self.pos..self.pos + 8].try_into().unwrap());
                let zeros = zero_bytes(word);
                // Four consecutive marked bytes. The lowest one is always a genuine zero: a
                // borrow-induced mark is always preceded by a real zero, which would have
                // made an earlier group of four match first.
                let runs = zeros & (zeros >> 8) & (zeros >> 16) & (zeros >> 24);
                if runs == 0 {
                    self.pos += RUN_SCAN_STRIDE;
                    continue;
                }
                self.pos += (runs.trailing_zeros() / 8) as usize;
                break;
            }
            if self.pos + 8 > end {
                // The last few bytes are always literals: a run there could not be coded as
                // a match reaching back far enough to be worth the symbols.
                break;
            }

            debug_assert_eq!(self.input[self.pos], 0);
            let run_start = self.pos;
            // Measure the run four words at a time. Long runs are the whole cost on flat
            // images, and OR-ing the words together tests thirty-two bytes for one branch.
            while self.pos + 32 <= end {
                let block = &self.input[self.pos..self.pos + 32];
                let combined = u64::from_le_bytes(block[0..8].try_into().unwrap())
                    | u64::from_le_bytes(block[8..16].try_into().unwrap())
                    | u64::from_le_bytes(block[16..24].try_into().unwrap())
                    | u64::from_le_bytes(block[24..32].try_into().unwrap());
                if combined != 0 {
                    break;
                }
                self.pos += 32;
            }
            while self.pos + 8 <= end
                && u64::from_le_bytes(self.input[self.pos..self.pos + 8].try_into().unwrap()) == 0
            {
                self.pos += 8;
            }
            while self.pos < end && self.input[self.pos] == 0 {
                self.pos += 1;
            }

            if self.pos - run_start < MIN_ZERO_RUN {
                // A borrow-induced false positive, or a run shorter than a match: leave it
                // in the literal span.
                continue;
            }

            // The run's first zero becomes the literal that the match copies from, and the
            // rest are handed out as matches by the calls that follow.
            let literals = &self.input[self.literals_from..run_start + 1];
            self.pending = self.pos - run_start - 1;
            self.literals_from = self.pos;
            return Some(Item::Literals(literals));
        }

        self.done = true;
        if self.literals_from < end {
            Some(Item::Literals(&self.input[self.literals_from..end]))
        } else {
            None
        }
    }
}

/// Counts the symbols [`walk_runs`] would produce.
///
/// Literals are tallied into four interleaved tables rather than one. Image data is full of
/// repeated bytes - a constant alpha channel, a flat region's zero residuals - and every
/// repeat of a byte increments the same counter again, so a single table turns the loop into
/// a chain of dependent read-modify-writes running at store-forwarding latency. Four tables
/// give the four bytes of each step independent counters, and merging them afterwards costs
/// a thousand adds per block.
fn count_runs(
    input: &[u8],
    start: usize,
    end: usize,
    litlen_freq: &mut [u32; LITLEN_SYMBOLS],
    dist_freq: &mut [u32; DIST_SYMBOLS],
) {
    let mut lanes = [[0u32; 256]; 4];

    for item in walk_runs(input, start, end) {
        match item {
            Item::Literals(bytes) => {
                let (chunks, remainder) = bytes.as_chunks::<4>();
                for chunk in chunks {
                    lanes[0][chunk[0] as usize] += 1;
                    lanes[1][chunk[1] as usize] += 1;
                    lanes[2][chunk[2] as usize] += 1;
                    lanes[3][chunk[3] as usize] += 1;
                }
                for (lane, &byte) in remainder.iter().enumerate() {
                    lanes[lane][byte as usize] += 1;
                }
            }
            Item::ZeroRun(length) => {
                litlen_freq[LEN_TO_SYM[length - 3] as usize] += 1;
                dist_freq[0] += 1;
            }
        }
    }

    for lane in &lanes {
        for (total, &count) in litlen_freq[..256].iter_mut().zip(lane.iter()) {
            *total += count;
        }
    }
}

/// Literal/length alphabet size actually used (286 of the 288 encodable symbols).
const LITLEN_SYMBOLS: usize = 286;
const DIST_SYMBOLS: usize = 30;
const CLCL_SYMBOLS: usize = 19;

// ---------------------------------------------------------------------------------------
// Bit writer
// ---------------------------------------------------------------------------------------

/// Little-endian bit writer over a `Vec<u8>`.
///
/// Holds fewer than eight bits between calls and flushes whole bytes with a single unaligned
/// eight-byte store, relying on spare capacity reserved in advance rather than a bounds
/// check per symbol.
struct BitWriter<'a> {
    out: &'a mut Vec<u8>,
    buffer: u64,
    nbits: u32,
}

/// Bytes of spare capacity the writer needs past the end of the data it will produce.
const WRITER_SLACK: usize = 8;

impl<'a> BitWriter<'a> {
    fn new(out: &'a mut Vec<u8>) -> Self {
        Self { out, buffer: 0, nbits: 0 }
    }

    /// Ensures `bytes` more bytes can be produced without reallocating mid-write.
    #[inline]
    fn reserve(&mut self, bytes: usize) {
        self.out.reserve(bytes + WRITER_SLACK);
    }

    /// Appends the low `count` bits of `bits`.
    ///
    /// `count` may be up to 56: the buffer never holds more than seven bits on entry.
    #[inline(always)]
    fn write(&mut self, bits: u64, count: u32) {
        debug_assert!(count <= 56);
        debug_assert!(self.nbits < 8);
        debug_assert!(count == 64 || bits < (1u64 << count));
        debug_assert!(self.out.len() + WRITER_SLACK <= self.out.capacity());

        self.buffer |= bits << self.nbits;
        self.nbits += count;

        let whole = (self.nbits >> 3) as usize;
        // SAFETY: `reserve` guarantees `WRITER_SLACK` spare bytes past the current length,
        // so the eight-byte store stays inside the allocation. Only the `whole` bytes that
        // are fully written become part of the vector; the rest are overwritten by the next
        // call before they can be observed.
        unsafe {
            let len = self.out.len();
            self.out
                .as_mut_ptr()
                .add(len)
                .cast::<[u8; 8]>()
                .write_unaligned(self.buffer.to_le_bytes());
            self.out.set_len(len + whole);
        }
        self.buffer >>= whole * 8;
        self.nbits &= 7;
    }

    /// Total bits written so far, including any still buffered.
    #[inline]
    fn bit_position(&self) -> u64 {
        self.out.len() as u64 * 8 + self.nbits as u64
    }

    /// Pads to a byte boundary with zeros.
    #[inline]
    fn align(&mut self) {
        if self.nbits > 0 {
            self.reserve(1);
            self.write(0, 8 - self.nbits);
        }
    }
}

// ---------------------------------------------------------------------------------------
// Compressor
// ---------------------------------------------------------------------------------------

/// A reusable DEFLATE compressor.
///
/// Holds the Huffman tables and header scratch between calls, so compressing a second buffer
/// allocates nothing for them.
pub struct Deflater {
    litlen_freq: [u32; LITLEN_SYMBOLS],
    dist_freq: [u32; DIST_SYMBOLS],
    litlen_lengths: [u8; LITLEN_SYMBOLS],
    litlen_codes: [u16; LITLEN_SYMBOLS],
    dist_lengths: [u8; DIST_SYMBOLS],
    dist_codes: [u16; DIST_SYMBOLS],
    builder: Builder,
    /// Run-length encoded code lengths for the block header, as `(symbol, extra bits)`.
    header_symbols: Vec<(u8, u8, u8)>,
    clcl_freq: [u32; CLCL_SYMBOLS],
    clcl_lengths: [u8; CLCL_SYMBOLS],
    clcl_codes: [u16; CLCL_SYMBOLS],
}

impl core::fmt::Debug for Deflater {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Deflater").finish_non_exhaustive()
    }
}

impl Default for Deflater {
    fn default() -> Self {
        Self::new()
    }
}

impl Deflater {
    /// A compressor with empty tables, ready for its first stream.
    pub fn new() -> Self {
        Self {
            litlen_freq: [0; LITLEN_SYMBOLS],
            dist_freq: [0; DIST_SYMBOLS],
            litlen_lengths: [0; LITLEN_SYMBOLS],
            litlen_codes: [0; LITLEN_SYMBOLS],
            dist_lengths: [0; DIST_SYMBOLS],
            dist_codes: [0; DIST_SYMBOLS],
            builder: Builder::new(),
            header_symbols: Vec::with_capacity(320),
            clcl_freq: [0; CLCL_SYMBOLS],
            clcl_lengths: [0; CLCL_SYMBOLS],
            clcl_codes: [0; CLCL_SYMBOLS],
        }
    }

    /// Compresses `input` into a zlib stream appended to `output`.
    pub fn zlib(&mut self, input: &[u8], output: &mut Vec<u8>) {
        // CM = deflate, CINFO = 32 KiB window, no preset dictionary, and a check byte that
        // makes the two-byte header a multiple of 31.
        output.extend_from_slice(&[0x78, 0x01]);
        self.raw(input, output);
        let mut checksum = Adler32::new();
        checksum.update(input);
        output.extend_from_slice(&checksum.finish().to_be_bytes());
    }

    /// Compresses `input` as a bare DEFLATE stream appended to `output`.
    pub fn raw(&mut self, input: &[u8], output: &mut Vec<u8>) {
        let mut writer = BitWriter::new(output);
        self.compress(input, &mut writer);
        writer.align();
    }

    /// Compresses `input` a block at a time.
    ///
    /// The first block is scanned twice, once to count its symbols and once to write them,
    /// so a small image gets a code fitted exactly to it. Every block after that is written
    /// in a single pass, using the code built from the block before while counting its own
    /// symbols for the block after.
    ///
    /// Neighbouring blocks of an image have near-identical byte distributions - they are
    /// adjacent bands of the same picture - so a code one block out of date costs a fraction
    /// of a percent, and it halves the number of times the data is walked.
    fn compress(&mut self, input: &[u8], writer: &mut BitWriter) {
        let mut start = 0usize;
        // Bits per input byte the previous block needed, in 8.8 fixed point. `None` means
        // there is no usable measurement, and the next block must be counted exactly.
        let mut density: Option<u64> = None;
        let mut blocks_since_exact = 0u32;

        loop {
            let end = (start + RUN_BLOCK_BYTES).min(input.len());
            let last = end == input.len();
            let bytes = end - start;

            // An exact count is needed to bootstrap, and worth repeating occasionally in
            // case the image changes character; the rest of the time the previous block's
            // measurement stands in, which is what lets an incompressible image skip the
            // counting pass entirely and go straight to stored blocks.
            let exact = density.is_none() || blocks_since_exact >= RUN_PROBE_INTERVAL;
            if exact {
                blocks_since_exact = 0;
                self.litlen_freq.fill(0);
                self.dist_freq.fill(0);
                count_runs(input, start, end, &mut self.litlen_freq, &mut self.dist_freq);
                self.litlen_freq[256] += 1;
            } else {
                blocks_since_exact += 1;
            }

            // Give every symbol a code, whether or not the counts say it occurs. A block
            // coded with the previous block's code will meet bytes that block never
            // contained, and a symbol with no code cannot be written at all. On a block of
            // tens of thousands of symbols the floor shifts the rarest codes by a bit and
            // leaves the common ones untouched.
            for count in self.litlen_freq.iter_mut() {
                *count += 1;
            }

            // Twelve-bit codes let four literals be packed into one write of the bit
            // buffer instead of three, which is most of the point of this compressor. The
            // ceiling costs a fraction of a percent against the fifteen bits DEFLATE allows.
            self.builder.code_lengths(&self.litlen_freq, &mut self.litlen_lengths, 12);
            self.builder.code_lengths(&self.dist_freq, &mut self.dist_lengths, 12);

            let header_bits = self.build_header();
            let compressed_bits = header_bits
                + match density {
                    Some(bits_per_byte) if !exact => bits_per_byte * bytes as u64 / 256,
                    _ => self.token_bits(),
                };
            let stored_bits = 3 + 7 + 32 + 8 * bytes as u64;

            if stored_bits < compressed_bits {
                self.emit_stored(writer, &input[start..end], last);
                // Nothing was coded, but what coding *would* have cost is still the best
                // estimate for the next block, and it is what keeps incompressible input
                // from being counted over and over.
                density = Some((compressed_bits * 256 / bytes.max(1) as u64).max(1));
            } else {
                let before = writer.bit_position();
                self.emit_runs(writer, input, start, end, last);
                let used = writer.bit_position() - before;
                density = Some((used * 256 / bytes.max(1) as u64).max(1));
            }

            if last {
                break;
            }
            start = end;
        }
    }

    fn emit_runs(
        &mut self,
        block_writer: &mut BitWriter,
        input: &[u8],
        start: usize,
        end: usize,
        last: bool,
    ) {
        let ok = canonical_codes(&self.litlen_lengths, &mut self.litlen_codes)
            && canonical_codes(&self.dist_lengths, &mut self.dist_codes)
            && canonical_codes(&self.clcl_lengths, &mut self.clcl_codes);
        debug_assert!(ok, "encoder produced an invalid huffman code");

        // Worst case is a literal per byte at the full code length, plus the header.
        block_writer.reserve((end - start) * 2 + 512);
        self.write_block_header(block_writer, last);

        // Disjoint field borrows: the code tables are read while the frequency tables are
        // rebuilt for the next block, and the two never overlap.
        let Self {
            litlen_codes,
            litlen_lengths,
            dist_codes,
            dist_lengths,
            litlen_freq,
            dist_freq,
            ..
        } = self;
        let dist_code = dist_codes[0] as u64;
        let dist_bits = dist_lengths[0] as u32;

        // The symbol loop writes through a writer of its own, living in this frame. Reached
        // through the caller's `&mut`, the bit buffer and bit count are memory the compiler
        // reloads on every symbol, behind a two-level pointer chase; as a local whose
        // address never escapes they stay in registers for the whole block.
        let mut local = BitWriter {
            out: &mut *block_writer.out,
            buffer: block_writer.buffer,
            nbits: block_writer.nbits,
        };
        let writer = &mut local;

        litlen_freq.fill(0);
        dist_freq.fill(0);
        // Four interleaved tallies, for the reason given on `count_runs`.
        let mut lanes = [[0u32; 256]; 4];

        for item in walk_runs(input, start, end) {
            match item {
                Item::Literals(bytes) => {
                    let (chunks, remainder) = bytes.as_chunks::<4>();
                    for chunk in chunks {
                        let (b0, b1, b2, b3) = (
                            chunk[0] as usize,
                            chunk[1] as usize,
                            chunk[2] as usize,
                            chunk[3] as usize,
                        );
                        let n0 = litlen_lengths[b0] as u32;
                        let n1 = litlen_lengths[b1] as u32;
                        let n2 = litlen_lengths[b2] as u32;
                        let n3 = litlen_lengths[b3] as u32;
                        writer.write(
                            litlen_codes[b0] as u64
                                | (litlen_codes[b1] as u64) << n0
                                | (litlen_codes[b2] as u64) << (n0 + n1)
                                | (litlen_codes[b3] as u64) << (n0 + n1 + n2),
                            n0 + n1 + n2 + n3,
                        );
                        lanes[0][b0] += 1;
                        lanes[1][b1] += 1;
                        lanes[2][b2] += 1;
                        lanes[3][b3] += 1;
                    }
                    for (lane, &byte) in remainder.iter().enumerate() {
                        writer.write(
                            litlen_codes[byte as usize] as u64,
                            litlen_lengths[byte as usize] as u32,
                        );
                        lanes[lane][byte as usize] += 1;
                    }
                }
                Item::ZeroRun(length) => {
                    let symbol = LEN_TO_SYM[length - 3] as usize;
                    let index = symbol - 257;
                    let extra_bits = LEN_EXTRA[index] as u32;
                    let extra = (length - LEN_BASE[index] as usize) as u64;
                    let code_bits = litlen_lengths[symbol] as u32;
                    // Length code, its extra bits, then the distance-one code: at most 12 +
                    // 5 + 12 bits, which fits a single write.
                    writer.write(
                        litlen_codes[symbol] as u64
                            | (extra << code_bits)
                            | (dist_code << (code_bits + extra_bits)),
                        code_bits + extra_bits + dist_bits,
                    );
                    litlen_freq[symbol] += 1;
                    dist_freq[0] += 1;
                }
            }
        }

        writer.write(litlen_codes[256] as u64, litlen_lengths[256] as u32);
        let (buffer, nbits) = (local.buffer, local.nbits);
        block_writer.buffer = buffer;
        block_writer.nbits = nbits;

        for lane in &lanes {
            for (total, &count) in litlen_freq[..256].iter_mut().zip(lane.iter()) {
                *total += count;
            }
        }
        litlen_freq[256] += 1;
    }

    // -----------------------------------------------------------------------------------
    // Block emission
    // -----------------------------------------------------------------------------------

    /// Emits `raw` as one or more uncompressed blocks.
    ///
    /// Choosing this over a coded block keeps incompressible input from growing, which is
    /// exactly the case that matters for photographic images.
    fn emit_stored(&mut self, writer: &mut BitWriter, raw: &[u8], last: bool) {
        // A stored block's length field is 16 bits, so long blocks are split.
        let mut chunks = raw.chunks(u16::MAX as usize).peekable();
        if chunks.peek().is_none() {
            writer.reserve(8);
            writer.write(last as u64, 1);
            writer.write(0, 2);
            writer.align();
            writer.write(0, 16);
            writer.write(0xffff, 16);
            return;
        }
        while let Some(chunk) = chunks.next() {
            let final_chunk = last && chunks.peek().is_none();
            writer.reserve(chunk.len() + 16);
            writer.write(final_chunk as u64, 1);
            writer.write(0, 2);
            writer.align();
            let len = chunk.len() as u16;
            writer.write(len as u64, 16);
            writer.write(!len as u64, 16);
            debug_assert_eq!(writer.nbits, 0);
            writer.out.extend_from_slice(chunk);
        }
    }

    /// Writes a dynamic block's opening bits: the block type, the two alphabet sizes, and
    /// the code that describes the two code length tables.
    fn write_block_header(&self, writer: &mut BitWriter, last: bool) {
        writer.write(last as u64, 1);
        writer.write(2, 2);

        let hlit = last_used(&self.litlen_lengths).max(256) + 1;
        let hdist = last_used(&self.dist_lengths) + 1;
        let hclen = clcl_count(&self.clcl_lengths);

        writer.write((hlit - 257) as u64, 5);
        writer.write((hdist - 1) as u64, 5);
        writer.write((hclen - 4) as u64, 4);
        for &symbol in CLCL_ORDER.iter().take(hclen) {
            writer.write(self.clcl_lengths[symbol as usize] as u64, 3);
        }
        for &(symbol, extra_bits, extra_value) in &self.header_symbols {
            let code = self.clcl_codes[symbol as usize] as u64;
            let bits = self.clcl_lengths[symbol as usize] as u32;
            writer.write(code | (extra_value as u64) << bits, bits + extra_bits as u32);
        }
    }

    /// Run-length encodes both code length tables into `header_symbols` and returns the
    /// number of bits the resulting header will occupy.
    fn build_header(&mut self) -> u64 {
        let hlit = last_used(&self.litlen_lengths).max(256) + 1;
        let hdist = last_used(&self.dist_lengths) + 1;

        let mut combined = [0u8; LITLEN_SYMBOLS + DIST_SYMBOLS];
        combined[..hlit].copy_from_slice(&self.litlen_lengths[..hlit]);
        combined[hlit..hlit + hdist].copy_from_slice(&self.dist_lengths[..hdist]);
        let combined = &combined[..hlit + hdist];

        self.header_symbols.clear();
        self.clcl_freq.fill(0);

        let mut i = 0;
        while i < combined.len() {
            let length = combined[i];
            let mut run = 1;
            while i + run < combined.len() && combined[i + run] == length {
                run += 1;
            }

            if length == 0 {
                // Zero runs have their own two symbols, covering 3..=10 and 11..=138.
                while run >= 3 {
                    let take = run.min(138);
                    if take >= 11 {
                        self.push_header(18, 7, (take - 11) as u8);
                    } else {
                        self.push_header(17, 3, (take - 3) as u8);
                    }
                    i += take;
                    run -= take;
                }
                for _ in 0..run {
                    self.push_header(0, 0, 0);
                    i += 1;
                }
            } else {
                // The length itself, then repeats of it in groups of 3..=6.
                self.push_header(length, 0, 0);
                i += 1;
                run -= 1;
                while run >= 3 {
                    let take = run.min(6);
                    self.push_header(16, 2, (take - 3) as u8);
                    i += take;
                    run -= take;
                }
                for _ in 0..run {
                    self.push_header(length, 0, 0);
                    i += 1;
                }
            }
        }

        // The code length alphabet is itself Huffman coded, with a seven-bit ceiling.
        self.builder.code_lengths(&self.clcl_freq, &mut self.clcl_lengths, 7);

        let hclen = clcl_count(&self.clcl_lengths);
        let mut bits = 3 + 5 + 5 + 4 + 3 * hclen as u64;
        for &(symbol, extra_bits, _) in &self.header_symbols {
            bits += self.clcl_lengths[symbol as usize] as u64 + extra_bits as u64;
        }
        bits
    }

    #[inline]
    fn push_header(&mut self, symbol: u8, extra_bits: u8, extra_value: u8) {
        self.clcl_freq[symbol as usize] += 1;
        self.header_symbols.push((symbol, extra_bits, extra_value));
    }

    /// Bits the counted symbols will occupy under the code just built.
    fn token_bits(&self) -> u64 {
        let mut bits = 0u64;
        for symbol in 0..LITLEN_SYMBOLS {
            bits += self.litlen_freq[symbol] as u64 * self.litlen_lengths[symbol] as u64;
        }
        for (&frequency, &extra) in self.litlen_freq[257..].iter().zip(LEN_EXTRA.iter()) {
            bits += frequency as u64 * extra as u64;
        }
        for ((&frequency, &length), &extra) in
            self.dist_freq.iter().zip(self.dist_lengths.iter()).zip(DIST_EXTRA.iter())
        {
            bits += frequency as u64 * (length as u64 + extra as u64);
        }
        bits
    }
}

/// Index of the last symbol with a non-zero code length.
fn last_used(lengths: &[u8]) -> usize {
    lengths.iter().rposition(|&length| length != 0).unwrap_or(0)
}

/// Number of code length code lengths that must be transmitted.
///
/// The trailing entries of the fixed transmission order can be dropped when they are zero.
fn clcl_count(lengths: &[u8; CLCL_SYMBOLS]) -> usize {
    let mut count = CLCL_SYMBOLS;
    while count > 4 && lengths[CLCL_ORDER[count - 1] as usize] == 0 {
        count -= 1;
    }
    count
}

/// Compresses `input` into a zlib stream.
pub fn compress_zlib(input: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len() / 3 + 64);
    Deflater::new().zlib(input, &mut output);
    output
}
