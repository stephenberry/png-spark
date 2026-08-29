//! PNG scanline filters (RFC 2083 section 6).
//!
//! Each scanline is prefixed by a filter byte selecting one of five predictors. Decoding
//! reverses the predictor; encoding applies it. Both directions are specialized on the pixel
//! stride with a const generic, because the stride is what determines the serial dependency
//! distance and therefore how the loops schedule.

/// The five filter types a scanline may use.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Filter {
    None = 0,
    Sub = 1,
    Up = 2,
    Average = 3,
    Paeth = 4,
}

impl Filter {
    /// Every filter, in the order their type bytes are numbered.
    pub const ALL: [Filter; 5] = [
        Filter::None,
        Filter::Sub,
        Filter::Up,
        Filter::Average,
        Filter::Paeth,
    ];

    #[inline]
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Filter::None),
            1 => Some(Filter::Sub),
            2 => Some(Filter::Up),
            3 => Some(Filter::Average),
            4 => Some(Filter::Paeth),
            _ => None,
        }
    }
}

/// The Paeth predictor: whichever of the three neighbours the linear estimate `a + b - c`
/// lands closest to.
///
/// Written as a running minimum rather than the specification's nested comparison. The two
/// forms select the same neighbour, including at ties, but this one is a chain of compares
/// and selects with no branches, and maps onto vector min/select when the surrounding loop
/// gets unrolled across a pixel.
#[inline(always)]
pub(crate) fn paeth(a: i16, b: i16, c: i16) -> u8 {
    // `p - a` is `b - c` and `p - b` is `a - c`, so the three distances need only two
    // differences between them.
    let da = b - c;
    let db = a - c;
    let pa = da.abs();
    let pb = db.abs();
    let pc = (da + db).abs();

    let mut nearest = a;
    let mut smallest = pa;
    if pb < smallest {
        nearest = b;
        smallest = pb;
    }
    if pc < smallest {
        nearest = c;
    }
    nearest as u8
}

/// The Paeth predictor over plain bytes, for callers outside the filter loops.
#[inline(always)]
pub fn paeth_predictor(left: u8, above: u8, upper_left: u8) -> u8 {
    paeth(left as i16, above as i16, upper_left as i16)
}

// ---------------------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------------------
//
// Every reconstruction loop below walks a whole pixel at a time and carries the neighbours
// it needs in fixed-size arrays. The obvious formulation, `row[i] += row[i - stride]`,
// reloads a byte the previous iteration has just stored, and store-to-load forwarding makes
// that dependency several cycles long; keeping the pixel in registers reduces it to a single
// add, and lets the compiler unroll the per-channel work across the pixel.
//
// A scanline's length is always a whole number of strides: for bit depths of 8 and above the
// stride is the pixel size, and below that the stride is one byte.

/// Reverses the filter on the first scanline of an image or interlace pass.
///
/// The row above is defined to be all zeros, which collapses `Up` and `None` to no-ops and
/// reduces `Paeth` to `Sub`.
fn unfilter_first_row<const BPP: usize>(filter: Filter, row: &mut [u8]) {
    match filter {
        Filter::None | Filter::Up => {}
        Filter::Sub | Filter::Paeth => {
            let mut left = [0u8; BPP];
            for pixel in row.chunks_exact_mut(BPP) {
                for k in 0..BPP {
                    left[k] = pixel[k].wrapping_add(left[k]);
                }
                pixel.copy_from_slice(&left);
            }
        }
        Filter::Average => {
            let mut left = [0u8; BPP];
            for pixel in row.chunks_exact_mut(BPP) {
                for k in 0..BPP {
                    left[k] = pixel[k].wrapping_add(left[k] >> 1);
                }
                pixel.copy_from_slice(&left);
            }
        }
    }
}

/// Reverses the filter on a scanline, given the already-reconstructed row above it.
///
/// Each filter reconstructs into `left`, its own running left neighbour, and copies that out
/// to the row. Every lane reads only its own index, so writing in place is sound, and it
/// matters: a version building each pixel in a separate temporary and assigning that to
/// `left` reconstructs a stride-8 row at an eighth of the speed, LLVM taking the assignment
/// for a memory move rather than a register one.
fn unfilter_row<const BPP: usize>(filter: Filter, prev: &[u8], row: &mut [u8]) {
    debug_assert_eq!(prev.len(), row.len());

    match filter {
        Filter::None => {}
        Filter::Sub => {
            let mut left = [0u8; BPP];
            for pixel in row.chunks_exact_mut(BPP) {
                for k in 0..BPP {
                    left[k] = pixel[k].wrapping_add(left[k]);
                }
                pixel.copy_from_slice(&left);
            }
        }
        Filter::Up => {
            // No serial dependency at all, so this vectorizes to a plain packed add.
            for (x, &b) in row.iter_mut().zip(prev.iter()) {
                *x = x.wrapping_add(b);
            }
        }
        Filter::Average => {
            let mut left = [0u8; BPP];
            for (pixel, above) in row.chunks_exact_mut(BPP).zip(prev.chunks_exact(BPP)) {
                for k in 0..BPP {
                    let sum = left[k] as u16 + above[k] as u16;
                    left[k] = pixel[k].wrapping_add((sum >> 1) as u8);
                }
                pixel.copy_from_slice(&left);
            }
        }
        Filter::Paeth => {
            let mut left = [0u8; BPP];
            let mut upper_left = [0u8; BPP];
            for (pixel, above) in row.chunks_exact_mut(BPP).zip(prev.chunks_exact(BPP)) {
                for k in 0..BPP {
                    left[k] = pixel[k].wrapping_add(paeth(
                        left[k] as i16,
                        above[k] as i16,
                        upper_left[k] as i16,
                    ));
                }
                pixel.copy_from_slice(&left);
                upper_left.copy_from_slice(above);
            }
        }
    }
}

/// One step of the two-row Paeth wavefront: both predictions, from state settled before
/// the step began.
///
/// The four operands are laid out one array per role rather than one per lane so that the
/// two lanes sit adjacent in memory. Both loop bounds are constants, so this unrolls into
/// `2 * BPP` independent lane computations that pack into single vector operations.
#[inline(always)]
fn paeth_step<const BPP: usize>(
    residual: &[[u8; BPP]; 2],
    left: &[[u8; BPP]; 2],
    up: &[[u8; BPP]; 2],
    upper_left: &[[u8; BPP]; 2],
) -> [[u8; BPP]; 2] {
    let mut value = [[0u8; BPP]; 2];
    for lane in 0..2 {
        for k in 0..BPP {
            value[lane][k] = residual[lane][k].wrapping_add(paeth(
                left[lane][k] as i16,
                up[lane][k] as i16,
                upper_left[lane][k] as i16,
            ));
        }
    }
    value
}

/// Reverses `Paeth` on two adjacent rows at once, as a wavefront one pixel deep.
///
/// Reconstruction is serial along a row, and the Paeth predictor is a long enough chain of
/// compares and selects that a single row leaves the machine mostly waiting rather than
/// working: the loop is bound by the latency of that chain, not by its throughput. Taking
/// the two rows on an anti-diagonal fills the idle slots. Row `r` pixel `x` needs row `r`
/// pixel `x - 1`, and row `r + 1` pixel `x - 1` needs row `r + 1` pixel `x - 2` along with
/// row `r` pixels `x - 1` and `x - 2`; every one of those settled before this step, so the
/// two predictions are independent of each other and issue together.
///
/// `above` is the reconstructed row `r - 1`. `first` and `second` hold rows `r` and `r + 1`,
/// filtered on entry and reconstructed on return.
fn unfilter_paeth_pair<const BPP: usize>(above: &[u8], first: &mut [u8], second: &mut [u8]) {
    debug_assert_eq!(above.len(), first.len());
    debug_assert_eq!(first.len(), second.len());

    let count = first.len() / BPP;
    if count == 0 {
        return;
    }

    // Lane 0 is row `r` at pixel `x`, lane 1 is row `r + 1` at pixel `x - 1`.
    let mut left = [[0u8; BPP]; 2];
    let mut up = [[0u8; BPP]; 2];
    let mut upper_left = [[0u8; BPP]; 2];
    let mut residual = [[0u8; BPP]; 2];

    // Row `r` pixel `x - 2`: the upper-left neighbour of lane 1, one step further back than
    // `left[0]` and so not recoverable from it.
    let mut behind = [0u8; BPP];

    // Every neighbour off the left edge or above the first row is defined to be zero, and
    // `paeth(0, b, 0)` is `b`, so the edges need no special case beyond starting from zeros.
    residual[0].copy_from_slice(&first[..BPP]);
    up[0].copy_from_slice(&above[..BPP]);
    let mut value = paeth_step::<BPP>(&residual, &left, &up, &upper_left);
    first[..BPP].copy_from_slice(&value[0]);
    left[0] = value[0];

    for x in 1..count {
        let at = x * BPP;
        let back = at - BPP;

        residual[0].copy_from_slice(&first[at..at + BPP]);
        residual[1].copy_from_slice(&second[back..at]);
        up[0].copy_from_slice(&above[at..at + BPP]);
        up[1] = left[0];
        upper_left[0].copy_from_slice(&above[back..at]);
        upper_left[1] = behind;

        value = paeth_step::<BPP>(&residual, &left, &up, &upper_left);

        first[at..at + BPP].copy_from_slice(&value[0]);
        second[back..at].copy_from_slice(&value[1]);

        behind = left[0];
        left = value;
    }

    // The second row trails the first by a pixel, so its last one is left over.
    let back = (count - 1) * BPP;
    residual[1].copy_from_slice(&second[back..back + BPP]);
    up[1] = left[0];
    upper_left[1] = behind;
    value = paeth_step::<BPP>(&residual, &left, &up, &upper_left);
    second[back..back + BPP].copy_from_slice(&value[1]);
}

/// Reverses filtering across a whole image and compacts it in place.
///
/// `buffer` holds `height` records of `1 + row_bytes` bytes, exactly as the compressed
/// stream produced them. On return the reconstructed image occupies `buffer[..height *
/// row_bytes]` with the filter bytes removed.
///
/// Each row is first moved down over the filter bytes that precede it, which is always a
/// forward move, and is then reconstructed against the row already sitting immediately
/// before it. That keeps the previous row where the next one needs it without a scratch
/// buffer.
pub fn unfilter_image(
    buffer: &mut [u8],
    row_bytes: usize,
    height: usize,
    bpp: usize,
) -> Result<(), usize> {
    match bpp {
        1 => unfilter_image_bpp::<1>(buffer, row_bytes, height),
        2 => unfilter_image_bpp::<2>(buffer, row_bytes, height),
        3 => unfilter_image_bpp::<3>(buffer, row_bytes, height),
        4 => unfilter_image_bpp::<4>(buffer, row_bytes, height),
        6 => unfilter_image_bpp::<6>(buffer, row_bytes, height),
        8 => unfilter_image_bpp::<8>(buffer, row_bytes, height),
        _ => unreachable!("PNG pixel strides are 1, 2, 3, 4, 6 or 8 bytes"),
    }
}

fn unfilter_image_bpp<const BPP: usize>(
    buffer: &mut [u8],
    row_bytes: usize,
    height: usize,
) -> Result<(), usize> {
    debug_assert!(buffer.len() >= height * (1 + row_bytes));

    let mut row = 0;
    while row < height {
        let source = row * (1 + row_bytes) + 1;
        let dest = row * row_bytes;

        let filter = Filter::from_byte(buffer[source - 1]).ok_or(row)?;

        // Two adjacent `Paeth` rows are worth reconstructing together; see
        // [`unfilter_paeth_pair`]. The first row of the image is excluded because it has no
        // row above it, and a row whose successor uses a different filter falls through to
        // the single-row path, which the next iteration then takes for the successor.
        if filter == Filter::Paeth && row > 0 && row + 1 < height {
            let next = (row + 1) * (1 + row_bytes) + 1;
            if buffer[next - 1] == Filter::Paeth as u8 {
                // In that order: compacting the second row first would overwrite the tail of
                // the first row's filtered bytes, which are still where they were written.
                buffer.copy_within(source..source + row_bytes, dest);
                buffer.copy_within(next..next + row_bytes, dest + row_bytes);

                let (above, rest) = buffer.split_at_mut(dest);
                let (first, second) = rest.split_at_mut(row_bytes);
                unfilter_paeth_pair::<BPP>(
                    &above[dest - row_bytes..],
                    first,
                    &mut second[..row_bytes],
                );
                row += 2;
                continue;
            }
        }

        buffer.copy_within(source..source + row_bytes, dest);

        if row == 0 {
            unfilter_first_row::<BPP>(filter, &mut buffer[..row_bytes]);
        } else {
            let (above, current) = buffer.split_at_mut(dest);
            unfilter_row::<BPP>(
                filter,
                &above[dest - row_bytes..],
                &mut current[..row_bytes],
            );
        }
        row += 1;
    }

    Ok(())
}

// ---------------------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------------------

/// Applies `filter` to `row`, writing the residuals to `out`.
///
/// `prev` is the unfiltered row above, or an all-zero slice for the first row.
pub fn filter_row<const BPP: usize>(filter: Filter, prev: &[u8], row: &[u8], out: &mut [u8]) {
    debug_assert_eq!(row.len(), out.len());
    debug_assert_eq!(prev.len(), row.len());
    let len = row.len();
    let head = BPP.min(len);

    match filter {
        Filter::None => out.copy_from_slice(row),
        Filter::Sub => {
            out[..head].copy_from_slice(&row[..head]);
            for i in BPP..len {
                out[i] = row[i].wrapping_sub(row[i - BPP]);
            }
        }
        Filter::Up => {
            for i in 0..len {
                out[i] = row[i].wrapping_sub(prev[i]);
            }
        }
        Filter::Average => {
            for i in 0..head {
                out[i] = row[i].wrapping_sub(prev[i] >> 1);
            }
            for i in BPP..len {
                let sum = row[i - BPP] as u16 + prev[i] as u16;
                out[i] = row[i].wrapping_sub((sum >> 1) as u8);
            }
        }
        Filter::Paeth => {
            for i in 0..head {
                out[i] = row[i].wrapping_sub(prev[i]);
            }
            for i in BPP..len {
                out[i] = row[i]
                    .wrapping_sub(paeth(row[i - BPP] as i16, prev[i] as i16, prev[i - BPP] as i16));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip<const BPP: usize>(filter: Filter, width_pixels: usize, height: usize) {
        let row_bytes = width_pixels * BPP;
        let image: Vec<u8> = (0..row_bytes * height)
            .map(|i| (i.wrapping_mul(97).wrapping_add(i / row_bytes * 13) % 256) as u8)
            .collect();

        // Build the filtered stream the way an encoder would.
        let mut stream = vec![0u8; height * (1 + row_bytes)];
        let zero_row = vec![0u8; row_bytes];
        for row in 0..height {
            let prev = if row == 0 {
                &zero_row[..]
            } else {
                &image[(row - 1) * row_bytes..row * row_bytes]
            };
            let base = row * (1 + row_bytes);
            stream[base] = filter as u8;
            let (before, after) = stream.split_at_mut(base + 1);
            let _ = before;
            filter_row::<BPP>(
                filter,
                prev,
                &image[row * row_bytes..(row + 1) * row_bytes],
                &mut after[..row_bytes],
            );
        }

        unfilter_image(&mut stream, row_bytes, height, BPP).unwrap();
        assert_eq!(&stream[..row_bytes * height], &image[..], "bpp {BPP} {filter:?}");
    }

    #[test]
    fn every_filter_round_trips() {
        for filter in Filter::ALL {
            round_trip::<1>(filter, 17, 5);
            round_trip::<2>(filter, 13, 4);
            round_trip::<3>(filter, 11, 6);
            round_trip::<4>(filter, 9, 7);
            round_trip::<6>(filter, 5, 3);
            round_trip::<8>(filter, 4, 3);
        }
    }

    /// Narrower than one pixel stride: the whole row is "off the left edge".
    #[test]
    fn rows_narrower_than_the_stride() {
        for filter in Filter::ALL {
            round_trip::<8>(filter, 1, 3);
            round_trip::<4>(filter, 1, 2);
        }
    }

    /// Round-trips an image whose rows use `filters[row % filters.len()]`.
    ///
    /// Reconstruction takes adjacent `Paeth` rows two at a time, so what matters is where
    /// the runs of them start and stop: an odd-length run leaves a row for the single-row
    /// path, and a run reaching the last row has no successor to pair with.
    fn round_trip_mixed<const BPP: usize>(filters: &[Filter], width_pixels: usize, height: usize) {
        let row_bytes = width_pixels * BPP;
        let image: Vec<u8> = (0..row_bytes * height)
            .map(|i| (i.wrapping_mul(31).wrapping_add(i / row_bytes * 7) % 251) as u8)
            .collect();

        let mut stream = vec![0u8; height * (1 + row_bytes)];
        let zero_row = vec![0u8; row_bytes];
        for row in 0..height {
            let filter = filters[row % filters.len()];
            let prev = if row == 0 {
                &zero_row[..]
            } else {
                &image[(row - 1) * row_bytes..row * row_bytes]
            };
            let base = row * (1 + row_bytes);
            stream[base] = filter as u8;
            let (_, after) = stream.split_at_mut(base + 1);
            filter_row::<BPP>(
                filter,
                prev,
                &image[row * row_bytes..(row + 1) * row_bytes],
                &mut after[..row_bytes],
            );
        }

        unfilter_image(&mut stream, row_bytes, height, BPP).unwrap();
        assert_eq!(
            &stream[..row_bytes * height],
            &image[..],
            "bpp {BPP} {filters:?} {width_pixels}x{height}"
        );
    }

    /// Every arrangement of `Paeth` runs the two-row wavefront has to cope with.
    #[test]
    fn paeth_runs_of_every_length_and_alignment() {
        use Filter::{Paeth as P, Sub as S, Up as U};

        let patterns: &[&[Filter]] = &[
            &[P],             // every row, so runs bounded only by the image
            &[P, S],          // no run longer than one
            &[P, P, S],       // an even run, then a break
            &[P, P, P, S],    // an odd run, so one row is left for the single-row path
            &[P, P, P, P, U], // a longer run whose remainder lands differently each time
            &[S, P, P],       // a run that does not start at the top
            &[U, U, P],       // isolated rows between other filters
        ];

        // Heights either side of each pattern's period, so runs end mid-pattern as well as
        // on it, and in particular so a run sometimes reaches the last row of the image.
        for pattern in patterns {
            for height in 1..=9 {
                round_trip_mixed::<1>(pattern, 13, height);
                round_trip_mixed::<2>(pattern, 7, height);
                round_trip_mixed::<3>(pattern, 11, height);
                round_trip_mixed::<4>(pattern, 9, height);
                round_trip_mixed::<6>(pattern, 5, height);
                round_trip_mixed::<8>(pattern, 3, height);
            }
        }
    }

    /// A single pixel per row leaves the wavefront with nothing but its prologue and tail.
    #[test]
    fn paeth_rows_one_pixel_wide() {
        for height in 1..=5 {
            round_trip_mixed::<3>(&[Filter::Paeth], 1, height);
            round_trip_mixed::<4>(&[Filter::Paeth], 1, height);
            round_trip_mixed::<1>(&[Filter::Paeth], 2, height);
        }
    }

    #[test]
    fn paeth_matches_the_specification() {
        // The reference formulation from RFC 2083, written out literally.
        fn reference(a: u8, b: u8, c: u8) -> u8 {
            let p = a as i32 + b as i32 - c as i32;
            let pa = (p - a as i32).abs();
            let pb = (p - b as i32).abs();
            let pc = (p - c as i32).abs();
            if pa <= pb && pa <= pc {
                a
            } else if pb <= pc {
                b
            } else {
                c
            }
        }

        for a in 0..=255u8 {
            for b in [0u8, 1, 63, 127, 128, 200, 255] {
                for c in [0u8, 1, 63, 127, 128, 200, 255] {
                    assert_eq!(paeth(a as i16, b as i16, c as i16), reference(a, b, c), "{a} {b} {c}");
                }
            }
        }
    }
}
