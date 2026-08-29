//! Static tables derived from the DEFLATE specification (RFC 1951).

/// Base match length for literal/length symbols 257..=285.
pub const LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];

/// Number of extra bits following literal/length symbols 257..=285.
pub const LEN_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];

/// Base match distance for distance symbols 0..=29.
pub const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];

/// Number of extra bits following distance symbols 0..=29.
pub const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// The order in which code length code lengths appear in a dynamic block header.
pub const CLCL_ORDER: [u8; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// Code lengths of the fixed literal/length Huffman code.
pub const FIXED_LITLEN_LENGTHS: [u8; 288] = {
    let mut lengths = [8u8; 288];
    let mut i = 144;
    while i < 256 {
        lengths[i] = 9;
        i += 1;
    }
    while i < 280 {
        lengths[i] = 7;
        i += 1;
    }
    lengths
};

/// Code lengths of the fixed distance Huffman code.
///
/// All 32 slots are five bits wide. Symbols 30 and 31 are part of the code space but may
/// never be emitted, so the decoder marks them unusable.
pub const FIXED_DIST_LENGTHS: [u8; 32] = [5; 32];

/// Maps a match length in `3..=258` to its literal/length symbol, offset by 3.
pub const LEN_TO_SYM: [u16; 256] = {
    let mut table = [0u16; 256];
    let mut sym = 0;
    while sym < 29 {
        let base = LEN_BASE[sym] as usize;
        let count = 1usize << LEN_EXTRA[sym];
        let mut i = 0;
        while i < count && base - 3 + i < 256 {
            table[base - 3 + i] = 257 + sym as u16;
            i += 1;
        }
        sym += 1;
    }
    table
};

/// Maps a match distance in `1..=32768` to its distance symbol.
///
/// Distances below 257 index directly by `distance - 1`; larger distances index by
/// `256 + ((distance - 1) >> 7)`, which is unambiguous because every symbol from 16 upward
/// spans at least 128 distances.
pub const DIST_TO_SYM: [u8; 512] = {
    let mut table = [0u8; 512];
    let mut sym = 0usize;
    while sym < 30 {
        let base = DIST_BASE[sym] as usize;
        let count = 1usize << DIST_EXTRA[sym];
        let mut i = 0;
        while i < count {
            let dist = base + i;
            let index = if dist <= 256 { dist - 1 } else { 256 + ((dist - 1) >> 7) };
            table[index] = sym as u8;
            i += 1;
        }
        sym += 1;
    }
    table
};

/// Returns the distance symbol for a match distance in `1..=32768`.
#[inline]
pub fn dist_symbol(distance: u16) -> u8 {
    let d = distance as usize;
    DIST_TO_SYM[if d <= 256 { d - 1 } else { 256 + ((d - 1) >> 7) }]
}
