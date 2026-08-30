//! Compressing a zlib stream in pieces, as `Encoder::encode_to` does band by band.

use png_spark::deflate::Deflater;
use png_spark::inflate::decompress_zlib_to_vec;

const LIMIT: usize = 16 << 20;

fn mixed(len: usize) -> Vec<u8> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    (0..len)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Long compressible stretches broken up by noise, so both block kinds appear.
            if (i / 4096).is_multiple_of(2) { (i / 311) as u8 } else { (state >> 32) as u8 }
        })
        .collect()
}

fn push_pieces(data: &[u8], cuts: &[usize]) -> Vec<u8> {
    let mut deflater = Deflater::new();
    let mut out = Vec::new();
    let mut stream = deflater.zlib_start(&mut out);

    let mut start = 0;
    for (index, &cut) in cuts.iter().enumerate() {
        let end = cut.min(data.len());
        let last = index + 1 == cuts.len();
        deflater.zlib_push(&mut stream, &data[start..end], last, &mut out);
        start = end;
    }
    out
}

#[test]
fn a_stream_pushed_in_pieces_inflates_to_what_went_in() {
    let data = mixed(900_000);
    let splittings: &[&[usize]] = &[
        &[900_000],
        &[1, 900_000],
        &[450_000, 900_000],
        &[262_144, 524_288, 900_000],
        &[100, 100, 200_000, 899_999, 900_000],
    ];
    for cuts in splittings {
        let stream = push_pieces(&data, cuts);
        let back =
            decompress_zlib_to_vec(&stream, LIMIT).unwrap_or_else(|e| panic!("{cuts:?}: {e}"));
        assert_eq!(back, data, "{cuts:?}");
    }
}

#[test]
fn empty_pieces_in_the_middle_are_harmless() {
    let data = mixed(50_000);
    let stream = push_pieces(&data, &[0, 20_000, 20_000, 50_000, 50_000]);
    assert_eq!(decompress_zlib_to_vec(&stream, LIMIT).unwrap(), data);
}

#[test]
fn a_stream_with_no_bytes_at_all_is_still_valid() {
    let stream = push_pieces(&[], &[0]);
    assert_eq!(decompress_zlib_to_vec(&stream, LIMIT).unwrap(), Vec::<u8>::new());
}

#[test]
fn one_piece_matches_the_whole_buffer_call() {
    let data = mixed(300_000);

    let mut whole = Vec::new();
    Deflater::new().zlib(&data, &mut whole);

    assert_eq!(push_pieces(&data, &[300_000]), whole);
}

/// Pieces the size of the compressor's own block cost almost nothing against one call. Much
/// smaller pieces do cost, because the Huffman code is fitted per block.
#[test]
fn band_sized_pieces_compress_about_as_well_as_one_call() {
    let data = mixed(1_000_000);

    let mut whole = Vec::new();
    Deflater::new().zlib(&data, &mut whole);

    let cuts: Vec<usize> = (1..=4).map(|i| (i * 262_144).min(data.len())).collect();
    let banded = push_pieces(&data, &cuts);

    assert_eq!(decompress_zlib_to_vec(&banded, LIMIT).unwrap(), data);
    let ratio = banded.len() as f64 / whole.len() as f64;
    assert!(ratio < 1.02, "banded output was {ratio:.3}x the one-shot size");
}
