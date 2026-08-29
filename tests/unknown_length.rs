//! Decompressing a zlib stream whose expanded length is not known in advance.
//!
//! `decompress_zlib` is shaped for PNG, where `IHDR` states the length. A caller without
//! that number must not invent one by trusting a length field in an untrusted file, which
//! is the same instruction to over-allocate wearing different clothes.

use png_spark::deflate::compress_zlib;
use png_spark::inflate::{InflateError, decompress_zlib_to_vec};

/// Something that compresses, at a few different scales relative to the first guess.
fn sample(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i / 7 % 251) as u8).collect()
}

#[test]
fn a_stream_round_trips_without_being_told_its_length() {
    for len in [0usize, 1, 100, 5_000, 300_000, 3_000_000] {
        let compressed = compress_zlib(&sample(len));
        let decoded = decompress_zlib_to_vec(&compressed, 8 << 20)
            .unwrap_or_else(|e| panic!("{len} bytes: {e}"));
        assert_eq!(decoded, sample(len), "{len} bytes round-tripped incorrectly");
    }
}

#[test]
fn the_ceiling_is_a_ceiling() {
    let compressed = compress_zlib(&sample(100_000));

    // Exactly the right size is enough, and it is the smallest size that is.
    assert_eq!(decompress_zlib_to_vec(&compressed, 100_000).unwrap().len(), 100_000);
    assert_eq!(
        decompress_zlib_to_vec(&compressed, 99_999),
        Err(InflateError::OutputOverflow),
        "a stream that does not fit under the ceiling must be refused, not truncated"
    );
    assert_eq!(decompress_zlib_to_vec(&compressed, 0), Err(InflateError::OutputOverflow));
}

#[test]
fn a_highly_compressible_stream_cannot_talk_its_way_past_the_ceiling() {
    // Four kilobytes of input expanding to eight megabytes: the case a guess drawn from the
    // compressed size gets wrong, and the case where the ceiling has to be the thing that
    // holds.
    let compressed = compress_zlib(&vec![0u8; 8 << 20]);
    assert!(compressed.len() < 64 * 1024, "the premise: this compresses hard");

    assert_eq!(decompress_zlib_to_vec(&compressed, 1 << 20), Err(InflateError::OutputOverflow));
    assert_eq!(decompress_zlib_to_vec(&compressed, 8 << 20).unwrap(), vec![0u8; 8 << 20]);
}

#[test]
fn corruption_is_still_caught_without_a_length_to_check_against() {
    // The argument for not carrying a length field at all: the zlib framing already covers
    // the data with an Adler-32, so a truncated or altered stream is caught regardless.
    let compressed = compress_zlib(&sample(50_000));

    for cut in [2usize, compressed.len() / 2, compressed.len() - 5] {
        assert!(
            decompress_zlib_to_vec(&compressed[..cut], 1 << 20).is_err(),
            "truncation at {cut} must not pass"
        );
    }

    let mut altered = compressed.clone();
    let middle = altered.len() / 2;
    altered[middle] ^= 0xFF;
    assert!(decompress_zlib_to_vec(&altered, 1 << 20).is_err());
}

#[test]
fn hostile_streams_terminate() {
    // Every payload carries a valid zlib header, so each iteration reaches the decompressor
    // and the retry loop rather than being turned away two bytes in. Random bytes clear that
    // header check about one time in a thousand, which is why they are not left to chance.
    let mut state = 0x0f1e_2d3c_4b5a_6978u64;
    for _ in 0..2000 {
        let length = (state % 200) as usize + 2;
        let mut data = vec![0x78u8, 0x9c];
        data.extend((0..length).map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u8
        }));
        let _ = decompress_zlib_to_vec(&data, 1 << 20);
    }
}

#[test]
fn an_empty_stream_needs_no_room_at_all() {
    // The corner of the ceiling: a stream expanding to nothing fits under a ceiling of
    // nothing, so a zero `max_output` is not by itself a refusal.
    let compressed = compress_zlib(b"");
    assert_eq!(decompress_zlib_to_vec(&compressed, 0).unwrap(), Vec::<u8>::new());
}
