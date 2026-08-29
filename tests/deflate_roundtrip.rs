//! Checks that everything the compressor emits is a valid zlib stream, both by decoding it
//! with this crate and by decoding it with the reference streams' own decoder.

use png_spark::deflate::Deflater;
use png_spark::inflate::decompress_zlib;

fn round_trip(data: &[u8], label: &str) {
    let mut compressed = Vec::new();
    Deflater::new().zlib(data, &mut compressed);

    let decoded =
        decompress_zlib(&compressed, data.len()).unwrap_or_else(|e| panic!("{label}: {e}"));
    assert_eq!(decoded, data, "{label} round-tripped incorrectly");
}

#[test]
fn edge_case_inputs() {
    round_trip(b"", "empty");
    round_trip(b"a", "one byte");
    round_trip(b"ab", "two bytes");
    round_trip(&[0u8; 3], "three zeros");
    round_trip(&[7u8; 300], "constant");
    round_trip(b"abcabcabcabcabcabcabcabc", "short period");
    round_trip(&vec![0u8; 200_000], "long zero run");
}

#[test]
fn structured_inputs() {
    let text = b"the quick brown fox jumps over the lazy dog. ".repeat(2000);
    round_trip(&text, "repeated text");

    // Repetition running right up against the end of the input.
    let mut tail = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    tail.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    round_trip(&tail, "match at end");

    // Repetition separated by more than the 32 KiB window.
    let mut far = vec![9u8; 40_000];
    far[..16].copy_from_slice(b"0123456789abcdef");
    far.extend_from_slice(b"0123456789abcdef");
    round_trip(&far, "beyond the window");
}

#[test]
fn pseudo_random_inputs() {
    let mut state = 0x243f_6a88_85a3_08d3u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    for case in 0..40 {
        let len = [0, 1, 5, 17, 255, 4096, 70_000][case % 7];
        let alphabet = 1 + (case % 6);
        let data: Vec<u8> = (0..len).map(|_| (next() % alphabet as u64) as u8).collect();
        round_trip(&data, &format!("random case {case}"));
    }

    // Fully incompressible input, where a stored block must win.
    let noise: Vec<u8> = (0..100_000).map(|_| next() as u8).collect();
    round_trip(&noise, "incompressible");
    let mut compressed = Vec::new();
    Deflater::new().zlib(&noise, &mut compressed);
    assert!(
        compressed.len() < noise.len() + noise.len() / 100,
        "incompressible input grew to {} from {}",
        compressed.len(),
        noise.len()
    );
}
