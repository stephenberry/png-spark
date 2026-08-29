//! The decompressor against arbitrary bytes, reached without going through a PNG container.
//!
//! `decode` only ever hands inflate a stream that survived chunk parsing and CRC checks,
//! which leaves most of the malformed-stream space unreachable from that target.
#![no_main]

use libfuzzer_sys::fuzz_target;

const MAX_OUTPUT: usize = 16 << 20;

fuzz_target!(|data: &[u8]| {
    // The unknown-length path, since it is the one that retries and reallocates.
    let _ = png_spark::inflate::decompress_zlib_to_vec(data, MAX_OUTPUT);
});
