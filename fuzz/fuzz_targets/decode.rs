//! The decoder against arbitrary bytes: a malformed file must be an error, never a crash.
#![no_main]

use libfuzzer_sys::fuzz_target;

/// Far below the library's own default.
///
/// The point of the limit here is not to model what a caller would choose but to keep the
/// fuzzer's budget going into parser states rather than into allocating whatever a hostile
/// header asked for. A run that spends its time in `alloc_zeroed` explores nothing.
const MAX_DECOMPRESSED: usize = 16 << 20;

fuzz_target!(|data: &[u8]| {
    let mut decoder = png_spark::Decoder::new();
    decoder
        // Both are off or minimal by default, and both are code the fuzzer should reach:
        // `Full` runs the Adler-32 verification, `All` runs chunk retention and copying.
        .checks(png_spark::Checks::Full)
        .keep(png_spark::Keep::All)
        .max_decompressed_size(Some(MAX_DECOMPRESSED));

    let _ = decoder.decode(data);
});
