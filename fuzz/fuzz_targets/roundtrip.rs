//! Anything the decoder accepts must survive being written back out and read again.
//!
//! The other two targets can only fail on a crash. This one fails on a wrong answer, which
//! is the more valuable kind of bug and the kind no amount of "did not panic" will find.
#![no_main]

use libfuzzer_sys::fuzz_target;

const MAX_DECOMPRESSED: usize = 16 << 20;

fuzz_target!(|data: &[u8]| {
    let mut decoder = png_spark::Decoder::new();
    decoder.max_decompressed_size(Some(MAX_DECOMPRESSED));

    // Most inputs are not PNGs at all; those are `decode`'s business, not this target's.
    let Ok(first) = decoder.decode(data) else {
        return;
    };

    let png = png_spark::encode(&first.info, &first.data)
        .expect("the encoder must accept what the decoder produced");
    let second = decoder.decode(&png).expect("the decoder must accept its own output");

    // Interlacing is deliberately absent from these: the encoder always writes a
    // non-interlaced file, so an Adam7 input legitimately comes back as `None`. The pixels
    // are what must not change.
    assert_eq!(first.data, second.data, "pixels differ across a round trip");
    assert_eq!(first.info.width, second.info.width);
    assert_eq!(first.info.height, second.info.height);
    assert_eq!(first.info.color_type, second.info.color_type);
    assert_eq!(first.info.bit_depth, second.info.bit_depth);
    assert_eq!(first.info.palette, second.info.palette);
    assert_eq!(first.info.transparency, second.info.transparency);
});
