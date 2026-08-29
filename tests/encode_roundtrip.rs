//! Encodes images in every colour type and bit depth, then decodes them back.

use png_spark::common::{BitDepth, ColorType, Info};
use png_spark::encoder::{Encoder, FilterStrategy};
use png_spark::filter::Filter;

fn make_info(width: u32, height: u32, color_type: ColorType, bit_depth: BitDepth) -> Info {
    let mut info = Info::new(width, height, color_type, bit_depth);
    info.palette = (color_type == ColorType::Indexed)
        .then(|| (0..256u32).flat_map(|i| [i as u8, (i * 7) as u8, (i * 13) as u8]).collect());
    info
}

/// Structured pixels: smooth in places, sharp in others, so filters actually differ.
fn pixels(info: &Info) -> Vec<u8> {
    (0..info.output_size())
        .map(|i| {
            let row = i / info.row_bytes().max(1);
            match i % 5 {
                0 => (i / 3) as u8,
                1 => (row * 2) as u8,
                2 => 0,
                3 => 0xff,
                _ => (i.wrapping_mul(2_654_435_761) >> 9) as u8,
            }
        })
        .collect()
}

fn combinations() -> Vec<(ColorType, BitDepth)> {
    let mut all = Vec::new();
    for depth in [BitDepth::One, BitDepth::Two, BitDepth::Four, BitDepth::Eight, BitDepth::Sixteen]
    {
        all.push((ColorType::Grayscale, depth));
    }
    for depth in [BitDepth::Eight, BitDepth::Sixteen] {
        all.push((ColorType::Rgb, depth));
        all.push((ColorType::GrayscaleAlpha, depth));
        all.push((ColorType::Rgba, depth));
    }
    for depth in [BitDepth::One, BitDepth::Two, BitDepth::Four, BitDepth::Eight] {
        all.push((ColorType::Indexed, depth));
    }
    all
}

#[test]
fn every_format_round_trips() {
    let mut encoder = Encoder::new();
    let mut decoder = png_spark::decoder::Decoder::new();

    for (color_type, bit_depth) in combinations() {
        for (width, height) in [(1, 1), (1, 17), (17, 1), (13, 9), (64, 40)] {
            let info = make_info(width, height, color_type, bit_depth);
            let data = pixels(&info);

            for strategy in [
                FilterStrategy::Adaptive,
                FilterStrategy::Sampled,
                FilterStrategy::Fixed(Filter::None),
                FilterStrategy::Fixed(Filter::Paeth),
            ] {
                let mut png = Vec::new();
                encoder.filter(strategy).encode(&info, &data, &mut png).unwrap();

                let decoded = decoder.decode(&png).unwrap_or_else(|e| {
                    panic!("{color_type:?}/{bit_depth:?} {width}x{height} {strategy:?}: {e}")
                });
                assert_eq!(decoded.info.width, width);
                assert_eq!(decoded.info.height, height);
                assert_eq!(decoded.info.color_type, color_type);
                assert_eq!(decoded.info.bit_depth, bit_depth);
                assert_eq!(
                    decoded.data, data,
                    "{color_type:?}/{bit_depth:?} {width}x{height} {strategy:?}"
                );
            }
        }
    }
}

#[test]
fn rejects_mismatched_buffers() {
    let info = make_info(4, 4, ColorType::Rgba, BitDepth::Eight);
    let mut output = Vec::new();
    assert!(Encoder::new().encode(&info, &[0; 10], &mut output).is_err());
}

#[test]
fn convenience_helpers_produce_decodable_files() {
    let rgba: Vec<u8> = (0..16 * 16 * 4).map(|i| (i * 3) as u8).collect();
    let png = png_spark::encoder::encode_rgba8(16, 16, &rgba).unwrap();
    let decoded = png_spark::decoder::decode(&png).unwrap();
    assert_eq!(decoded.data, rgba);

    let rgb: Vec<u8> = (0..16 * 16 * 3).map(|i| (i * 5) as u8).collect();
    let png = png_spark::encoder::encode_rgb8(16, 16, &rgb).unwrap();
    assert_eq!(png_spark::decoder::decode(&png).unwrap().data, rgb);
}

#[test]
fn rejects_colour_chunks_the_decoder_would_reject() {
    let mut info = make_info(4, 4, ColorType::Rgb, BitDepth::Eight);
    let data = pixels(&info);
    let mut output = Vec::new();

    // tRNS for a truecolour image is three 16-bit samples; anything else is invalid.
    info.transparency = Some(vec![0, 1, 0]);
    assert!(Encoder::new().encode(&info, &data, &mut output).is_err());

    info.transparency = Some(vec![0, 1, 0, 2, 0, 3]);
    output.clear();
    assert!(Encoder::new().encode(&info, &data, &mut output).is_ok());

    // A palette must be whole RGB triples.
    let mut indexed = make_info(4, 4, ColorType::Indexed, BitDepth::Eight);
    indexed.palette = Some(vec![1, 2, 3, 4]);
    output.clear();
    let indexed_data = vec![0u8; indexed.output_size()];
    assert!(Encoder::new().encode(&indexed, &indexed_data, &mut output).is_err());
}
