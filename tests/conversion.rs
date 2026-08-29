//! Checks pixel-format conversion against hand-computed expectations.

use png_spark::common::{BitDepth, ColorType, Info};
use png_spark::decoder::Image;

fn image(info: Info, data: Vec<u8>) -> Image {
    Image { info, data }
}

fn info(width: u32, height: u32, color_type: ColorType, bit_depth: BitDepth) -> Info {
    Info::new(width, height, color_type, bit_depth)
}

#[test]
fn grayscale_bit_depths_scale_to_the_full_range() {
    // 1-bit: 0b1010_0000 is white, black, white, black across four pixels.
    let one = image(info(4, 1, ColorType::Grayscale, BitDepth::One), vec![0b1010_0000]);
    assert_eq!(
        one.to_rgba8().unwrap(),
        [255, 255, 255, 255, 0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 255,]
    );

    // 2-bit: the four levels must land on 0, 85, 170, 255.
    let two = image(info(4, 1, ColorType::Grayscale, BitDepth::Two), vec![0b00_01_10_11]);
    let rgb = two.to_rgb8().unwrap();
    assert_eq!(rgb.iter().step_by(3).copied().collect::<Vec<_>>(), [0, 85, 170, 255]);

    // 4-bit: each nibble repeats into a byte.
    let four = image(info(2, 1, ColorType::Grayscale, BitDepth::Four), vec![0x0F]);
    assert_eq!(four.to_rgb8().unwrap(), [0, 0, 0, 255, 255, 255]);

    // 16-bit keeps the high byte.
    let sixteen =
        image(info(2, 1, ColorType::Grayscale, BitDepth::Sixteen), vec![0x12, 0x34, 0xAB, 0xCD]);
    assert_eq!(sixteen.to_rgb8().unwrap(), [0x12, 0x12, 0x12, 0xAB, 0xAB, 0xAB]);
}

#[test]
fn transparency_becomes_alpha() {
    let mut grey = info(3, 1, ColorType::Grayscale, BitDepth::Eight);
    grey.transparency = Some(vec![0x00, 0x40]);
    let converted = image(grey, vec![0x10, 0x40, 0x80]).to_rgba8().unwrap();
    assert_eq!(converted[3], 255);
    assert_eq!(converted[7], 0, "the sample matching tRNS is transparent");
    assert_eq!(converted[11], 255);

    let mut rgb = info(2, 1, ColorType::Rgb, BitDepth::Eight);
    rgb.transparency = Some(vec![0, 1, 0, 2, 0, 3]);
    let converted = image(rgb, vec![1, 2, 3, 4, 5, 6]).to_rgba8().unwrap();
    assert_eq!(converted[3], 0, "the pixel matching tRNS is transparent");
    assert_eq!(converted[7], 255);
}

#[test]
fn palettes_resolve_with_per_entry_alpha() {
    let mut indexed = info(4, 1, ColorType::Indexed, BitDepth::Two);
    indexed.palette = Some(vec![10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42]);
    indexed.transparency = Some(vec![0, 128]);

    let converted = image(indexed, vec![0b00_01_10_11]).to_rgba8().unwrap();
    assert_eq!(
        converted,
        [
            10, 11, 12, 0, // entry 0, tRNS 0
            20, 21, 22, 128, // entry 1, tRNS 128
            30, 31, 32, 255, // entry 2, past the end of tRNS
            40, 41, 42, 255,
        ]
    );
}

#[test]
fn alpha_channels_pass_through() {
    let rgba = image(
        info(1, 1, ColorType::Rgba, BitDepth::Sixteen),
        vec![0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88],
    );
    assert_eq!(rgba.to_rgba8().unwrap(), [0x11, 0x33, 0x55, 0x77]);
    assert_eq!(rgba.to_rgb8().unwrap(), [0x11, 0x33, 0x55]);

    let grey_alpha =
        image(info(2, 1, ColorType::GrayscaleAlpha, BitDepth::Eight), vec![9, 200, 60, 0]);
    assert_eq!(grey_alpha.to_rgba8().unwrap(), [9, 9, 9, 200, 60, 60, 60, 0]);
}

/// Every image in the generated corpus must convert without panicking, and the result must
/// have exactly one pixel per pixel.
#[test]
fn corpus_converts_cleanly() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tmp/png");
    if !dir.exists() {
        eprintln!("skipping: {} not generated", dir.display());
        return;
    }
    let mut decoder = png_spark::decoder::Decoder::new();
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("png") {
            continue;
        }
        let decoded = decoder.decode(&std::fs::read(&path).unwrap()).unwrap();
        let pixels = decoded.width() as usize * decoded.height() as usize;
        assert_eq!(decoded.to_rgba8().unwrap().len(), pixels * 4, "{}", path.display());
        assert_eq!(decoded.to_rgb8().unwrap().len(), pixels * 3, "{}", path.display());
    }
}
