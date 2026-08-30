//! The `io::Write` encoder path: same image, several `IDAT` chunks, errors reported.

use png_spark::{BitDepth, ColorType, Decoder, Encoder, Info, WriteError};

/// Pseudo-random bytes, so the image does not compress down to a single chunk.
fn noise(len: usize) -> Vec<u8> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 24) as u8
        })
        .collect()
}

fn rgba(width: u32, height: u32) -> (Info, Vec<u8>) {
    let info = Info::new(width, height, ColorType::Rgba, BitDepth::Eight);
    let pixels = noise(info.output_size());
    (info, pixels)
}

/// Chunk types in order, so a test can see how the image data was split.
fn chunk_kinds(png: &[u8]) -> Vec<[u8; 4]> {
    let mut kinds = Vec::new();
    let mut pos = 8;
    while pos + 12 <= png.len() {
        let length = u32::from_be_bytes(png[pos..pos + 4].try_into().unwrap()) as usize;
        let kind: [u8; 4] = png[pos + 4..pos + 8].try_into().unwrap();
        kinds.push(kind);
        pos += 12 + length;
    }
    kinds
}

#[test]
fn a_streamed_image_decodes_to_the_pixels_that_went_in() {
    for (width, height) in [(1, 1), (17, 3), (64, 64), (300, 200)] {
        let (info, pixels) = rgba(width, height);
        let mut png = Vec::new();
        Encoder::new().encode_to(&info, &pixels, &mut png).unwrap();

        let image = png_spark::decode(&png).unwrap();
        assert_eq!(image.data, pixels, "{width}x{height}");
        assert_eq!(image.width(), width);
        assert_eq!(image.height(), height);
    }
}

#[test]
fn an_image_larger_than_one_chunk_is_split_across_several_idats() {
    // A megabyte of noise cannot fit in one 64 KiB chunk however it is filtered.
    let (info, pixels) = rgba(512, 512);
    let mut png = Vec::new();
    Encoder::new().encode_to(&info, &pixels, &mut png).unwrap();

    let kinds = chunk_kinds(&png);
    let idats = kinds.iter().filter(|kind| *kind == b"IDAT").count();
    assert!(idats > 1, "expected several IDAT chunks, got {idats}");

    // Order still has to be right: header first, image data in the middle, IEND last.
    assert_eq!(kinds.first(), Some(b"IHDR"));
    assert_eq!(kinds.last(), Some(b"IEND"));

    assert_eq!(png_spark::decode(&png).unwrap().data, pixels);
}

#[test]
fn every_colour_type_survives_the_streamed_path() {
    let cases = [
        (ColorType::Grayscale, BitDepth::Eight),
        (ColorType::Grayscale, BitDepth::Sixteen),
        (ColorType::Rgb, BitDepth::Eight),
        (ColorType::Rgb, BitDepth::Sixteen),
        (ColorType::GrayscaleAlpha, BitDepth::Eight),
        (ColorType::Rgba, BitDepth::Eight),
        (ColorType::Rgba, BitDepth::Sixteen),
        (ColorType::Grayscale, BitDepth::One),
        (ColorType::Grayscale, BitDepth::Two),
        (ColorType::Grayscale, BitDepth::Four),
    ];
    for (color_type, bit_depth) in cases {
        let info = Info::new(70, 50, color_type, bit_depth);
        let pixels = noise(info.output_size());
        let mut png = Vec::new();
        Encoder::new().encode_to(&info, &pixels, &mut png).unwrap();
        let image = png_spark::decode(&png).unwrap();
        assert_eq!(image.data, pixels, "{color_type:?} {bit_depth:?}");
    }
}

#[test]
fn a_palette_and_its_transparency_survive_the_streamed_path() {
    let mut info = Info::new(40, 40, ColorType::Indexed, BitDepth::Eight);
    info.palette = Some((0..256).flat_map(|i| [i as u8, 255 - i as u8, 128]).collect());
    info.transparency = Some((0..256).map(|i| i as u8).collect());
    let pixels = noise(info.output_size());

    let mut png = Vec::new();
    Encoder::new().encode_to(&info, &pixels, &mut png).unwrap();

    let image = png_spark::decode(&png).unwrap();
    assert_eq!(image.data, pixels);
    assert_eq!(image.info.palette, info.palette);
    assert_eq!(image.info.transparency, info.transparency);
}

#[test]
fn metadata_survives_the_streamed_path() {
    let (mut info, pixels) = rgba(20, 20);
    info.metadata.push(png_spark::Chunk::new(*b"apPd", b"carried through".to_vec()));

    let mut png = Vec::new();
    Encoder::new().encode_to(&info, &pixels, &mut png).unwrap();

    let mut decoder = Decoder::new();
    decoder.keep(png_spark::Keep::All);
    let image = decoder.decode(&png).unwrap();
    assert_eq!(image.info.chunk(b"apPd"), Some(&b"carried through"[..]));
}

/// A sink that accepts `allow` bytes and then refuses, like a disk filling up mid-file.
struct ShortSink {
    allow: usize,
}

impl std::io::Write for ShortSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.allow == 0 {
            return Err(std::io::Error::other("sink is full"));
        }
        let taken = buf.len().min(self.allow);
        self.allow -= taken;
        Ok(taken)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn a_sink_that_fails_is_reported_rather_than_panicking() {
    let (info, pixels) = rgba(512, 512);
    // Enough for the header chunks, not for the image data.
    let mut sink = ShortSink { allow: 100 };
    let error = Encoder::new().encode_to(&info, &pixels, &mut sink).unwrap_err();
    assert!(matches!(error, WriteError::Io(_)), "{error:?}");
}

#[test]
fn a_sink_that_fails_immediately_is_reported() {
    let (info, pixels) = rgba(8, 8);
    let mut sink = ShortSink { allow: 0 };
    let error = Encoder::new().encode_to(&info, &pixels, &mut sink).unwrap_err();
    assert!(matches!(error, WriteError::Io(_)), "{error:?}");
}

#[test]
fn a_rejected_image_reaches_the_sink_not_at_all() {
    let info = Info::new(4, 4, ColorType::Rgba, BitDepth::Eight);
    let mut written = Vec::new();
    // One byte short of what the header describes.
    let error = Encoder::new()
        .encode_to(&info, &vec![0u8; info.output_size() - 1], &mut written)
        .unwrap_err();

    assert!(matches!(error, WriteError::Encode(_)), "{error:?}");
    assert!(written.is_empty(), "a rejected image wrote {} bytes", written.len());
}

/// `encode` is `encode_to` with a `Vec` sink. Pinned so that if the two are ever split apart
/// again, they are held to producing the same file.
#[test]
fn the_buffered_and_streamed_paths_agree() {
    for (width, height) in [(1, 1), (63, 65), (256, 300)] {
        let (info, pixels) = rgba(width, height);

        let mut buffered = Vec::new();
        Encoder::new().encode(&info, &pixels, &mut buffered).unwrap();
        let mut streamed = Vec::new();
        Encoder::new().encode_to(&info, &pixels, &mut streamed).unwrap();

        assert_eq!(buffered, streamed, "{width}x{height}");
        assert_eq!(png_spark::decode(&buffered).unwrap().data, pixels);
    }
}

/// A smooth ramp, which compresses hard enough that a failure to compress is unmistakable.
fn ramp(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i / 977) as u8).collect()
}

/// A row wider than a band forces the band to split, and a split that leaves a runt block
/// used to poison the density estimate the next block is chosen by: every block after it was
/// written stored. At width 65535 this compressed 270x and at 65536 it compressed 1.1x.
#[test]
fn a_row_wider_than_a_band_still_compresses() {
    for width in [65535u32, 65536, 65537, 65600, 131072] {
        let info = Info::new(width, 4, ColorType::Rgba, BitDepth::Eight);
        let pixels = ramp(info.output_size());

        let mut png = Vec::new();
        Encoder::new().encode_to(&info, &pixels, &mut png).unwrap();

        let ratio = pixels.len() as f64 / png.len() as f64;
        assert!(ratio > 50.0, "width {width} compressed only {ratio:.2}x");
        assert_eq!(png_spark::decode(&png).unwrap().data, pixels);
    }
}

/// Compressible data is coded rather than stored, so its blocks do not end on byte
/// boundaries. That is the only way the bit carried across a band boundary gets exercised:
/// incompressible data is stored, and stored blocks are byte aligned.
#[test]
fn a_compressible_image_spanning_many_bands_round_trips() {
    let info = Info::new(1024, 1024, ColorType::Rgba, BitDepth::Eight);
    let pixels = ramp(info.output_size());

    let mut png = Vec::new();
    Encoder::new().encode_to(&info, &pixels, &mut png).unwrap();

    let ratio = pixels.len() as f64 / png.len() as f64;
    assert!(ratio > 20.0, "compressed only {ratio:.2}x");
    assert_eq!(png_spark::decode(&png).unwrap().data, pixels);
}

#[test]
fn an_encoder_reused_across_streamed_images_keeps_no_state() {
    let mut encoder = Encoder::new();
    let mut first = Vec::new();
    let (info_a, pixels_a) = rgba(40, 40);
    encoder.encode_to(&info_a, &pixels_a, &mut first).unwrap();

    let (info_b, pixels_b) = rgba(31, 47);
    let mut second = Vec::new();
    encoder.encode_to(&info_b, &pixels_b, &mut second).unwrap();

    // The second image must be exactly what a fresh encoder would have written.
    let mut fresh = Vec::new();
    Encoder::new().encode_to(&info_b, &pixels_b, &mut fresh).unwrap();
    assert_eq!(second, fresh);
    assert_eq!(png_spark::decode(&second).unwrap().data, pixels_b);
}
