//! Verifies png-spark's output against independent implementations.
//!
//! Round-tripping through our own code proves self-consistency; these tests prove the
//! streams and files are actually what the formats specify, by handing them to `fdeflate`
//! and the `png` crate.

use png_spark::deflate::Deflater;

fn corpus() -> Vec<(String, Vec<u8>)> {
    let mut cases: Vec<(String, Vec<u8>)> = vec![
        ("empty".into(), Vec::new()),
        ("one".into(), vec![42]),
        ("zeros".into(), vec![0; 100_000]),
        ("text".into(), b"the quick brown fox. ".repeat(5000)),
    ];

    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for case in 0..12 {
        let len = [3usize, 100, 5000, 200_000][case % 4];
        let alphabet = 1 + case % 5;
        cases.push((
            format!("random{case}"),
            (0..len).map(|_| (next() % alphabet as u64) as u8).collect(),
        ));
    }
    cases.push(("noise".into(), (0..80_000).map(|_| next() as u8).collect()));
    cases
}

#[test]
fn deflate_output_decodes_with_fdeflate() {
    let mut deflater = Deflater::new();
    for (name, data) in corpus() {
        let mut compressed = Vec::new();
        deflater.zlib(&data, &mut compressed);
        let decoded =
            fdeflate::decompress_to_vec(&compressed).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert_eq!(decoded, data, "{name}");
    }
}

#[test]
fn inflate_accepts_fdeflate_output() {
    for (name, data) in corpus() {
        let compressed = fdeflate::compress_to_vec(&data);
        let decoded = png_spark::inflate::decompress_zlib(&compressed, data.len())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(decoded, data, "{name}");
    }
}

#[test]
fn encoded_pngs_decode_with_the_png_crate() {
    use png_spark::common::{BitDepth, ColorType, Info};
    use png_spark::encoder::{Encoder, FilterStrategy};

    let mut encoder = Encoder::new();
    let cases = [
        (ColorType::Grayscale, BitDepth::Eight),
        (ColorType::Grayscale, BitDepth::Sixteen),
        (ColorType::Grayscale, BitDepth::Four),
        (ColorType::Rgb, BitDepth::Eight),
        (ColorType::Rgb, BitDepth::Sixteen),
        (ColorType::GrayscaleAlpha, BitDepth::Eight),
        (ColorType::Rgba, BitDepth::Eight),
        (ColorType::Indexed, BitDepth::Eight),
        (ColorType::Indexed, BitDepth::Two),
    ];

    for (color_type, bit_depth) in cases {
        let palette = (color_type == ColorType::Indexed)
            .then(|| (0..256u32).flat_map(|i| [i as u8, (i * 3) as u8, (i * 11) as u8]).collect());
        let mut info = Info::new(61, 37, color_type, bit_depth);
        info.palette = palette;
        let data: Vec<u8> =
            (0..info.output_size()).map(|i| (i.wrapping_mul(2_654_435_761) >> 11) as u8).collect();

        for strategy in [FilterStrategy::Adaptive, FilterStrategy::Sampled] {
            let mut out = Vec::new();
            encoder.filter(strategy).encode(&info, &data, &mut out).unwrap();

            let decoder = png::Decoder::new(std::io::Cursor::new(&out));
            let mut reader =
                decoder.read_info().unwrap_or_else(|e| panic!("{color_type:?}/{bit_depth:?}: {e}"));
            let mut buffer = vec![0; reader.output_buffer_size().unwrap()];
            let frame = reader.next_frame(&mut buffer).unwrap();
            assert_eq!(
                &buffer[..frame.buffer_size()],
                &data[..],
                "{color_type:?}/{bit_depth:?} {strategy:?}"
            );
        }
    }
}

/// Application data in an ancillary chunk must be invisible to every other PNG reader.
#[test]
fn private_chunks_are_ignored_by_the_png_crate() {
    use png_spark::common::{BitDepth, Chunk, ColorType, Info};

    let mut info = Info::new(48, 31, ColorType::Rgba, BitDepth::Eight);
    info.metadata = vec![
        Chunk::new(*b"apPd", (0..=255u8).collect()),
        Chunk::new(*b"bnDl", b"arbitrary bytes\0with a null".to_vec()),
    ];
    let data: Vec<u8> =
        (0..info.output_size()).map(|i| (i.wrapping_mul(2_654_435_761) >> 11) as u8).collect();
    let png = png_spark::encode(&info, &data).unwrap();

    let decoder = png::Decoder::new(std::io::Cursor::new(&png));
    let mut reader = decoder.read_info().unwrap();
    let mut buffer = vec![0; reader.output_buffer_size().unwrap()];
    let frame = reader.next_frame(&mut buffer).unwrap();
    assert_eq!(&buffer[..frame.buffer_size()], &data[..]);
}
