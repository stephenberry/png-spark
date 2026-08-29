//! Reading a file's header without decoding it, and asking what that header means.

use png_spark::{BitDepth, Chunk, ColorType, Decoder, Error, Info, Keep};

/// Encodes an image described by `info`, filling the pixels with something reproducible.
fn encoded(info: &Info) -> Vec<u8> {
    let data: Vec<u8> = (0..info.output_size()).map(|i| (i * 31) as u8).collect();
    png_spark::encode(info, &data).unwrap()
}

fn palette_info() -> Info {
    let mut info = Info::new(4, 3, ColorType::Indexed, BitDepth::Eight);
    info.palette = Some((0..=255u8).flat_map(|i| [i, i / 2, 255 - i]).collect());
    info
}

#[test]
fn a_well_formed_file_reads_the_same_header_either_way() {
    for info in [
        Info::new(7, 5, ColorType::Rgba, BitDepth::Eight),
        Info::new(7, 5, ColorType::Grayscale, BitDepth::One),
        Info::new(7, 5, ColorType::Rgb, BitDepth::Sixteen),
        palette_info(),
    ] {
        let png = encoded(&info);
        assert_eq!(png_spark::read_info(&png).unwrap(), png_spark::decode(&png).unwrap().info);
    }
}

/// Inserts a chunk immediately before the trailing `IEND`.
fn splice_before_end(png: &[u8], kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut chunk = (body.len() as u32).to_be_bytes().to_vec();
    chunk.extend_from_slice(kind);
    chunk.extend_from_slice(body);
    let crc = png_spark::crc32::crc32(&chunk[4..]);
    chunk.extend_from_slice(&crc.to_be_bytes());

    let split = png.len() - 12;
    let mut spliced = png[..split].to_vec();
    spliced.extend_from_slice(&chunk);
    spliced.extend_from_slice(&png[split..]);
    spliced
}

#[test]
fn the_header_read_is_the_weaker_check_of_the_two() {
    // Reading stops at the first IDAT, so a fault behind it is invisible here and shows up
    // only on decode. The asymmetry runs one way and has to: a file this accepts may still
    // fail to decode, but a file this rejects would never have decoded.
    let png = encoded(&Info::new(4, 4, ColorType::Rgba, BitDepth::Eight));
    assert!(png_spark::decode(&png).is_ok());

    // An unknown critical chunk after the image data.
    let hidden = splice_before_end(&png, b"ZzZz", b"");
    assert!(png_spark::read_info(&hidden).is_ok(), "the header is intact and is read as such");
    assert_eq!(png_spark::decode(&hidden), Err(Error::UnknownCriticalChunk { chunk: *b"ZzZz" }));

    // A truncated trailer after the image data.
    let cut = &png[..png.len() - 4];
    assert!(png_spark::read_info(cut).is_ok());
    assert!(matches!(png_spark::decode(cut), Err(Error::TruncatedChunk)));
}

#[test]
fn transparency_is_read_because_it_precedes_the_image_data() {
    // The reason to parse past IHDR rather than stopping at it: `tRNS` is where a palette
    // or greyscale image keeps its alpha, and a header without it answers `has_alpha`
    // wrongly for exactly the colour types that most need the question asked.
    let mut info = palette_info();
    info.transparency = Some(vec![0, 64, 128]);
    let png = encoded(&info);

    let read = png_spark::read_info(&png).unwrap();
    assert_eq!(read.transparency, info.transparency);
    assert_eq!(read.palette, info.palette);
    assert!(read.has_alpha());
}

#[test]
fn metadata_before_the_image_data_is_kept_when_asked_for() {
    let mut info = Info::new(2, 2, ColorType::Rgb, BitDepth::Eight);
    info.metadata.push(Chunk::new(*b"apPd", b"header side".to_vec()));
    let png = encoded(&info);

    let read = Decoder::new().keep(Keep::All).read_info(&png).unwrap();
    assert_eq!(read.chunk(b"apPd"), Some(&b"header side"[..]));
    assert_eq!(png_spark::read_info(&png).unwrap().metadata, Vec::new());
}

#[test]
fn a_file_with_no_image_data_is_not_a_header_either() {
    let png = encoded(&Info::new(2, 2, ColorType::Rgb, BitDepth::Eight));

    // Signature and IHDR alone: the header parses, but the file promises an image it does
    // not contain, and saying so is more useful than describing pixels that are not there.
    assert!(matches!(png_spark::read_info(&png[..33]), Err(Error::MissingImageData)));
    assert!(matches!(png_spark::read_info(b"not a png"), Err(Error::NotAPng)));
}

#[test]
fn an_indexed_file_with_no_palette_is_refused_at_the_header() {
    let png = encoded(&palette_info());
    // Strip PLTE, which sits between IHDR and IDAT.
    let start = 33;
    let length = u32::from_be_bytes(png[start..start + 4].try_into().unwrap()) as usize;
    assert_eq!(&png[start + 4..start + 8], b"PLTE");
    let mut stripped = png[..start].to_vec();
    stripped.extend_from_slice(&png[start + length + 12..]);

    assert!(matches!(png_spark::read_info(&stripped), Err(Error::MissingPalette)));
}

#[test]
fn has_alpha_follows_the_format_and_not_just_the_colour_type() {
    let opaque = [ColorType::Grayscale, ColorType::Rgb, ColorType::Indexed];
    for color_type in opaque {
        let info = Info::new(1, 1, color_type, BitDepth::Eight);
        assert!(!info.has_alpha(), "{color_type:?} carries no alpha of its own");

        // The same colour type with a `tRNS` chunk does.
        let mut keyed = Info::new(1, 1, color_type, BitDepth::Eight);
        keyed.transparency = Some(vec![0; 2]);
        assert!(keyed.has_alpha(), "{color_type:?} with tRNS is transparent");
    }

    for color_type in [ColorType::GrayscaleAlpha, ColorType::Rgba] {
        assert!(Info::new(1, 1, color_type, BitDepth::Eight).has_alpha());
    }
}
