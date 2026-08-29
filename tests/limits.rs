//! The decoder must not size an allocation on a number it read out of an untrusted file.
//!
//! `IHDR` is thirteen bytes and states the image's dimensions, and the buffer they imply is
//! allocated before a single compressed byte has been read. Nothing in the rest of the file
//! has to corroborate them, so the header alone decides how much memory a decode asks for.

use png_spark::{BitDepth, ColorType, Decoder, Error, Info};

fn valid_png() -> Vec<u8> {
    let info = Info::new(23, 17, ColorType::Rgba, BitDepth::Eight);
    let data: Vec<u8> = (0..info.output_size()).map(|i| (i * 7) as u8).collect();
    png_spark::encode(&info, &data).unwrap()
}

/// The same file with `IHDR` rewritten to claim `width` by `height`, checksum repaired.
///
/// The image data is left alone: this is exactly the file an attacker can make from a real
/// PNG by editing eight bytes and four more.
fn with_dimensions(width: u32, height: u32) -> Vec<u8> {
    let mut png = valid_png();
    png[16..20].copy_from_slice(&width.to_be_bytes());
    png[20..24].copy_from_slice(&height.to_be_bytes());
    let crc = png_spark::crc32::crc32(&png[12..29]);
    png[29..33].copy_from_slice(&crc.to_be_bytes());
    png
}

#[test]
fn a_header_claiming_more_than_the_limit_is_refused() {
    // 65535 by 65535 RGBA8 is under the addressable ceiling `Info::validate` enforces, so
    // nothing but the limit stands between this file and a 17 GB allocation.
    let png = with_dimensions(65535, 65535);

    match png_spark::decode(&png) {
        Err(Error::SizeLimitExceeded { size, limit }) => {
            assert!(size > limit, "the reported size must be what exceeded the limit");
            assert_eq!(limit, png_spark::decoder::DEFAULT_MAX_DECOMPRESSED_SIZE);
        }
        other => panic!("expected the size limit to refuse this file, got {other:?}"),
    }
}

#[test]
fn the_refusal_costs_nothing_and_does_not_depend_on_the_rest_of_the_file() {
    // Truncated to the signature and `IHDR`: the header is refused before the decoder has
    // any reason to look further, so there is nothing else for it to have read.
    let png = with_dimensions(65535, 65535);
    assert!(matches!(png_spark::decode(&png[..33]), Err(Error::SizeLimitExceeded { .. })));
}

#[test]
fn a_real_image_decodes_under_the_default_limit() {
    assert!(png_spark::decode(&valid_png()).is_ok());
}

#[test]
fn the_limit_can_be_tightened_and_loosened() {
    let png = valid_png();
    let size = Decoder::new().read_info(&png).unwrap().decompressed_size();

    // Exactly the size the header describes is allowed; one byte less is not.
    assert!(Decoder::new().max_decompressed_size(Some(size)).decode(&png).is_ok());
    assert!(matches!(
        Decoder::new().max_decompressed_size(Some(size - 1)).decode(&png),
        Err(Error::SizeLimitExceeded { .. })
    ));

    // And it can be given up altogether.
    assert!(Decoder::new().max_decompressed_size(None).decode(&png).is_ok());
}

#[test]
fn an_interlaced_image_is_bounded_too() {
    // Adam7 sizes its passes separately and unfilters them into a second buffer, so it is
    // the path where the limit and the allocation are least obviously the same number.
    let mut info = Info::new(40, 30, ColorType::Rgba, BitDepth::Eight);
    info.interlacing = png_spark::Interlacing::Adam7;
    let data: Vec<u8> = (0..info.output_size()).map(|i| (i * 11) as u8).collect();
    let png = png_spark::encode(&info, &data).unwrap();

    let size = png_spark::read_info(&png).unwrap().decompressed_size();
    assert!(size > info.output_size(), "the passes carry their own filter bytes");
    assert_eq!(Decoder::new().max_decompressed_size(Some(size)).decode(&png).unwrap().data, data);
    assert!(matches!(
        Decoder::new().max_decompressed_size(Some(size - 1)).decode(&png),
        Err(Error::SizeLimitExceeded { .. })
    ));
}

#[test]
fn a_hostile_header_still_parses_as_a_header() {
    // The point of the seam: reading the header allocates nothing that depends on the
    // dimensions, so a caller can apply a policy of its own to a file the decoder would
    // refuse, without the decoder having to allocate first to find that out.
    let info = png_spark::read_info(&with_dimensions(65535, 65535)).unwrap();
    assert_eq!((info.width, info.height), (65535, 65535));
    assert!(info.decompressed_size() > png_spark::decoder::DEFAULT_MAX_DECOMPRESSED_SIZE);
}
