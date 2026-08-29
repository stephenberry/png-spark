//! Arbitrary application data carried inside a PNG, in ancillary chunks.

use png_spark::common::{BitDepth, Chunk, ColorType, Info};
use png_spark::{Checks, Decoder, Error, Keep};

/// A small RGBA image with whatever metadata the caller wants attached.
fn image(metadata: Vec<Chunk>) -> (Info, Vec<u8>) {
    let mut info = Info::new(16, 9, ColorType::Rgba, BitDepth::Eight);
    info.metadata = metadata;
    let data = (0..info.output_size()).map(|i| (i.wrapping_mul(37) >> 2) as u8).collect();
    (info, data)
}

/// Every chunk in a file, as the offset of its type field and its type.
///
/// Walked rather than searched for: a palette or a payload can contain any four bytes it
/// likes, including `IDAT`.
fn chunks(png: &[u8]) -> Vec<(usize, [u8; 4])> {
    let mut found = Vec::new();
    let mut pos = png_spark::common::SIGNATURE.len();
    while pos + 12 <= png.len() {
        let length = u32::from_be_bytes(png[pos..pos + 4].try_into().unwrap()) as usize;
        found.push((pos + 4, png[pos + 4..pos + 8].try_into().unwrap()));
        pos += 12 + length;
    }
    found
}

/// Byte offset of a chunk's type field within an encoded file.
fn offset_of(png: &[u8], kind: &[u8; 4]) -> Option<usize> {
    chunks(png).into_iter().find(|(_, found)| found == kind).map(|(at, _)| at)
}

/// Inserts a chunk, correct CRC and all, immediately before `IDAT`.
fn splice_before_idat(png: &[u8], kind: [u8; 4], body: &[u8]) -> Vec<u8> {
    let at = offset_of(png, b"IDAT").unwrap() - 4;
    let mut chunk = (body.len() as u32).to_be_bytes().to_vec();
    chunk.extend_from_slice(&kind);
    chunk.extend_from_slice(body);
    chunk.extend_from_slice(&png_spark::crc32::crc32(&chunk[4..]).to_be_bytes());
    [&png[..at], &chunk[..], &png[at..]].concat()
}

#[test]
fn metadata_survives_a_round_trip_verbatim() {
    // Every byte value, including the nulls and the PNG signature itself, so that nothing in
    // the payload can be mistaken for structure on the way back.
    let mut payload: Vec<u8> = (0..=255u8).collect();
    payload.extend_from_slice(&png_spark::common::SIGNATURE);

    let (info, data) = image(vec![
        Chunk::new(*b"apPd", payload.clone()),
        Chunk::new(*b"apPd", Vec::new()),
        Chunk::new(*b"bnDl", b"second type".to_vec()),
    ]);
    let png = png_spark::encode(&info, &data).unwrap();

    let decoded = Decoder::new().keep(Keep::All).decode(&png).unwrap();
    assert_eq!(decoded.data, data);
    assert_eq!(decoded.info.metadata, info.metadata);

    // The accessor reaches the first chunk of a type; duplicates stay visible in the list.
    assert_eq!(decoded.info.chunk(b"apPd"), Some(&payload[..]));
    assert_eq!(decoded.info.chunk(b"bnDl"), Some(&b"second type"[..]));
    assert_eq!(decoded.info.chunk(b"noPe"), None);
}

#[test]
fn a_payload_of_the_size_an_asset_blob_actually_is_survives() {
    // Every other case here is a few bytes, which leaves the chunk's four-byte length field
    // exercised only in its bottom byte.
    let payload: Vec<u8> =
        (0..1 << 20).map(|i: usize| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect();
    let (info, data) = image(vec![Chunk::new(*b"apPd", payload.clone())]);
    let png = png_spark::encode(&info, &data).unwrap();

    let decoded = Decoder::new().keep(Keep::All).decode(&png).unwrap();
    assert_eq!(decoded.info.chunk(b"apPd"), Some(&payload[..]));
}

#[test]
fn metadata_is_dropped_unless_it_is_asked_for() {
    let (info, data) = image(vec![Chunk::new(*b"apPd", b"payload".to_vec())]);
    let png = png_spark::encode(&info, &data).unwrap();

    let decoded = png_spark::decode(&png).unwrap();
    assert!(decoded.info.metadata.is_empty());
    assert_eq!(decoded.data, data);
}

#[test]
fn keep_only_retains_the_listed_types() {
    let (info, data) = image(vec![
        Chunk::new(*b"apPd", b"wanted".to_vec()),
        Chunk::new(*b"bnDl", b"unwanted".to_vec()),
        Chunk::new(*b"apPd", b"also wanted".to_vec()),
    ]);
    let png = png_spark::encode(&info, &data).unwrap();

    let decoded = Decoder::new().keep(Keep::Only(vec![*b"apPd"])).decode(&png).unwrap();
    assert_eq!(decoded.info.metadata.len(), 2);
    assert!(decoded.info.metadata.iter().all(|chunk| chunk.kind == *b"apPd"));
    assert_eq!(decoded.info.chunk(b"apPd"), Some(&b"wanted"[..]));
}

#[test]
fn re_encoding_a_decoded_image_preserves_its_metadata() {
    // Load, edit, save: the metadata rides along on `Info`, so it does not have to be
    // carried by hand.
    let (info, data) = image(vec![Chunk::new(*b"apPd", b"asset id 41".to_vec())]);
    let png = png_spark::encode(&info, &data).unwrap();

    let mut decoded = Decoder::new().keep(Keep::All).decode(&png).unwrap();
    decoded.data[0] ^= 0xff;
    let again = png_spark::encode(&decoded.info, &decoded.data).unwrap();

    let round_tripped = Decoder::new().keep(Keep::All).decode(&again).unwrap();
    assert_eq!(round_tripped.info.chunk(b"apPd"), Some(&b"asset id 41"[..]));
    assert_eq!(round_tripped.data, decoded.data);
}

#[test]
fn chunks_are_placed_on_the_side_of_the_palette_the_specification_requires() {
    let mut info = Info::new(8, 8, ColorType::Indexed, BitDepth::Eight);
    info.palette =
        Some((0..256u32).flat_map(|i| [i as u8, (i * 5) as u8, (i * 9) as u8]).collect());
    // Given in the order that is wrong for both: `gAMA` must precede `PLTE`, `bKGD` follow it.
    info.metadata =
        vec![Chunk::new(*b"bKGD", vec![7]), Chunk::new(*b"gAMA", 45455u32.to_be_bytes().to_vec())];
    let data = vec![3u8; info.output_size()];

    let png = png_spark::encode(&info, &data).unwrap();
    let gama = offset_of(&png, b"gAMA").unwrap();
    let plte = offset_of(&png, b"PLTE").unwrap();
    let bkgd = offset_of(&png, b"bKGD").unwrap();
    let idat = offset_of(&png, b"IDAT").unwrap();
    assert!(gama < plte, "gAMA must precede PLTE");
    assert!(plte < bkgd, "bKGD must follow PLTE");
    assert!(bkgd < idat, "metadata must precede IDAT");

    // Reordering for the file does not reorder the list the decoder hands back, which is
    // file order.
    let decoded = Decoder::new().keep(Keep::All).decode(&png).unwrap();
    assert_eq!(decoded.info.metadata[0].kind, *b"gAMA");
    assert_eq!(decoded.info.metadata[1].kind, *b"bKGD");
}

#[test]
fn the_encoder_refuses_chunk_types_it_must_not_write() {
    let refused = [
        // Critical: a decoder that does not know the type has to fail on it, so writing one
        // would produce a file png-spark could not read back.
        *b"ApPd",
        *b"IDAT",
        // The third byte is reserved, and must be upper case.
        *b"appd",
        // Chunk types are four ASCII letters.
        *b"ap0d",
        *b"ap d",
    ];
    for kind in refused {
        let (info, data) = image(vec![Chunk::new(kind, b"payload".to_vec())]);
        assert_eq!(
            png_spark::encode(&info, &data).unwrap_err(),
            Error::InvalidChunkType { chunk: kind },
            "{:?} should be refused",
            core::str::from_utf8(&kind)
        );
    }

    // The fourth byte carries the safe-to-copy bit, and either case is meaningful.
    for kind in [*b"apPd", *b"apPD"] {
        let (info, data) = image(vec![Chunk::new(kind, b"payload".to_vec())]);
        assert!(png_spark::encode(&info, &data).is_ok());
    }
}

#[test]
fn a_corrupt_metadata_chunk_is_dropped_and_the_image_still_decodes() {
    let (info, data) = image(vec![Chunk::new(*b"apPd", b"payload".to_vec())]);
    let mut png = png_spark::encode(&info, &data).unwrap();

    // Corrupt the payload, leaving its length and the CRC alone.
    let start = offset_of(&png, b"apPd").unwrap() + 4;
    png[start] ^= 0x01;

    let decoded = Decoder::new().keep(Keep::All).decode(&png).unwrap();
    assert!(decoded.info.metadata.is_empty(), "a chunk failing its CRC must not be handed back");
    assert_eq!(decoded.data, data);

    // Unchecked, the corrupted bytes come through as they are.
    let unchecked = Decoder::new().checks(Checks::None).keep(Keep::All).decode(&png).unwrap();
    assert_eq!(unchecked.info.chunk(b"apPd"), Some(&b"qayload"[..]));
}

#[test]
fn transparency_may_not_be_smuggled_in_as_metadata() {
    // `tRNS` is the one ancillary chunk png-spark interprets, and the encoder writes it from
    // `Info::transparency`, where its length is checked against the colour type. A second
    // one would be an illegal duplicate, would displace the real transparency on read-back,
    // and with an unsuitable length would make a file png-spark itself rejects.
    let (info, data) = image(vec![Chunk::new(*b"tRNS", vec![0, 1, 2, 3, 4, 5])]);
    assert_eq!(
        png_spark::encode(&info, &data).unwrap_err(),
        Error::InvalidChunkType { chunk: *b"tRNS" }
    );

    // The supported route is unaffected, and the chunk it writes is not echoed back as
    // metadata by a decode that keeps everything.
    let mut info = Info::new(8, 8, ColorType::Rgb, BitDepth::Eight);
    info.transparency = Some(vec![0, 1, 0, 2, 0, 3]);
    let png = png_spark::encode(&info, &vec![0u8; info.output_size()]).unwrap();
    assert_eq!(chunks(&png).iter().filter(|(_, kind)| kind == b"tRNS").count(), 1);

    let decoded = Decoder::new().keep(Keep::All).decode(&png).unwrap();
    assert_eq!(decoded.info.transparency, info.transparency);
    assert!(decoded.info.metadata.is_empty());
}

#[test]
fn a_chunk_the_encoder_could_not_write_is_never_retained() {
    // Anything in `info.metadata` has to survive a trip back to the encoder, so a type the
    // encoder would refuse is dropped on the way in, however broadly chunks were asked for.
    // These are all well formed otherwise, CRC included: only the type is wrong.
    let (info, data) = image(Vec::new());
    let png = png_spark::encode(&info, &data).unwrap();

    for kind in [*b"ab1D", *b"abcd", [b'a', b'p', 0, b'd']] {
        let spliced = splice_before_idat(&png, kind, b"payload");
        let decoded = Decoder::new().keep(Keep::All).decode(&spliced).unwrap();
        assert!(decoded.info.metadata.is_empty(), "{kind:?} should not be retained");
        assert_eq!(decoded.data, data);
        png_spark::encode(&decoded.info, &decoded.data)
            .expect("a decoded image must always re-encode");
    }

    // A well-formed private type spliced into the same position is retained.
    let spliced = splice_before_idat(&png, *b"apPd", b"payload");
    let decoded = Decoder::new().keep(Keep::All).decode(&spliced).unwrap();
    assert_eq!(decoded.info.chunk(b"apPd"), Some(&b"payload"[..]));
}
