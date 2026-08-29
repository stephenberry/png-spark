//! Decodes a corpus of PNGs covering every colour type, bit depth, filter and interlace
//! mode, and compares the result against the pixel data the generator started from.
//!
//! Run `python3 tools/gen_testdata.py` to produce the corpus.

use std::path::Path;

#[test]
fn decodes_reference_images() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tmp/png");
    if !dir.exists() {
        eprintln!("skipping: {} not generated", dir.display());
        return;
    }

    let mut decoder = png_spark::decoder::Decoder::new();
    let mut checked = 0;
    let mut failures = Vec::new();

    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("png"))
        .collect();
    paths.sort();

    for path in paths {
        let png = std::fs::read(&path).unwrap();
        let expected = std::fs::read(path.with_extension("raw")).unwrap();
        let name = path.file_name().unwrap().to_string_lossy().to_string();

        match decoder.decode(&png) {
            Ok(image) if image.data == expected => {}
            Ok(image) => failures.push(format!(
                "{name}: {} bytes decoded, {} expected, first difference at {:?}",
                image.data.len(),
                expected.len(),
                image.data.iter().zip(&expected).position(|(a, b)| a != b)
            )),
            Err(error) => failures.push(format!("{name}: {error}")),
        }
        checked += 1;
    }

    assert!(checked > 0, "no images found in {}", dir.display());
    assert!(failures.is_empty(), "{} of {checked} failed:\n{}", failures.len(), failures.join("\n"));
    eprintln!("checked {checked} images");
}
