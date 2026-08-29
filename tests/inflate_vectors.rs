//! Round-trips the decompressor against zlib streams produced by a reference implementation.
//!
//! The corpus lives outside the crate (see `tools/gen_testdata.py`); when it is absent the
//! test reports what is missing rather than silently passing.

use std::path::Path;

#[test]
fn decompresses_reference_streams() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tmp/z");
    if !dir.exists() {
        eprintln!("skipping: {} not generated", dir.display());
        return;
    }

    let mut checked = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("z") {
            continue;
        }
        let compressed = std::fs::read(&path).unwrap();
        let expected = std::fs::read(path.with_extension("raw")).unwrap();

        let actual = png_spark::inflate::decompress_zlib(&compressed, expected.len())
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(actual, expected, "{}", path.display());
        checked += 1;
    }
    assert!(checked > 0, "no vectors found in {}", dir.display());
    eprintln!("checked {checked} streams");
}
