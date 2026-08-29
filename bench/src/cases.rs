//! Loads the benchmark corpus from `tmp/bench` and derives the intermediate forms the
//! individual benchmarks need.

pub struct TestImage {
    pub name: String,
    /// The original PNG file.
    pub png: Vec<u8>,
    /// Decoded pixels in the image's native format.
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub color_type: png_spark::common::ColorType,
    pub bit_depth: png_spark::common::BitDepth,
    /// Filter bytes plus filtered scanlines, exactly as the file's `IDAT` decompresses to.
    pub raw_stream: Vec<u8>,
    /// The file's `IDAT` payload, concatenated.
    pub zlib_stream: Vec<u8>,
}

pub fn load_images() -> Vec<TestImage> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tmp/bench");
    if !dir.exists() {
        eprintln!("run tools/gen_bench_images.py first ({} missing)", dir.display());
        std::process::exit(1);
    }

    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("png"))
        .collect();
    paths.sort();

    paths
        .into_iter()
        .map(|path| {
            let png = std::fs::read(&path).unwrap();
            let image = png_spark::decoder::decode(&png).unwrap();
            let zlib_stream = extract_idat(&png);
            let raw_stream =
                fdeflate::decompress_to_vec(&zlib_stream).expect("reference stream decodes");

            TestImage {
                name: path.file_stem().unwrap().to_string_lossy().into_owned(),
                png,
                pixels: image.data,
                width: image.info.width,
                height: image.info.height,
                color_type: image.info.color_type,
                bit_depth: image.info.bit_depth,
                raw_stream,
                zlib_stream,
            }
        })
        .collect()
}

fn extract_idat(png: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut pos = 8;
    while pos + 8 <= png.len() {
        let length = u32::from_be_bytes(png[pos..pos + 4].try_into().unwrap()) as usize;
        let kind = &png[pos + 4..pos + 8];
        if kind == b"IDAT" {
            out.extend_from_slice(&png[pos + 8..pos + 8 + length]);
        }
        if kind == b"IEND" {
            break;
        }
        pos += 12 + length;
    }
    out
}
