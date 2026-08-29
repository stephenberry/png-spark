//! Encodes and decodes a directory tree of real PNG files, reporting per directory.
//!
//! This is the counterpart to the `png` crate's `examples/corpus-bench.rs`, laid out so the
//! two can be read side by side: same normalisation of the input pixels, same per-directory
//! grouping, same ratio and megapixels-per-second figures. Run `tools/fetch_corpus.py`
//! first to populate `tmp/corpus`.
//!
//! Every image is timed once rather than best-of-many. A corpus run covers thousands of
//! files, so repetition would cost hours, and the per-directory aggregate over hundreds of
//! images is already steadier than a best-of-three on any one of them. That makes these
//! numbers the right tool for comparing two libraries on the same run, and the wrong tool
//! for chasing a two-percent regression: use the synthetic modes for that.
//!
//! Correctness is checked as it goes. Every re-encoded image is decoded back and compared
//! against the pixels that went in, and png-spark's output is additionally decoded by the
//! `png` crate, so a corpus run doubles as a round-trip sweep over real files. The
//! verification happens outside the timed regions.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use png_spark::common::{BitDepth, ColorType, Info, Interlacing};

/// One image, normalised to a form both encoders accept, exactly as `corpus-bench` does it.
struct Source {
    info: Info,
    pixels: Vec<u8>,
}

#[derive(Default)]
struct Stats {
    files: usize,
    pixels: u64,
    raw: u64,
    our_size: u64,
    their_size: u64,
    our_encode: Duration,
    our_decode: Duration,
    their_encode: Duration,
    their_decode: Duration,
}

impl Stats {
    fn add(&mut self, other: &Stats) {
        self.files += other.files;
        self.pixels += other.pixels;
        self.raw += other.raw;
        self.our_size += other.our_size;
        self.their_size += other.their_size;
        self.our_encode += other.our_encode;
        self.our_decode += other.our_decode;
        self.their_encode += other.their_encode;
        self.their_decode += other.their_decode;
    }
}

/// Megapixels per second, or zero when nothing was measured.
fn megapixels(pixels: u64, time: Duration) -> f64 {
    if time.is_zero() {
        return 0.0;
    }
    pixels as f64 / time.as_secs_f64() / 1e6
}

fn ratio(compressed: u64, raw: u64) -> f64 {
    if raw == 0 {
        0.0
    } else {
        100.0 * compressed as f64 / raw as f64
    }
}

/// Decodes one file and converts it to the pixel layout the benchmark compares on.
///
/// `corpus-bench` expands indexed images through the palette and, following `qoibench`,
/// widens eight-bit grey to RGBA. Doing the same here keeps the input bytes identical to
/// the ones the reference numbers were measured on. Anything else is left in the file's own
/// format, which is what both encoders would really be handed.
fn load(path: &Path) -> Result<Source, String> {
    let file = std::fs::read(path).map_err(|e| e.to_string())?;
    let image = png_spark::decoder::decode(&file).map_err(|e| e.to_string())?;

    let mut info = Info {
        interlacing: Interlacing::None,
        palette: None,
        transparency: None,
        ..image.info.clone()
    };

    let pixels = match (image.info.color_type, image.info.bit_depth) {
        (ColorType::Indexed, _) => {
            let data = if image.info.transparency.is_some() {
                info.color_type = ColorType::Rgba;
                image.to_rgba8().map_err(|e| e.to_string())?
            } else {
                info.color_type = ColorType::Rgb;
                image.to_rgb8().map_err(|e| e.to_string())?
            };
            info.bit_depth = BitDepth::Eight;
            data
        }
        (ColorType::Grayscale, BitDepth::Eight) => {
            info.color_type = ColorType::Rgba;
            image.data.iter().flat_map(|&v| [v, v, v, 255]).collect()
        }
        (ColorType::GrayscaleAlpha, BitDepth::Eight) => {
            info.color_type = ColorType::Rgba;
            // Not `chunks_exact`: `as_chunks` would need a newer compiler than the library
            // asks for, and clippy prefers it wherever `chunks_exact` appears.
            let mut widened = Vec::with_capacity(image.data.len() * 2);
            for pixel in image.data.windows(2).step_by(2) {
                widened.extend_from_slice(&[pixel[0], pixel[0], pixel[0], pixel[1]]);
            }
            widened
        }
        _ => image.data,
    };

    Ok(Source { info, pixels })
}

fn encode_ours(encoder: &mut png_spark::encoder::Encoder, source: &Source, out: &mut Vec<u8>) {
    out.clear();
    encoder.encode(&source.info, &source.pixels, out).expect("png-spark encodes");
}

fn encode_theirs(source: &Source, speed: png::Compression, out: &mut Vec<u8>) {
    out.clear();
    let (color, depth) = crate::png_color(source.info.color_type, source.info.bit_depth);
    let mut encoder = png::Encoder::new(&mut *out, source.info.width, source.info.height);
    encoder.set_color(color);
    encoder.set_depth(depth);
    encoder.set_compression(speed);
    let mut writer = encoder.write_header().expect("png writes a header");
    writer.write_image_data(&source.pixels).expect("png encodes");
    writer.finish().expect("png finishes");
}

fn decode_theirs(png: &[u8], out: &mut Vec<u8>) {
    let decoder = png::Decoder::new(std::io::Cursor::new(png));
    let mut reader = decoder.read_info().expect("png reads its own header");
    out.resize(reader.output_buffer_size().expect("output size is known"), 0);
    reader.next_frame(out).expect("png decodes its own output");
}

/// Runs both libraries over one image and folds the result into `stats`.
///
/// Returns the reason the file was passed over, if it could not be read at all. A failure
/// after that point is a bug rather than a limitation, so it panics.
fn measure_image(
    path: &Path,
    encoder: &mut png_spark::encoder::Encoder,
    decoder: &mut png_spark::decoder::Decoder,
    speed: png::Compression,
    stats: &mut Stats,
) -> Result<(), String> {
    let source = load(path)?;
    let mut ours = Vec::new();
    let mut theirs = Vec::new();
    let mut back = Vec::new();

    let start = Instant::now();
    encode_ours(encoder, &source, &mut ours);
    let our_encode = start.elapsed();

    let start = Instant::now();
    let decoded = decoder.decode(&ours).expect("png-spark decodes its own output");
    let our_decode = start.elapsed();
    assert_eq!(decoded.data, source.pixels, "png-spark round trip differs: {}", path.display());

    let start = Instant::now();
    encode_theirs(&source, speed, &mut theirs);
    let their_encode = start.elapsed();

    let start = Instant::now();
    decode_theirs(&theirs, &mut back);
    let their_decode = start.elapsed();
    assert_eq!(back, source.pixels, "png round trip differs: {}", path.display());

    // A file only png-spark can read would be a bug the round trip above cannot see.
    decode_theirs(&ours, &mut back);
    assert_eq!(back, source.pixels, "png disagrees with png-spark's output: {}", path.display());

    stats.add(&Stats {
        files: 1,
        pixels: source.info.width as u64 * source.info.height as u64,
        raw: source.pixels.len() as u64,
        our_size: ours.len() as u64,
        their_size: theirs.len() as u64,
        our_encode,
        our_decode,
        their_encode,
        their_decode,
    });
    Ok(())
}

/// Collects `root` and every directory beneath it, deepest last, in a stable order.
fn directories(root: &Path) -> Vec<PathBuf> {
    let mut found = vec![root.to_path_buf()];
    let mut index = 0;
    while index < found.len() {
        let mut children: Vec<PathBuf> = match std::fs::read_dir(&found[index]) {
            Ok(entries) => entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .collect(),
            Err(_) => Vec::new(),
        };
        children.sort();
        found.extend(children);
        index += 1;
    }
    found
}

fn pngs_in(directory: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = match std::fs::read_dir(directory) {
        Ok(entries) => entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("png"))
            .collect(),
        Err(_) => Vec::new(),
    };
    paths.sort();
    paths
}

/// Shortens a directory label to fit the table, keeping the tail that distinguishes it.
fn label(root: &Path, directory: &Path, width: usize) -> String {
    let relative = directory.strip_prefix(root).unwrap_or(directory);
    let text = relative.to_string_lossy();
    let text = if text.is_empty() { ".".into() } else { text };
    if text.chars().count() <= width {
        return text.into_owned();
    }
    let tail: String = text.chars().skip(text.chars().count() - (width - 3)).collect();
    format!("...{tail}")
}

const LABEL_WIDTH: usize = 38;

/// Width of one library's three columns, so the group labels sit over the right ones.
const GROUP_WIDTH: usize = 7 + 1 + 9 + 1 + 9;

fn header(speed: &str) {
    println!(
        "{:<LABEL_WIDTH$} {:>6} | {:<GROUP_WIDTH$} | png 0.18 ({speed})",
        "", "", "png-spark",
    );
    println!(
        "{:<LABEL_WIDTH$} {:>6} | {:>7} {:>9} {:>9} | {:>7} {:>9} {:>9}",
        "directory", "files", "ratio", "enc MP/s", "dec MP/s", "ratio", "enc MP/s", "dec MP/s",
    );
}

fn print_row(name: &str, stats: &Stats) {
    println!(
        "{:<LABEL_WIDTH$} {:>6} | {:>6.2}% {:>9.1} {:>9.1} | {:>6.2}% {:>9.1} {:>9.1}",
        name,
        stats.files,
        ratio(stats.our_size, stats.raw),
        megapixels(stats.pixels, stats.our_encode),
        megapixels(stats.pixels, stats.our_decode),
        ratio(stats.their_size, stats.raw),
        megapixels(stats.pixels, stats.their_encode),
        megapixels(stats.pixels, stats.their_decode),
    );
}

/// The `png` crate's compression setting to measure against.
///
/// Its own default is `Balanced`, which is what a user of that crate gets without asking,
/// so that is the default here too. `fast` selects the fdeflate path, which is the closest
/// thing in that crate to png-spark's single design point and the fairer speed comparison.
fn compression(name: &str) -> Option<png::Compression> {
    Some(match name {
        "none" => png::Compression::NoCompression,
        "fastest" => png::Compression::Fastest,
        "fast" => png::Compression::Fast,
        "balanced" => png::Compression::Balanced,
        "high" => png::Compression::High,
        _ => return None,
    })
}

/// Entry point for `bench corpus [dir] [speed]`.
pub fn run(root: Option<&str>, speed: Option<&str>) {
    let speed_name = speed.unwrap_or("balanced");
    let Some(speed) = compression(speed_name) else {
        eprintln!("unknown speed {speed_name:?}: expected none, fastest, fast, balanced or high");
        std::process::exit(1);
    };
    let root = match root {
        Some(path) => PathBuf::from(path),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../tmp/corpus"),
    };
    // Only for the heading: the default path is written relative to the bench crate.
    let root = std::fs::canonicalize(&root).unwrap_or(root);
    if !root.is_dir() {
        eprintln!(
            "no corpus at {}\nrun: python3 tools/fetch_corpus.py image-png",
            root.display()
        );
        std::process::exit(1);
    }

    println!("\n=== corpus: {} ===", root.display());
    header(speed_name);

    let mut encoder = png_spark::encoder::Encoder::new();
    let mut decoder = png_spark::decoder::Decoder::new();
    let mut total = Stats::default();
    let mut skipped: Vec<(PathBuf, String)> = Vec::new();

    for directory in directories(&root) {
        let paths = pngs_in(&directory);
        if paths.is_empty() {
            continue;
        }
        let mut stats = Stats::default();
        for path in &paths {
            if let Err(reason) = measure_image(path, &mut encoder, &mut decoder, speed, &mut stats)
            {
                skipped.push((path.clone(), reason));
            }
        }
        if stats.files == 0 {
            continue;
        }
        print_row(&label(&root, &directory, LABEL_WIDTH), &stats);
        total.add(&stats);
    }

    if total.files == 0 {
        println!("no decodable PNGs found");
        return;
    }

    println!();
    print_row("TOTAL", &total);
    println!(
        "{:<LABEL_WIDTH$} {:>6} | encode {:.2}x   decode {:.2}x   size {:.3}x",
        "vs png",
        "",
        total.their_encode.as_secs_f64() / total.our_encode.as_secs_f64(),
        total.their_decode.as_secs_f64() / total.our_decode.as_secs_f64(),
        total.our_size as f64 / total.their_size as f64,
    );
    // Named, not counted: a file png-spark turns away is either a format it does not claim to
    // support or a gap in the decoder, and a bare tally hides which.
    if !skipped.is_empty() {
        println!("\n{} skipped:", skipped.len());
        for (path, reason) in &skipped {
            println!("  {}: {reason}", label(&root, path, 60));
        }
    }
}
