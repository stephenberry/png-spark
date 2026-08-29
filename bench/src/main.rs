//! Benchmarks png-spark against the `png` + `fdeflate` stack it aims to replace.
//!
//! Usage: `cargo run --release -p png-spark-bench -- <mode>`
//!
//! Modes:
//!   `all` (default)  decode, encode, inflate, deflate and unfilter
//!   `decode`         whole-PNG decode against the `png` crate
//!   `encode`         whole-PNG encode against the `png` crate
//!   `inflate`        zlib decompression against `fdeflate`
//!   `deflate`        zlib compression against `fdeflate`
//!   `unfilter`       scanline reconstruction on its own
//!   `baseline`       what the reference compressors achieve, for context
//!   `sizes`          output size per filter strategy, for tracking size regressions
//!   `dump`           write each image's filtered stream to `tmp/filtered`
//!   `files [filter]` compress the dumped streams, optionally matching a name
//!   `corpus [dir] [speed]`
//!                    encode and decode a tree of real PNGs, reported per directory;
//!                    `speed` picks what the `png` crate is asked for, default `balanced`
//!
//! Every mode but `corpus` runs on the synthetic set from `tools/gen_bench_images.py`.
//! `corpus` runs on real files fetched by `tools/fetch_corpus.py`.

use std::time::{Duration, Instant};

mod cases;
mod corpus;
use cases::{load_images, TestImage};

/// Runs `body` enough times to get a stable figure and returns the best wall time.
///
/// The minimum is the right statistic here: every source of noise on a shared machine adds
/// time, so the fastest observed run is the closest estimate of the work actually done.
fn measure(mut body: impl FnMut() -> usize) -> (Duration, usize) {
    // One warm-up pass to fault in pages and settle the branch predictors.
    let mut output = body();
    let mut best = Duration::MAX;

    let deadline = Instant::now() + Duration::from_millis(600);
    let mut runs = 0;
    while runs < 3 || (Instant::now() < deadline && runs < 200) {
        let start = Instant::now();
        output = body();
        best = best.min(start.elapsed());
        runs += 1;
    }
    (best, output)
}

fn throughput(bytes: usize, time: Duration) -> f64 {
    bytes as f64 / time.as_secs_f64() / 1e6
}

struct Row {
    label: String,
    ours: Duration,
    theirs: Duration,
    bytes: usize,
    our_size: Option<usize>,
    their_size: Option<usize>,
}

fn report(title: &str, rows: &[Row]) {
    println!("\n=== {title} ===");
    println!(
        "{:<22} {:>11} {:>11} {:>8}   {:>11} {:>11} {:>8}",
        "case", "png-spark", "reference", "speedup", "png-spark", "reference", "ratio"
    );
    let mut total_ours = Duration::ZERO;
    let mut total_theirs = Duration::ZERO;
    let (mut total_our_size, mut total_their_size) = (0usize, 0usize);

    for row in rows {
        let speedup = row.theirs.as_secs_f64() / row.ours.as_secs_f64();
        let sizes = match (row.our_size, row.their_size) {
            (Some(a), Some(b)) => format!(
                "{:>11} {:>11} {:>7.3}x",
                a,
                b,
                a as f64 / b as f64
            ),
            _ => format!("{:>11} {:>11} {:>8}", "-", "-", "-"),
        };
        println!(
            "{:<22} {:>8.1} MB/s {:>8.1} MB/s {:>7.2}x   {sizes}",
            row.label,
            throughput(row.bytes, row.ours),
            throughput(row.bytes, row.theirs),
            speedup,
        );
        total_ours += row.ours;
        total_theirs += row.theirs;
        total_our_size += row.our_size.unwrap_or(0);
        total_their_size += row.their_size.unwrap_or(0);
    }

    println!(
        "{:<22} {:>36.2}x total",
        "OVERALL",
        total_theirs.as_secs_f64() / total_ours.as_secs_f64()
    );
    if total_their_size > 0 {
        println!(
            "{:<22} {:>47.3}x size",
            "",
            total_our_size as f64 / total_their_size as f64
        );
    }
}

fn bench_inflate(images: &[TestImage]) {
    let mut rows = Vec::new();
    for image in images {
        // Compare on the zlib stream the reference encoder produced for this image.
        let stream = &image.zlib_stream;
        let expected = image.raw_stream.len();

        let (ours, _) = measure(|| {
            png_spark::inflate::decompress_zlib(stream, expected).unwrap().len()
        });
        let (theirs, _) = measure(|| {
            fdeflate::decompress_to_vec(stream).unwrap().len()
        });

        // Split out the two halves of the work so a regression can be attributed.
        let mut buffer = vec![0u8; expected + png_spark::inflate::OUTPUT_SLACK];
        let mut plain = png_spark::inflate::Inflater::new();
        plain.verify_checksum(false);
        let (no_checksum, _) = measure(|| plain.zlib(stream, &mut buffer).unwrap());
        let (checksum_only, _) = measure(|| {
            std::hint::black_box(png_spark::adler32::adler32(std::hint::black_box(
                &buffer[..expected],
            ))) as usize
        });
        println!(
            "    {:<18} decode {:>8.1} MB/s   adler32 {:>8.1} MB/s",
            image.name,
            throughput(expected, no_checksum),
            throughput(expected, checksum_only),
        );

        rows.push(Row {
            label: image.name.clone(),
            ours,
            theirs,
            bytes: expected,
            our_size: None,
            their_size: None,
        });
    }
    report("inflate (fdeflate)", &rows);
}

fn bench_decode(images: &[TestImage]) {
    let mut rows = Vec::new();
    for image in images {
        let png = &image.png;
        let mut decoder = png_spark::decoder::Decoder::new();
        let (ours, size) = measure(|| decoder.decode(png).unwrap().data.len());

        let (theirs, other) = measure(|| {
            let decoder = png::Decoder::new(std::io::Cursor::new(png));
            let mut reader = decoder.read_info().unwrap();
            let mut buffer = vec![0; reader.output_buffer_size().unwrap()];
            let info = reader.next_frame(&mut buffer).unwrap();
            info.buffer_size()
        });
        assert_eq!(size, other, "{} decoded sizes differ", image.name);

        rows.push(Row {
            label: image.name.clone(),
            ours,
            theirs,
            bytes: size,
            our_size: None,
            their_size: None,
        });
    }
    report("decode (png crate)", &rows);
}

/// Times unfiltering on its own, so decode regressions can be attributed to the right stage.
fn bench_unfilter(images: &[TestImage]) {
    let mut rows = Vec::new();
    for image in images {
        let info = png_spark::common::Info::new(
            image.width,
            image.height,
            image.color_type,
            image.bit_depth,
        );
        let row_bytes = info.row_bytes();
        let height = image.height as usize;
        let stride = info.filter_stride();

        let mut scratch = vec![0u8; image.raw_stream.len() + 16];
        let (ours, _) = measure(|| {
            scratch[..image.raw_stream.len()].copy_from_slice(&image.raw_stream);
            png_spark::filter::unfilter_image(&mut scratch, row_bytes, height, stride).unwrap();
            scratch[0] as usize
        });
        // Subtract the cost of restoring the input, which is not part of unfiltering.
        let (copy_only, _) = measure(|| {
            scratch[..image.raw_stream.len()].copy_from_slice(&image.raw_stream);
            scratch[0] as usize
        });

        rows.push(Row {
            label: image.name.clone(),
            ours: ours.saturating_sub(copy_only).max(std::time::Duration::from_nanos(1)),
            theirs: copy_only,
            bytes: row_bytes * height,
            our_size: None,
            their_size: None,
        });
    }
    report("unfilter (vs a plain copy of the same data)", &rows);
}

/// Measures the reference compressors on the exact byte stream a PNG encoder must compress,
/// establishing the speed and ratio targets.
fn bench_reference_compressors(images: &[TestImage]) {
    println!("\n=== reference compressors (on the filtered IDAT stream) ===");
    println!(
        "{:<18} {:>10} {:>13} {:>10}  {:>13} {:>10}",
        "case", "raw", "fdeflate", "ratio", "png encode", "size"
    );
    for image in images {
        let raw = &image.raw_stream;
        let (fd_time, fd_size) = measure(|| fdeflate::compress_to_vec(raw).len());

        // The whole `png` encode, which is what a user actually pays: adaptive filtering
        // plus fdeflate.
        let pixels = &image.pixels;
        let (color, depth) = png_color(image.color_type, image.bit_depth);
        let (png_time, png_size) = measure(|| {
            let mut out = Vec::new();
            let mut encoder = png::Encoder::new(&mut out, image.width, image.height);
            encoder.set_color(color);
            encoder.set_depth(depth);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(pixels).unwrap();
            drop(writer);
            out.len()
        });

        println!(
            "{:<18} {:>10} {:>8.1} MB/s {:>9.3} {:>8.1} MB/s {:>10}",
            image.name,
            raw.len(),
            throughput(raw.len(), fd_time),
            fd_size as f64 / raw.len() as f64,
            throughput(pixels.len(), png_time),
            png_size,
        );
    }
}

/// Translates png-spark's colour type and bit depth into the `png` crate's equivalents, so
/// both encoders are asked for the same output format.
fn png_color(
    color_type: png_spark::common::ColorType,
    bit_depth: png_spark::common::BitDepth,
) -> (png::ColorType, png::BitDepth) {
    let color = match color_type {
        png_spark::common::ColorType::Grayscale => png::ColorType::Grayscale,
        png_spark::common::ColorType::Rgb => png::ColorType::Rgb,
        png_spark::common::ColorType::Indexed => png::ColorType::Indexed,
        png_spark::common::ColorType::GrayscaleAlpha => png::ColorType::GrayscaleAlpha,
        png_spark::common::ColorType::Rgba => png::ColorType::Rgba,
    };
    let depth = match bit_depth {
        png_spark::common::BitDepth::One => png::BitDepth::One,
        png_spark::common::BitDepth::Two => png::BitDepth::Two,
        png_spark::common::BitDepth::Four => png::BitDepth::Four,
        png_spark::common::BitDepth::Eight => png::BitDepth::Eight,
        png_spark::common::BitDepth::Sixteen => png::BitDepth::Sixteen,
    };
    (color, depth)
}

fn bench_deflate(images: &[TestImage]) {
    let mut rows = Vec::new();
    for image in images {
        let raw = &image.raw_stream;
        let mut deflater = png_spark::deflate::Deflater::new();
        let mut buffer = Vec::with_capacity(raw.len());
        let (ours, our_size) = measure(|| {
            buffer.clear();
            deflater.zlib(raw, &mut buffer);
            buffer.len()
        });
        let (theirs, their_size) = measure(|| fdeflate::compress_to_vec(raw).len());

        rows.push(Row {
            label: image.name.clone(),
            ours,
            theirs,
            bytes: raw.len(),
            our_size: Some(our_size),
            their_size: Some(their_size),
        });
    }
    report("deflate (vs fdeflate)", &rows);
}

fn bench_encode(images: &[TestImage]) {
    let mut rows = Vec::new();
    for image in images {
        let info = png_spark::common::Info::new(
            image.width,
            image.height,
            image.color_type,
            image.bit_depth,
        );
        let pixels = &image.pixels;

        let mut encoder = png_spark::encoder::Encoder::new();
        let mut buffer = Vec::new();
        let (ours, our_size) = measure(|| {
            buffer.clear();
            encoder.encode(&info, pixels, &mut buffer).unwrap();
            buffer.len()
        });

        let (color, depth) = png_color(image.color_type, image.bit_depth);
        let (theirs, their_size) = measure(|| {
            let mut out = Vec::new();
            let mut encoder = png::Encoder::new(&mut out, image.width, image.height);
            encoder.set_color(color);
            encoder.set_depth(depth);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(pixels).unwrap();
            drop(writer);
            out.len()
        });

        rows.push(Row {
            label: image.name.clone(),
            ours,
            theirs,
            bytes: pixels.len(),
            our_size: Some(our_size),
            their_size: Some(their_size),
        });
    }
    report("encode (vs png crate)", &rows);
}

/// Prints the size each filter strategy reaches, to show where a size regression comes from.
fn bench_sizes(images: &[TestImage]) {
    use png_spark::encoder::{Encoder, FilterStrategy};
    use png_spark::filter::Filter;

    println!("\n=== encoded size by strategy ===");
    let strategies: Vec<(String, FilterStrategy)> = vec![
        ("none".into(), FilterStrategy::Fixed(Filter::None)),
        ("sub".into(), FilterStrategy::Fixed(Filter::Sub)),
        ("up".into(), FilterStrategy::Fixed(Filter::Up)),
        ("paeth".into(), FilterStrategy::Fixed(Filter::Paeth)),
        ("sampled".into(), FilterStrategy::Sampled),
        ("adaptive".into(), FilterStrategy::Adaptive),
    ];
    print!("{:<16}", "case");
    for (name, _) in &strategies {
        print!("{name:>10}");
    }
    println!("{:>10}", "png");
    for image in images {
        let info = png_spark::common::Info::new(
            image.width,
            image.height,
            image.color_type,
            image.bit_depth,
        );
        print!("{:<16}", image.name);
        let mut encoder = Encoder::new();
        for (_, strategy) in &strategies {
            let mut out = Vec::new();
            encoder.filter(*strategy);
            encoder.encode(&info, &image.pixels, &mut out).unwrap();
            print!("{:>10}", out.len());
        }

        let (color, depth) = png_color(image.color_type, image.bit_depth);
        let mut reference = Vec::new();
        let mut png_encoder = png::Encoder::new(&mut reference, image.width, image.height);
        png_encoder.set_color(color);
        png_encoder.set_depth(depth);
        let mut writer = png_encoder.write_header().unwrap();
        writer.write_image_data(&image.pixels).unwrap();
        drop(writer);
        println!("{:>10}", reference.len());
    }
}

/// Writes the filtered scanline stream for each image so an external compressor can be run
/// against exactly the same input.
fn dump_filtered(images: &[TestImage]) {
    use png_spark::common::Info;
    use png_spark::filter::{filter_row, Filter};

    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tmp/filtered");
    std::fs::create_dir_all(&dir).unwrap();

    for image in images {
        let info = Info::new(image.width, image.height, image.color_type, image.bit_depth);
        let row_bytes = info.row_bytes();
        let height = image.height as usize;
        let zero = vec![0u8; row_bytes];

        for (name, filter) in [
            ("none", Filter::None),
            ("sub", Filter::Sub),
            ("up", Filter::Up),
            ("paeth", Filter::Paeth),
        ] {
            let mut stream = vec![0u8; height * (1 + row_bytes)];
            for row in 0..height {
                let current = &image.pixels[row * row_bytes..(row + 1) * row_bytes];
                let previous = if row == 0 {
                    &zero[..]
                } else {
                    &image.pixels[(row - 1) * row_bytes..row * row_bytes]
                };
                let base = row * (1 + row_bytes);
                stream[base] = filter as u8;
                let out = &mut stream[base + 1..base + 1 + row_bytes];
                match info.filter_stride() {
                    1 => filter_row::<1>(filter, previous, current, out),
                    2 => filter_row::<2>(filter, previous, current, out),
                    3 => filter_row::<3>(filter, previous, current, out),
                    4 => filter_row::<4>(filter, previous, current, out),
                    6 => filter_row::<6>(filter, previous, current, out),
                    _ => filter_row::<8>(filter, previous, current, out),
                }
            }
            std::fs::write(dir.join(format!("{}_{name}.bin", image.name)), &stream).unwrap();
        }
    }
    println!("wrote filtered streams to {}", dir.display());
}

/// Compresses each dumped filtered stream on its own, so behaviour on one kind of data can
/// be looked at in isolation. Run `bench dump` first.
fn bench_files(filter: Option<&str>) {
    use png_spark::deflate::Deflater;

    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tmp/filtered");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .expect("run `bench dump` first")
        .map(|e| e.unwrap().path())
        .collect();
    paths.sort();

    println!("\n=== per-file compression ===");
    println!(
        "{:<26} {:>9} {:>10} {:>7}",
        "file", "raw", "compressed", "MB/s"
    );
    for path in paths {
        let data = std::fs::read(&path).unwrap();
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        if let Some(want) = filter {
            if !name.contains(want) {
                continue;
            }
        }
        let mut line = format!("{:<26} {:>9}", name, data.len());
        let mut deflater = Deflater::new();
        let mut out = Vec::new();
        let (time, size) = measure(|| {
            out.clear();
            deflater.zlib(&data, &mut out);
            out.len()
        });
        line += &format!(" {:>10} {:>6.0}", size, throughput(data.len(), time));
        println!("{line}");
    }
}

fn main() {
    let what = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    // These two read their own inputs, so neither needs the synthetic image set loaded.
    if what == "files" {
        bench_files(std::env::args().nth(2).as_deref());
        return;
    }
    if what == "corpus" {
        corpus::run(
            std::env::args().nth(2).as_deref(),
            std::env::args().nth(3).as_deref(),
        );
        return;
    }
    let images = load_images();
    println!("{} images loaded", images.len());

    if what == "all" || what == "inflate" {
        bench_inflate(&images);
    }
    if what == "all" || what == "decode" {
        bench_decode(&images);
    }
    if what == "files" {
        bench_files(std::env::args().nth(2).as_deref());
        return;
    }
    if what == "dump" {
        dump_filtered(&images);
    }
    if what == "sizes" {
        bench_sizes(&images);
    }
    if what == "all" || what == "encode" {
        bench_encode(&images);
    }
    if what == "all" || what == "unfilter" {
        bench_unfilter(&images);
    }
    if what == "all" || what == "baseline" {
        bench_reference_compressors(&images);
    }
    if what == "all" || what == "deflate" {
        bench_deflate(&images);
    }
}
