//! Conversion from a PNG's native pixel layout to the eight-bit formats callers usually
//! want.
//!
//! Decoding leaves pixels exactly as the file stores them, because that is the only form
//! that is always correct and always free. These conversions are separate so that a caller
//! who already handles the native layout never pays for them.

use crate::common::{BitDepth, ColorType};
use crate::decoder::Image;
use crate::error::Error;

impl Image {
    /// Converts to tightly packed 8-bit RGBA.
    ///
    /// Sub-byte grey levels are scaled to the full range, 16-bit samples are truncated to
    /// their high byte, palette indices are resolved, and `tRNS` becomes real alpha. Fully
    /// opaque images get an alpha of 255.
    pub fn to_rgba8(&self) -> Result<Vec<u8>, Error> {
        let pixels = self.pixel_count();
        let mut output = vec![0u8; pixels * 4];
        self.expand(&mut output, true)?;
        Ok(output)
    }

    /// Converts to tightly packed 8-bit RGB, discarding any alpha.
    pub fn to_rgb8(&self) -> Result<Vec<u8>, Error> {
        let pixels = self.pixel_count();
        let mut output = vec![0u8; pixels * 3];
        self.expand(&mut output, false)?;
        Ok(output)
    }

    fn pixel_count(&self) -> usize {
        self.info.width as usize * self.info.height as usize
    }

    fn expand(&self, output: &mut [u8], with_alpha: bool) -> Result<(), Error> {
        if with_alpha { self.expand_rows::<4>(output) } else { self.expand_rows::<3>(output) }
    }

    fn expand_rows<const CHANNELS: usize>(&self, output: &mut [u8]) -> Result<(), Error> {
        let info = &self.info;
        let width = info.width as usize;
        let height = info.height as usize;
        let row_bytes = info.row_bytes();

        let palette = match info.color_type {
            ColorType::Indexed => Some(info.palette.as_deref().ok_or(Error::MissingPalette)?),
            _ => None,
        };
        let transparency = info.transparency.as_deref();

        // The `tRNS` colour is a property of the image, so it is read once here rather than
        // re-parsed for every pixel that has to be compared against it.
        let grey_key = match transparency {
            Some(t) if t.len() >= 2 => Some(be16(t, 0)),
            _ => None,
        };
        let rgb_key = match transparency {
            Some(t) if t.len() >= 6 => Some([be16(t, 0), be16(t, 2), be16(t, 4)]),
            _ => None,
        };

        for y in 0..height {
            let row = &self.data[y * row_bytes..(y + 1) * row_bytes];
            let target = &mut output[y * width * CHANNELS..(y + 1) * width * CHANNELS];

            // Colour type and bit depth belong to the file, not to the pixel, so they are
            // resolved once per row. Each helper is then straight-line code over a fixed
            // stride, which is what lets the offsets fold and the whole-row cases become a
            // single copy. Deciding per pixel instead cost more than the conversion did.
            match (info.color_type, info.bit_depth) {
                (ColorType::Grayscale, BitDepth::One) => {
                    grey_row::<CHANNELS, 1>(row, target, grey_key)
                }
                (ColorType::Grayscale, BitDepth::Two) => {
                    grey_row::<CHANNELS, 2>(row, target, grey_key)
                }
                (ColorType::Grayscale, BitDepth::Four) => {
                    grey_row::<CHANNELS, 4>(row, target, grey_key)
                }
                (ColorType::Grayscale, BitDepth::Eight) => {
                    grey_row::<CHANNELS, 8>(row, target, grey_key)
                }
                (ColorType::Grayscale, BitDepth::Sixteen) => {
                    grey_row::<CHANNELS, 16>(row, target, grey_key)
                }

                (ColorType::GrayscaleAlpha, BitDepth::Sixteen) => {
                    grey_alpha_row::<CHANNELS, true>(row, target)
                }
                (ColorType::GrayscaleAlpha, _) => grey_alpha_row::<CHANNELS, false>(row, target),

                (ColorType::Rgb, BitDepth::Sixteen) => {
                    rgb_row::<CHANNELS, true>(row, target, rgb_key)
                }
                (ColorType::Rgb, _) => rgb_row::<CHANNELS, false>(row, target, rgb_key),

                (ColorType::Rgba, BitDepth::Sixteen) => rgba_row::<CHANNELS, true>(row, target),
                (ColorType::Rgba, _) => rgba_row::<CHANNELS, false>(row, target),

                (ColorType::Indexed, depth) => {
                    let palette = palette.ok_or(Error::MissingPalette)?;
                    match depth {
                        BitDepth::One => {
                            indexed_row::<CHANNELS, 1>(row, target, palette, transparency)?
                        }
                        BitDepth::Two => {
                            indexed_row::<CHANNELS, 2>(row, target, palette, transparency)?
                        }
                        BitDepth::Four => {
                            indexed_row::<CHANNELS, 4>(row, target, palette, transparency)?
                        }
                        BitDepth::Eight => {
                            indexed_row::<CHANNELS, 8>(row, target, palette, transparency)?
                        }
                        BitDepth::Sixteen => unreachable!("indexed PNGs have depths 1, 2, 4 or 8"),
                    }
                }
            }
        }

        Ok(())
    }
}

/// Grey levels, scaled to the full byte range, with `tRNS` matched against the raw sample.
fn grey_row<const CHANNELS: usize, const BITS: usize>(
    row: &[u8],
    target: &mut [u8],
    key: Option<u16>,
) {
    for (x, pixel) in target.as_chunks_mut::<CHANNELS>().0.iter_mut().enumerate() {
        let raw = sample::<BITS>(row, x);
        let grey = scale_to_byte::<BITS>(raw);
        let alpha = if key == Some(raw) { 0 } else { 255 };
        pixel.copy_from_slice(&[grey, grey, grey, alpha][..CHANNELS]);
    }
}

/// Grey plus alpha. `tRNS` does not apply: the alpha channel is already explicit.
fn grey_alpha_row<const CHANNELS: usize, const WIDE: bool>(row: &[u8], target: &mut [u8]) {
    let (stride, step) = if WIDE { (4, 2) } else { (2, 1) };
    for (x, pixel) in target.as_chunks_mut::<CHANNELS>().0.iter_mut().enumerate() {
        let base = x * stride;
        let (grey, alpha) = (row[base], row[base + step]);
        pixel.copy_from_slice(&[grey, grey, grey, alpha][..CHANNELS]);
    }
}

fn rgb_row<const CHANNELS: usize, const WIDE: bool>(
    row: &[u8],
    target: &mut [u8],
    key: Option<[u16; 3]>,
) {
    // Eight-bit RGB asked for as RGB, with no `tRNS` to test, is the same bytes in the same
    // order.
    if CHANNELS == 3 && !WIDE && key.is_none() {
        target.copy_from_slice(&row[..target.len()]);
        return;
    }

    let (stride, step) = if WIDE { (6, 2) } else { (3, 1) };
    for (x, pixel) in target.as_chunks_mut::<CHANNELS>().0.iter_mut().enumerate() {
        let base = x * stride;
        let (r, g, b) = (row[base], row[base + step], row[base + 2 * step]);
        // `tRNS` names a colour at the file's own depth, so a 16-bit image is compared at
        // 16 bits: two colours can differ there and still truncate to the same byte.
        let alpha = match key {
            Some(key) => {
                let raw = if WIDE {
                    [be16(row, base), be16(row, base + 2), be16(row, base + 4)]
                } else {
                    [r as u16, g as u16, b as u16]
                };
                if raw == key { 0 } else { 255 }
            }
            None => 255,
        };
        pixel.copy_from_slice(&[r, g, b, alpha][..CHANNELS]);
    }
}

fn rgba_row<const CHANNELS: usize, const WIDE: bool>(row: &[u8], target: &mut [u8]) {
    // Eight-bit RGBA asked for as RGBA needs no conversion at all.
    if CHANNELS == 4 && !WIDE {
        target.copy_from_slice(&row[..target.len()]);
        return;
    }

    let (stride, step) = if WIDE { (8, 2) } else { (4, 1) };
    for (x, pixel) in target.as_chunks_mut::<CHANNELS>().0.iter_mut().enumerate() {
        let base = x * stride;
        let rgba = [row[base], row[base + step], row[base + 2 * step], row[base + 3 * step]];
        pixel.copy_from_slice(&rgba[..CHANNELS]);
    }
}

fn indexed_row<const CHANNELS: usize, const BITS: usize>(
    row: &[u8],
    target: &mut [u8],
    palette: &[u8],
    transparency: Option<&[u8]>,
) -> Result<(), Error> {
    for (x, pixel) in target.as_chunks_mut::<CHANNELS>().0.iter_mut().enumerate() {
        let index = sample::<BITS>(row, x) as usize;
        let entry = index * 3;
        let rgb = palette.get(entry..entry + 3).ok_or(Error::PaletteIndexOutOfRange)?;
        // A `tRNS` shorter than the palette leaves the entries past its end opaque.
        let alpha = transparency.and_then(|t| t.get(index).copied()).unwrap_or(255);
        pixel.copy_from_slice(&[rgb[0], rgb[1], rgb[2], alpha][..CHANNELS]);
    }
    Ok(())
}

/// Reads the `x`-th sample of a row stored at `BITS` bits per sample.
#[inline(always)]
fn sample<const BITS: usize>(row: &[u8], x: usize) -> u16 {
    match BITS {
        8 => row[x] as u16,
        16 => be16(row, x * 2),
        _ => {
            let offset = x * BITS;
            let mask = (1u8 << BITS) - 1;
            ((row[offset / 8] >> (8 - BITS - offset % 8)) & mask) as u16
        }
    }
}

/// Scales a sample to the full eight-bit range.
///
/// The multiplier is chosen so the maximum value maps to 255 exactly: a one-bit sample
/// becomes 0 or 255, a four-bit sample repeats its nibble, and so on.
#[inline(always)]
fn scale_to_byte<const BITS: usize>(value: u16) -> u8 {
    match BITS {
        1 => (value as u8) * 255,
        2 => (value as u8) * 85,
        4 => (value as u8) * 17,
        16 => (value >> 8) as u8,
        _ => value as u8,
    }
}

#[inline]
fn be16(data: &[u8], offset: usize) -> u16 {
    u16::from_be_bytes([data[offset], data[offset + 1]])
}
