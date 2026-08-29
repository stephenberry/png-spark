//! Adler-32, as used by the zlib wrapper around the DEFLATE stream inside `IDAT`.
//!
//! Every byte of a PNG passes through this checksum twice over an encode/decode round trip,
//! so it is worth real attention: a naive implementation is slower than the DEFLATE decoder
//! it accompanies.
//!
//! All the implementations here share one reformulation. Over a run of `n` bytes,
//!
//! ```text
//! a' = a + sum(x[j])
//! b' = b + n * a + sum((n - j) * x[j])
//! ```
//!
//! which replaces the textbook `a += x; b += a;` per-byte recurrence with two independent
//! reductions plus one multiply. The modulo is deferred until the accumulators are close to
//! overflowing.

/// Largest prime below 65536; the modulus for both halves of the sum.
const BASE: u32 = 65521;

/// Number of bytes that can be accumulated before the `b` half risks overflowing a `u32`.
///
/// This is the standard zlib constant: the largest `n` for which
/// `255 * n * (n + 1) / 2 + (n + 1) * (BASE - 1)` still fits in 32 bits.
const NMAX: usize = 5552;

/// The largest multiple of 64 that fits in `NMAX`.
///
/// Using it as the block size keeps every vector block exactly full, so the scalar tail only
/// runs once at the very end of the input rather than once per block.
///
/// Only the NEON path blocks this way; the portable one works in `NMAX` directly.
#[cfg(target_arch = "aarch64")]
const BLOCK: usize = 5504;

/// Incremental Adler-32 hasher.
#[derive(Clone, Copy, Debug)]
pub struct Adler32 {
    a: u32,
    b: u32,
}

impl Default for Adler32 {
    fn default() -> Self {
        Self::new()
    }
}

impl Adler32 {
    /// A checksum over no bytes, which Adler-32 defines as 1 rather than 0.
    #[inline]
    pub const fn new() -> Self {
        Self { a: 1, b: 0 }
    }

    /// The checksum of everything fed in so far.
    #[inline]
    pub const fn finish(&self) -> u32 {
        (self.b << 16) | self.a
    }

    /// Folds `data` into the running checksum. Any split into calls gives the same result.
    #[inline]
    pub fn update(&mut self, data: &[u8]) {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is part of the aarch64 baseline, so no runtime check is needed.
        unsafe {
            aarch64::update_neon(&mut self.a, &mut self.b, data)
        };

        #[cfg(not(target_arch = "aarch64"))]
        update_portable(&mut self.a, &mut self.b, data);
    }
}

/// Scalar implementation, folding sixteen bytes at a time.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn update_portable(a_out: &mut u32, b_out: &mut u32, data: &[u8]) {
    let (mut a, mut b) = (*a_out, *b_out);

    for block in data.chunks(NMAX) {
        let (chunks, remainder) = block.as_chunks::<16>();
        for chunk in chunks {
            b += a * 16;

            let mut sum = 0u32;
            let mut weighted = 0u32;
            for (j, &byte) in chunk.iter().enumerate() {
                sum += byte as u32;
                weighted += (16 - j) as u32 * byte as u32;
            }

            a += sum;
            b += weighted;
        }

        for &byte in remainder {
            a += byte as u32;
            b += a;
        }

        a %= BASE;
        b %= BASE;
    }

    *a_out = a;
    *b_out = b;
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::{BASE, BLOCK};
    use core::arch::aarch64::*;

    /// Descending weights `n .. 1` across a block of `n` bytes, as four 16-byte vectors.
    const WEIGHTS_64: [u8; 64] = {
        let mut weights = [0u8; 64];
        let mut i = 0;
        while i < 64 {
            weights[i] = (64 - i) as u8;
            i += 1;
        }
        weights
    };

    /// Adds the bytes accumulated so far to the running scalar pair and resets the vectors.
    #[inline(always)]
    unsafe fn fold(a: &mut u32, b: &mut u32, tail: &[u8]) {
        for &byte in tail {
            *a += byte as u32;
            *b += *a;
        }
        *a %= BASE;
        *b %= BASE;
    }

    /// 64 bytes per iteration.
    ///
    /// The weighted sum needs a widening multiply and a pairwise accumulate per eight
    /// bytes; splitting those across four independent accumulators keeps every dependency
    /// chain one instruction long, which is what takes this from roughly 8 GB/s to over 30.
    ///
    /// ARMv8.4's `UDOT` would do the same reduction in a third of the instructions, and is
    /// deliberately left alone: the checksum is not where the time goes. It does not run at
    /// all on the default decode path, where the chunk CRC already covers the same bytes,
    /// and it is about 2% of an encode. It would also need Rust 1.98, above this crate's
    /// floor, but that is the lesser reason and the one that will expire.
    ///
    /// # Safety
    /// Requires the `neon` target feature, which is baseline on aarch64.
    pub unsafe fn update_neon(a_out: &mut u32, b_out: &mut u32, data: &[u8]) {
        unsafe {
            let (mut a, mut b) = (*a_out, *b_out);

            for block in data.chunks(BLOCK) {
                let (chunks, remainder) = block.as_chunks::<64>();
                let full = chunks.len() as u32;

                if full > 0 {
                    let weights = [
                        vld1q_u8(WEIGHTS_64.as_ptr()),
                        vld1q_u8(WEIGHTS_64.as_ptr().add(16)),
                        vld1q_u8(WEIGHTS_64.as_ptr().add(32)),
                        vld1q_u8(WEIGHTS_64.as_ptr().add(48)),
                    ];

                    let mut s1 = [vdupq_n_u32(0); 2];
                    let mut s2 = [vdupq_n_u32(0); 4];
                    let mut carry = vdupq_n_u32(0);

                    for chunk in chunks {
                        let v = [
                            vld1q_u8(chunk.as_ptr()),
                            vld1q_u8(chunk.as_ptr().add(16)),
                            vld1q_u8(chunk.as_ptr().add(32)),
                            vld1q_u8(chunk.as_ptr().add(48)),
                        ];

                        let running = vaddq_u32(s1[0], s1[1]);
                        carry = vaddq_u32(carry, vshlq_n_u32(running, 6));

                        // Pairwise widening keeps every partial sum in a lane wide enough for a
                        // whole block; the two accumulators split the dependency chain.
                        s1[0] = vpadalq_u16(s1[0], vaddq_u16(vpaddlq_u8(v[0]), vpaddlq_u8(v[1])));
                        s1[1] = vpadalq_u16(s1[1], vaddq_u16(vpaddlq_u8(v[2]), vpaddlq_u8(v[3])));

                        // Each product is at most 255 * 64 = 16320, which still fits a `u16`.
                        for i in 0..4 {
                            let low = vmull_u8(vget_low_u8(v[i]), vget_low_u8(weights[i]));
                            let high = vmull_u8(vget_high_u8(v[i]), vget_high_u8(weights[i]));
                            s2[i] = vpadalq_u16(s2[i], vaddq_u16(low, high));
                        }
                    }

                    b += a * (full * 64);
                    b += vaddvq_u32(carry);
                    b += vaddvq_u32(vaddq_u32(vaddq_u32(s2[0], s2[1]), vaddq_u32(s2[2], s2[3])));
                    a += vaddvq_u32(vaddq_u32(s1[0], s1[1]));
                }

                fold(&mut a, &mut b, remainder);
            }

            *a_out = a;
            *b_out = b;
        }
    }
}

/// Computes the Adler-32 of `data` in one shot.
#[inline]
pub fn adler32(data: &[u8]) -> u32 {
    let mut hasher = Adler32::new();
    hasher.update(data);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(data: &[u8]) -> u32 {
        let (mut a, mut b) = (1u32, 0u32);
        for &byte in data {
            a = (a + byte as u32) % BASE;
            b = (b + a) % BASE;
        }
        (b << 16) | a
    }

    const LENGTHS: [usize; 20] = [
        0, 1, 5, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 5503, 5504, 5505, 11_007, 11_008, 11_009,
        20_000,
    ];

    #[test]
    fn known_vectors() {
        assert_eq!(adler32(b""), 1);
        assert_eq!(adler32(b"a"), 0x0062_0062);
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }

    fn varied_data(len: usize) -> Vec<u8> {
        (0..len as u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 11) as u8).collect()
    }

    #[test]
    fn matches_reference() {
        let data = varied_data(20_000);
        for len in LENGTHS {
            assert_eq!(adler32(&data[..len]), reference(&data[..len]), "len {len}");
        }
    }

    /// The `b` half is what can overflow; saturating the input exercises that bound.
    #[test]
    fn saturated_input_does_not_overflow() {
        let data = vec![0xffu8; 40_000];
        assert_eq!(adler32(&data), reference(&data));
    }

    /// Every implementation must agree, not just whichever one this CPU selects.
    #[test]
    fn all_implementations_agree() {
        let data = varied_data(20_000);
        for len in LENGTHS {
            let slice = &data[..len];
            let expected = reference(slice);

            let (mut a, mut b) = (1u32, 0u32);
            update_portable(&mut a, &mut b, slice);
            assert_eq!((b << 16) | a, expected, "portable, len {len}");

            #[cfg(target_arch = "aarch64")]
            {
                let (mut a, mut b) = (1u32, 0u32);
                unsafe { aarch64::update_neon(&mut a, &mut b, slice) };
                assert_eq!((b << 16) | a, expected, "neon, len {len}");
            }
        }
    }

    #[test]
    fn incremental_matches_one_shot() {
        let data: Vec<u8> = (0..9000u32).map(|i| (i % 251) as u8).collect();
        for split in [0, 1, 17, 64, 65, 5504, 6000, 9000] {
            let mut hasher = Adler32::new();
            hasher.update(&data[..split]);
            hasher.update(&data[split..]);
            assert_eq!(hasher.finish(), reference(&data), "split {split}");
        }
    }
}
