//! CRC-32/ISO-HDLC, as used by PNG chunk checksums.
//!
//! The portable implementation is slice-by-16, which consumes sixteen input bytes per
//! iteration using sixteen independent table lookups. On aarch64 the dedicated `crc32x`
//! instruction is used instead when the CPU reports support for it.

/// Reflected polynomial for CRC-32/ISO-HDLC (`0x04C11DB7` reversed).
const POLY: u32 = 0xEDB8_8320;

/// The 16 lookup tables used by the slice-by-16 algorithm.
///
/// `TABLES[0]` is the classic byte-at-a-time table; `TABLES[n]` gives the contribution of a
/// byte that still has `n` further bytes shifted in after it.
static TABLES: [[u32; 256]; 16] = build_tables();

const fn build_tables() -> [[u32; 256]; 16] {
    let mut tables = [[0u32; 256]; 16];

    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = (crc >> 1) ^ (POLY & (0u32.wrapping_sub(crc & 1)));
            bit += 1;
        }
        tables[0][i] = crc;
        i += 1;
    }

    let mut t = 1;
    while t < 16 {
        let mut i = 0;
        while i < 256 {
            let prev = tables[t - 1][i];
            tables[t][i] = (prev >> 8) ^ tables[0][(prev & 0xff) as usize];
            i += 1;
        }
        t += 1;
    }

    tables
}

/// Incremental CRC-32 hasher.
#[derive(Clone, Debug)]
pub struct Crc32 {
    state: u32,
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

impl Crc32 {
    /// A checksum over no bytes.
    #[inline]
    pub const fn new() -> Self {
        Self { state: !0 }
    }

    /// Folds `data` into the running checksum. Any split into calls gives the same result.
    #[inline]
    pub fn update(&mut self, data: &[u8]) {
        self.state = update(self.state, data);
    }

    /// The checksum of everything fed in so far.
    #[inline]
    pub const fn finish(&self) -> u32 {
        !self.state
    }
}

/// Computes the CRC-32 of `data` in one shot.
#[inline]
pub fn crc32(data: &[u8]) -> u32 {
    !update(!0, data)
}

#[inline]
fn update(state: u32, data: &[u8]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        if crc_hw::available() {
            // SAFETY: guarded by a runtime check for the `crc` feature.
            return unsafe { crc_hw::update(state, data) };
        }
    }
    update_portable(state, data)
}

fn update_portable(mut state: u32, data: &[u8]) -> u32 {
    let (chunks, remainder) = data.as_chunks::<16>();
    for chunk in chunks {
        // The first four bytes are mixed into the running state; the remaining twelve are
        // looked up directly since nothing precedes them.
        let a = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) ^ state;
        let b = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        let c = u32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
        let d = u32::from_le_bytes([chunk[12], chunk[13], chunk[14], chunk[15]]);

        state = TABLES[15][(a & 0xff) as usize]
            ^ TABLES[14][((a >> 8) & 0xff) as usize]
            ^ TABLES[13][((a >> 16) & 0xff) as usize]
            ^ TABLES[12][(a >> 24) as usize]
            ^ TABLES[11][(b & 0xff) as usize]
            ^ TABLES[10][((b >> 8) & 0xff) as usize]
            ^ TABLES[9][((b >> 16) & 0xff) as usize]
            ^ TABLES[8][(b >> 24) as usize]
            ^ TABLES[7][(c & 0xff) as usize]
            ^ TABLES[6][((c >> 8) & 0xff) as usize]
            ^ TABLES[5][((c >> 16) & 0xff) as usize]
            ^ TABLES[4][(c >> 24) as usize]
            ^ TABLES[3][(d & 0xff) as usize]
            ^ TABLES[2][((d >> 8) & 0xff) as usize]
            ^ TABLES[1][((d >> 16) & 0xff) as usize]
            ^ TABLES[0][(d >> 24) as usize];
    }

    for &byte in remainder {
        state = (state >> 8) ^ TABLES[0][((state ^ byte as u32) & 0xff) as usize];
    }

    state
}

#[cfg(target_arch = "aarch64")]
mod crc_hw {
    use core::arch::aarch64::{__crc32b, __crc32d};
    use core::sync::atomic::{AtomicU8, Ordering};

    const UNKNOWN: u8 = 0;
    const YES: u8 = 1;
    const NO: u8 = 2;

    static SUPPORT: AtomicU8 = AtomicU8::new(UNKNOWN);

    pub fn available() -> bool {
        match SUPPORT.load(Ordering::Relaxed) {
            YES => true,
            NO => false,
            _ => {
                let detected = std::arch::is_aarch64_feature_detected!("crc");
                SUPPORT.store(if detected { YES } else { NO }, Ordering::Relaxed);
                detected
            }
        }
    }

    /// # Safety
    /// The `crc` target feature must be available on the current CPU.
    #[target_feature(enable = "crc")]
    pub unsafe fn update(mut state: u32, data: &[u8]) -> u32 {
        let (chunks, remainder) = data.as_chunks::<8>();
        for chunk in chunks {
            state = __crc32d(state, u64::from_le_bytes(*chunk));
        }
        for &byte in remainder {
            state = __crc32b(state, byte);
        }
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vectors() {
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"a"), 0xe8b7_be43);
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
        assert_eq!(crc32(b"IEND"), 0xae42_6082);
    }

    #[test]
    fn matches_bytewise_reference() {
        let data: Vec<u8> =
            (0..1024u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();

        let mut reference: u32 = !0;
        for &byte in &data {
            reference = (reference >> 8) ^ TABLES[0][((reference ^ byte as u32) & 0xff) as usize];
        }
        assert_eq!(crc32(&data), !reference);

        // Splitting the input must not change the result.
        for split in [0, 1, 7, 8, 15, 16, 17, 511, 1023, 1024] {
            let mut hasher = Crc32::new();
            hasher.update(&data[..split]);
            hasher.update(&data[split..]);
            assert_eq!(hasher.finish(), crc32(&data), "split at {split}");
        }
    }
}
