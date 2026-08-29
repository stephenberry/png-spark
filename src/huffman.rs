//! Construction of canonical Huffman codes for the encoder.
//!
//! DEFLATE describes a code purely by its symbol lengths, so the encoder's job is to turn
//! symbol frequencies into a set of lengths that is optimal, obeys the format's 15-bit
//! ceiling, and forms a complete tree.

/// Reverses the low `count` bits of `value`.
///
/// Huffman codes go into the stream most-significant bit first, while everything else in
/// DEFLATE is least-significant bit first. Reversing once here lets the bit writer stay
/// uniformly little-endian.
#[inline]
pub fn reverse_bits(value: u16, count: u32) -> u16 {
    value.reverse_bits() >> (16 - count)
}

/// Assigns canonical codes to symbols from their lengths, pre-reversed for the bit writer.
///
/// Returns `false` if the lengths do not form a complete tree.
pub fn canonical_codes(lengths: &[u8], codes: &mut [u16]) -> bool {
    let mut count = [0u16; 16];
    for &length in lengths {
        count[length as usize] += 1;
    }
    count[0] = 0;

    let mut next = [0u16; 16];
    let mut code = 0u16;
    let mut total = 0u32;
    for bits in 1..16 {
        code = (code + count[bits - 1]) << 1;
        next[bits] = code;
        total += (count[bits] as u32) << (15 - bits);
    }
    if total != 1 << 15 && total != 0 {
        return false;
    }

    for (symbol, &length) in lengths.iter().enumerate() {
        if length != 0 {
            let assigned = next[length as usize];
            next[length as usize] += 1;
            codes[symbol] = reverse_bits(assigned, length as u32);
        } else {
            codes[symbol] = 0;
        }
    }
    true
}

/// Scratch space for building a code, sized for the largest DEFLATE alphabet.
///
/// Kept in the encoder so that building a code for every block costs no allocation.
pub struct Builder {
    /// Used symbols, sorted by frequency ascending.
    order: [u16; 288],
    /// Frequencies of the tree's leaves followed by its internal nodes.
    weights: [u64; 576],
    /// Index of each node's parent, for walking a leaf back to the root.
    parents: [u16; 576],
}

impl core::fmt::Debug for Builder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("Builder")
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

impl Builder {
    /// A builder with its scratch arrays allocated, ready to fit its first code.
    pub fn new() -> Self {
        Self { order: [0; 288], weights: [0; 576], parents: [0; 576] }
    }

    /// Computes code lengths for `frequencies`, none longer than `max_length`.
    ///
    /// Writes to `lengths`, which must be at least as long as `frequencies`; unused symbols
    /// get length zero. The result always describes a complete tree, so a code is emitted
    /// for at least two symbols even when the data uses fewer.
    // The loops below count in code lengths, which are used arithmetically as well as to
    // index the histogram; iterating the histogram instead would obscure that.
    #[allow(clippy::needless_range_loop)]
    pub fn code_lengths(&mut self, frequencies: &[u32], lengths: &mut [u8], max_length: usize) {
        debug_assert!(frequencies.len() <= 288);
        debug_assert!((1..=15).contains(&max_length));

        let alphabet = frequencies.len();
        lengths[..alphabet].fill(0);

        let mut used = 0usize;
        for (symbol, &frequency) in frequencies.iter().enumerate() {
            if frequency > 0 {
                self.order[used] = symbol as u16;
                used += 1;
            }
        }

        // A complete tree needs at least two codes. Pad with the lowest symbols that are
        // not already in use, which costs one bit of header and keeps every decoder happy:
        // a one-symbol code leaves half the code space unreachable, and some decoders
        // reject that.
        if used < 2 {
            let first = if used == 1 { self.order[0] } else { 0 };
            let second = if first == 0 { 1 } else { 0 };
            debug_assert!(alphabet >= 2);
            lengths[first as usize] = 1;
            lengths[second as usize] = 1;
            return;
        }

        // Sort by frequency so the tree build can take the two smallest in order, and so
        // the final length assignment can hand the shortest codes to the commonest symbols.
        self.order[..used].sort_unstable_by_key(|&symbol| frequencies[symbol as usize]);

        for i in 0..used {
            self.weights[i] = frequencies[self.order[i] as usize] as u64;
        }

        // Classic two-queue Huffman: leaves are already sorted, and merged nodes come out in
        // non-decreasing order, so the two smallest are always at the front of one queue or
        // the other.
        let mut leaf = 0usize;
        let mut node = used;
        let mut next_node = used;
        while (used - leaf) + (next_node - node) > 1 {
            let mut pick = || {
                if leaf < used && (node >= next_node || self.weights[leaf] <= self.weights[node]) {
                    leaf += 1;
                    leaf - 1
                } else {
                    node += 1;
                    node - 1
                }
            };
            let first = pick();
            let second = pick();
            self.weights[next_node] = self.weights[first] + self.weights[second];
            self.parents[first] = next_node as u16;
            self.parents[second] = next_node as u16;
            next_node += 1;
        }
        let root = next_node - 1;

        // Depth of each leaf is its code length before the ceiling is applied.
        let mut histogram = [0u32; 16];
        for i in 0..used {
            let mut depth = 0usize;
            let mut current = i;
            while current != root {
                current = self.parents[current] as usize;
                depth += 1;
            }
            histogram[depth.min(max_length)] += 1;
        }

        // Clamping over-long codes leaves the tree over-subscribed. Repair it by repeatedly
        // dropping one longest code and splitting a shorter one in its place, which is the
        // cheapest single step that restores the Kraft equality.
        let limit = max_length;
        let mut kraft: u64 = (1..=limit).map(|len| (histogram[len] as u64) << (limit - len)).sum();
        let target: u64 = 1 << limit;
        while kraft > target {
            histogram[limit] -= 1;
            let mut shorter = limit - 1;
            while histogram[shorter] == 0 {
                shorter -= 1;
            }
            histogram[shorter] -= 1;
            histogram[shorter + 1] += 2;
            kraft -= 1;
        }

        // Hand the shortest codes to the most frequent symbols.
        let mut index = used;
        for length in 1..=limit {
            for _ in 0..histogram[length] {
                index -= 1;
                lengths[self.order[index] as usize] = length as u8;
            }
        }
        debug_assert_eq!(index, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_complete(lengths: &[u8]) -> bool {
        let total: u64 = lengths.iter().filter(|&&l| l > 0).map(|&l| 1u64 << (15 - l)).sum();
        total == 1 << 15
    }

    fn weighted_bits(frequencies: &[u32], lengths: &[u8]) -> u64 {
        frequencies.iter().zip(lengths).map(|(&f, &l)| f as u64 * l as u64).sum()
    }

    #[test]
    fn produces_complete_trees() {
        let mut builder = Builder::new();
        let mut lengths = [0u8; 288];

        let cases: Vec<Vec<u32>> = vec![
            vec![0; 30],
            {
                let mut f = vec![0u32; 30];
                f[7] = 5;
                f
            },
            {
                let mut f = vec![0u32; 30];
                f[3] = 5;
                f[9] = 1;
                f
            },
            (0..286u32).map(|i| i % 7).collect(),
            (0..286u32).map(|i| if i == 0 { 1_000_000 } else { 1 }).collect(),
            (0..286u32).map(|i| 1u32 << (i % 28)).collect(),
        ];

        for frequencies in cases {
            builder.code_lengths(&frequencies, &mut lengths, 15);
            let used = &lengths[..frequencies.len()];
            assert!(is_complete(used), "incomplete for {frequencies:?}");
            for (symbol, &length) in used.iter().enumerate() {
                assert!(length <= 15);
                if frequencies[symbol] > 0 {
                    assert!(length > 0, "symbol {symbol} used but has no code");
                }
            }
            let mut codes = [0u16; 288];
            assert!(canonical_codes(used, &mut codes));
        }
    }

    /// Highly skewed frequencies naturally produce codes longer than DEFLATE allows; the
    /// repair must bring them under the ceiling without breaking the tree.
    #[test]
    fn respects_the_length_ceiling() {
        let mut builder = Builder::new();
        let mut lengths = [0u8; 288];
        // Fibonacci frequencies are the classic worst case: they force a maximally
        // unbalanced tree, one level per symbol.
        let mut frequencies = vec![0u32; 40];
        let (mut a, mut b) = (1u32, 1u32);
        for slot in frequencies.iter_mut() {
            *slot = a;
            let next = a.saturating_add(b);
            a = b;
            b = next;
        }

        for limit in [7, 15] {
            builder.code_lengths(&frequencies, &mut lengths, limit);
            assert!(lengths[..40].iter().all(|&l| l as usize <= limit));
            assert!(is_complete(&lengths[..40]));
        }
    }

    /// The assignment must be at least as good as giving every symbol the same length.
    #[test]
    fn beats_a_flat_code() {
        let mut builder = Builder::new();
        let mut lengths = [0u8; 288];
        let frequencies: Vec<u32> = (0..256u32).map(|i| 1 + (i % 5) * (i % 13)).collect();
        builder.code_lengths(&frequencies, &mut lengths, 15);

        let flat = vec![8u8; 256];
        assert!(weighted_bits(&frequencies, &lengths[..256]) <= weighted_bits(&frequencies, &flat));
    }
}
