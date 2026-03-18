/// Shard dispatch: map a virtual address to a worker index.
///
/// We use Fibonacci hashing (multiplicative) to avoid clustering on
/// naturally-aligned addresses (ARM64 instructions are always 4-byte
/// aligned, so `addr % N` would cluster badly for small N).
///
/// The same address always maps to the same worker — this gives
/// ownership semantics: each worker owns its addresses, no locking
/// on the hot path.
#[inline(always)]
#[cfg(test)]
pub fn worker_for(addr: u64, n_workers: usize) -> usize {
    // Knuth multiplicative hash — good distribution for aligned addrs
    let h = addr.wrapping_mul(0x9e3779b97f4a7c15u64);
    (h >> 32) as usize % n_workers
}

/// Split an address range into N shards, each assigned to a worker.
/// Returns (start_va, end_va) pairs — exclusive end.
///
/// `boundary_align`: alignment in bytes for shard start addresses.
/// For code shards this is the instruction alignment (ARM64=4, x86=1) so
/// that fixed-width decoders always begin on a valid instruction boundary.
/// For data shards this is the pointer size so that byte-scan steps stay
/// on pointer-aligned boundaries.
///
/// We add `overlap` bytes of lookahead to each shard so that
/// cross-boundary instruction pairs (e.g. ADRP at end of shard,
/// ADD at start of next) are visible to both workers. Duplicates
/// from the overlap are deduplicated at merge time.
pub fn split_range(
    start: u64,
    end: u64,
    n: usize,
    overlap: u64,
    boundary_align: u64,
) -> Vec<(u64, u64)> {
    if n == 0 || start >= end {
        return vec![];
    }
    let align = boundary_align.max(1);
    let total = end - start;
    let chunk = total.div_ceil(n as u64);

    (0..n as u64)
        .filter_map(|i| {
            let raw_start = start + i * chunk;
            // Round up to instruction alignment
            let shard_start = if align <= 1 {
                raw_start
            } else {
                (raw_start + align - 1) & !(align - 1)
            };
            if shard_start >= end {
                return None; // alignment pushed us past the end
            }
            let raw_end = start.saturating_add((i + 1).saturating_mul(chunk));
            // End does not need alignment — the arch scanner will handle tail bytes
            let shard_end = raw_end.saturating_add(overlap).min(end);
            Some((shard_start, shard_end))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_distribution() {
        // Check that ARM64-aligned addresses (multiples of 4) distribute evenly
        let n = 8;
        let mut counts = vec![0usize; n];
        for i in 0..1000u64 {
            let addr = i * 4; // ARM64 aligned
            counts[worker_for(addr, n)] += 1;
        }
        // Each bucket should get roughly 125 ± 50
        for c in &counts {
            assert!(*c > 50, "bucket got only {c} entries — poor distribution");
        }
    }

    #[test]
    fn test_split_range_coverage() {
        let shards = split_range(0x1000, 0x9000, 4, 0x40, 1);
        assert_eq!(shards.len(), 4);
        assert_eq!(shards[0].0, 0x1000);
        assert_eq!(shards[3].1, 0x9000);
        for (s, e) in &shards {
            assert!(e > s, "empty shard: {s:#x}..{e:#x}");
        }
    }

    #[test]
    fn test_split_range_arm64_alignment() {
        // Simulate ARM64: 3097000-byte segment at 0x400000, 14 workers
        let start = 0x400000u64;
        let size = 3097000u64;
        let shards = split_range(start, start + size, 14, 64, 4);
        for (i, (s, _e)) in shards.iter().enumerate() {
            assert_eq!(s % 4, 0, "shard {i} start {s:#x} is not 4-byte aligned");
        }
        // Coverage: first shard starts at segment start
        assert_eq!(shards[0].0, start);
    }
}
