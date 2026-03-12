pub mod arm64;
pub mod arm64_decode;
pub(crate) mod x86_64;

use crate::loader::{DecodeMode, Segment};
use crate::va::Va;
use crate::xref::{Confidence, Xref, XrefKind};

/// Minimum target VA for exec-segment byte-scan pointers.
///
/// Values below this threshold are almost always 3-byte ASCII strings whose
/// little-endian representation accidentally falls in the code segment VA range
/// (e.g. `"ode\0…"` → `0x65646f`). No real 64-bit code pointer is below 16 MiB.
const EXEC_TARGET_MIN_VA: u64 = 0x0100_0000;

/// Everything a single-pass xref extractor needs to know about
/// the region it's scanning.
pub(crate) struct ScanRegion<'a> {
    pub data: &'a [u8],
    pub base_va: Va,
    /// Decode mode for this region (Default, Thumb, Arm32).
    /// Reserved for arm32/Thumb scanners — not used by arm64/x86-64.
    #[allow(dead_code)]
    pub mode: DecodeMode,
    /// True if the source segment is writable (e.g. .data, .got).
    /// Reserved for future use by architecture-specific scanners.
    #[allow(dead_code)]
    pub writable: bool,
}

impl<'a> ScanRegion<'a> {
    pub fn new(seg: &'a Segment, start_va: Va, end_va: Va) -> Self {
        debug_assert!(
            start_va >= seg.va && end_va >= start_va,
            "ScanRegion::new: start_va ({start_va:#x}) must be >= seg.va ({:#x}) \
             and end_va ({end_va:#x}) must be >= start_va",
            seg.va,
        );
        let offset = (start_va - seg.va) as usize;
        let len = ((end_va - start_va) as usize).min(seg.data.len() - offset);
        Self {
            data: &seg.data[offset..offset + len],
            base_va: start_va,
            mode: seg.mode,
            writable: seg.writable,
        }
    }
}



// ── Segment flags ─────────────────────────────────────────────────────────────

/// Per-segment attribute bitmask stored in [`SegmentIndex`] and
/// [`SegmentDataIndex`] entries.
///
/// Newtype over `u8` so the flags cannot be accidentally mixed with other
/// byte values.  Use the provided `contains` / constant methods instead of
/// raw bit-twiddling at call sites.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub(crate) struct SegFlags(u8);

impl SegFlags {
    pub(crate) const EMPTY: Self = Self(0);
    pub(crate) const EXEC: Self = Self(0b01);
    pub(crate) const WRITE: Self = Self(0b10);

    /// Merge two flag sets (bitwise OR).
    #[inline]
    pub(crate) const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// True if `self` contains all bits in `other`.
    #[inline]
    pub(crate) const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

/// Compute the attribute flags for a segment.
#[inline]
fn segment_flags(s: &Segment) -> SegFlags {
    let mut flags = SegFlags::EMPTY;
    if s.executable {
        flags = flags.union(SegFlags::EXEC);
    }
    if s.writable {
        flags = flags.union(SegFlags::WRITE);
    }
    flags
}

// ── Segment index ─────────────────────────────────────────────────────────────

/// Sorted, copy-friendly VA interval table for O(log n) membership and
/// attribute queries. Built once per pass from `LoadedBinary::segments`.
///
/// Each entry is `(start, end, flags)` where [`SegFlags`] stores the segment
/// attributes as a bitmask.
pub(crate) struct SegmentIndex {
    /// Sorted by `start`.
    entries: Vec<(Va, Va, SegFlags)>,
}

/// Warn (once per build) if sorted intervals overlap — binary search may give
/// wrong results on malformed binaries.
fn check_disjoint<T, F: Fn(&T) -> (Va, Va)>(label: &str, entries: &[T], extract: F) {
    let overlaps = entries
        .windows(2)
        .any(|w| extract(&w[0]).1 > extract(&w[1]).0);
    if overlaps {
        eprintln!("warning: {label}: overlapping segments detected (binary search may give wrong results)");
    }
}

impl SegmentIndex {
    pub(crate) fn build(segments: &[Segment]) -> Self {
        let mut entries: Vec<(Va, Va, SegFlags)> = segments
            .iter()
            .map(|s| (s.va, s.va + s.data.len() as u64, segment_flags(s)))
            .collect();
        entries.sort_unstable_by_key(|e| e.0);
        // Overlap check is done once in SegmentDataIndex::build (built from
        // the same segments), no need to duplicate the warning here.
        Self { entries }
    }

    /// True if `va` falls within any mapped segment.
    #[inline]
    pub(crate) fn contains(&self, va: Va) -> bool {
        self.entry_at(va).is_some()
    }

    /// True if `va` falls within an executable segment.
    #[inline]
    pub(crate) fn is_exec(&self, va: Va) -> bool {
        self.entry_at(va).is_some_and(|f| f.contains(SegFlags::EXEC))
    }

    /// Returns the flags for the segment covering `va`, or None if unmapped.
    #[inline]
    pub(crate) fn flags_at(&self, va: Va) -> Option<SegFlags> {
        self.entry_at(va)
    }

    /// Binary-search for the entry covering `va`. Returns the flags.
    #[inline]
    fn entry_at(&self, va: Va) -> Option<SegFlags> {
        let idx = self.entries.partition_point(|e| e.0 <= va);
        if idx == 0 {
            return None;
        }
        let (start, end, flags) = self.entries[idx - 1];
        if va >= start && va < end {
            Some(flags)
        } else {
            None
        }
    }
}

// ── SegmentDataIndex ──────────────────────────────────────────────────────────

/// A single entry in a [`SegmentDataIndex`]: VA range, flags, and data slice.
struct DataIndexEntry<'a> {
    start: Va,
    end: Va,
    flags: SegFlags,
    data: &'a [u8],
}

/// Sorted VA interval table that also carries the raw data slice for each
/// segment. Used for O(log n) `read_u64_at` in the hot byte-scan and
/// ADRP/LDR pointer-follow paths — replacing the previous O(n_segments)
/// `segments.iter().find()` linear scan.
///
/// The lifetime `'a` ties the data slices to the `&'a LoadedBinary` that
/// this index was built from, which is the borrow held by `XrefPass::run()`.
/// No raw pointers or unsafe `Send`/`Sync` impls needed.
pub(crate) struct SegmentDataIndex<'a> {
    /// Sorted by `start`.
    entries: Vec<DataIndexEntry<'a>>,
}

impl<'a> SegmentDataIndex<'a> {
    pub(crate) fn build(segments: &'a [crate::loader::Segment]) -> Self {
        let mut entries: Vec<DataIndexEntry<'a>> = segments
            .iter()
            .map(|s| DataIndexEntry {
                start: s.va,
                end: s.va + s.data.len() as u64,
                flags: segment_flags(s),
                data: s.data,
            })
            .collect();
        entries.sort_unstable_by_key(|e| e.start);
        check_disjoint("SegmentDataIndex", &entries, |e| (e.start, e.end));
        Self { entries }
    }

    /// Binary-search for the entry covering `va`.
    #[inline]
    fn entry_at(&self, va: Va) -> Option<&DataIndexEntry<'a>> {
        let idx = self.entries.partition_point(|e| e.start <= va);
        if idx == 0 {
            return None;
        }
        let e = &self.entries[idx - 1];
        if va >= e.start && va < e.end {
            Some(e)
        } else {
            None
        }
    }

    /// Read a little-endian `u64` at `va` in a **non-executable** segment.
    /// Returns `None` if `va` is unmapped, executable, or the read would
    /// overflow the segment.
    #[inline]
    pub(crate) fn read_u64_at_nonexec(&self, va: Va) -> Option<u64> {
        let e = self.entry_at(va)?;
        if e.flags.contains(SegFlags::EXEC) {
            return None;
        }
        let offset = (va - e.start) as usize;
        let bytes = e.data.get(offset..offset + 8)?;
        Some(u64::from_le_bytes(
            bytes.try_into().expect("slice is exactly 8 bytes"),
        ))
    }

    /// Returns the flags for the segment covering `va`, or `None` if unmapped.
    #[inline]
    pub(crate) fn flags_at(&self, va: Va) -> Option<SegFlags> {
        self.entry_at(va).map(|e| e.flags)
    }

    /// Read a little-endian `i32` at `va` in any mapped segment.
    /// Returns `None` if `va` is unmapped or the read would overflow the segment.
    ///
    /// Unlike `read_u64_at_nonexec`, this does NOT filter by exec flag — jump
    /// tables may reside in `.rodata` (non-exec) or, rarely, inline in `.text`.
    #[inline]
    pub(crate) fn read_i32_at(&self, va: Va) -> Option<i32> {
        let e = self.entry_at(va)?;
        let offset = (va - e.start) as usize;
        let bytes = e.data.get(offset..offset + 4)?;
        Some(i32::from_le_bytes(
            bytes.try_into().expect("slice is exactly 4 bytes"),
        ))
    }

    /// Read a single byte at the given VA, returning `None` if unmapped.
    pub(crate) fn read_u8_at(&self, va: Va) -> Option<u8> {
        let e = self.entry_at(va)?;
        let offset = (va - e.start) as usize;
        e.data.get(offset).copied()
    }

    /// Read a little-endian `u16` at the given VA, returning `None` if unmapped.
    pub(crate) fn read_u16_at(&self, va: Va) -> Option<u16> {
        let e = self.entry_at(va)?;
        let offset = (va - e.start) as usize;
        let bytes = e.data.get(offset..offset + 2)?;
        Some(u16::from_le_bytes(
            bytes.try_into().expect("slice is exactly 2 bytes"),
        ))
    }
}

/// Pointer-width byte scan over a data region.
///
/// Emits `DataPointer` xrefs for every aligned pointer-sized slot whose value
/// lands in a mapped segment, subject to:
///
/// - **Non-exec target**: always emitted — data→data pointers are unambiguous.
/// - **Exec target**: emitted only when `target >= 0x0100_0000`. Values below
///   that threshold are almost always 3-byte ASCII strings whose little-endian
///   representation accidentally falls in the code segment VA range (e.g.
///   `"ode\0…"` → `0x65646f`). No real 64-bit code pointer is below 16 MiB.
pub(crate) fn byte_scan_pointers(
    region: &ScanRegion,
    data_idx: &SegmentDataIndex,
    pointer_size: usize,
) -> Vec<Xref> {
    let mut xrefs = Vec::new();
    let data = region.data;
    let step = pointer_size;

    // Only scan on pointer-aligned boundaries.
    // Guard: if data is shorter than one pointer we have nothing to scan.
    if data.len() < pointer_size {
        return xrefs;
    }
    let end = data.len() - pointer_size; // safe: checked above

    let mut i = 0usize;
    while i <= end {
        // Loop bound guarantees `i + pointer_size <= data.len()`, so slices
        // are always exactly the right length for the array conversion.
        let target = if pointer_size == 8 {
            u64::from_le_bytes(data[i..i + 8].try_into().expect("8-byte slice"))
        } else {
            u32::from_le_bytes(data[i..i + 4].try_into().expect("4-byte slice")) as u64
        };

        if target != 0 {
            let target_va = Va::new(target);
            let emit = match data_idx.flags_at(target_va) {
                // Target in a non-exec segment: always emit.
                Some(f) if !f.contains(SegFlags::EXEC) => true,
                // Target in an exec segment: vtables and function pointer tables
                // legitimately hold code pointers, so allow them — but require at
                // least 4 significant bytes to reject ASCII-string false positives.
                // Strings in data sections can form a value in the exec VA range
                // from just 3 bytes (e.g. "ode\0…" → 0x65646f in a binary loaded
                // at 0x400000). No real 64-bit code pointer is below 0x0100_0000.
                Some(_) => target >= EXEC_TARGET_MIN_VA,
                // Target not in any mapped segment: skip.
                None => false,
            };
            if emit {
                let from = region.base_va + i as u64;
                xrefs.push(Xref {
                    from,
                    to: target_va,
                    kind: XrefKind::DataPointer,
                    confidence: Confidence::ByteScan,
                });
            }
        }
        i += step;
    }
    xrefs
}
