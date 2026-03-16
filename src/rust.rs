//! Rust-specific binary analysis: string literal extraction.
//!
//! Rust concatenates string literals into large blobs in `.rodata`. Each `&str`
//! reference is a `(ptr, len)` pair where `ptr` points into the blob and `len`
//! gives the byte length. This module finds those blobs and extracts individual
//! strings by following `data_ptr` xrefs and reading the adjacent length field.

use crate::loader::{LoadedBinary, Segment};
use crate::va::Va;

/// Minimum blob size to consider (bytes).
pub const DEFAULT_MIN_BLOB_LEN: usize = 16;

/// A contiguous UTF-8 blob found in read-only data.
#[derive(Debug, Clone)]
pub struct StringBlob {
    /// Start VA of the blob.
    pub va: Va,
    /// Raw UTF-8 bytes.
    pub data: Vec<u8>,
}

impl StringBlob {
    /// End VA (exclusive) of the blob.
    #[inline]
    pub fn end_va(&self) -> Va {
        self.va + self.data.len() as u64
    }

    /// Check if a VA falls within this blob.
    #[inline]
    pub fn contains(&self, va: Va) -> bool {
        va >= self.va && va < self.end_va()
    }

    /// Extract a string slice starting at `va` with the given `len`.
    /// Returns `None` if out of bounds or invalid UTF-8.
    pub fn extract(&self, va: Va, len: usize) -> Option<&str> {
        if va < self.va {
            return None;
        }
        let offset = (va - self.va) as usize;
        let end = offset.checked_add(len)?;
        let bytes = self.data.get(offset..end)?;
        std::str::from_utf8(bytes).ok()
    }
}

/// Index of UTF-8 string blobs for fast point queries.
///
/// Blobs are sorted by VA and non-overlapping, enabling O(log n) lookup.
#[derive(Debug)]
pub struct StringBlobIndex {
    blobs: Vec<StringBlob>,
}

impl StringBlobIndex {
    /// Build an index by scanning read-only segments for UTF-8 blobs.
    ///
    /// `min_len` is the minimum blob size to include (default: [`DEFAULT_MIN_BLOB_LEN`], 16 bytes).
    pub fn build(binary: &LoadedBinary, min_len: usize) -> Self {
        let mut blobs = Vec::new();

        // Scan all readable, non-writable segments for string data.
        // On Mach-O, __cstring and __const are inside the executable __TEXT
        // segment, so we can't just use data_segments() which filters by !exec.
        // Instead we scan anything that's readable and not writable.
        for seg in &binary.segments {
            if !seg.readable || seg.writable {
                continue;
            }
            blobs.extend(scan_segment_for_blobs(seg, min_len));
        }

        // Sort by VA for binary search
        blobs.sort_by_key(|b| b.va);

        Self { blobs }
    }

    /// Find the blob containing `va`, if any.
    pub fn lookup(&self, va: Va) -> Option<&StringBlob> {
        // Binary search: find the last blob with start <= va
        let idx = self.blobs.partition_point(|b| b.va <= va);
        if idx == 0 {
            return None;
        }
        let blob = &self.blobs[idx - 1];
        if blob.contains(va) {
            Some(blob)
        } else {
            None
        }
    }

    /// Number of blobs found.
    pub fn len(&self) -> usize {
        self.blobs.len()
    }

    /// Total bytes across all blobs.
    pub fn total_bytes(&self) -> usize {
        self.blobs.iter().map(|b| b.data.len()).sum()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.blobs.is_empty()
    }

    /// Iterate over all blobs.
    pub fn iter(&self) -> impl Iterator<Item = &StringBlob> {
        self.blobs.iter()
    }
}

/// Scan a segment for contiguous UTF-8 spans >= `min_len` bytes.
fn scan_segment_for_blobs(seg: &Segment, min_len: usize) -> Vec<StringBlob> {
    let data = seg.data();
    let mut blobs = Vec::new();
    let mut start: Option<usize> = None;

    for (i, &b) in data.iter().enumerate() {
        if is_string_byte(b) {
            if start.is_none() {
                start = Some(i);
            }
        } else {
            if let Some(s) = start {
                let len = i - s;
                if len >= min_len {
                    let span = &data[s..i];
                    // Validate it's actually valid UTF-8
                    if std::str::from_utf8(span).is_ok() {
                        blobs.push(StringBlob {
                            va: seg.va + s as u64,
                            data: span.to_vec(),
                        });
                    }
                }
            }
            start = None;
        }
    }

    // Handle span at end of segment
    if let Some(s) = start {
        let len = data.len() - s;
        if len >= min_len {
            let span = &data[s..];
            if std::str::from_utf8(span).is_ok() {
                blobs.push(StringBlob {
                    va: seg.va + s as u64,
                    data: span.to_vec(),
                });
            }
        }
    }

    blobs
}

/// Check if a byte is plausibly part of a UTF-8 string.
///
/// Accepts printable ASCII, common whitespace, valid UTF-8 continuation
/// bytes (`0x80..=0xbf`), and valid UTF-8 leading bytes (`0xc2..=0xf4`).
/// Rejects `0xc0`/`0xc1` (overlong 2-byte encodings) and `0xf5..=0xff`
/// (above Unicode max), which can never appear in valid UTF-8.
///
/// This is a fast pre-filter; spans that pass are still validated with
/// `str::from_utf8` before being emitted as blobs.
#[inline]
fn is_string_byte(b: u8) -> bool {
    match b {
        // Printable ASCII
        0x20..=0x7e => true,
        // Common whitespace
        0x09 | 0x0a | 0x0d => true,
        // UTF-8 continuation bytes
        0x80..=0xbf => true,
        // UTF-8 leading bytes (2-byte: 0xc2..=0xdf, 3-byte: 0xe0..=0xef, 4-byte: 0xf0..=0xf4)
        // Excludes 0xc0/0xc1 (overlong) and 0xf5..=0xff (above U+10FFFF).
        0xc2..=0xf4 => true,
        _ => false,
    }
}

/// Read a pointer-sized unsigned integer at the given VA.
pub fn read_usize_at(binary: &LoadedBinary, va: Va, ptr_size: usize) -> Option<usize> {
    let seg = binary.segment_at(va)?;
    let bytes = seg.bytes_at(va, ptr_size)?;

    Some(match ptr_size {
        8 => u64::from_le_bytes(bytes.try_into().ok()?) as usize,
        4 => u32::from_le_bytes(bytes.try_into().ok()?) as usize,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_string_byte() {
        // Printable ASCII
        assert!(is_string_byte(b'a'));
        assert!(is_string_byte(b'Z'));
        assert!(is_string_byte(b' '));
        assert!(is_string_byte(b'~'));

        // Whitespace
        assert!(is_string_byte(b'\t'));
        assert!(is_string_byte(b'\n'));
        assert!(is_string_byte(b'\r'));

        // UTF-8 continuation bytes
        assert!(is_string_byte(0x80));
        assert!(is_string_byte(0xbf));

        // UTF-8 start bytes
        assert!(is_string_byte(0xc2)); // 2-byte start
        assert!(is_string_byte(0xe0)); // 3-byte start
        assert!(is_string_byte(0xf0)); // 4-byte start

        // Not string bytes
        assert!(!is_string_byte(0x00)); // null
        assert!(!is_string_byte(0x01)); // control
        assert!(!is_string_byte(0x7f)); // DEL
        assert!(!is_string_byte(0xc0)); // overlong 2-byte
        assert!(!is_string_byte(0xc1)); // overlong 2-byte
        assert!(!is_string_byte(0xf5)); // above U+10FFFF
        assert!(!is_string_byte(0xfe)); // invalid
        assert!(!is_string_byte(0xff)); // invalid
    }

    #[test]
    fn test_string_blob_extract() {
        let blob = StringBlob {
            va: Va::new(0x1000),
            data: b"Hello, World!".to_vec(),
        };

        assert_eq!(blob.extract(Va::new(0x1000), 5), Some("Hello"));
        assert_eq!(blob.extract(Va::new(0x1007), 5), Some("World"));
        assert_eq!(blob.extract(Va::new(0x1000), 13), Some("Hello, World!"));

        // Out of bounds
        assert_eq!(blob.extract(Va::new(0x1000), 14), None);
        assert_eq!(blob.extract(Va::new(0x0fff), 5), None);
        assert_eq!(blob.extract(Va::new(0x100d), 1), None);
    }

    #[test]
    fn test_string_blob_contains() {
        let blob = StringBlob {
            va: Va::new(0x1000),
            data: b"test".to_vec(),
        };

        assert!(blob.contains(Va::new(0x1000)));
        assert!(blob.contains(Va::new(0x1003)));
        assert!(!blob.contains(Va::new(0x0fff)));
        assert!(!blob.contains(Va::new(0x1004)));
    }
}
