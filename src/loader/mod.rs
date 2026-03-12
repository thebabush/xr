mod dyld;
mod elf;
mod macho;
mod pe;

use crate::va::Va;
use anyhow::{anyhow, Result};
use goblin::Object;
use memmap2::Mmap;
use rustc_hash::FxHashSet;
use std::any::Any;
use std::ops::Range;
use std::path::Path;

// ── Shared types ──────────────────────────────────────────────────────────────

/// A relocation-derived pointer: a pointer-sized slot at `from` whose value
/// (after relocation) points to `to`.
///
/// These are authoritative — the relocation table says exactly which slots
/// are pointers and what they point to.  Emitted as `DataPointer` xrefs.
#[derive(Clone, Copy, Debug)]
pub struct RelocPointer {
    /// Address of the pointer slot.
    pub from: Va,
    /// Value the pointer resolves to.
    pub to: Va,
}

/// A named symbol exported by (or defined in) the binary.
#[derive(Clone, Debug)]
pub struct Symbol {
    pub name: String,
    pub va: Va,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    X86_64,
    Arm64,
    X86,
    Arm32,
    Unknown,
}

/// Decode mode — some architectures have multiple modes active simultaneously.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeMode {
    /// Normal mode (ARM64, x86, x86-64)
    Default,
    /// ARM Thumb (2-byte aligned, mixed 16/32-bit instructions)
    Thumb,
    /// ARM32 classic (4-byte aligned)
    Arm32,
}

/// Segment byte data with externally managed lifetime.
///
/// Internally stores a `&'static [u8]` whose true lifetime is tied to the
/// backing store (mmap / BSS buffer / dyld context) in [`LoadedBinary`].
/// The `'static` is hidden by this newtype so it cannot leak through safe
/// code.
///
/// [`Deref<Target = [u8]>`] provides zero-cost transparent access to all
/// slice methods.  The resulting `&[u8]` lifetime is tied to `&self`, not
/// `'static`, so callers cannot hold references beyond the segment's
/// lifetime.
pub struct SegData {
    /// The `'static` lifetime here is a lie — see type-level docs.
    /// This field is private so only code in this module can observe
    /// the `'static` lifetime.
    inner: &'static [u8],
}

impl SegData {
    /// Create a `SegData` wrapping the given byte slice.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `data` remains valid for the lifetime
    /// of any [`Segment`] that holds this `SegData`.  In practice this means
    /// the backing mmap / BSS buffer must outlive the `Segment` — enforced
    /// by the field-ordering invariant of [`LoadedBinary`].
    ///
    /// Restricted to the `loader` module — all segment construction happens
    /// here.  Test code should use [`new_for_test`](Self::new_for_test).
    #[inline]
    pub(in crate::loader) unsafe fn new(data: &[u8]) -> Self {
        Self {
            inner: unsafe { &*(data as *const [u8]) },
        }
    }

    /// Test-only constructor with crate-wide visibility.
    ///
    /// # Safety
    ///
    /// Same as [`new`](Self::new) — `data` must remain valid for the
    /// lifetime of any `Segment` holding this `SegData`.  In tests this
    /// is typically satisfied by `&'static [u8]` (static arrays or
    /// `Box::leak`).
    #[cfg(test)]
    #[inline]
    pub(crate) unsafe fn new_for_test(data: &[u8]) -> Self {
        // Safety: forwarded from caller.
        unsafe { Self::new(data) }
    }
}

impl std::ops::Deref for SegData {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        self.inner
    }
}

impl std::fmt::Debug for SegData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{} bytes]", self.inner.len())
    }
}

/// A single mapped segment of the binary — executable or data.
/// Backed by a byte slice into the mmap'd file. Zero copy.
#[derive(Debug)]
pub struct Segment {
    /// Virtual address of the segment start.
    pub va: Va,
    /// Raw bytes — a [`SegData`] wrapping a slice into the mmap (zero-copy).
    ///
    /// Access via [`Deref`]: `seg.data.len()`, `&seg.data[range]`, etc.
    /// The resulting `&[u8]` lifetime is tied to `&seg`, not `'static`.
    pub(crate) data: SegData,
    /// Whether this segment contains executable code.
    pub executable: bool,
    /// Whether this segment contains readable data.
    pub readable: bool,
    /// Whether this segment contains writable data.
    pub writable: bool,
    /// Whether this segment should be byte-scanned for pointer values.
    /// False for sections like `.data.rel.ro` which contain relocatable
    /// data that produces too many false-positive pointer hits without
    /// relocation-table context.
    pub byte_scannable: bool,
    /// Decode mode for this segment (relevant for ARM).
    pub mode: DecodeMode,
    /// Human-readable name (e.g. "__TEXT", ".text", "LOAD[0]").
    pub name: String,
}

impl Segment {
    /// Raw byte data backing this segment.
    ///
    /// The returned reference is tied to `&self`, so callers cannot outlive
    /// the owning [`LoadedBinary`].
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Byte slice at a given virtual address range, if within this segment.
    pub fn bytes_at(&self, va: Va, len: usize) -> Option<&[u8]> {
        let offset = va.raw().checked_sub(self.va.raw())? as usize;
        self.data.get(offset..offset + len)
    }

    /// VA range covered by this segment.
    pub fn va_range(&self) -> Range<Va> {
        self.va..self.va + self.data.len() as u64
    }

    /// True if the given VA falls within this segment.
    pub fn contains(&self, va: Va) -> bool {
        self.va_range().contains(&va)
    }
}

// ── LoadedBinary ──────────────────────────────────────────────────────────────

/// A loaded binary — mmap'd file + parsed segment list.
/// The mmap is kept alive for the lifetime of this struct.
/// All segment data slices are zero-copy into the mmap.
///
/// # Safety — field ordering invariant
///
/// `Segment::data` slices are `&'static [u8]` whose true lifetime is tied to
/// the backing stores in this struct (`_mmap`, `_bss_bufs`, `_dyld_ctx`).
/// Rust drops struct fields in declaration order, so `segments` (field 1) is
/// dropped **before** the backing stores (fields 8–10).  This guarantees that
/// no `Segment::data` slice is accessed after its backing memory is freed.
///
/// **Do not reorder fields** such that `_mmap`, `_bss_bufs`, or `_dyld_ctx`
/// appear before `segments`.  The `test_backing_fields_after_segments` test
/// enforces this invariant at compile time.
pub struct LoadedBinary {
    // ── public data (fields 0–6) ──────────────────────────────────────────
    /// Architecture detected from the binary.
    pub arch: Arch,
    /// All mapped segments (code + data).
    ///
    /// # Safety
    /// Must be declared before the backing-store fields (`_mmap`, `_bss_bufs`,
    /// `_dyld_ctx`) so it is dropped first.
    pub segments: Vec<Segment>,
    /// Entry points / known function seeds.
    pub entry_points: Vec<Va>,
    /// Exports / named symbols with their addresses.
    pub symbols: Vec<Symbol>,
    /// Non-zero if this is a PIE ELF that was rebased by the loader.
    /// All segment VAs, entry points, and symbols have already had this
    /// value added. 0 for non-PIE binaries and non-ELF formats.
    pub pie_base: u64,
    /// Set of GOT slot VAs (from GLOB_DAT / JUMP_SLOT relocations).
    ///
    /// Used by the x86-64 scanner to restrict `CALL [RIP+disp32]` /
    /// `JMP [RIP+disp32]` xref emission to actual import-GOT slots
    /// (vs arbitrary RIP-relative indirect calls through non-GOT pointers).
    /// Populated for ELF binaries; empty for Mach-O and PE.
    pub got_slots: FxHashSet<Va>,
    /// Relocation-derived pointer entries.
    ///
    /// Each entry represents a pointer-sized slot that the relocation table
    /// says points to a mapped address.  Emitted as `DataPointer` xrefs in
    /// the scan pass.  Populated from ELF `.rela.dyn` / `.rel.dyn`
    /// (R_*_RELATIVE, R_*_64, R_*_ABS64) and PE base relocations + IAT.
    /// Empty for formats without relocation tables.
    pub reloc_pointers: Vec<RelocPointer>,

    // ── backing stores (fields 7–9) — MUST come after `segments` ──────────
    /// The underlying mmap — kept alive so `Segment::data` slices remain valid.
    _mmap: Mmap,
    /// Zero-filled BSS buffers — `Segment::data` slices may point into these.
    _bss_bufs: Vec<Box<[u8]>>,
    /// For dyld shared caches: the `DyldContext` (owns the subcache mmaps).
    /// Segment slices borrow from these mmaps zero-copy.
    _dyld_ctx: Option<Box<dyn Any + Send + Sync>>,
}

impl LoadedBinary {
    /// Load a binary from disk. Mmap's the file, parses segments.
    /// Returns a `LoadedBinary` whose segment slices are zero-copy into the mmap.
    /// Supports ELF, single-arch Mach-O, and PE. Fat (universal) Mach-O binaries
    /// are not supported — extract the desired arch slice first (e.g. with `lipo`).
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_with_base(path, None)
    }

    /// Like `load`, but allows overriding the base VA for PIE ELF binaries.
    ///
    /// `base` is the virtual address at which the binary's first PT_LOAD segment
    /// (the one with `p_vaddr == 0`) is assumed to be mapped.  For non-PIE ELF,
    /// Mach-O, and PE binaries the preferred base is already baked into the binary
    /// and `base` is ignored.
    ///
    /// When `base` is `None` the default PIE base (`0x0040_0000`) is used, which
    /// matches IDA's default load address for Linux PIE ELF shared libraries.
    pub fn load_with_base(path: &Path, base: Option<u64>) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Detect dyld shared cache by magic prefix before handing off to goblin.
        if mmap.starts_with(b"dyld_v1 ") {
            let result = dyld::parse_dyld_cache(path)?;
            let p = result.parsed;
            return Ok(LoadedBinary {
                arch: p.arch,
                segments: p.segments,
                entry_points: p.entry_points,
                symbols: p.symbols,
                pie_base: p.pie_base,
                got_slots: p.got_slots,
                reloc_pointers: p.reloc_pointers,
                _mmap: mmap,
                _bss_bufs: vec![],
                _dyld_ctx: Some(Box::new(result.dyld_ctx)),
            });
        }

        let bytes: &[u8] = &mmap[..];

        let mut bss_bufs: Vec<Box<[u8]>> = Vec::new();
        let p = parse_binary(bytes, &mut bss_bufs, base)?;

        Ok(LoadedBinary {
            arch: p.arch,
            segments: p.segments,
            entry_points: p.entry_points,
            symbols: p.symbols,
            pie_base: p.pie_base,
            got_slots: p.got_slots,
            reloc_pointers: p.reloc_pointers,
            _mmap: mmap,
            _bss_bufs: bss_bufs,
            _dyld_ctx: None,
        })
    }

    /// Construct a LoadedBinary directly from segments, for use in tests.
    #[cfg(test)]
    pub fn from_segments(arch: Arch, segments: Vec<Segment>) -> Self {
        use std::io::Write;
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&[0u8]).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(tmp.as_file()).unwrap() };
        Self {
            arch,
            segments,
            entry_points: vec![],
            symbols: vec![],
            pie_base: 0,
            got_slots: FxHashSet::default(),
            reloc_pointers: Vec::new(),
            _mmap: mmap,
            _bss_bufs: vec![],
            _dyld_ctx: None,
        }
    }

    /// Find the segment containing a given virtual address.
    pub fn segment_at(&self, va: Va) -> Option<&Segment> {
        self.segments.iter().find(|s| s.contains(va))
    }

    /// True if the given VA is in any mapped segment.
    pub fn is_mapped(&self, va: Va) -> bool {
        self.segment_at(va).is_some()
    }

    /// True if the given VA is in an executable segment.
    pub fn is_executable(&self, va: Va) -> bool {
        self.segment_at(va).is_some_and(|s| s.executable)
    }

    /// All executable segments.
    pub fn code_segments(&self) -> impl Iterator<Item = &Segment> {
        self.segments.iter().filter(|s| s.executable)
    }

    /// All data (non-executable) segments.
    pub fn data_segments(&self) -> impl Iterator<Item = &Segment> {
        self.segments.iter().filter(|s| !s.executable && s.readable)
    }

    /// All segments worth byte-scanning for pointers.
    /// Excludes executable segments and segments marked as not byte-scannable
    /// (e.g. `.data.rel.ro` which has a very high FP rate without relocation
    /// context). Only non-executable, readable, byte_scannable segments
    /// (.data, .bss, .got, etc.) are scanned.
    pub fn scannable_data_segments(&self) -> impl Iterator<Item = &Segment> {
        self.segments
            .iter()
            .filter(|s| s.readable && !s.executable && s.byte_scannable)
    }

    /// Lowest virtual address of any mapped segment.
    /// Useful as a minimum bound for filtering out-of-range ref targets.
    pub fn min_va(&self) -> Va {
        self.segments.iter().map(|s| s.va).min().unwrap_or(Va::ZERO)
    }
}

// ── Shared utilities ──────────────────────────────────────────────────────────

/// Return type shared by all binary-format parsers.
struct ParseResult {
    arch: Arch,
    segments: Vec<Segment>,
    entry_points: Vec<Va>,
    symbols: Vec<Symbol>,
    /// Non-zero only for PIE ELF binaries rebased by `parse_elf`.
    pie_base: u64,
    /// GOT slot VAs from GLOB_DAT / JUMP_SLOT relocs. Empty for non-ELF.
    got_slots: FxHashSet<Va>,
    /// Relocation-derived pointer entries.
    reloc_pointers: Vec<RelocPointer>,
}

/// Allocate a zero-filled buffer of `size` bytes, store it in `bufs` for
/// lifetime management, and return a [`SegData`] wrapping it.
///
/// # Safety (internal)
///
/// The returned `SegData` is valid as long as the owning `bufs` Vec (and
/// thus `LoadedBinary::_bss_bufs`) is alive — same drop-order guarantee as
/// `LoadedBinary::_mmap` for the mmap-backed slices.
fn alloc_bss(size: usize, bufs: &mut Vec<Box<[u8]>>) -> SegData {
    let buf: Box<[u8]> = vec![0u8; size].into_boxed_slice();
    let ptr = buf.as_ptr();
    bufs.push(buf);
    // Safety: ptr points into the Box we just pushed, which lives in `bufs`
    // (ultimately in `LoadedBinary::_bss_bufs`). The struct drops `_bss_bufs`
    // after `segments`, so the SegData outlives all Segment references.
    unsafe { SegData::new(std::slice::from_raw_parts(ptr, size)) }
}

/// Sorted VA range set for O(log n) containment checks.
struct VaRangeSet {
    /// Sorted by start VA; assumed disjoint (same invariant as SegmentIndex).
    ranges: Vec<(Va, Va)>,
}

impl VaRangeSet {
    fn build(segments: &[Segment]) -> Self {
        let mut ranges: Vec<(Va, Va)> = segments
            .iter()
            .map(|s| (s.va, s.va + s.data.len() as u64))
            .collect();
        ranges.sort_unstable_by_key(|&(start, _)| start);
        Self { ranges }
    }

    #[inline]
    fn contains(&self, va: Va) -> bool {
        let idx = self.ranges.partition_point(|&(start, _)| start <= va);
        if idx == 0 {
            return false;
        }
        let (start, end) = self.ranges[idx - 1];
        va >= start && va < end
    }
}

// ── Format dispatch ───────────────────────────────────────────────────────────

/// # Safety (internal)
///
/// `bytes` must remain valid for the lifetime of any `Segment` in the
/// returned `ParseResult`.  This is guaranteed by the caller
/// (`load_with_base`) which stores the mmap in `LoadedBinary::_mmap` and
/// drops it after `segments`.
fn parse_binary(
    bytes: &[u8],
    bss_bufs: &mut Vec<Box<[u8]>>,
    base: Option<u64>,
) -> Result<ParseResult> {
    match Object::parse(bytes)? {
        Object::Elf(ref elf_obj) => elf::parse_elf(bytes, elf_obj, bss_bufs, base),
        Object::Mach(goblin::mach::Mach::Binary(ref macho)) => {
            macho::parse_macho(bytes, macho, bss_bufs)
        }
        Object::Mach(goblin::mach::Mach::Fat(_)) => {
            Err(anyhow!("fat (universal) Mach-O binaries are not supported; extract the desired arch slice first (e.g. with `lipo -extract`)"))
        }
        Object::PE(ref pe_obj) => pe::parse_pe(bytes, pe_obj, bss_bufs),
        Object::Unknown(_) => {
            // Safety: `bytes` is the mmap kept alive by `LoadedBinary::_mmap`.
            let seg = Segment {
                va: Va::ZERO,
                data: unsafe { SegData::new(bytes) },
                executable: true,
                readable: true,
                writable: false,
                mode: DecodeMode::Default,
                name: "raw".to_string(),
                byte_scannable: true,
            };
            Ok(ParseResult {
                arch: Arch::Unknown,
                segments: vec![seg],
                entry_points: vec![Va::ZERO],
                symbols: vec![],
                pie_base: 0,
                got_slots: FxHashSet::default(),
                reloc_pointers: Vec::new(),
            })
        }
        _ => Err(anyhow!("unsupported binary format")),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Build a minimal valid 64-bit PIE ELF (ET_DYN, EM_X86_64) with a single
    /// PT_LOAD at p_vaddr=0 and a .text section at raw vaddr 0x1000.
    fn make_minimal_pie_elf() -> Vec<u8> {
        let shstrtab_off: u64 = 0x78;
        let shstrtab_content: &[u8] = b"\x00.text\x00";
        let shstrtab_size = shstrtab_content.len() as u64;

        let text_off: u64 = shstrtab_off + shstrtab_size; // 0x80
        let text_size: u64 = 64;
        let text_vaddr: u64 = text_off; // 0x80

        let shoff: u64 = text_off + text_size; // 0xc0
        let file_size: u64 = shoff + 3 * 64; // 0x180

        let mut buf = vec![0u8; file_size as usize];

        fn w(buf: &mut [u8], off: usize, val: u64, n: usize) {
            buf[off..off + n].copy_from_slice(&val.to_le_bytes()[..n]);
        }

        buf[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
        buf[4] = 2; // ELFCLASS64
        buf[5] = 1; // ELFDATA2LSB
        buf[6] = 1; // EV_CURRENT

        let h = 16usize;
        w(&mut buf, h, 3, 2); // e_type = ET_DYN
        w(&mut buf, h + 2, 62, 2); // e_machine = EM_X86_64
        w(&mut buf, h + 4, 1, 4); // e_version
        w(&mut buf, h + 8, 0, 8); // e_entry
        w(&mut buf, h + 16, 0x40, 8); // e_phoff
        w(&mut buf, h + 24, shoff, 8); // e_shoff
        w(&mut buf, h + 32, 0, 4); // e_flags
        w(&mut buf, h + 36, 64, 2); // e_ehsize
        w(&mut buf, h + 38, 56, 2); // e_phentsize
        w(&mut buf, h + 40, 1, 2); // e_phnum
        w(&mut buf, h + 42, 64, 2); // e_shentsize
        w(&mut buf, h + 44, 3, 2); // e_shnum
        w(&mut buf, h + 46, 1, 2); // e_shstrndx = 1

        let ph = 0x40usize;
        w(&mut buf, ph, 1, 4); // PT_LOAD
        w(&mut buf, ph + 4, 0x5, 4); // PF_R|PF_X
        w(&mut buf, ph + 8, 0, 8); // p_offset
        w(&mut buf, ph + 16, 0, 8); // p_vaddr = 0 (PIE)
        w(&mut buf, ph + 24, 0, 8); // p_paddr
        w(&mut buf, ph + 32, file_size, 8); // p_filesz
        w(&mut buf, ph + 40, file_size, 8); // p_memsz
        w(&mut buf, ph + 48, 0x1000, 8); // p_align

        buf[shstrtab_off as usize..shstrtab_off as usize + shstrtab_content.len()]
            .copy_from_slice(shstrtab_content);

        // Section 1: .shstrtab
        let s1 = shoff as usize + 64;
        w(&mut buf, s1, 0, 4);
        w(&mut buf, s1 + 4, 3, 4); // SHT_STRTAB
        w(&mut buf, s1 + 24, shstrtab_off, 8);
        w(&mut buf, s1 + 32, shstrtab_size, 8);
        w(&mut buf, s1 + 48, 1, 8);

        // Section 2: .text
        let s2 = shoff as usize + 128;
        w(&mut buf, s2, 1, 4); // sh_name = 1 → ".text"
        w(&mut buf, s2 + 4, 1, 4); // SHT_PROGBITS
        w(&mut buf, s2 + 8, 0x6, 8); // SHF_ALLOC|SHF_EXECINSTR
        w(&mut buf, s2 + 16, text_vaddr, 8);
        w(&mut buf, s2 + 24, text_off, 8);
        w(&mut buf, s2 + 32, text_size, 8);
        w(&mut buf, s2 + 48, 0x10, 8);

        buf
    }

    fn write_tmp(bytes: &[u8]) -> tempfile::NamedTempFile {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(bytes).unwrap();
        tmp.flush().unwrap();
        tmp
    }

    fn segment_vas(elf: &[u8], base: Option<u64>) -> Vec<Va> {
        let tmp = write_tmp(elf);
        let bin = LoadedBinary::load_with_base(tmp.path(), base).unwrap();
        let mut vas: Vec<Va> = bin.segments.iter().map(|s| s.va).collect();
        vas.sort_unstable();
        vas
    }

    #[test]
    fn test_load_pie_default_base() {
        let elf = make_minimal_pie_elf();
        let tmp = write_tmp(&elf);
        let binary = LoadedBinary::load(tmp.path()).unwrap();
        assert_eq!(binary.pie_base, 0x0040_0000);
        assert!(
            binary.segments.iter().all(|s| s.va >= Va::new(0x0040_0000)),
            "all segment VAs should be rebased above 0x400000, got: {:?}",
            binary.segments.iter().map(|s| s.va).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_load_pie_base_zero() {
        let elf = make_minimal_pie_elf();
        let raw_vas = segment_vas(&elf, Some(0));
        let tmp = write_tmp(&elf);
        let binary = LoadedBinary::load_with_base(tmp.path(), Some(0)).unwrap();
        assert_eq!(binary.pie_base, 0);
        let mut got: Vec<Va> = binary.segments.iter().map(|s| s.va).collect();
        got.sort_unstable();
        assert_eq!(got, raw_vas, "base=0 should not shift VAs");
    }

    #[test]
    fn test_load_pie_custom_base() {
        let elf = make_minimal_pie_elf();
        let raw_vas = segment_vas(&elf, Some(0));
        let custom_base = 0x7f00_0000u64;
        let tmp = write_tmp(&elf);
        let binary = LoadedBinary::load_with_base(tmp.path(), Some(custom_base)).unwrap();
        assert_eq!(binary.pie_base, custom_base);
        let mut got: Vec<Va> = binary.segments.iter().map(|s| s.va).collect();
        got.sort_unstable();
        let expected: Vec<Va> = raw_vas.iter().map(|v| *v + custom_base).collect();
        assert_eq!(
            got, expected,
            "custom base should shift all VAs by exactly custom_base"
        );
    }

    #[test]
    fn test_load_pie_base_shift_consistency() {
        let elf = make_minimal_pie_elf();
        let base_a = 0x0040_0000u64;
        let base_b = 0x0080_0000u64;
        let vas_a = segment_vas(&elf, Some(base_a));
        let vas_b = segment_vas(&elf, Some(base_b));
        assert_eq!(vas_a.len(), vas_b.len(), "same number of segments");
        let delta = base_b - base_a;
        for (a, b) in vas_a.iter().zip(vas_b.iter()) {
            assert_eq!(*b - *a, delta, "VA shift mismatch: {a:#x} vs {b:#x}");
        }
    }

    #[test]
    fn test_backing_fields_after_segments() {
        let seg = std::mem::offset_of!(LoadedBinary, segments);
        let mmap = std::mem::offset_of!(LoadedBinary, _mmap);
        let bss = std::mem::offset_of!(LoadedBinary, _bss_bufs);
        let dyld = std::mem::offset_of!(LoadedBinary, _dyld_ctx);
        assert!(
            seg < mmap && seg < bss && seg < dyld,
            "LoadedBinary field order violated: `segments` (offset {seg}) must be \
             declared before _mmap ({mmap}), _bss_bufs ({bss}), _dyld_ctx ({dyld}) \
             so it is dropped first (Rust drops fields in declaration order)"
        );
    }
}
