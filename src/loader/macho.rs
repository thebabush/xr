use super::{alloc_bss, ParseResult, SegData, Segment, Symbol, VaRangeSet};
use crate::loader::{Arch, DecodeMode, RelocPointer};
use crate::va::Va;
use anyhow::Result;
use rustc_hash::FxHashSet;

/// # Safety (internal)
///
/// `bytes` must remain valid for the lifetime of any `Segment` in the result.
/// Guaranteed by `LoadedBinary` field-ordering invariant.
pub(super) fn parse_macho(
    bytes: &[u8],
    macho: &goblin::mach::MachO,
    bss_bufs: &mut Vec<Box<[u8]>>,
) -> Result<ParseResult> {
    use goblin::mach::constants::cputype::*;
    let arch = match macho.header.cputype() {
        CPU_TYPE_X86_64 => Arch::X86_64,
        CPU_TYPE_ARM64 => Arch::Arm64,
        CPU_TYPE_X86 => Arch::X86,
        CPU_TYPE_ARM => Arch::Arm32,
        m => {
            eprintln!("warning: unknown Mach-O cputype {m:#x}, treating as unknown");
            Arch::Unknown
        }
    };

    const NO_SCAN_SECTIONS: &[&str] = &[
        "__DATA_CONST,__got",
        "__DATA,__got",
        "__DATA_CONST,__auth_got",
        "__DATA,__la_symbol_ptr",
        "__DATA,__nl_symbol_ptr",
        "__DATA_CONST,__cfstring",
        "__DATA,__cfstring",
    ];

    let mut segments = Vec::new();
    for seg in macho.segments.iter() {
        let seg_name = seg.name().unwrap_or("?").to_string();
        for section in seg.sections().map_err(|e| anyhow::anyhow!("{e}"))? {
            let (sect, data) = section;
            let sect_name = sect.name().unwrap_or("?").to_string();
            let size = sect.size as usize;
            if size == 0 {
                continue;
            }
            let exec = seg.initprot & goblin::mach::constants::VM_PROT_EXECUTE != 0;
            let read = seg.initprot & goblin::mach::constants::VM_PROT_READ != 0;
            let write = seg.initprot & goblin::mach::constants::VM_PROT_WRITE != 0;
            let full_name = format!("{seg_name},{sect_name}");
            let byte_scannable = !exec && !NO_SCAN_SECTIONS.iter().any(|&n| n == full_name);

            let section_data = if data.is_empty() {
                alloc_bss(size, bss_bufs)
            } else {
                let file_offset = sect.offset as usize;
                if file_offset + size > bytes.len() {
                    continue;
                }
                // Safety: `bytes` is the mmap kept alive by LoadedBinary.
                unsafe { SegData::new(&bytes[file_offset..file_offset + size]) }
            };

            segments.push(Segment {
                va: Va::new(sect.addr),
                data: section_data,
                executable: exec,
                readable: read,
                writable: write,
                mode: DecodeMode::Default,
                name: full_name,
                byte_scannable,
            });
        }
    }

    let entry_points = vec![Va::new(macho.entry)];
    let mut symbols = Vec::new();
    if let Some(syms) = macho.symbols.as_ref() {
        for (name, nlist) in syms.iter().flatten() {
            if !name.is_empty() && nlist.n_value != 0 {
                symbols.push(Symbol {
                    name: name.to_string(),
                    va: Va::new(nlist.n_value),
                });
            }
        }
    }

    let preferred_base = macho
        .segments
        .iter()
        .find(|s| s.name().is_ok_and(|n| n == "__TEXT"))
        .map_or(0, |s| s.vmaddr);
    let reloc_pointers = build_macho_fixup_pointers(bytes, macho, preferred_base, &segments);

    Ok(ParseResult {
        arch,
        segments,
        entry_points,
        symbols,
        pie_base: 0,
        got_slots: FxHashSet::default(),
        reloc_pointers,
    })
}

// ── Chained fixup pointer formats ─────────────────────────────────────────────
//
// Each format defines how to decode a 64-bit chained fixup entry in a Mach-O
// binary.  The outer parsing (LC_DYLD_CHAINED_FIXUPS → segments → pages →
// chain walking) is format-independent; only the per-entry decode differs.
//
// Bit layouts (from Apple's `mach-o/fixup-chains.h`):
//
//   Formats 2 & 6 (DYLD_CHAINED_PTR_64 / _OFFSET):
//     [63]    bind       [62:51] next (12 bits, stride 4)
//     [50:44] reserved   [43:36] high8   [35:0] target (36 bits)
//
//   Formats 1, 9, 12 (ARM64E family):
//     [63]    auth       [62]    bind    [61:51] next (11 bits, stride 8)
//     Non-auth rebase:   [50:43] high8   [42:0]  target (43 bits)
//     Auth rebase:       [50:49] key     [48] addrDiv  [47:32] diversity
//                        [31:0]  target (32 bits)

/// Recognized chained-fixup pointer formats.
#[derive(Clone, Copy)]
enum ChainedPtrFormat {
    /// Format 2: target is absolute vmaddr.  Stride 4.
    Ptr64,
    /// Format 6: target is offset from preferred base.  Stride 4.
    Ptr64Offset,
    /// Format 1: ARM64E, target is absolute vmaddr.  Stride 8.
    Arm64e,
    /// Format 9: ARM64E userland, target is offset from preferred base.  Stride 8.
    Arm64eUserland,
    /// Format 12: ARM64E userland 24-bit bind, target is offset.  Stride 8.
    Arm64eUserland24,
}

impl ChainedPtrFormat {
    /// Map the raw `pointer_format` field to a known format, or `None`.
    fn from_raw(raw: u16) -> Option<Self> {
        match raw {
            2 => Some(Self::Ptr64),
            6 => Some(Self::Ptr64Offset),
            1 => Some(Self::Arm64e),
            9 => Some(Self::Arm64eUserland),
            12 => Some(Self::Arm64eUserland24),
            _ => None,
        }
    }

    /// Byte stride per "next" unit (4 for generic 64-bit, 8 for ARM64E).
    fn stride(self) -> usize {
        match self {
            Self::Ptr64 | Self::Ptr64Offset => 4,
            Self::Arm64e | Self::Arm64eUserland | Self::Arm64eUserland24 => 8,
        }
    }

    /// Extract the `next` delta (in stride units) from a raw entry.
    /// Returns 0 when this is the last entry in the chain.
    fn next_delta(self, val: u64) -> usize {
        match self {
            // Formats 2/6: next = bits [62:51], 12 bits
            Self::Ptr64 | Self::Ptr64Offset => ((val >> 51) & 0xFFF) as usize,
            // ARM64E: next = bits [61:51], 11 bits
            Self::Arm64e | Self::Arm64eUserland | Self::Arm64eUserland24 => {
                ((val >> 51) & 0x7FF) as usize
            }
        }
    }

    /// Decode a rebase entry into a target VA, or `None` if this is a bind.
    fn decode_rebase(self, val: u64, preferred_base: u64) -> Option<Va> {
        match self {
            Self::Ptr64 => {
                // bind = bit 63
                if (val >> 63) & 1 != 0 {
                    return None;
                }
                let target = val & 0xF_FFFF_FFFF; // bits [35:0]
                let high8 = (val >> 36) & 0xFF;
                Some(Va::new((high8 << 56) | target))
            }
            Self::Ptr64Offset => {
                if (val >> 63) & 1 != 0 {
                    return None;
                }
                let target_off = val & 0xF_FFFF_FFFF;
                let high8 = (val >> 36) & 0xFF;
                Some(Va::new((high8 << 56) | (preferred_base + target_off)))
            }
            Self::Arm64e => {
                // auth = bit 63, bind = bit 62
                let auth = (val >> 63) & 1 != 0;
                let bind = (val >> 62) & 1 != 0;
                if bind {
                    return None;
                }
                if auth {
                    // Auth rebase: target = bits [31:0], absolute vmaddr (32-bit)
                    let target = val & 0xFFFF_FFFF;
                    Some(Va::new(target))
                } else {
                    // Rebase: target = bits [42:0], high8 = bits [50:43]
                    let target = val & 0x7FF_FFFF_FFFF; // 43 bits
                    let high8 = (val >> 43) & 0xFF;
                    Some(Va::new((high8 << 56) | target))
                }
            }
            Self::Arm64eUserland | Self::Arm64eUserland24 => {
                let auth = (val >> 63) & 1 != 0;
                let bind = (val >> 62) & 1 != 0;
                if bind {
                    return None;
                }
                if auth {
                    // Auth rebase: target = bits [31:0], offset from preferred base
                    let target_off = val & 0xFFFF_FFFF;
                    Some(Va::new(preferred_base + target_off))
                } else {
                    // Rebase: target = bits [42:0], high8 = bits [50:43]
                    let target_off = val & 0x7FF_FFFF_FFFF;
                    let high8 = (val >> 43) & 0xFF;
                    Some(Va::new((high8 << 56) | (preferred_base + target_off)))
                }
            }
        }
    }
}

/// Walk LC_DYLD_CHAINED_FIXUPS to extract rebase pointers.
///
/// Supported pointer formats:
///   - `DYLD_CHAINED_PTR_64` (2) — x86_64, target is absolute vmaddr
///   - `DYLD_CHAINED_PTR_64_OFFSET` (6) — arm64, target is offset from base
///   - `DYLD_CHAINED_PTR_ARM64E` (1) — arm64e, target is absolute vmaddr
///   - `DYLD_CHAINED_PTR_ARM64E_USERLAND` (9) — arm64e, target is offset
///   - `DYLD_CHAINED_PTR_ARM64E_USERLAND24` (12) — arm64e, 24-bit bind ordinal
///
/// Each rebase entry encodes a pointer-sized slot whose value (after relocation)
/// resolves to a virtual address.  These are emitted as `DataPointer` xrefs.
/// Bind entries (imports from other dylibs) are skipped.
fn build_macho_fixup_pointers(
    bytes: &[u8],
    macho: &goblin::mach::MachO,
    preferred_base: u64,
    segments: &[Segment],
) -> Vec<RelocPointer> {
    use goblin::mach::load_command::CommandVariant;

    let seg_set = VaRangeSet::build(segments);

    // Build a map from segment index → segment vmaddr so we can convert
    // chain offsets (which are file offsets within a segment) to VAs.
    // The chained fixups header lists segments by index matching the Mach-O
    // LC_SEGMENT_64 order.  `segment_offset` in each starts-in-segment
    // record is the segment's file offset, so:
    //   slot_va = seg_vmaddr + (chain_off - segment_offset)
    let seg_vmaddrs: Vec<u64> = macho.segments.iter().map(|s| s.vmaddr).collect();

    let Some(lc) = macho.load_commands.iter().find_map(|lc| {
        if let CommandVariant::DyldChainedFixups(ref cmd) = lc.command {
            Some(cmd)
        } else {
            None
        }
    }) else {
        return Vec::new();
    };

    let data_off = lc.dataoff as usize;
    let data_size = lc.datasize as usize;
    if data_off + data_size > bytes.len() || data_size < 28 {
        return Vec::new();
    }

    let starts_offset = u32::from_le_bytes(
        bytes[data_off + 4..data_off + 8]
            .try_into()
            .expect("checked above"),
    ) as usize;

    let si_off = data_off + starts_offset;
    if si_off + 4 > bytes.len() {
        return Vec::new();
    }
    let seg_count =
        u32::from_le_bytes(bytes[si_off..si_off + 4].try_into().expect("checked above")) as usize;

    let mut result = Vec::new();

    for seg_idx in 0..seg_count {
        let off_off = si_off + 4 + seg_idx * 4;
        if off_off + 4 > bytes.len() {
            break;
        }
        let seg_info_off = u32::from_le_bytes(
            bytes[off_off..off_off + 4]
                .try_into()
                .expect("checked above"),
        ) as usize;
        if seg_info_off == 0 {
            continue;
        }

        let ss_off = si_off + seg_info_off;
        if ss_off + 22 > bytes.len() {
            continue;
        }

        let page_size =
            u16::from_le_bytes(bytes[ss_off + 4..ss_off + 6].try_into().expect("checked")) as usize;
        let pointer_format =
            u16::from_le_bytes(bytes[ss_off + 6..ss_off + 8].try_into().expect("checked"));
        let segment_offset =
            u64::from_le_bytes(bytes[ss_off + 8..ss_off + 16].try_into().expect("checked"));
        let page_count =
            u16::from_le_bytes(bytes[ss_off + 20..ss_off + 22].try_into().expect("checked"))
                as usize;

        let fmt = match ChainedPtrFormat::from_raw(pointer_format) {
            Some(f) => f,
            None => continue,
        };

        // Resolve this segment's vmaddr for file-offset → VA conversion.
        // `segment_offset` is a file offset; `seg_vmaddr` is where the
        // segment is mapped.  slot_va = seg_vmaddr + (chain_off - segment_offset).
        let seg_vmaddr = seg_vmaddrs.get(seg_idx).copied().unwrap_or(0);

        for p_idx in 0..page_count {
            let ps_off = ss_off + 22 + p_idx * 2;
            if ps_off + 2 > bytes.len() {
                break;
            }
            let page_start =
                u16::from_le_bytes(bytes[ps_off..ps_off + 2].try_into().expect("checked"));
            const DYLD_CHAINED_PTR_START_NONE: u16 = 0xFFFF;
            if page_start == DYLD_CHAINED_PTR_START_NONE {
                continue;
            }

            let mut chain_off = segment_offset as usize + p_idx * page_size + page_start as usize;

            loop {
                if chain_off + 8 > bytes.len() {
                    break;
                }
                let val = u64::from_le_bytes(
                    bytes[chain_off..chain_off + 8]
                        .try_into()
                        .expect("checked above"),
                );

                if let Some(target_va) = fmt.decode_rebase(val, preferred_base) {
                    // Convert file offset to VA:
                    // chain_off is a file offset within this segment;
                    // segment_offset is the segment's file offset.
                    let offset_in_seg = chain_off as u64 - segment_offset;
                    let slot_va = Va::new(seg_vmaddr + offset_in_seg);
                    if seg_set.contains(target_va) {
                        result.push(RelocPointer {
                            from: slot_va,
                            to: target_va,
                        });
                    }
                }

                let next = fmt.next_delta(val);
                if next == 0 {
                    break;
                }
                chain_off += next * fmt.stride();
            }
        }
    }

    result
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: u64 = 0x1_0000_0000; // typical Mach-O __TEXT vmaddr

    // ── Format helpers ────────────────────────────────────────────────────

    /// Build a DYLD_CHAINED_PTR_64 (format 2) rebase entry.
    /// Layout: [63] bind=0  [62:51] next  [43:36] high8  [35:0] target (abs vmaddr)
    fn ptr64_rebase(target: u64, high8: u64, next: u64) -> u64 {
        (target & 0xF_FFFF_FFFF) | ((high8 & 0xFF) << 36) | ((next & 0xFFF) << 51)
    }

    /// Build a DYLD_CHAINED_PTR_64_OFFSET (format 6) rebase entry.
    /// Same bit layout as format 2, but target is an offset from preferred base.
    fn ptr64_offset_rebase(target_off: u64, high8: u64, next: u64) -> u64 {
        ptr64_rebase(target_off, high8, next) // same encoding
    }

    /// Build a DYLD_CHAINED_PTR_64 bind entry (bit 63 set).
    fn ptr64_bind(next: u64) -> u64 {
        (1u64 << 63) | ((next & 0xFFF) << 51)
    }

    /// ARM64E non-auth rebase: auth=0 bind=0.
    /// [42:0] target  [50:43] high8  [61:51] next
    fn arm64e_rebase(target: u64, high8: u64, next: u64) -> u64 {
        (target & 0x7FF_FFFF_FFFF) | ((high8 & 0xFF) << 43) | ((next & 0x7FF) << 51)
    }

    /// ARM64E auth rebase: auth=1 bind=0.
    /// [31:0] target  [47:32] diversity  [48] addrDiv  [50:49] key  [61:51] next
    fn arm64e_auth_rebase(target: u64, next: u64) -> u64 {
        (target & 0xFFFF_FFFF) | ((next & 0x7FF) << 51) | (1u64 << 63)
    }

    /// ARM64E bind: bind=1 (bit 62).
    fn arm64e_bind(next: u64) -> u64 {
        (1u64 << 62) | ((next & 0x7FF) << 51)
    }

    /// ARM64E auth bind: auth=1 bind=1.
    fn arm64e_auth_bind(next: u64) -> u64 {
        (1u64 << 63) | (1u64 << 62) | ((next & 0x7FF) << 51)
    }

    // ── Format 2: DYLD_CHAINED_PTR_64 ────────────────────────────────────

    #[test]
    fn test_ptr64_rebase_absolute() {
        let fmt = ChainedPtrFormat::Ptr64;
        let val = ptr64_rebase(0x1_0000_1000, 0, 5);
        let result = fmt.decode_rebase(val, BASE);
        assert_eq!(result, Some(Va::new(0x1_0000_1000)));
        assert_eq!(fmt.next_delta(val), 5);
        assert_eq!(fmt.stride(), 4);
    }

    #[test]
    fn test_ptr64_rebase_with_high8() {
        let fmt = ChainedPtrFormat::Ptr64;
        let val = ptr64_rebase(0x1_0000_1000, 0x80, 0);
        let result = fmt.decode_rebase(val, BASE).unwrap();
        assert_eq!(result.raw() >> 56, 0x80);
        assert_eq!(result.raw() & 0x00FF_FFFF_FFFF_FFFF, 0x1_0000_1000);
    }

    #[test]
    fn test_ptr64_bind_skipped() {
        let fmt = ChainedPtrFormat::Ptr64;
        let val = ptr64_bind(3);
        assert_eq!(fmt.decode_rebase(val, BASE), None);
        assert_eq!(fmt.next_delta(val), 3);
    }

    // ── Format 6: DYLD_CHAINED_PTR_64_OFFSET ─────────────────────────────

    #[test]
    fn test_ptr64_offset_rebase() {
        let fmt = ChainedPtrFormat::Ptr64Offset;
        let offset = 0x1000u64;
        let val = ptr64_offset_rebase(offset, 0, 2);
        let result = fmt.decode_rebase(val, BASE).unwrap();
        assert_eq!(result, Va::new(BASE + offset));
        assert_eq!(fmt.next_delta(val), 2);
    }

    // ── Format 1: DYLD_CHAINED_PTR_ARM64E ────────────────────────────────

    #[test]
    fn test_arm64e_rebase_absolute() {
        let fmt = ChainedPtrFormat::Arm64e;
        let target = 0x1_0000_2000u64;
        let val = arm64e_rebase(target, 0, 7);
        let result = fmt.decode_rebase(val, BASE).unwrap();
        assert_eq!(result, Va::new(target));
        assert_eq!(fmt.next_delta(val), 7);
        assert_eq!(fmt.stride(), 8);
    }

    #[test]
    fn test_arm64e_rebase_with_high8() {
        let fmt = ChainedPtrFormat::Arm64e;
        let val = arm64e_rebase(0x1_0000_2000, 0xAB, 0);
        let result = fmt.decode_rebase(val, BASE).unwrap();
        assert_eq!(result.raw() >> 56, 0xAB);
        assert_eq!(result.raw() & 0x00FF_FFFF_FFFF_FFFF, 0x1_0000_2000);
    }

    #[test]
    fn test_arm64e_auth_rebase_absolute() {
        let fmt = ChainedPtrFormat::Arm64e;
        // Auth rebase: 32-bit absolute target
        let val = arm64e_auth_rebase(0x1234_5678, 4);
        let result = fmt.decode_rebase(val, BASE).unwrap();
        assert_eq!(result, Va::new(0x1234_5678));
        assert_eq!(fmt.next_delta(val), 4);
    }

    #[test]
    fn test_arm64e_bind_skipped() {
        let fmt = ChainedPtrFormat::Arm64e;
        assert_eq!(fmt.decode_rebase(arm64e_bind(1), BASE), None);
    }

    #[test]
    fn test_arm64e_auth_bind_skipped() {
        let fmt = ChainedPtrFormat::Arm64e;
        assert_eq!(fmt.decode_rebase(arm64e_auth_bind(2), BASE), None);
    }

    // ── Format 9: DYLD_CHAINED_PTR_ARM64E_USERLAND ───────────────────────

    #[test]
    fn test_arm64e_userland_rebase_offset() {
        let fmt = ChainedPtrFormat::Arm64eUserland;
        let offset = 0x5000u64;
        let val = arm64e_rebase(offset, 0, 3);
        let result = fmt.decode_rebase(val, BASE).unwrap();
        assert_eq!(result, Va::new(BASE + offset));
    }

    #[test]
    fn test_arm64e_userland_auth_rebase_offset() {
        let fmt = ChainedPtrFormat::Arm64eUserland;
        let offset = 0xABCD_0000u64;
        let val = arm64e_auth_rebase(offset, 1);
        let result = fmt.decode_rebase(val, BASE).unwrap();
        assert_eq!(result, Va::new(BASE + offset));
    }

    #[test]
    fn test_arm64e_userland_bind_skipped() {
        let fmt = ChainedPtrFormat::Arm64eUserland;
        assert_eq!(fmt.decode_rebase(arm64e_bind(0), BASE), None);
        assert_eq!(fmt.decode_rebase(arm64e_auth_bind(0), BASE), None);
    }

    // ── Format 12: DYLD_CHAINED_PTR_ARM64E_USERLAND24 ────────────────────

    #[test]
    fn test_arm64e_userland24_rebase_same_as_userland() {
        // Rebase encoding is identical to format 9; only bind ordinal width differs.
        let fmt = ChainedPtrFormat::Arm64eUserland24;
        let offset = 0x8000u64;
        let val = arm64e_rebase(offset, 0, 5);
        let result = fmt.decode_rebase(val, BASE).unwrap();
        assert_eq!(result, Va::new(BASE + offset));
        assert_eq!(fmt.stride(), 8);
    }

    // ── from_raw ──────────────────────────────────────────────────────────

    #[test]
    fn test_from_raw_known_formats() {
        assert!(ChainedPtrFormat::from_raw(1).is_some());
        assert!(ChainedPtrFormat::from_raw(2).is_some());
        assert!(ChainedPtrFormat::from_raw(6).is_some());
        assert!(ChainedPtrFormat::from_raw(9).is_some());
        assert!(ChainedPtrFormat::from_raw(12).is_some());
    }

    #[test]
    fn test_from_raw_unknown_formats() {
        for f in [0, 3, 4, 5, 7, 8, 10, 11, 13, 100] {
            assert!(
                ChainedPtrFormat::from_raw(f).is_none(),
                "format {f} should be None"
            );
        }
    }

    // ── next_delta field width ────────────────────────────────────────────

    #[test]
    fn test_next_delta_max_ptr64() {
        let fmt = ChainedPtrFormat::Ptr64;
        // All 12 bits set in the next field
        let val = 0xFFF_u64 << 51;
        assert_eq!(fmt.next_delta(val), 0xFFF);
    }

    #[test]
    fn test_next_delta_max_arm64e() {
        let fmt = ChainedPtrFormat::Arm64e;
        // All 11 bits set in the next field, plus auth+bind bits above
        let val = (0x7FF_u64 << 51) | (1u64 << 63) | (1u64 << 62);
        assert_eq!(fmt.next_delta(val), 0x7FF);
    }
}
