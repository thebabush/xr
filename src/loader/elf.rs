use super::{alloc_bss, ParseResult, SegData, Segment, Symbol};
use crate::loader::{Arch, DecodeMode, RelocPointer};
use crate::va::Va;
use anyhow::Result;
use rustc_hash::FxHashSet;

/// Vote over the first [`PROBE_WORDS`] 4-byte-aligned words of an executable
/// section to decide between ARM32 and Thumb mode.
///
/// ## Why voting works
///
/// ARM32 instructions encode a condition code in bits\[31:28\].  The "always"
/// condition (AL = `0xE`) covers virtually every unconditional instruction, so
/// a typical ARM32 function prologue has ≥ 70 % of its words with top nibble
/// `0xE`.  Thumb code read as 32-bit LE words has top nibble `0xE` only for
/// 16-bit `B T2` instructions sitting in the high halfword of an aligned pair —
/// empirically ≤ 6 % of all Thumb words in our test suite.
///
/// ## Why we cap at 16 words (64 bytes)
///
/// ARM32 PIC armel shared libraries embed literal-pool data *between* functions
/// — GOT offsets, AES S-boxes, lookup tables — all with uniformly distributed
/// nibbles.  Sampling beyond the first prologue dilutes the ARM32 signal until
/// it falls to Thumb-like levels:
///
/// | section                | n=16 | n=64 | n=256 |
/// |------------------------|------|------|-------|
/// | ssl-rand `.text` (A32) | 0.81 | 0.34 | 0.12  |
/// | armhf `.text`  (Thumb) | 0.06 | 0.17 | 0.12  |
///
/// At n=16, all A32 sections in our corpus score ≥ 0.73 and all Thumb sections
/// score ≤ 0.06 — a comfortable margin for a 50 % majority threshold.
///
/// Returns `Arm32` if strictly more than half of the sampled words have top
/// nibble `0xE`; `Thumb` otherwise.
const PROBE_WORDS: usize = 16;

fn probe_arm32_section_mode(file: &[u8], offset: usize, size: usize) -> DecodeMode {
    if size < 4 || offset + 4 > file.len() {
        return DecodeMode::Thumb;
    }
    let available = size.min(file.len().saturating_sub(offset));
    let n = (available / 4).min(PROBE_WORDS);

    let arm32_votes = (0..n)
        .filter(|&i| {
            let w = u32::from_le_bytes(
                file[offset + i * 4..offset + i * 4 + 4]
                    .try_into()
                    .unwrap(),
            );
            w >> 28 == 0xE
        })
        .count();

    // Strict majority: more than half of the sampled words carry the ARM32
    // "always" condition in their top nibble.
    if arm32_votes * 2 > n {
        DecodeMode::Arm32
    } else {
        DecodeMode::Thumb
    }
}

/// Default PIE base for ET_DYN ELF binaries whose lowest PT_LOAD has `p_vaddr == 0`.
/// Matches the traditional Linux x86-64 / AArch64 PIE base and IDA default.
const DEFAULT_PIE_BASE: u64 = 0x0040_0000;

/// # Safety (internal)
///
/// `bytes` must remain valid for the lifetime of any `Segment` in the result.
/// Guaranteed by `LoadedBinary` field-ordering invariant.
pub(super) fn parse_elf(
    bytes: &[u8],
    elf: &goblin::elf::Elf,
    bss_bufs: &mut Vec<Box<[u8]>>,
    base_override: Option<u64>,
) -> Result<ParseResult> {
    let arch = match elf.header.e_machine {
        goblin::elf::header::EM_X86_64 => Arch::X86_64,
        goblin::elf::header::EM_AARCH64 => Arch::Arm64,
        goblin::elf::header::EM_386 => Arch::X86,
        goblin::elf::header::EM_ARM => Arch::Arm32,
        m => {
            eprintln!("warning: unknown ELF e_machine {m:#x}, treating as unknown");
            Arch::Unknown
        }
    };

    // Sections that should NOT be byte-scanned for pointers.
    const NO_SCAN_SECTIONS: &[&str] = &[".data.rel.ro", ".data.rel.ro.local"];

    // Sections that are NOT machine code, even when they appear in an exec PT_LOAD.
    const NON_CODE_SECTIONS: &[&str] = &[
        ".rodata",
        ".rodata1",
        ".eh_frame_hdr",
        ".eh_frame",
        ".gcc_except_table",
        ".note.gnu.build-id",
        ".note.ABI-tag",
    ];

    struct SectionInfo {
        va: u64,
        end: u64,
        file_offset: usize,
        file_size: usize,
        name: String,
        byte_scannable: bool,
        is_code: bool,
        /// Per-section decode mode (Thumb vs ARM32 for ARM32 ELF; Default otherwise).
        mode: DecodeMode,
    }
    let mut section_infos: Vec<SectionInfo> = Vec::new();
    for sh in &elf.section_headers {
        use goblin::elf::section_header::*;
        if sh.sh_type == SHT_NULL || sh.sh_type == SHT_NOBITS || sh.sh_addr == 0 || sh.sh_size == 0
        {
            continue;
        }
        if let Some(name) = elf.shdr_strtab.get_at(sh.sh_name) {
            section_infos.push(SectionInfo {
                va: sh.sh_addr,
                end: sh.sh_addr + sh.sh_size,
                file_offset: sh.sh_offset as usize,
                file_size: sh.sh_size as usize,
                name: name.to_string(),
                byte_scannable: !NO_SCAN_SECTIONS.contains(&name),
                is_code: !NON_CODE_SECTIONS.contains(&name),
                mode: DecodeMode::Default, // filled in below for ARM32
            });
        }
    }
    section_infos.sort_by_key(|s| s.va);

    // For ARM32 ELF, assign per-section Thumb vs ARM32 decode mode.
    // For all other architectures keep DecodeMode::Default.
    let default_mode = if arch == Arch::Arm32 {
        // Collect ELF mapping symbols ($t = Thumb, $a = ARM32).
        // These are local STT_NOTYPE symbols present in non-stripped binaries.
        let mut map_syms: Vec<(u64, DecodeMode)> = elf
            .syms
            .iter()
            .filter_map(|sym| {
                elf.strtab.get_at(sym.st_name).and_then(|name| {
                    // Match bare "$t" / "$a" and indexed variants like "$t.0".
                    if name == "$t" || name.starts_with("$t.") {
                        Some((sym.st_value, DecodeMode::Thumb))
                    } else if name == "$a" || name.starts_with("$a.") {
                        Some((sym.st_value, DecodeMode::Arm32))
                    } else {
                        None
                    }
                })
            })
            .collect();

        if !map_syms.is_empty() {
            map_syms.sort_unstable_by_key(|e| e.0);
            // Assign each section's mode from the last mapping symbol at or
            // before its start address.
            for si in &mut section_infos {
                let idx = map_syms.partition_point(|e| e.0 <= si.va);
                si.mode = if idx > 0 {
                    map_syms[idx - 1].1
                } else {
                    DecodeMode::Thumb // default if no mapping symbol precedes
                };
            }
            DecodeMode::Thumb // used for unsectioned LOAD fallback
        } else {
            // Stripped binary: majority-vote probe over the first 16 words of
            // each section (see `probe_arm32_section_mode`).
            // Correctly classifies:
            //   armel (soft-float):  .text / .init / .plt → all ARM32
            //   armhf (hard-float):  .init / .plt → ARM32, .text → Thumb
            for si in &mut section_infos {
                si.mode = probe_arm32_section_mode(bytes, si.file_offset, si.file_size);
            }
            DecodeMode::Thumb // fallback for unsectioned LOAD regions
        }
    } else {
        for si in &mut section_infos {
            si.mode = DecodeMode::Default;
        }
        DecodeMode::Default
    };

    use goblin::elf::header::ET_DYN;
    use goblin::elf::program_header::PT_LOAD;
    let pie_base: u64 = if elf.header.e_type == ET_DYN {
        let min_load_va = elf
            .program_headers
            .iter()
            .filter(|ph| ph.p_type == PT_LOAD)
            .map(|ph| ph.p_vaddr)
            .min()
            .unwrap_or(1);
        if min_load_va == 0 {
            base_override.unwrap_or(DEFAULT_PIE_BASE)
        } else {
            0
        }
    } else {
        0
    };

    if pie_base != 0 {
        for si in &mut section_infos {
            si.va += pie_base;
            si.end += pie_base;
        }
    }

    let mut segments = Vec::new();
    for ph in &elf.program_headers {
        use goblin::elf::program_header::*;
        if ph.p_type != PT_LOAD {
            continue;
        }
        let exec = ph.p_flags & PF_X != 0;
        let read = ph.p_flags & PF_R != 0;
        let write = ph.p_flags & PF_W != 0;
        let ph_va = ph.p_vaddr + pie_base;

        if exec && !section_infos.is_empty() {
            let ph_va_start = ph_va;
            let ph_va_end = ph_va + ph.p_memsz;
            let secs: Vec<&SectionInfo> = section_infos
                .iter()
                .filter(|s| s.va >= ph_va_start && s.end <= ph_va_end)
                .collect();
            if !secs.is_empty() {
                for sec in &secs {
                    if sec.file_offset + sec.file_size > bytes.len() {
                        eprintln!(
                            "warning: ELF section '{}' at offset {:#x}+{:#x} exceeds file size, skipping",
                            sec.name, sec.file_offset, sec.file_size
                        );
                        continue;
                    }
                    let data = &bytes[sec.file_offset..sec.file_offset + sec.file_size];
                    // Safety: `bytes` is the mmap kept alive by LoadedBinary.
                    segments.push(Segment {
                        va: Va::new(sec.va),
                        data: unsafe { SegData::new(data) },
                        executable: sec.is_code,
                        readable: read,
                        writable: write,
                        byte_scannable: sec.byte_scannable,
                        mode: sec.mode,
                        name: sec.name.clone(),
                    });
                }
                let last_end = secs.iter().map(|s| s.end).max().unwrap_or(ph_va_end);
                if last_end < ph_va_end {
                    let bss_sz = (ph_va_end - last_end) as usize;
                    let bss_data = alloc_bss(bss_sz, bss_bufs);
                    segments.push(Segment {
                        va: Va::new(last_end),
                        data: bss_data,
                        executable: false,
                        readable: read,
                        writable: write,
                        byte_scannable: false,
                        mode: default_mode,
                        name: format!("BSS[{:#x}]", last_end),
                    });
                }
                continue;
            }
        }

        if ph.p_filesz > 0 {
            let offset = ph.p_offset as usize;
            let filesz = ph.p_filesz as usize;
            if offset + filesz <= bytes.len() {
                let data = &bytes[offset..offset + filesz];
                let ph_va_end = ph_va + ph.p_filesz;
                let byte_scannable = !section_infos
                    .iter()
                    .any(|s| !s.byte_scannable && s.va < ph_va_end && s.end > ph_va);
                // Safety: `bytes` is the mmap kept alive by LoadedBinary.
                segments.push(Segment {
                    va: Va::new(ph_va),
                    data: unsafe { SegData::new(data) },
                    executable: exec,
                    readable: read,
                    writable: write,
                    byte_scannable,
                    mode: default_mode,
                    name: format!("LOAD[{:#x}]", ph_va),
                });
            }
        }

        if ph.p_memsz > ph.p_filesz {
            let bss_va = ph_va + ph.p_filesz;
            let bss_sz = (ph.p_memsz - ph.p_filesz) as usize;
            let bss_data = alloc_bss(bss_sz, bss_bufs);
            segments.push(Segment {
                va: Va::new(bss_va),
                data: bss_data,
                executable: false,
                readable: read,
                writable: write,
                byte_scannable: false,
                mode: default_mode,
                name: format!("BSS[{:#x}]", bss_va),
            });
        }
    }

    let entry_points = if elf.entry != 0 {
        vec![Va::new(elf.entry + pie_base)]
    } else {
        vec![]
    };
    let mut symbols = Vec::new();
    for sym in &elf.syms {
        if sym.st_value == 0 {
            continue;
        }
        if let Some(name) = elf.strtab.get_at(sym.st_name) {
            if !name.is_empty() {
                let va = Va::new((sym.st_value & !1) + pie_base);
                symbols.push(Symbol {
                    name: name.to_string(),
                    va,
                });
            }
        }
    }

    let got_slots = build_elf_got_slots(elf, pie_base);
    let reloc_pointers = build_elf_reloc_pointers(elf, bytes, pie_base, &segments);

    Ok(ParseResult {
        arch,
        segments,
        entry_points,
        symbols,
        pie_base,
        got_slots,
        reloc_pointers,
    })
}

fn build_elf_got_slots(elf: &goblin::elf::Elf, pie_base: u64) -> FxHashSet<Va> {
    const R_X86_64_GLOB_DAT: u32 = 6;
    const R_X86_64_JUMP_SLOT: u32 = 7;
    const R_AARCH64_GLOB_DAT: u32 = 1025;
    const R_AARCH64_JUMP_SLOT: u32 = 1026;
    const R_ARM_GLOB_DAT: u32 = 21;
    const R_ARM_JUMP_SLOT: u32 = 22;

    let is_got_reloc = |r_type: u32| {
        matches!(
            r_type,
            R_X86_64_GLOB_DAT
                | R_X86_64_JUMP_SLOT
                | R_AARCH64_GLOB_DAT
                | R_AARCH64_JUMP_SLOT
                | R_ARM_GLOB_DAT
                | R_ARM_JUMP_SLOT
        )
    };

    elf.dynrelas
        .iter()
        .chain(elf.dynrels.iter())
        .chain(elf.pltrelocs.iter())
        .filter(|rel| is_got_reloc(rel.r_type) && rel.r_sym != 0)
        .map(|rel| Va::new(rel.r_offset + pie_base))
        .collect()
}

fn build_elf_reloc_pointers(
    elf: &goblin::elf::Elf,
    bytes: &[u8],
    pie_base: u64,
    segments: &[Segment],
) -> Vec<RelocPointer> {
    use goblin::elf::program_header::PT_LOAD;
    use goblin::elf::section_header::SHN_UNDEF;

    // x86-64
    const R_X86_64_RELATIVE: u32 = 8;
    const R_X86_64_64: u32 = 1;
    // AArch64
    const R_AARCH64_RELATIVE: u32 = 1027;
    const R_AARCH64_ABS64: u32 = 257;
    // ARM32 — REL format: addend lives in the word at *r_offset in the file
    const R_ARM_RELATIVE: u32 = 23;
    const R_ARM_ABS32: u32 = 2;

    // Translate an ELF VMA to its raw file byte offset via PT_LOAD headers.
    // Used to read the implicit in-place addend for ARM32 REL relocations.
    let vma_to_file = |vma: u64| -> Option<usize> {
        elf.program_headers.iter().find_map(|ph| {
            if ph.p_type == PT_LOAD
                && vma >= ph.p_vaddr
                && vma < ph.p_vaddr + ph.p_filesz
            {
                Some((vma - ph.p_vaddr + ph.p_offset) as usize)
            } else {
                None
            }
        })
    };

    // Read a little-endian u32 from the file at the given ELF VMA.
    let read_u32_at = |vma: u64| -> Option<u32> {
        let off = vma_to_file(vma)?;
        bytes
            .get(off..off + 4)
            .and_then(|s| s.try_into().ok())
            .map(u32::from_le_bytes)
    };

    // Simple linear membership test over the (small) segment list.
    // Called once per relocation entry during load; O(n_segments) is fine.
    let is_mapped = |va: Va| segments.iter().any(|s| s.contains(va));
    let mut result = Vec::new();

    for rel in elf.dynrelas.iter().chain(elf.dynrels.iter()) {
        let from = rel.r_offset + pie_base;
        let r_type = rel.r_type;

        if r_type == R_X86_64_RELATIVE || r_type == R_AARCH64_RELATIVE {
            // RELA: explicit addend encodes the pre-link target.
            let target = Va::new((rel.r_addend.unwrap_or(0) as u64).wrapping_add(pie_base));
            if is_mapped(target) {
                result.push(RelocPointer { from: Va::new(from), to: target });
            }
        } else if r_type == R_ARM_RELATIVE {
            // REL: the word at *place in the file IS the pre-link target VMA.
            if let Some(addend) = read_u32_at(rel.r_offset) {
                let target = Va::new((addend as u64).wrapping_add(pie_base));
                if is_mapped(target) {
                    result.push(RelocPointer { from: Va::new(from), to: target });
                }
            }
        } else if (r_type == R_X86_64_64 || r_type == R_AARCH64_ABS64) && rel.r_sym != 0 {
            // RELA with symbol: target = sym.st_value + addend.
            let sym = elf
                .dynsyms
                .get(rel.r_sym)
                .or_else(|| elf.syms.get(rel.r_sym));
            if let Some(sym) = sym {
                if sym.st_shndx != SHN_UNDEF as usize && sym.st_value != 0 {
                    let target = Va::new(
                        sym.st_value
                            .wrapping_add(pie_base)
                            .wrapping_add(rel.r_addend.unwrap_or(0) as u64),
                    );
                    if is_mapped(target) {
                        result.push(RelocPointer { from: Va::new(from), to: target });
                    }
                }
            }
        } else if r_type == R_ARM_ABS32 && rel.r_sym != 0 {
            // REL with symbol: target = sym.st_value + implicit_addend.
            let sym = elf
                .dynsyms
                .get(rel.r_sym)
                .or_else(|| elf.syms.get(rel.r_sym));
            if let Some(sym) = sym {
                if sym.st_shndx != SHN_UNDEF as usize && sym.st_value != 0 {
                    let implicit_addend = read_u32_at(rel.r_offset).unwrap_or(0);
                    let target = Va::new(
                        sym.st_value
                            .wrapping_add(pie_base)
                            .wrapping_add(implicit_addend as u64),
                    );
                    if is_mapped(target) {
                        result.push(RelocPointer { from: Va::new(from), to: target });
                    }
                }
            }
        }
    }

    result
}
