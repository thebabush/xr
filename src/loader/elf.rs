use super::{alloc_bss, ParseResult, SegData, Segment, Symbol, VaRangeSet};
use crate::loader::{Arch, DecodeMode, RelocPointer};
use crate::va::Va;
use anyhow::Result;
use rustc_hash::FxHashSet;

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
            });
        }
    }
    section_infos.sort_by_key(|s| s.va);

    let mode = if arch == Arch::Arm32 {
        DecodeMode::Arm32
    } else {
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
                        mode,
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
                        mode,
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
                    mode,
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
                mode,
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
    let reloc_pointers = build_elf_reloc_pointers(elf, pie_base, &segments);

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

    let is_got_reloc = |r_type: u32| {
        matches!(
            r_type,
            R_X86_64_GLOB_DAT | R_X86_64_JUMP_SLOT | R_AARCH64_GLOB_DAT | R_AARCH64_JUMP_SLOT
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
    pie_base: u64,
    segments: &[Segment],
) -> Vec<RelocPointer> {
    use goblin::elf::section_header::SHN_UNDEF;

    const R_X86_64_RELATIVE: u32 = 8;
    const R_X86_64_64: u32 = 1;
    const R_AARCH64_RELATIVE: u32 = 1027;
    const R_AARCH64_ABS64: u32 = 257;

    let seg_set = VaRangeSet::build(segments);
    let mut result = Vec::new();

    for rel in elf.dynrelas.iter().chain(elf.dynrels.iter()) {
        let from = rel.r_offset + pie_base;
        let r_type = rel.r_type;

        if r_type == R_X86_64_RELATIVE || r_type == R_AARCH64_RELATIVE {
            let target = Va::new((rel.r_addend.unwrap_or(0) as u64).wrapping_add(pie_base));
            if seg_set.contains(target) {
                result.push(RelocPointer {
                    from: Va::new(from),
                    to: target,
                });
            }
        } else if (r_type == R_X86_64_64 || r_type == R_AARCH64_ABS64) && rel.r_sym != 0 {
            let sym = &elf
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
                    if seg_set.contains(target) {
                        result.push(RelocPointer {
                            from: Va::new(from),
                            to: target,
                        });
                    }
                }
            }
        }
    }

    result
}
