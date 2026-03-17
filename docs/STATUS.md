# xr — Status & Architecture

## Goal

Build `xr`: a standalone Rust crate for ultra-fast, parallel cross-reference
extraction from stripped binaries (ELF, Mach-O, PE). Maximise F1 score across
all xref kinds.

---

## Current Scores (Paired depth)

Tested against ground truth on 26 binaries across ELF (x86-64,
AArch64), Mach-O (ARM64), and PE (x86-64, ARM64).

| Category | n | F1 range | Notes |
|----------|----|----------|-------|
| ELF x86-64 | 8 | 0.867–0.969 | Low end: libharlem-shake.so (PLT call FNs). High: curl-amd64 (static, fully resolved) |
| ELF ARM64 | 5 | 0.840–0.952 | Low end: libziggy.so / libssl3 (unresolved ADRP pairs, jump table FNs) |
| Mach-O ARM64 | 1 | 0.982 | Fixup chain parsing recovers most data_ptr |
| PE x86-64 (MinGW/Rust) | 2 | 0.954–0.995 | .pdata + UNWIND_INFO very effective; win32kbase_rs.sys near-perfect |
| PE x86-64 (MSVC) | 8 | 0.561–0.918 | Low end: concrt140.dll (.pdata FPs). 32-bit RVA EH/RTTI invisible to 8-byte scanner |
| PE ARM64 (MSVC) | 2 | 0.623–0.752 | Limited by data_ptr gaps in ADRP-heavy code |
| COFF object (x86-64 / x86-32) | — | — | Format newly supported; no ground-truth benchmark yet |
| ARM32 ELF armel (A32) | 4 | 0.715–0.763 | call/jump prec ≥0.999/0.967; data_ptr prec ≥0.954 (R_ARM_RELATIVE + LDR+ADD PC pairs); recall capped by IDA's GOT-indirect chain analysis |
| ARM32 ELF armhf (Thumb-2) | 2 | 0.627–0.651 | call prec ≥0.998; jump FPs from literal-pool bytes; data_ptr recall low (Thumb LDR+ADD non-adjacent, pre-link VA pools) |
| ARM32 ELF (Android, mixed) | 1 | 0.336 | Heavy intra-section ARM↔Thumb interleaving causes jump FPs; calls still F1=0.932 |

Call xref precision is near-perfect (F1 ≥0.995) on all tested binaries.

---

## Architecture

### Binary

- **Languages**: Rust (core), Python (analysis scripts)
- **Parallelism**: Custom Rayon thread pool (`n_workers` threads); shard-per-segment dispatch
- **Zero-copy**: `memmap2` mmap, `SegData` newtype segment slices (hides `&'static` lifetime)
- **Depth levels**:
  - `ByteScan` (0): pointer-aligned 8-byte scan of data segments
  - `Linear` (1): sequential instruction decode of exec segments
  - `Paired` (2): ADRP+ADD/LDR pair resolution (ARM64), register const-prop (x86-64)

### Threading model

```
scan workers (rayon custom pool, n_workers threads)
    └─ tx (mpsc unbounded) ──► drain relay thread
                                    └─ out_tx (sync_channel, n_workers*4) ──► output thread
                                                                                  └─ pool.install(on_batch)
```

- **Scan workers**: `n_workers` rayon threads; each scans one shard, sends `Vec<Xref>` via `tx`
- **Drain relay**: pure channel relay; counts xrefs, forwards to output thread without blocking on I/O; sets stop flag on `Break`
- **Output thread**: calls `on_batch` under `pool.install` so any `par_iter` inside shares the scan pool (no oversubscription); formats in parallel chunks of 8192 records via `fold`+`reduce` into `Vec<u8>`, then one `write_all` per chunk through a 4 MiB `BufWriter`
- **Bounded output channel** (`sync_channel(n_workers*4)`): applies backpressure to scan if output falls behind, bounding peak memory

### ARM64 hot-path decode

`scan_adrp` uses a two-level dispatch:
1. `Arm64Insn::is_tracked(word)` — cheap bitmask union of all tracked encoding families (BL/B/ADRP/ADD/LDR/STR/branches); ~60–70% of instructions return `false`
2. For untracked words: `rd = word & 0x1F`; invalidate `adrp_state[rd]`; `continue` — no enum allocation
3. Only tracked words go through full `Arm64Insn::decode`

### Segment model

Each binary is split into `Segment` structs with:
- `executable: bool` — whether to instruction-scan
- `byte_scannable: bool` — whether to byte-scan for pointers
- For ELF: exec PT_LOADs are split per-section so `.rodata`/`.eh_frame*` inside
  the exec PT_LOAD are `executable=false` (not instruction-scanned)
- `.data.rel.ro` / `.data.rel.ro.local` → `byte_scannable=false`
  (relocation tables produce ~5–29x FP:TP ratio without reloc context)
- PIE ELFs (ET_DYN with first PT_LOAD at p_vaddr==0) are rebased to `0x0040_0000`
  (the default load address used by common disassemblers)
- ARM32 ELF: per-section Thumb/A32 mode from ELF mapping symbols (`$t`/`$a`)
  when present; otherwise each executable section is probed — if its first 4
  bytes form a LE u32 with top nibble `0xE` (ARM32 "always" condition) it is
  decoded as A32, otherwise Thumb.  Correctly distinguishes armhf (`.text`=Thumb,
  `.plt`/`.init`=A32) from armel (all sections A32) without mapping symbols.

### Xref kinds

| Kind | Source insns (ARM64) | Source insns (x86-64) |
|------|----------------------|-----------------------|
| Call | BL (exec target only), BLR (resolved) | CALL rel32, CALL r/m64 |
| Jump | B, B.cond, CBZ, CBNZ, TBZ, TBNZ, BR | Jcc, JMP rel, JMP r/m64 |
| DataRead | LDR/LDRB/LDRH + ADRP resolve | MOV [RIP+d], LEA reads |
| DataWrite | STR/STRB/STRH + ADRP resolve | MOV [RIP+d] writes |
| DataPointer | ADRP (emit at ADRP VA, not ADD VA) | LEA RIP+d, byte-scan, CMP/SUB/MOV imm32 |

### Type system

Strong typing throughout:
- `Va` newtype for virtual addresses (not raw `u64`)
- `Reg` newtype (0–30) for ARM64 registers, validated at construction
- `CmpBound`, `JumpTableEntrySize`, `JumpTableAddInfo`, `JumpTablePattern`, `JumpTableCtx` — ARM64 jump table recovery types
- `SegFlags` newtype for segment permission bitmasks
- `RelocPointer`, `Symbol` structs (not bare tuples)

### GOT-indirect call/jump resolution

xr emits `to=got_slot_va` (the real address the CPU dereferences) for
GOT-indirect calls/jumps. The benchmark normalizes extern-target xrefs back
to GOT slot VAs by decoding instruction bytes at each `from`.

### Relocation-derived data_ptr recovery

Relocation tables are parsed to extract authoritative pointer pairs:
- **ELF**: `.rela.dyn` / `.rel.dyn` — `R_*_RELATIVE`, `R_*_64` / `R_*_ABS64`
- **PE**: base relocation table (`IMAGE_REL_BASED_DIR64`), `.pdata` exception
  directory, UNWIND_INFO handler RVAs, IAT slots
- **Mach-O**: `LC_DYLD_CHAINED_FIXUPS` — formats 1, 2, 6, 9, 12 (including ARM64E)

These are emitted as `DataPointer` xrefs and bypass `min_ref_va` filtering
(authoritative metadata, not heuristic).

### Jump table recovery

**x86-64**: Recognises `CMP+JA+LEA+MOVSXD+ADD+JMP` pattern. Reads i32
offset tables from `.rodata`, computes targets, emits `Jump` xrefs. CMP
bound tracking per register limits table size precisely.

**ARM64**: Recognises `ADRP+ADD+CMP+LDRB/LDRH+ADD+BR` patterns with
backward scan from BR. Uses `Reg`-indexed `ScanState`, first-wins
semantics, `JUMP_TABLE_LOOKBACK` window, register chain verification.

---

## File Map

```
src/
  lib.rs                         ← public API re-exports
  main.rs                        ← CLI entry point, output formatting
  va.rs                          ← Va newtype (virtual address)
  xref.rs                        ← Xref, XrefKind, Confidence
  shard.rs                       ← split_range: parallel shard boundaries
  pass.rs                        ← XrefPass: orchestrates parallel scan
  disasm.rs                      ← disassembly context for -A/-B output
  output.rs                      ← Printer trait, text/json/csv formatters
  loader/
    mod.rs                       ← Segment, LoadedBinary, shared types, dispatch
    elf.rs                       ← ELF parsing, GOT slots, reloc pointers
    macho.rs                     ← Mach-O parsing, LC_DYLD_CHAINED_FIXUPS
    pe.rs                        ← PE parsing, .pdata, IAT, base relocations
    coff.rs                      ← COFF object file parsing (sequential VA layout)
    dyld.rs                      ← dyld shared cache
  arch/
    mod.rs                       ← byte_scan_pointers, SegmentDataIndex
    arm32.rs                     ← Thumb-2 + ARM32 (A32) scanners
    arm64.rs                     ← ADRP pair scan, jump table recovery
    arm64_decode.rs              ← pure bitmask ARM64 decoder
    x86_64.rs                    ← x86-64 scanner, jump table recovery
  bin/
    benchmark.rs                 ← benchmark vs ground truth
    fuzz_arm64.rs                ← ARM64 decoder fuzzer

scripts/
  ida_extract_xrefs_binary.py    ← ground-truth extraction script
  batch_extract_xrefs.sh         ← batch ground truth for all testcases
  score_all.sh                   ← run benchmark on all testcases
  eval.py                        ← quick eval without rebuild

testcases/                       ← test binaries + .xrefs.json (gitignored)
```

---

## Remaining Gaps & Root Causes

### ARM64 jump FNs (~495 on curl-aarch64)

Patterns without CMP bound in the backward scan window, or table base
register set outside the lookback window. Diminishing returns.

### ARM64 data_ptr FNs

- **ADD-VA mismatch**: ground truth records xref at ADD VA, xr at ADRP VA.
  Re-enabling ADD-VA gives +6496 TPs / +6981 FPs (net negative).
- **LDR through unresolved registers**: needs interprocedural data flow.
- **Byte-scan pointers to exec segment**: suppressed (10–14x FP:TP ratio).

### x86-64 jump FPs (~5807 on curl-amd64)

~4881 in a 153KB dead zone within `.text` where ground truth records only 13 xrefs.
FDE filtering would remove ~5780 FPs but add ~4946 FNs (net +0.002 F1).

### PE MSVC C++ EH/RTTI data_ptr FNs

MSVC exception handling and RTTI metadata stores references as 32-bit
image-relative RVAs (not 64-bit pointers), invisible to the 8-byte scanner.
Blind 32-bit RVA scanning has 14.5% precision. No tractable fix without
deep MSVC EH metadata parsing.

### PLT call resolution (x86-64 ELF)

`CALL rel32` through PLT stubs → ground truth records `to=extern_va`, xr records
`to=PLT_stub_va`. Causes ~711 call FNs on libharlem-shake.so.

### ARM32 jump FPs from literal pools (~259k on libamp.so)

Thumb-2 code sections embed literal pool data between functions. The linear
scanner treats these bytes as instructions, and 16-bit values in `0xE000–0xE7FF`
match the `B T2` (unconditional branch) encoding, producing spurious jumps.
Intra-section ARM↔Thumb interleaving (no mapping symbols, stripped Android
binary) compounds the problem. The only general fix is mapping-symbol-aware or
CFG-guided disassembly.

### ARM32 Thumb data_ptr recall ~15% (armhf) / ~22% (armel)

**armel (A32):** R_ARM_RELATIVE + adjacent LDR+ADD pairs cover the common PIC
GOT-pointer idiom.  The remaining FNs are non-adjacent LDR+ADD sequences and
multi-step GOT-indirect chains that require register-state tracking (analogous
to the ARM64 ADRP scanner).

**armhf (Thumb-2):** Literal pool words in `.text` contain pre-link absolute
VAs (`word = IDA_target`).  byte_scan only covers non-exec sections and sees
`word < pie_base → no match`.  Fix requires either (a) PIE-aware byte-scan of
exec sections (`word + pie_base`) or (b) tracking which LDR pool loads are
followed by an `ADD Rt, PC` (non-adjacent register state).  `.dynsym` xrefs
(≈13k per binary) are IDA-specific symbol-table references unlikely to match.

### data_write FNs

All register-based stores where the base register was set far earlier
(function arg or overwritten beyond the ADRP window). Requires
interprocedural data flow.

---

## What Was Tried & Outcome

### Fixes that worked

| Fix | Impact | Notes |
|-----|--------|-------|
| ARM32 section-probe mode detection | armel binaries 0.000→0.715–0.763 | Top-nibble probe replaces name heuristic; correctly classifies armel (A32) vs armhf (Thumb) |
| ARM32 R_ARM_RELATIVE reloc parsing | data_ptr 0 TPs → 2k–24k TPs | REL in-place addend; vma_to_file helper for PT_LOAD mapping |
| ARM32 LDR+ADD PC pair detection | armel data_ptr F1 0.189–0.292 → 0.268–0.361 | Adjacent `LDR Rd,[PC,#N]; ADD Rd,PC,Rd` pair; emits DataPointer from LDR, ADD, and pool |
| GOT slot VA approach | blackcat call F1 0.644→0.964 | Emit to=got_slot_va, normalize in benchmark |
| ELF reloc data_ptr | +24k TPs across all PIE ELFs | R_*_RELATIVE + R_*_64/ABS64 |
| Mach-O fixup chain parsing | hello.aarch64 F1 0.946→0.980 | Formats 2, 6, 1, 9, 12 |
| PE .pdata + UNWIND_INFO | win32kbase F1 0.894→0.995 | 4 xrefs per RUNTIME_FUNCTION |
| PE IAT slot population | PE indirect calls work | `got_slots` from PE import table |
| x86-64 jump table recovery | curl-amd64 jump FN 3448→403 | CMP+MOVSXD+ADD+JMP pattern |
| ARM64 jump table recovery | curl-aarch64 jump FN 3310→2815 | ADRP+ADD+CMP+LDR+ADD+BR pattern |
| Exec PT_LOAD section split | −1915 ARM64 jump FPs | .rodata in exec PT_LOAD → non-exec |
| Suppress ADD-VA data_ptr | ARM64 +0.015 | Emit at ADRP VA only |
| `.data.rel.ro` byte scan suppress | x86-64 +0.096 | 5:1 FP:TP ratio without reloc context |
| BLR/BR exec-target suppression | −77 call FP, −118 jump FP | Non-exec targets suppressed |
| Pure Rust ARM64 decoder | −28% CPU (was memset) | Replaced bad64 C FFI |
| is_tracked fast-path | −18% decode cost | Skip 65% of instructions |

### Fixes that were tried and reverted

| Fix | Why abandoned |
|-----|---------------|
| Re-enable ADD-VA data_ptr | +6496 TPs but +6981 FPs; net F1 +0.001 |
| .pdata xrefs with field-offset `from` | Ground truth uses entry start VA, not field offsets. 0 TP. |
| Blind 32-bit RVA scan of .rdata | 14.5% precision — too many random u32 matches |
| UNWIND_INFO scope table parsing | Layout varies by handler type. 2937 FP. Handler RVA alone is 100% precise. |
| FDE/`.eh_frame` coverage filter | −5780 FPs but +4946 FNs; net +0.002 F1 |
| Forward register tracker for data_write | F1 0.407 vs 0.541 — register reuse causes massive FPs |
| Extern VA replication algorithm | Binary-dependent layout; 5184 FP on blackcat.elf |
