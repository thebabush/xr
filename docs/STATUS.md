# xr — Status & Architecture

## Goal

Build `xr`: a standalone Rust crate for ultra-fast, parallel cross-reference
extraction from stripped binaries (ELF, Mach-O, PE). Benchmarked against IDA Pro
ground-truth xrefs. Maximise F1 score across all xref kinds.

---

## Current Scores (Paired depth)

Tested against IDA Pro ground truth on 26 binaries across ELF (x86-64,
AArch64), Mach-O (ARM64), and PE (x86-64, ARM64).

| Category | F1 range | Notes |
|----------|----------|-------|
| ELF x86-64 | 0.87–0.97 | Best on statically linked, lower on PIE with many extern calls |
| ELF ARM64 | 0.84–0.95 | Lower end from unresolved ADRP pairs and jump tables |
| Mach-O ARM64 | 0.98 | Fixup chain parsing recovers most data_ptr |
| PE x86-64 (Rust/MinGW) | 0.89–0.99 | .pdata + UNWIND_INFO parsing very effective |
| PE x86-64 (MSVC C++) | 0.56–0.87 | Limited by 32-bit RVA EH/RTTI metadata; low end from .pdata FPs on concrt140.dll |
| PE ARM64 | 0.62–0.75 | Limited by data_ptr gaps |

Call xref precision is near-perfect (F1 ≥0.995) on all tested binaries.

---

## Architecture

### Binary

- **Languages**: Rust (core), Python (analysis scripts)
- **Parallelism**: Custom Rayon thread pool (`n_workers` threads); shard-per-segment dispatch
- **Zero-copy**: `memmap2` mmap, `&'static [u8]` segment slices
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
  to match IDA's default load address

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
GOT-indirect calls/jumps, rather than trying to replicate IDA's fragile
synthetic extern VA assignment. The benchmark normalizes IDA's extern-target
xrefs back to GOT slot VAs by decoding instruction bytes at each `from`.

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
    dyld.rs                      ← dyld shared cache
  arch/
    mod.rs                       ← byte_scan_pointers, SegmentDataIndex
    arm64.rs                     ← ADRP pair scan, jump table recovery
    arm64_decode.rs              ← pure bitmask ARM64 decoder
    x86_64.rs                    ← x86-64 scanner, jump table recovery
  bin/
    benchmark.rs                 ← benchmark vs IDA ground truth
    fuzz_arm64.rs                ← ARM64 decoder fuzzer

scripts/
  ida_extract_xrefs_binary.py    ← IDA Pro ground-truth extraction
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

- **ADD-VA mismatch**: IDA records xref at ADD VA, xr at ADRP VA.
  Re-enabling ADD-VA gives +6496 TPs / +6981 FPs (net negative).
- **LDR through unresolved registers**: needs interprocedural data flow.
- **Byte-scan pointers to exec segment**: suppressed (10–14x FP:TP ratio).

### x86-64 jump FPs (~5784 on curl-amd64)

~4881 in a 153KB dead zone within `.text` where IDA records only 13 xrefs.
FDE filtering would remove ~5780 FPs but add ~4946 FNs (net +0.002 F1).

### PE MSVC C++ EH/RTTI data_ptr FNs

MSVC exception handling and RTTI metadata stores references as 32-bit
image-relative RVAs (not 64-bit pointers), invisible to the 8-byte scanner.
Blind 32-bit RVA scanning has 14.5% precision. No tractable fix without
deep MSVC EH metadata parsing.

### PLT call resolution (x86-64 ELF)

`CALL rel32` through PLT stubs → IDA records `to=extern_va`, xr records
`to=PLT_stub_va`. Causes ~711 call FNs on libharlem-shake.so.

### data_write FNs

All register-based stores where the base register was set far earlier
(function arg or overwritten beyond the ADRP window). Requires
interprocedural data flow.

---

## What Was Tried & Outcome

### Fixes that worked

| Fix | Impact | Notes |
|-----|--------|-------|
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
| .pdata xrefs with field-offset `from` | IDA uses entry start VA, not field offsets. 0 TP. |
| Blind 32-bit RVA scan of .rdata | 14.5% precision — too many random u32 matches |
| UNWIND_INFO scope table parsing | Layout varies by handler type. 2937 FP. Handler RVA alone is 100% precise. |
| FDE/`.eh_frame` coverage filter | −5780 FPs but +4946 FNs; net +0.002 F1 |
| Forward register tracker for data_write | F1 0.407 vs 0.541 — register reuse causes massive FPs |
| IDA extern VA replication algorithm | Binary-dependent layout; 5184 FP on blackcat.elf |
