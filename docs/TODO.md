# TODO

## Architecture & Design

- [ ] No GOT/IAT map for Mach-O binaries. `parse_macho` returns empty
  `got_slots`, so indirect `BLR Xn` / `CALL [RIP+got]` xrefs to extern
  symbols are never resolved for Mach-O. Needs `__got`/`__la_symbol_ptr`
  parsing.
  (`src/loader/macho.rs`)

- [ ] `build_elf_got_slots` is missing x86 (32-bit) relocation types
  (`R_386_GLOB_DAT`=6, `R_386_JMP_SLOT`=7) — GOT-indirect calls on x86
  produce no extern-VA xrefs. ARM32 and x86-64/AArch64 are already handled.
  (`src/loader/elf.rs`)

- [ ] `src/arch/arm64.rs` is 1316 lines and growing. Consider splitting into
  `arm64_scan.rs` (linear + ADRP pair scanner) and `arm64_jump_table.rs`
  (jump table recovery) once a natural seam presents itself.
  (`src/arch/arm64.rs`)

## Performance

- [ ] `ContextLine` allocates a `String` for `hex` on every disassembly line
  via `bytes_to_hex`. Writing hex directly into the output buffer would
  avoid the intermediate allocation.
  (`src/output.rs`)

- [ ] `ContextLine::from_disasm` clones `line.text` into a new `String` on
  every context line. Since `DisasmLine` is consumed immediately after
  `from_disasm` is called, the field could be moved instead of cloned.
  (`src/output.rs`)

## Testing

- [ ] Test helpers in `pass.rs` and `arch/arm32.rs` use `Box::leak` to create
  `&'static [u8]` slices for synthetic segments. This leaks memory on
  every test invocation. Harmless for correctness but accumulates with
  `--test-threads=1` under sanitisers. A `ManuallyDrop`+destructor or
  a test-scoped arena would avoid the leaks.
  (`src/pass.rs`, `src/arch/arm32.rs`)

## Minor

- [ ] `benchmark.rs` uses `ahash` while the main crate uses `rustc_hash`.
  Inconsistent but harmless. Consider standardising on one.
  (`src/bin/benchmark.rs`)

- [ ] `benchmark.rs` `run_pass` discards the `PassResult` returned by
  `XrefPass::run` (assigned to `_result`). The `elapsed_ms` and
  `confidence_counts` fields could replace the manual `Instant` timing
  and provide a per-confidence breakdown in the benchmark output.
  (`src/bin/benchmark.rs`)
