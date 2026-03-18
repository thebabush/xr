# TODO

## Error Handling

- [ ] Thread pool build uses `.expect()` instead of returning an error.
  `rayon::ThreadPoolBuilder::build().expect(...)` panics on failure (e.g.
  OOM). Since `XrefPass::run` doesn't return `Result`, this is consistent
  but not ideal. Consider making `run` return `Result<PassResult>`.
  (`src/pass.rs`)

## Architecture & Design

- [ ] No GOT/IAT map for Mach-O binaries. `parse_macho` returns empty
  `got_slots`, so indirect `BLR Xn` / `CALL [RIP+got]` xrefs to extern
  symbols are never resolved for Mach-O. Needs `__got`/`__la_symbol_ptr`
  parsing.
  (`src/loader/macho.rs`)

- [ ] `build_elf_got_slots` only handles x86-64 and AArch64 relocation
  types (`R_X86_64_GLOB_DAT/JUMP_SLOT`, `R_AARCH64_GLOB_DAT/JUMP_SLOT`).
  ARM32 (`R_ARM_GLOB_DAT`=21, `R_ARM_JUMP_SLOT`=22) and x86
  (`R_386_GLOB_DAT`=6, `R_386_JMP_SLOT`=7) are missing — GOT-indirect
  calls on those arches produce no extern-VA xrefs.
  (`src/loader/elf.rs`)

- [ ] `ScanRegion` has two `#[allow(dead_code)]` fields: `mode` and
  `writable`. They are reserved for future ARM32/Thumb and
  write-tracking passes. Either implement the passes that use them or
  remove the fields and add them back when needed.
  (`src/arch/mod.rs`)

- [ ] Split `loader.rs` into per-format modules — ✅ DONE (now
  `src/loader/{mod,elf,macho,pe,dyld}.rs`). Consider further splitting
  `src/arch/arm64.rs` (~1253 lines) if it grows further.

## Performance

- [ ] `ContextLine` allocates a `String` for `hex` on every disassembly line
  via `bytes_to_hex`. Since context is rendered in parallel via `par_iter`,
  this is many small allocations. Writing hex directly to the output buffer
  would avoid the intermediate `String`.
  (`src/output.rs`)

- [ ] `disasm_x86` and `build_window` clone `Vec<u8>` and `String` for
  every instruction line (`bytes.clone()`, `text.clone()`). The tuples
  could be consumed (moved) instead of cloned when building the final
  `DisasmLine` vec — the source vecs are not used after conversion.
  (`src/disasm.rs`)

## Testing

- [ ] Test helpers in `pass.rs` and `arch/*.rs` use `Box::leak` to create
  `&'static [u8]` slices for synthetic segments. This leaks memory on
  every test invocation. Harmless for correctness but accumulates with
  `--test-threads=1` under sanitisers. A `ManuallyDrop`+destructor or
  a test-scoped arena would avoid the leaks.
  (`src/pass.rs`, `src/arch/arm64.rs`, `src/arch/x86_64.rs`)

## Minor

- [ ] `benchmark.rs` uses `ahash` while the main crate uses `rustc_hash`.
  Inconsistent but harmless. Consider standardising on one.
  (`src/bin/benchmark.rs`)

- [ ] `benchmark.rs` `run_pass` discards the `PassResult` returned by
  `XrefPass::run` (assigned to `_result`). The `elapsed_ms` and
  `confidence_counts` fields could replace the manual `Instant` timing
  and provide a per-confidence breakdown in the benchmark output.
  (`src/bin/benchmark.rs`)
