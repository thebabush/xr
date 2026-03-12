# xr — fast binary cross-reference extractor

`xr` is a standalone Rust CLI tool for ultra-fast, parallel extraction of
cross-references from stripped binaries (ELF, Mach-O, PE).  It emits
`(from_va, to_va, kind)` tuples and targets IDA Pro ground-truth fidelity at
orders-of-magnitude faster speed.

## Quick Start

```sh
cargo build --release

# Analyse a binary at the recommended depth
./target/release/xr /path/to/binary --depth paired

# Output as JSONL or CSV
./target/release/xr /path/to/binary --depth paired --format jsonl
./target/release/xr /path/to/binary --depth paired --format csv

# Filter to a specific xref kind
./target/release/xr /path/to/binary --depth paired --kind call

# Show disasm context around each xref site (like grep -A/-B)
./target/release/xr /path/to/binary --depth paired -A 3 -B 2

```

## Analysis Depths

| Flag | Name | What it does |
|------|------|-------------|
| `--depth scan` | ByteScan | Pointer-sized byte scan of data sections |
| `--depth linear` | Linear | Linear disasm — immediate targets + RIP-relative |
| `--depth paired` | Paired | ADRP+ADD/LDR pairs (ARM64) or register prop (x86-64) — **recommended** |

## Performance

206 million xrefs from a 4.6 GB dyld shared cache (3240 images) in 43 seconds:

```
$ xr /System/Library/dyld/dyld_shared_cache_x86_64 > /dev/null
dyld shared cache: arch=X86_64  mappings=24  images=3240  subcaches=5
xrefs: 206432528  |  43.3s  |  4620.3 MB scanned  |  24 segments
```

## Accuracy

Tested against IDA Pro ground truth on 26 binaries across ELF, Mach-O, and PE
(x86-64 and ARM64). Overall F1 ranges from **0.56–0.99** depending on binary
complexity (lowest on MSVC C++ PE binaries with dense EH/RTTI metadata). Call
xref precision is near-perfect (F1 ≥0.995) on all tested binaries.

See [docs/STATUS.md](docs/STATUS.md) for architecture details and known gaps.

## Supported Formats

- ELF (x86-64, AArch64) — including PIE (ET_DYN)
- Single-arch Mach-O (x86-64, ARM64) — fat binaries require `lipo -extract` first
- PE / COFF (x86-64, ARM64)
- Apple dyld shared cache
- Raw flat binary (treated as single executable segment)

x86-32 and ARM32 binaries are loaded but not scanned (architecture stubs only).

## Options

```
USAGE:
    xr [OPTIONS] <BINARY>

OPTIONS:
    -d, --depth <DEPTH>         Analysis depth: scan | linear | paired [default: paired]
    -j, --workers <N>           Worker threads; 0 = all CPUs [default: 0]
    -f, --format <FORMAT>       Output format: text | jsonl | csv [default: text]
    -k, --kind <KIND>           Filter by kind: call | jump | data_read | data_write | data_ptr
        --base <VA>             Override PIE ELF load base (hex or decimal)
        --min-ref-va <VA>       Drop xrefs whose 'to' VA is below this value
        --start <VA>            Scan only 'from' addresses >= VA
        --end <VA>              Scan only 'from' addresses < VA
        --ref-start <VA>        Retain only xrefs with 'to' >= VA
        --ref-end <VA>          Retain only xrefs with 'to' < VA
        --limit <N>             Cap output at N xrefs (0 = unlimited)
    -A, --after-context <N>     Show N instructions after each xref site
    -B, --before-context <N>    Show N instructions before each xref site
```

## Benchmarking

```sh
# Build the benchmark binary
cargo build --release --bin benchmark

# Run against a ground-truth file (JSON exported from IDA)
./target/release/benchmark \
    --binary /path/to/binary \
    --ground-truth /path/to/binary.xrefs.json \
    --depth paired
```

Ground-truth JSON files are generated with `scripts/ida_extract_xrefs_binary.py`
(requires IDA Pro with idalib).
