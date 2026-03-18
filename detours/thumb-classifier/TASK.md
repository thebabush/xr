# Thumb / ARM32 Mode Classifier

## Problem

`xr` is a binary cross-reference extractor.  For ARM32 ELF binaries it must
decide, for each executable section, whether the code is ARM32 (A32) or
Thumb-2 (T32).  The two ISAs have different instruction widths and encodings;
misidentifying the mode produces large numbers of spurious xrefs.

The current approach probes the **first 16 four-byte-aligned words** of each
section and counts how many have top nibble `0xE` (the ARM32 "always"
condition code).  If more than half do, the section is classified ARM32;
otherwise Thumb.

This works for sections that are homogenously one mode, but fails for
**mixed sections** — common in Android NDK libraries and some armel shared
libraries — where ARM32 and Thumb-2 functions are interleaved inside the same
`.text` section.  The probe fires once at the top and stamps the whole section,
so ARM32 functions that start a few kilobytes in are decoded as Thumb,
producing hundreds of thousands of false-positive branch xrefs.

### Concrete example

`testcases/libamp.so` is a 3.5 MB stripped Android ARM32 ELF.  Its `.text`
section opens with Thumb-2 code (correct probe result), but contains ARM32
functions deeper in.  The current scanner emits ~258 000 spurious B T2 jumps
because it decodes ARM32 words as pairs of 16-bit Thumb halfwords.

### Why ARM32 words look like Thumb branches

An ARM32 instruction word `0xE...xxxx` has the 32-bit Thumb B T2 encoding
(`0xE000–0xE7FF`) in its low halfword roughly 3% of the time — often enough
that dense ARM32 code produces a false branch at almost every instruction
address when decoded as Thumb.  The correct fix is to detect the mode switch
before decoding begins.

---

## Goal

Train a **small, explainable classifier** that predicts ARM32 vs Thumb-2 mode
for a sliding window of bytes within an executable ARM32 ELF section.  The
output should be:

1. A Python script that collects training data, trains the model, and prints
   the learned decision rules.
2. A hard-coded decision function (a few nested `if` branches) derived from
   the trained tree, ready to drop into `src/arch/arm32.rs` as a replacement
   for `probe_arm32_section_mode`.
3. Accuracy numbers on a held-out test set of binaries **not seen during
   training**.

The classifier must be **explainable** — a decision tree or a small set of
threshold rules, not a neural network or random forest black box.  The goal
is to understand *why* it fires, so it can be translated to a deterministic
Rust function and reasoned about during code review.

---

## Ground-Truth Annotation Strategy

### The annotation source: ELF mapping symbols

Every ARM32 ELF produced by GCC, Clang, or the GNU assembler contains local
`STT_NOTYPE` symbols named `$t` (start of a Thumb region) and `$a` (start of
an ARM32 region) in `.symtab`.  These are inserted by the assembler/linker at
every mode transition and are **byte-exact and authoritative** — they are what
IDA uses internally (`Alt+G` sets the `T` segment register, which IDA infers
from `$t`/`$a` when present).

These symbols survive in:
- Non-stripped debug builds
- Debian / Ubuntu `-dbgsym` packages
- Any library compiled with `-g` or without `-s`

Stripped binaries (like `testcases/libamp.so`) have had `.symtab` removed;
only `.dynsym` remains, which does not contain mapping symbols.  **Only use
non-stripped binaries for training.**

### Corpus to collect

Download non-stripped ARM32 shared libraries from Debian's two ARM32 ports:

- **armhf** (ARM hard-float ABI, `-mthumb` default): predominantly Thumb-2
- **armel** (ARM soft-float ABI, `-marm` default): predominantly ARM32 with
  some Thumb

```bash
# Example — adjust mirror and suite as needed
for arch in armhf armel; do
  for pkg in libc6 libssl3 libstdc++6 libm libpthread \
             libcurl4 libbz2-1.0 libz1 libpng16-16 \
             libjpeg62-turbo libglib2.0-0 libgnutls30 \
             libpcre2-8-0 libsystemd0 libdbus-1-3; do
    apt-get download ${pkg}-dbgsym:${arch}   2>/dev/null || \
    apt-get download ${pkg}:${arch}          2>/dev/null || true
  done
done
```

Alternatively, cross-compile a handful of well-known open-source projects
(OpenSSL, curl, zlib, pjsip) for `arm-linux-gnueabi` and
`arm-linux-gnueabihf` without stripping.  This gives full control over
compiler flags and a clean variety of code patterns.

**Target corpus size**: 100–200 ELF files covering both ABIs.  A single 3 MB
`.text` section yields ~750 000 labelled windows, so even 20 files gives
ample training data.

### Additional non-Debian sources

Debian armhf/armel is the easiest starting point but risks training on a
single toolchain (GCC + glibc).  Collect from additional sources to improve
generalisation:

| Source | Why useful |
|---|---|
| **Alpine Linux armv7** | musl libc, different struct layouts and calling conventions; `apk fetch` + extract dbg packages |
| **Arch Linux ARM** (armv7h) | Aggressively optimised Thumb-2; ALARM mirror or a running Pi |
| **Android AOSP** | Heavy Clang, NEON intrinsics, hand-written ARM32 asm in media codecs — covers the libamp.so failure case directly |
| **FreeBSD ARM** | LLVM-only toolchain, different ABI |
| **Buildroot** | Compile a root FS with `BR2_ARM_INSTRUCTION_SET=arm` vs `=thumb` for synthetic adversarial cases |

**Windows ARM32 (PE)**: exists (WOW32 on ARM64 Windows, Windows IoT Core) but
Microsoft uses Thumb-2 exclusively for user-mode code — no ARM32 (A32)
instructions in practice, and PE has no mapping symbols.  Not useful for
training.  If PE validation is desired, compile the same open-source code for
both ELF (annotated via `$t`/`$a`) and PE, and use the ELF labels to verify
the PE predictions — the byte-level features are format-agnostic.

### Extracting labels from mapping symbols

```python
import struct, elftools.elf.elffile as ef  # pyelftools

def extract_mapping_symbols(elf_path):
    """
    Returns a list of (va, mode) sorted by va, where mode is 'thumb' or 'arm32'.
    Only includes $t / $a symbols; ignores $d (data) for the mode predictor.
    """
    with open(elf_path, 'rb') as f:
        elf = ef.ELFFile(f)
        symtab = elf.get_section_by_name('.symtab')
        if symtab is None:
            return []   # stripped — skip this file
        entries = []
        for sym in symtab.iter_symbols():
            name = sym.name
            va   = sym['st_value'] & ~1   # strip Thumb bit from value if present
            if name == '$t' or name.startswith('$t.'):
                entries.append((va, 'thumb'))
            elif name == '$a' or name.startswith('$a.'):
                entries.append((va, 'arm32'))
        entries.sort()
        return entries

def mode_at(va, mapping_symbols, default='thumb'):
    """Return the mode in effect at `va` given a sorted list of (va, mode)."""
    import bisect
    idx = bisect.bisect_right(mapping_symbols, (va,)) - 1
    if idx < 0:
        return default
    return mapping_symbols[idx][1]
```

---

## Feature Extraction

For each **4-byte-aligned** address `a` in an executable section, extract a
feature vector from the surrounding window of `N` four-byte-aligned words.

```python
import struct

def read_u32_le(data, offset):
    return struct.unpack_from('<I', data, offset)[0]

def read_u16_le(data, offset):
    return struct.unpack_from('<H', data, offset)[0]

def window_features(data, section_offset, N=32):
    """
    data            : bytes of the full section
    section_offset  : byte offset within `data`, must be 4-byte aligned
    N               : window size in 4-byte words (default 32 = 128 bytes)

    Returns a dict of scalar features.
    """
    n_bytes = len(data)
    # Clamp window so it doesn't run off either end of the section
    start = max(0, section_offset - (N // 2) * 4)
    start = start & ~3   # keep 4-byte aligned
    end   = min(n_bytes, start + N * 4)
    actual_N = (end - start) // 4
    if actual_N == 0:
        return None

    words     = [read_u32_le(data, start + i * 4) for i in range(actual_N)]
    # Read halfwords at 2-byte stride across the same window
    hw_end    = min(n_bytes, start + N * 4)
    halfwords = [read_u16_le(data, start + i * 2)
                 for i in range((hw_end - start) // 2)]

    top_nibbles = [w >> 28 for w in words]
    top_bytes   = [w >> 24 for w in words]

    frac_top_E     = top_nibbles.count(0xE) / actual_N
    frac_top_F     = top_nibbles.count(0xF) / actual_N
    frac_top_0_9   = sum(1 for n in top_nibbles if n <= 9) / actual_N

    # 32-bit Thumb-2 first halfwords (bits 15:13 = 111, i.e. >= 0xE800)
    frac_hw_thumb32 = sum(1 for h in halfwords if h >= 0xE800) / len(halfwords)

    # Thumb-2 MOV-register / NOP family (very common in Thumb prologues)
    frac_hw_46xx   = sum(1 for h in halfwords if h >> 8 == 0x46) / len(halfwords)

    # ARM32 PUSH prologue: STMDB SP!, {regs}  — top 16 bits == 0xE92D
    has_arm32_push = int(any(w >> 16 == 0xE92D for w in words))

    # ARM32 BL family: condition | 1011 | imm24
    frac_arm32_bl  = sum(1 for w in words
                         if (w >> 24) & 0x0F == 0x0B) / actual_N

    # Thumb-2 PUSH.W: first halfword == 0xE92D
    has_thumb_push = int(any(h == 0xE92D for h in halfwords))

    return {
        'frac_top_E':       frac_top_E,
        'frac_top_F':       frac_top_F,
        'frac_top_0_9':     frac_top_0_9,
        'frac_hw_thumb32':  frac_hw_thumb32,
        'frac_hw_46xx':     frac_hw_46xx,
        'has_arm32_push':   has_arm32_push,
        'frac_arm32_bl':    frac_arm32_bl,
        'has_thumb_push':   has_thumb_push,
    }
```

**Stride**: generate one sample per 16 bytes (every 4 words) rather than
every 4 bytes.  Adjacent samples share most of their window and are highly
correlated; sampling at stride 16 gives better effective diversity without
ballooning the dataset.

---

## Training

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GroupShuffleSplit
import numpy as np

# --- build dataset -----------------------------------------------------------
rows   = []
labels = []
groups = []   # binary filename — used for group-aware train/test split

for binary_path in all_binary_paths:
    mapping_syms = extract_mapping_symbols(binary_path)
    if not mapping_syms:
        continue   # skip stripped

    with open(binary_path, 'rb') as f:
        elf = ELFFile(f)
        for section in elf.iter_sections():
            if not (section['sh_flags'] & SHF_EXECINSTR):
                continue
            data      = section.data()
            base_va   = section['sh_addr']
            stride    = 16   # bytes between samples

            for off in range(0, len(data) - 4, stride):
                if off % 4 != 0:
                    continue
                va   = base_va + off
                mode = mode_at(va, mapping_syms)
                feat = window_features(data, off, N=32)
                if feat is None:
                    continue
                rows.append(feat)
                labels.append(0 if mode == 'arm32' else 1)   # 0=ARM32, 1=Thumb
                groups.append(binary_path)

df = pd.DataFrame(rows)
y  = np.array(labels)
g  = np.array(groups)

# --- train/test split by binary (not by row!) --------------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, y, g))

X_train, X_test = df.iloc[train_idx], df.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# --- train -------------------------------------------------------------------
clf = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=500,   # avoid overfitting on rare encodings
    class_weight='balanced',
)
clf.fit(X_train, y_train)

# --- evaluate ----------------------------------------------------------------
from sklearn.metrics import classification_report
print(classification_report(y_test, clf.predict(X_test),
                             target_names=['ARM32', 'Thumb']))

# --- print the tree (this is the deliverable) --------------------------------
print(export_text(clf, feature_names=list(df.columns)))
```

---

## Expected Output / Deliverables

### 1. Printed decision tree

Something like:

```
|--- frac_top_E <= 0.31
|   |--- frac_hw_thumb32 <= 0.41
|   |   |--- class: ARM32
|   |   |--- class: Thumb
|   |--- class: Thumb
|--- frac_top_E > 0.31
|   |--- has_arm32_push <= 0.50
|   |   |--- class: Thumb   (VFP/NEON-heavy ARM32 with few 0xE words)
|   |--- class: ARM32
```

### 2. Rust translation

The tree translates directly into a Rust function replacing
`probe_arm32_section_mode` in `src/loader/elf.rs`:

```rust
/// Predict ARM32 vs Thumb-2 mode for a window of bytes.
/// Returns `DecodeMode::Arm32` or `DecodeMode::Thumb`.
fn predict_mode(data: &[u8], offset: usize, n_words: usize) -> DecodeMode {
    // ... compute frac_top_E, frac_hw_thumb32, has_arm32_push ...
    if frac_top_e <= 0.31 {
        if frac_hw_thumb32 <= 0.41 { DecodeMode::Arm32 } else { DecodeMode::Thumb }
    } else {
        if has_arm32_push { DecodeMode::Arm32 } else { DecodeMode::Thumb }
    }
}
```

The sliding-window integration in `src/arch/arm32.rs` then calls this every
64 bytes of scan position and switches decode mode when the prediction changes.

### 3. Accuracy numbers

Report per-class precision / recall / F1 on held-out binaries.  Minimum bar:

| Mode   | Precision | Recall |
|--------|-----------|--------|
| ARM32  | > 0.95    | > 0.90 |
| Thumb  | > 0.98    | > 0.99 |

The asymmetry is intentional: ARM32 misclassified as Thumb produces hundreds
of spurious jumps (high cost); Thumb misclassified as ARM32 mostly just misses
some branches (lower cost).

---

## Integration Notes

### Where the classifier plugs in

File: `src/loader/elf.rs`, function `parse_elf`.

Current call site:
```rust
// Stripped binary: majority-vote probe over the first 16 words of each section
for si in &mut section_infos {
    si.mode = probe_arm32_section_mode(bytes, si.file_offset, si.file_size);
}
```

The replacement:
1. Keep `probe_arm32_section_mode` as a fallback for very short sections
   (< 64 bytes).
2. For longer sections, assign an initial mode from the probe, then re-predict
   every 64 bytes during the actual Thumb scan in `src/arch/arm32.rs`,
   switching mode when the prediction flips for two consecutive windows
   (hysteresis to avoid thrashing at boundaries).

### Hysteresis

Do not switch mode on a single window flip.  Require two (or three) consecutive
windows to agree before changing.  This prevents a short ARM32 stub embedded in
Thumb code (e.g., a trampoline) from corrupting the surrounding decode.

### What to do with $d (data) symbols

Mapping symbols also include `$d` marking inline data (literal pools).
For the mode classifier, ignore `$d` regions when generating training labels
— they are neither ARM32 nor Thumb and would confuse the model.  During
inference, the classifier will naturally produce low-confidence predictions
in data regions, which is acceptable because the scanner should already skip
non-instruction words via the `byte_scannable` flag on data sections.

---

## Repository Layout for This Detour

```
detours/thumb-classifier/
  TASK.md                  ← this file
  collect_corpus.sh        ← download non-stripped ARM32 ELF packages
  extract_training_data.py ← build labelled CSV from corpus
  train.py                 ← train decision tree, print rules, save model
  evaluate.py              ← per-binary accuracy on held-out set
  rules.txt                ← printed decision tree (output of train.py)
  thumb_classifier.rs      ← generated Rust translation of the tree
```

The final output handed back to the main project is `thumb_classifier.rs` and
the accuracy report.  The Python scripts and corpus stay in this detour
directory and do not become part of the main `xr` build.
