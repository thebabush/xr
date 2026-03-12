//! Minimal ARM64 bitmask decoder — no allocation, no FFI, no struct zeroing.
//!
//! Classifies a 32-bit instruction word into the ~15 encoding families we care
//! about.  Every field is extracted on demand via `#[inline]` bitmask methods
//! on the raw `u32`.  Unrecognised encodings are returned as `Other` so callers
//! can still extract the destination register for state invalidation.
//!
//! # Encoding references (ARM DDI 0487)
//!
//! | Mnemonic     | Mask         | Match        |
//! |---|---|---|
//! | BL           | FC00_0000    | 9400_0000    |
//! | B            | FC00_0000    | 1400_0000    |
//! | B.cond       | FF00_0010    | 5400_0000    |
//! | CBZ          | 7F00_0000    | 3400_0000    |
//! | CBNZ         | 7F00_0000    | 3500_0000    |
//! | TBZ          | 7F00_0000    | 3600_0000    |
//! | TBNZ         | 7F00_0000    | 3700_0000    |
//! | ADRP         | 9F00_0000    | 9000_0000    |
//! | ADR          | 9F00_0000    | 1000_0000    |
//! | ADD imm      | 7F80_0000    | 1100_0000    | (sf=0 or 1, op=0, S=0, shift=00/01)
//! | LDR (unsigned)| FFC0_0000   | F940_0000    | (64-bit, unsigned offset)
//! | STR (unsigned)| FFC0_0000   | F900_0000    | (64-bit, unsigned offset)
//! | BLR          | FFFF_FC1F    | D63F_0000    |
//! | BR           | FFFF_FC1F    | D61F_0000    |

/// Classification of a 32-bit ARM64 word into the encodings we care about.
///
/// Every variant stores the raw `u32` word only — all fields are extracted
/// by the accessor methods below, so there is zero extra allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arm64Insn {
    /// BL #imm26 — direct call
    Bl(u32),
    /// B #imm26 — unconditional jump
    B(u32),
    /// B.cond #imm19 — conditional jump (any condition code)
    BCond(u32),
    /// CBZ Xn/Wn, #imm19
    Cbz(u32),
    /// CBNZ Xn/Wn, #imm19
    Cbnz(u32),
    /// TBZ Xn/Wn, #bit, #imm14
    Tbz(u32),
    /// TBNZ Xn/Wn, #bit, #imm14
    Tbnz(u32),
    /// ADRP Xn, #page
    Adrp(u32),
    /// ADR Xn, #label
    Adr(u32),
    /// ADD (immediate) Xd/Wd, Xn/Wn, #imm
    AddImm(u32),
    /// LDR (unsigned offset) — any size (byte/half/word/doubleword).
    /// Use `ldr_str_size()` for the access width (0=B, 1=H, 2=W, 3=X).
    Ldr(u32),
    /// STR (unsigned offset) — any size.
    Str(u32),
    /// LDR (literal) — PC-relative, imm19 scaled by 4.
    /// Covers LDR Xt/Wt, `label` and LDRSW Xt, `label`.
    LdrLiteral(u32),
    /// BLR Xn
    Blr(u32),
    /// BR Xn
    Br(u32),
    /// Anything else — dest reg (bits [4:0]) available for state invalidation.
    Other(u32),
}

impl Arm64Insn {
    /// Quick test: does `word` match any of the ~13 tracked encoding families?
    ///
    /// Returns `false` for instructions that would decode to `Other(word)`.
    /// Callers can skip `decode` entirely for those and handle register
    /// invalidation with `word & 0x1F` directly, saving ~19% CPU.
    ///
    /// The test is a union of all the guards inside `decode`, ordered from
    /// cheapest / most-common to most-specific.
    #[inline]
    pub fn is_tracked(word: u32) -> bool {
        // BL / B — bits[31:26] = 100101 / 000101  (very common, check first)
        let top6 = word & 0xFC00_0000;
        if top6 == 0x9400_0000 || top6 == 0x1400_0000 {
            return true;
        }
        // ADRP / ADR — bits{31,28:24} checked via 0x9F00_0000
        let adr_check = word & 0x9F00_0000;
        if adr_check == 0x9000_0000 || adr_check == 0x1000_0000 {
            return true;
        }
        // B.cond / BC.cond — bits[31:24] = 0x54
        if word & 0xFF00_0000 == 0x5400_0000 {
            return true;
        }
        // CBZ/CBNZ/TBZ/TBNZ — bits[30:24] ∈ {0x34,0x35,0x36,0x37}
        match word & 0x7F00_0000 {
            0x3400_0000 | 0x3500_0000 | 0x3600_0000 | 0x3700_0000 => return true,
            _ => {}
        }
        // BLR / BR — very specific masks
        if word & 0xFFFF_FC1F == 0xD63F_0000 || word & 0xFFFF_FC1F == 0xD61F_0000 {
            return true;
        }
        // ADD (immediate) — bits[28:24]=10001, bit[29]=0
        if word & 0x1F00_0000 == 0x1100_0000 && word & 0x2000_0000 == 0 {
            return true;
        }
        // LDR (unsigned offset, any size) — bits[29:27]=111, bit[26]=0 (not SIMD),
        // bits[25:24]=01, bits[23:22]=01 (opc=LDR).
        // Covers: LDRB(00), LDRH(01), LDR W(10), LDR X(11).
        // Common mask: 0x3FC0_0000, match: 0x3940_0000
        if word & 0x3FC0_0000 == 0x3940_0000 {
            return true;
        }
        // STR (unsigned offset, any size) — same layout, opc=00 (STR).
        // Common mask: 0x3FC0_0000, match: 0x3900_0000
        if word & 0x3FC0_0000 == 0x3900_0000 {
            return true;
        }
        // LDR (literal) — PC-relative: LDR Wt (0x18), LDR Xt (0x58), LDRSW (0x98)
        // bits[29:24]=011000, V=0 (bit[26]=0). Mask bits[29:24,26]: 0x3F00_0000=0x18
        // opc(bits[31:30]): 00=LDR W, 01=LDR X, 10=LDRSW
        if word & 0x3F00_0000 == 0x1800_0000 {
            return true;
        }
        false
    }

    /// Classify a raw instruction word.  Always succeeds — unknown encodings
    /// become `Other`.  `pc` is only needed for PC-relative target resolution
    /// (call `.bl_target(pc)` etc. separately).
    #[inline]
    pub fn decode(word: u32) -> Self {
        // Test from most-specific to least-specific so a more-specific mask
        // wins when multiple patterns would match.

        // BLR / BR — very specific masks
        if word & 0xFFFF_FC1F == 0xD63F_0000 {
            return Arm64Insn::Blr(word);
        }
        if word & 0xFFFF_FC1F == 0xD61F_0000 {
            return Arm64Insn::Br(word);
        }

        // ADRP / ADR — bits[28:24]=10000 (0x10), bit31 distinguishes them:
        // ADRP: bit31=1 → 0x90000000 after masking 0x9F000000
        // ADR:  bit31=0 → 0x10000000 after masking 0x9F000000
        // Correct mask: bits[28:24] and bit31 only — use 0x9F000000.
        // Note: immlo lives in bits[30:29] so we must NOT include those in the mask.
        match word & 0x9F00_0000 {
            0x9000_0000 => return Arm64Insn::Adrp(word),
            0x1000_0000 => return Arm64Insn::Adr(word),
            _ => {}
        }
        // The above match handles all ADRP/ADR encodings because:
        //   mask 0x9F000000 = bits{31,28,27,26,25,24} — immlo bits{30,29} are not masked.

        // BL / B — bits [31:26] = op1 || op = 1001_01 / 0001_01
        match word & 0xFC00_0000 {
            0x94000000 => return Arm64Insn::Bl(word),
            0x14000000 => return Arm64Insn::B(word),
            _ => {}
        }

        // B.cond / BC.cond — bits[31:24]=0101_0100
        // bit[4] is 'o0': 0 for B.cond (v8.0+), 1 for BC.cond (v8.8+).
        // We accept both — bad64 decodes both as BCond.
        // Mask 0xFF00_0000, match 0x5400_0000
        if word & 0xFF00_0000 == 0x5400_0000 {
            return Arm64Insn::BCond(word);
        }

        // CBZ/CBNZ/TBZ/TBNZ — bits [30:25]
        // CBZ:  sf|011_0100  mask 0x7F00_0000 match 0x3400_0000
        // CBNZ: sf|011_0101  mask 0x7F00_0000 match 0x3500_0000
        // TBZ:  b5|011_0110  mask 0x7F00_0000 match 0x3600_0000
        // TBNZ: b5|011_0111  mask 0x7F00_0000 match 0x3700_0000
        match word & 0x7F00_0000 {
            0x3400_0000 => return Arm64Insn::Cbz(word),
            0x3500_0000 => return Arm64Insn::Cbnz(word),
            0x3600_0000 => return Arm64Insn::Tbz(word),
            0x3700_0000 => return Arm64Insn::Tbnz(word),
            _ => {}
        }

        // ADD (immediate) — sf|001_0001|shift(2)|imm12|Rn|Rd
        // bits[28:24]=1_0001, bit[29]=0 (op=ADD), bit[23:22]=shift (00 or 01)
        // We accept both sf=0 and sf=1 (W and X forms).
        // Mask: 0x7F80_0000, match: 0x1100_0000 (sf=0) or 0x9100_0000 (sf=1)
        // Simplification: bits[28:24]=10001 and bit[29]=0:
        //   mask  = 0b0_1111_1_10_0000_0000_0000_0000_0000_0000 = 0x3FC0_0000? No.
        // Cleaner: just check bits[28:23] = 100010x (shift=00) or 100011x (shift=01)
        // i.e. (word >> 23) & 0x3F == 0x22 or 0x23  (sf irrelevant, lives in bit31)
        // Even simpler: mask away sf and the shift field, match the fixed bits.
        //   Fixed bits [28:24] = 1_0001 = 0x11
        //   bit[29] = 0 (S=0, ADD not ADDS)
        //   bits[23:22] = shift (00 or 01 are the only valid shifts for ADD imm)
        // Mask  = 0x1F80_0000 (bits [28:23]), match = 0x1100_0000 (shift=00)
        //      or match 0x1180_0000 (shift=01, lsl#12)
        // Accept both shifts:
        let add_check = word & 0x1F00_0000; // bits[28:24], sf/op stripped
        if add_check == 0x1100_0000 {
            // bits[29]=0 means ADD (not ADDS), bits[28:24]=10001
            // but we also need to ensure bit29=0 (not ADDS)
            if word & 0x2000_0000 == 0 {
                return Arm64Insn::AddImm(word);
            }
        }

        // LDR (unsigned offset, any size): size|111_001_01|imm12|Rn|Rt
        // Covers LDRB (size=00), LDRH (01), LDR W (10), LDR X (11).
        // V=0 (bit26) excludes SIMD/FP variants.
        // Mask: 0x3FC0_0000 (bits[29:22]), match: 0x3940_0000
        if word & 0x3FC0_0000 == 0x3940_0000 {
            return Arm64Insn::Ldr(word);
        }

        // STR (unsigned offset, any size): size|111_001_00|imm12|Rn|Rt
        // V=0 (bit26) excludes SIMD/FP variants.
        // Mask: 0x3FC0_0000, match: 0x3900_0000
        if word & 0x3FC0_0000 == 0x3900_0000 {
            return Arm64Insn::Str(word);
        }

        // LDR (literal) — PC-relative: imm19 at bits[23:5], scaled ×4.
        // bits[29:24]=011000, V=0. opc=bits[31:30]: 00=W, 01=X, 10=LDRSW.
        if word & 0x3F00_0000 == 0x1800_0000 {
            return Arm64Insn::LdrLiteral(word);
        }

        Arm64Insn::Other(word)
    }

    // ── Destination / base register ───────────────────────────────────────────

    /// Rd/Rt: bits [4:0].  Present for most encodings.
    /// Returns 31 for XZR/SP (we don't track those).
    #[inline]
    pub fn rd(&self) -> u8 {
        (self.word() & 0x1F) as u8
    }

    /// Rn (base/source register): bits [9:5].
    #[inline]
    pub fn rn(&self) -> u8 {
        ((self.word() >> 5) & 0x1F) as u8
    }

    // ── Branch targets ────────────────────────────────────────────────────────

    /// BL / B: sign-extended imm26, scaled by 4, added to PC.
    #[inline]
    pub fn imm26_target(&self, pc: u64) -> u64 {
        let imm26 = self.word() & 0x03FF_FFFF;
        // Sign-extend from bit 25
        let sext = (imm26 as i32) << 6 >> 6; // shift left to fill, arithmetic right to extend
        pc.wrapping_add((sext as i64 * 4) as u64)
    }

    /// B.cond: sign-extended imm19, bits [23:5], scaled by 4.
    #[inline]
    pub fn imm19_target(&self, pc: u64) -> u64 {
        let imm19 = (self.word() >> 5) & 0x0007_FFFF;
        let sext = (imm19 as i32) << 13 >> 13;
        pc.wrapping_add((sext as i64 * 4) as u64)
    }

    /// CBZ/CBNZ: same imm19 field as B.cond (bits [23:5]).
    #[inline]
    pub fn cbz_target(&self, pc: u64) -> u64 {
        self.imm19_target(pc)
    }

    /// TBZ/TBNZ: sign-extended imm14, bits [18:5], scaled by 4.
    #[inline]
    pub fn imm14_target(&self, pc: u64) -> u64 {
        let imm14 = (self.word() >> 5) & 0x0000_3FFF;
        let sext = (imm14 as i32) << 18 >> 18;
        pc.wrapping_add((sext as i64 * 4) as u64)
    }

    /// ADRP: page address.
    /// immhi = bits [23:5] (19 bits), immlo = bits [30:29] (2 bits).
    /// Full imm = immhi:immlo (21 bits), sign-extended, shifted left 12 → added to (PC & ~0xFFF).
    #[inline]
    pub fn adrp_page(&self, pc: u64) -> u64 {
        let word = self.word();
        let immlo = (word >> 29) & 0x3;
        let immhi = (word >> 5) & 0x0007_FFFF;
        let imm21 = (immhi << 2) | immlo;
        // Sign-extend from bit 20
        let sext = (imm21 as i32) << 11 >> 11;
        let page_pc = pc & !0xFFF;
        page_pc.wrapping_add((sext as i64 * 4096) as u64)
    }

    /// ADR: precise PC-relative address.
    /// Same split-imm layout as ADRP but NOT page-aligned, NOT scaled.
    #[inline]
    pub fn adr_target(&self, pc: u64) -> u64 {
        let word = self.word();
        let immlo = (word >> 29) & 0x3;
        let immhi = (word >> 5) & 0x0007_FFFF;
        let imm21 = (immhi << 2) | immlo;
        // Sign-extend from bit 20
        let sext = (imm21 as i32) << 11 >> 11;
        pc.wrapping_add(sext as u64)
    }

    // ── Memory operands ───────────────────────────────────────────────────────

    /// ADD imm: unsigned imm12, bits [21:10], with the shift field applied.
    ///
    /// The ARM64 ADD immediate encoding has a 2-bit shift field at [23:22]:
    ///   00 → imm12 is the literal immediate
    ///   01 → `ADD Xd, Xn, #imm, lsl #12` (imm12 shifted left 12)
    ///
    /// Unlike bad64 (which reports raw imm12 and exposes the shift separately),
    /// we return the fully resolved offset so callers can simply add it to the
    /// ADRP page address without worrying about the shift field.
    #[inline]
    pub fn add_imm(&self) -> u64 {
        let word = self.word();
        let imm12 = ((word >> 10) & 0xFFF) as u64;
        let shift = (word >> 22) & 0x3;
        if shift == 1 { imm12 << 12 } else { imm12 }
    }

    /// Access size for LDR/STR (unsigned offset): bits [31:30].
    /// 0 = byte, 1 = halfword, 2 = word (32-bit), 3 = doubleword (64-bit).
    #[inline]
    pub fn ldr_str_size(&self) -> u8 {
        (self.word() >> 30) as u8
    }

    /// LDR/STR (unsigned offset): unsigned imm12, bits [21:10], scaled by access size.
    /// Scale: byte=1, half=2, word=4, doubleword=8.
    #[inline]
    pub fn ldr_str_offset(&self) -> u64 {
        let imm12 = (self.word() >> 10) & 0xFFF;
        let scale = 1u64 << self.ldr_str_size();
        imm12 as u64 * scale
    }

    /// LDR (literal): sign-extended imm19 at bits [23:5], scaled by 4, added to PC.
    #[inline]
    pub fn ldr_literal_target(&self, pc: u64) -> u64 {
        let imm19 = (self.word() >> 5) & 0x0007_FFFF;
        let sext = (imm19 as i32) << 13 >> 13;
        pc.wrapping_add((sext as i64 * 4) as u64)
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    #[inline]
    fn word(&self) -> u32 {
        match self {
            Arm64Insn::Bl(w)
            | Arm64Insn::B(w)
            | Arm64Insn::BCond(w)
            | Arm64Insn::Cbz(w)
            | Arm64Insn::Cbnz(w)
            | Arm64Insn::Tbz(w)
            | Arm64Insn::Tbnz(w)
            | Arm64Insn::Adrp(w)
            | Arm64Insn::Adr(w)
            | Arm64Insn::AddImm(w)
            | Arm64Insn::Ldr(w)
            | Arm64Insn::Str(w)
            | Arm64Insn::LdrLiteral(w)
            | Arm64Insn::Blr(w)
            | Arm64Insn::Br(w)
            | Arm64Insn::Other(w) => *w,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Spot-check a few known encodings against hand-computed values.
    // The exhaustive fuzzer in src/bin/fuzz_arm64.rs checks all 2^32 words.

    #[test]
    fn bl_forward() {
        // BL +16 from 0x1000: imm26=4 → 0x94000004
        let w = 0x94000004u32;
        let insn = Arm64Insn::decode(w);
        assert!(matches!(insn, Arm64Insn::Bl(_)));
        assert_eq!(insn.imm26_target(0x1000), 0x1010);
    }

    #[test]
    fn bl_backward() {
        // BL -4 from 0x1004: imm26 = 0x3FFFFFF (−1 in signed 26-bit) → 0x97FFFFFF
        let w = 0x97FFFFFFu32;
        let insn = Arm64Insn::decode(w);
        assert!(matches!(insn, Arm64Insn::Bl(_)));
        assert_eq!(insn.imm26_target(0x1004), 0x1000);
    }

    #[test]
    fn b_jump() {
        // B +8 from 0x1008: imm26=2 → 0x14000002
        let w = 0x14000002u32;
        let insn = Arm64Insn::decode(w);
        assert!(matches!(insn, Arm64Insn::B(_)));
        assert_eq!(insn.imm26_target(0x1008), 0x1010);
    }

    #[test]
    fn adrp_page() {
        // ADRP X0, 1 (next page from 0x1000 → page 0x2000)
        // Encoding: 0xB0000000 (immhi=0, immlo=0 → imm21=1? No — let's compute)
        // ADRP X0, #0x1000: imm = +1 page = immlo=0b00, immhi=0b000_0000_0000_0000_0001
        // bits: [31]=1, [30:29]=immlo=00, [28:24]=10000, [23:5]=immhi=1, [4:0]=Rd=0
        // = 1_00_10000_0000000000000000001_00000 = 0x9000_0020? Let me be precise:
        // bit31=1, bits30:29=00, bits28:24=10000, bits23:5=0b000_0000_0000_0000_0001=1<<0, bits4:0=0
        // = 0b1_00_10000_0000000000000000001_00000
        //   31 30 29 28..24   23..5(19bits)       4..0
        // immhi at bits[23:5]: value=1 → bit5 set
        // So word = (1<<31)|(0<<29)|(0b10000<<24)|(1<<5)|(0) = 0x8000_0000 | 0x1000_0000 | 0x20
        // = 0x9000_0020
        let w = 0x9000_0020u32; // ADRP X0, 1 (immhi=1, immlo=0 → imm21=4 → +4 pages? wait)
                                // Actually: imm21 = (immhi<<2)|immlo = (1<<2)|0 = 4? That's +4 pages, not 1.
                                // For +1 page: imm21=1 = immhi:immlo where immlo=bits[30:29], immhi=bits[23:5]
                                // imm21=1 → immhi=0, immlo=1 (bits[30:29]=01)
                                // word = (1<<31)|(0b01<<29)|(0b10000<<24)|(0<<5)|(0) = 0x80000000|0x20000000|0x10000000 = 0xB0000000
        let w2 = 0xB000_0000u32; // ADRP X0, #0x1000 (from pc=0x1000, page_pc=0x1000, +1 page = 0x2000)
        let insn2 = Arm64Insn::decode(w2);
        assert!(matches!(insn2, Arm64Insn::Adrp(_)));
        assert_eq!(insn2.adrp_page(0x1000), 0x2000);
        let _ = w; // suppress unused warning from the exploratory calculation above
    }

    #[test]
    fn adrp_rd() {
        // ADRP X5, 0 — Rd=5, imm21=0
        // word = (1<<31)|(0b10000<<24)|(0<<5)|5 = 0x90000005
        let w = 0x9000_0005u32;
        let insn = Arm64Insn::decode(w);
        assert!(matches!(insn, Arm64Insn::Adrp(_)));
        assert_eq!(insn.rd(), 5);
    }

    #[test]
    fn add_imm_no_shift() {
        // ADD X0, X0, #0x100  (no shift)
        // sf=1, op=0, S=0 → bits[31:29]=100, bits[28:24]=10001
        // shift=00, imm12=0x100=256, Rn=0, Rd=0
        // word = (1<<31)|(0b10001<<24)|(0b00<<22)|(0x100<<10)|(0<<5)|0
        // = 0x80000000 | 0x11000000 | 0x00000000 | 0x00040000 | 0 | 0 = 0x91040000
        let w = 0x9104_0000u32;
        let insn = Arm64Insn::decode(w);
        assert!(matches!(insn, Arm64Insn::AddImm(_)));
        assert_eq!(insn.add_imm(), 0x100);
        assert_eq!(insn.rd(), 0);
        assert_eq!(insn.rn(), 0);
    }

    #[test]
    fn blr_rn() {
        // BLR X8 — Rn=8, Rd=0 (must be 0), bits[9:5]=8
        // word = 0xD63F0100
        let w = 0xD63F_0100u32;
        let insn = Arm64Insn::decode(w);
        assert!(matches!(insn, Arm64Insn::Blr(_)));
        assert_eq!(insn.rn(), 8);
    }

    #[test]
    fn ldr_offset() {
        // LDR X0, [X1, #8]  — 64-bit unsigned offset: imm12=1, scaled by 8 → offset=8
        // size=11, V=0, opc=01: bits[31:24]=0xF9, then bits[23:22]=01 (opc)
        // imm12=1 at bits[21:10], Rn=1 at bits[9:5], Rt=0 at bits[4:0]
        // word = 0xF940_0020? Let's compute:
        // 0xF9=1111_1001 for bits[31:24], opc=01 bits[23:22] → 0xF940_0000 base
        // imm12=1 at bits[21:10]: 1<<10 = 0x400
        // Rn=1 at bits[9:5]: 1<<5 = 0x20
        // Rt=0
        // word = 0xF940_0000 | 0x400 | 0x20 = 0xF940_0420
        let w = 0xF940_0420u32;
        let insn = Arm64Insn::decode(w);
        assert!(matches!(insn, Arm64Insn::Ldr(_)));
        assert_eq!(insn.ldr_str_offset(), 8);
        assert_eq!(insn.rn(), 1);
        assert_eq!(insn.rd(), 0);
    }

    #[test]
    fn bcond_target() {
        // B.EQ -4 from 0x1008: imm19 = 0x7FFFF (−1 in signed 19-bit)
        // bits[31:24]=0x54, bit[4]=0 (cond=EQ=0b0000)
        // imm19 at bits[23:5], cond at bits[3:0]
        // imm19=0x7FFFF → bits[23:5] = 0x7FFFF<<5 = 0x00FFFFE0
        // word = 0x54000000 | 0x00FFFFE0 | 0x0 = 0x54FFFFE0
        let w = 0x54FF_FFE0u32;
        let insn = Arm64Insn::decode(w);
        assert!(matches!(insn, Arm64Insn::BCond(_)));
        assert_eq!(insn.imm19_target(0x1008), 0x1004);
    }
}
