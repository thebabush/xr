//! ARM32 and Thumb-2 linear cross-reference scanners.
//!
//! # ARM32 (A32) mode
//!
//! All instructions are 4 bytes, little-endian. PC reads return
//! `instruction_address + 8`.
//!
//! Instructions decoded:
//! - `B`   `cond 1010 imm24` — Jump
//! - `BL`  `cond 1011 imm24` — Call
//! - `BLX` `1111 101H imm24` — Call (unconditional, crosses to Thumb)
//! - `LDR` PC-relative `cond 0101 U001 1111 Rt imm12` — DataRead
//!
//! # Thumb-2 mode
//!
//! Variable-width: 16-bit halfwords when `hw < 0xE800`, 32-bit pairs otherwise.
//! PC reads return `instruction_address + 4`.
//!
//! 16-bit instructions decoded:
//! - `B` T1 conditional `1101 cond imm8` — Jump
//! - `B` T2 unconditional `11100 imm11` — Jump
//! - `LDR` T1 literal `01001 Rt imm8` — DataRead
//!
//! 32-bit instructions decoded (first halfword `& 0xF800 == 0xF000`):
//! - `BL`      hw2 `& 0xD000 == 0xD000` — Call
//! - `BLX`     hw2 `& 0xD000 == 0xC000` — Call (to ARM32)
//! - `B.W`     hw2 `& 0xD000 == 0x9000` — Jump
//! - `B.cond.W` hw2 `& 0xD000 == 0x8000` — Jump
//! - `LDR.W` literal `hw1 & 0xFF7F == 0xF85F` — DataRead

use crate::arch::{SegmentIndex, ScanRegion};
use crate::va::Va;
use crate::xref::{Confidence, Xref, XrefKind};

// ── ARM32 (A32) scanner ───────────────────────────────────────────────────────

/// Linear scan of a 4-byte-aligned ARM32 code region.
pub(crate) fn scan_arm32(region: &ScanRegion, seg_idx: &SegmentIndex) -> Vec<Xref> {
    let data = region.data;
    let base = region.base_va;
    let mut xrefs = Vec::new();

    let mut i = 0usize;
    while i + 3 < data.len() {
        let pc = base + i as u64;
        let word = u32::from_le_bytes(data[i..i + 4].try_into().unwrap());
        i += 4;

        // ── B / BL ────────────────────────────────────────────────────────
        // Bits[27:25] = 101; bit[24] = link bit.
        // Condition code in bits[31:28]; BLX immediate uses cond=1111 so skip.
        let op24 = word & 0x0F000000;
        if op24 == 0x0B000000 {
            // BL — call
            let target = a32_branch_target(pc.raw(), word);
            if seg_idx.is_exec(Va::new(target)) {
                xrefs.push(xref(pc, target, XrefKind::Call));
            }
        } else if op24 == 0x0A000000 {
            // B — jump (conditional or unconditional depending on cond field)
            let target = a32_branch_target(pc.raw(), word);
            if seg_idx.is_exec(Va::new(target)) {
                xrefs.push(xref(pc, target, XrefKind::Jump));
            }
        } else if word & 0xFE000000 == 0xFA000000 {
            // BLX immediate — unconditional call that switches to Thumb.
            // Encoding: `1111 101H imm24`; H adds a half-word offset.
            let h = ((word >> 24) & 1) as i64;
            let imm24 = word & 0x00FF_FFFF;
            let offset = a32_signext24(imm24) * 4 + h * 2;
            // Clear bit 0: the target is a Thumb address whose LSB indicates
            // mode in the symbol table but is not part of the actual address.
            let target = ((pc.raw() as i64 + 8 + offset) as u64) & !1u64;
            if seg_idx.is_exec(Va::new(target)) {
                xrefs.push(xref(pc, target, XrefKind::Call));
            }
        } else if word & 0x0F7F_0000 == 0x051F_0000 {
            // LDR word, PC-relative: `cond 0101 U001 1111 Rt imm12`
            // Bit[23] = U (1 = add, 0 = subtract).
            let u = (word >> 23) & 1;
            let imm12 = (word & 0xFFF) as u64;
            let target = if u != 0 {
                pc.raw() + 8 + imm12
            } else {
                pc.raw() + 8 - imm12
            };
            if seg_idx.contains(Va::new(target)) {
                xrefs.push(xref(pc, target, XrefKind::DataRead));
            }
        }
    }

    xrefs
}

/// Decode a B/BL branch target. `pc` is the instruction address.
/// ARM32 PC = instruction_address + 8 when read by an instruction.
#[inline]
fn a32_branch_target(pc: u64, word: u32) -> u64 {
    let imm24 = word & 0x00FF_FFFF;
    let offset = a32_signext24(imm24) * 4;
    (pc as i64 + 8 + offset) as u64
}

/// Sign-extend a 24-bit value to 64-bit signed.
#[inline]
fn a32_signext24(imm24: u32) -> i64 {
    if imm24 & 0x80_0000 != 0 {
        // Set all upper bits.
        (imm24 as i64) | ((!0u64 as i64) << 24)
    } else {
        imm24 as i64
    }
}

// ── Thumb-2 scanner ───────────────────────────────────────────────────────────

/// Linear scan of a Thumb-2 code region (mixed 16/32-bit halfwords).
pub(crate) fn scan_thumb(region: &ScanRegion, seg_idx: &SegmentIndex) -> Vec<Xref> {
    let data = region.data;
    let base = region.base_va;
    let mut xrefs = Vec::new();

    let mut i = 0usize;
    while i + 1 < data.len() {
        let pc = base + i as u64;
        let hw1 = u16::from_le_bytes(data[i..i + 2].try_into().unwrap());

        // A halfword ≥ 0xE800 is the first half of a 32-bit instruction.
        if hw1 >= 0xE800 {
            if i + 3 >= data.len() {
                break;
            }
            let hw2 = u16::from_le_bytes(data[i + 2..i + 4].try_into().unwrap());
            i += 4;

            if hw1 & 0xF800 == 0xF000 {
                // BL / BLX / B.W / B.cond.W — classified by bits[15:12] of hw2.
                match hw2 & 0xD000 {
                    0xD000 => {
                        // BL T1 — call to Thumb
                        let target = thumb_bl_target(pc.raw(), hw1, hw2, false);
                        if seg_idx.is_exec(Va::new(target)) {
                            xrefs.push(xref(pc, target, XrefKind::Call));
                        }
                    }
                    0xC000 => {
                        // BLX T2 — call to ARM32; target must be 4-byte aligned.
                        let target = thumb_bl_target(pc.raw(), hw1, hw2, true);
                        if seg_idx.is_exec(Va::new(target)) {
                            xrefs.push(xref(pc, target, XrefKind::Call));
                        }
                    }
                    0x9000 => {
                        // B.W T4 — unconditional jump
                        let target = thumb_bl_target(pc.raw(), hw1, hw2, false);
                        if seg_idx.is_exec(Va::new(target)) {
                            xrefs.push(xref(pc, target, XrefKind::Jump));
                        }
                    }
                    0x8000 => {
                        // B.cond.W T3 — conditional jump
                        // cond lives in hw1 bits[9:6]; 0xE/0xF are reserved.
                        let cond = (hw1 >> 6) & 0xF;
                        if cond != 0xE && cond != 0xF {
                            let target = thumb_bcond_w_target(pc.raw(), hw1, hw2);
                            if seg_idx.is_exec(Va::new(target)) {
                                xrefs.push(xref(pc, target, XrefKind::Jump));
                            }
                        }
                    }
                    _ => {}
                }
            } else if hw1 & 0xFF7F == 0xF85F {
                // LDR.W T2 literal: `11111000 U101 1111` — DataRead.
                // U = bit[7] of hw1; imm12 from hw2 bits[11:0].
                let u = (hw1 >> 7) & 1;
                let imm12 = (hw2 & 0xFFF) as u64;
                // PC is aligned to 4 bytes for literal loads.
                let align_pc = (pc.raw() + 4) & !3u64;
                let target = if u != 0 {
                    align_pc + imm12
                } else {
                    align_pc.wrapping_sub(imm12)
                };
                if seg_idx.contains(Va::new(target)) {
                    xrefs.push(xref(pc, target, XrefKind::DataRead));
                }
            }
            // All other 32-bit instructions: advance past them (i already incremented).
        } else {
            i += 2;
            // 16-bit instruction.
            if hw1 & 0xF000 == 0xD000 {
                // B T1 — conditional branch: `1101 cond imm8`.
                let cond = (hw1 >> 8) & 0xF;
                if cond != 0xE && cond != 0xF {
                    let imm8 = (hw1 & 0xFF) as i8; // sign-extend 8 bits
                    let offset = (imm8 as i32) * 2;
                    let target = (pc.raw() as i64 + 4 + offset as i64) as u64;
                    if seg_idx.is_exec(Va::new(target)) {
                        xrefs.push(xref(pc, target, XrefKind::Jump));
                    }
                }
            } else if hw1 & 0xF800 == 0xE000 {
                // B T2 — unconditional branch: `11100 imm11`.
                let imm11 = hw1 & 0x7FF;
                let offset = thumb_signext11(imm11) * 2;
                let target = (pc.raw() as i64 + 4 + offset) as u64;
                if seg_idx.is_exec(Va::new(target)) {
                    xrefs.push(xref(pc, target, XrefKind::Jump));
                }
            } else if hw1 & 0xF800 == 0x4800 {
                // LDR T1 — PC-relative literal: `01001 Rt imm8`.
                // Effective address = Align(PC, 4) + imm8 * 4.
                let imm8 = (hw1 & 0xFF) as u64;
                let align_pc = (pc.raw() + 4) & !3u64;
                let target = align_pc + imm8 * 4;
                if seg_idx.contains(Va::new(target)) {
                    xrefs.push(xref(pc, target, XrefKind::DataRead));
                }
            }
            // All other 16-bit instructions: already advanced, nothing to emit.
        }
    }

    xrefs
}

// ── Thumb-2 offset helpers ────────────────────────────────────────────────────

/// Compute the BL/BLX/B.W target using the S:I1:I2:imm10:imm11:0 formula.
///
/// For BLX (`is_blx = true`) the target is 4-byte aligned (ARM32 mode entry);
/// for BL/B.W it is 2-byte aligned (Thumb entry).
#[inline]
fn thumb_bl_target(pc: u64, hw1: u16, hw2: u16, is_blx: bool) -> u64 {
    let s = ((hw1 >> 10) & 1) as u32;
    let imm10 = (hw1 & 0x3FF) as u32;
    let j1 = ((hw2 >> 13) & 1) as u32;
    let j2 = ((hw2 >> 11) & 1) as u32;
    // For BLX bit[0] of hw2 is always 0 per the encoding, but masking to 0x7FE
    // is equivalent: the last '0' bit makes the offset 4-byte aligned.
    let imm11 = if is_blx {
        (hw2 & 0x7FE) as u32
    } else {
        (hw2 & 0x7FF) as u32
    };

    let i1 = 1 ^ (j1 ^ s);
    let i2 = 1 ^ (j2 ^ s);

    // 25-bit signed value: S : I1 : I2 : imm10 : imm11 : 0
    let raw = (s << 24) | (i1 << 23) | (i2 << 22) | (imm10 << 12) | (imm11 << 1);
    let offset: i32 = if s != 0 {
        raw as i32 - (1i32 << 25)
    } else {
        raw as i32
    };

    let target = (pc as i64 + 4 + offset as i64) as u64;
    // BLX targets ARM32; bit[1] must be 0 (4-byte aligned).
    if is_blx { target & !3u64 } else { target }
}

/// Compute B.cond.W (T3) target using the S:J2:J1:imm6:imm11:0 formula.
#[inline]
fn thumb_bcond_w_target(pc: u64, hw1: u16, hw2: u16) -> u64 {
    let s = ((hw1 >> 10) & 1) as u32;
    let imm6 = (hw1 & 0x3F) as u32;
    let j1 = ((hw2 >> 13) & 1) as u32;
    let j2 = ((hw2 >> 11) & 1) as u32;
    let imm11 = (hw2 & 0x7FF) as u32;

    // 21-bit signed value: S : J2 : J1 : imm6 : imm11 : 0
    let raw = (s << 20) | (j2 << 19) | (j1 << 18) | (imm6 << 12) | (imm11 << 1);
    let offset: i32 = if s != 0 {
        raw as i32 - (1i32 << 21)
    } else {
        raw as i32
    };

    (pc as i64 + 4 + offset as i64) as u64
}

/// Sign-extend an 11-bit Thumb immediate to a 64-bit signed value.
#[inline]
fn thumb_signext11(imm11: u16) -> i64 {
    if imm11 & 0x400 != 0 {
        // Negative: set all upper bits.
        (imm11 as i64) | ((!0u64 as i64) << 11)
    } else {
        imm11 as i64
    }
}

// ── Shared helper ─────────────────────────────────────────────────────────────

#[inline]
fn xref(from: Va, to: u64, kind: XrefKind) -> Xref {
    Xref {
        from,
        to: Va::new(to),
        kind,
        confidence: Confidence::LinearImmediate,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::SegmentIndex;
    use crate::loader::{Arch, DecodeMode, LoadedBinary, SegData, Segment};
    use crate::va::Va;

    fn make_exec_seg(va: u64, data: Vec<u8>) -> Segment {
        let leaked: &'static [u8] = Box::leak(data.into_boxed_slice());
        Segment {
            va: Va::new(va),
            data: unsafe { SegData::new_for_test(leaked) },
            executable: true,
            readable: true,
            writable: false,
            byte_scannable: false,
            mode: DecodeMode::Default,
            name: ".text".to_string(),
        }
    }

    // ── ARM32 tests ───────────────────────────────────────────────────────────

    #[test]
    fn arm32_bl_forward() {
        // BL +8 at pc=0x1000: target = 0x1000 + 8 + 8 = 0x1010
        // imm24 = 1 (offset = 4), so BL #4 at 0x1000 -> target 0x100C
        // Let's do BL #8 at 0x1000 -> imm24 = 2 -> target = 0x1000+8+8 = 0x1010
        let imm24: u32 = 2; // offset = 8
        let word = 0xEB00_0000u32 | imm24; // cond=0xE (always), L=1
        let code = word.to_le_bytes().to_vec();
        let seg = make_exec_seg(0x1000, [code, vec![0u8; 0x20]].concat());
        let binary = LoadedBinary::from_segments(Arch::Arm32, vec![seg]);
        let seg_idx = SegmentIndex::build(&binary.segments);
        let region = ScanRegion::new(&binary.segments[0], Va::new(0x1000), Va::new(0x1020));
        let xrefs = scan_arm32(&region, &seg_idx);
        assert_eq!(xrefs.len(), 1);
        assert_eq!(xrefs[0].from, Va::new(0x1000));
        assert_eq!(xrefs[0].to, Va::new(0x1010));
        assert_eq!(xrefs[0].kind, XrefKind::Call);
    }

    #[test]
    fn arm32_b_backward() {
        // B -8 at pc=0x1010: target = 0x1010 + 8 + (-8) = 0x1010
        // imm24 = -3 = 0xFFFFFF + (-3 + 1) = sign-extended
        // offset = -8 means imm24*4 = -8 => imm24 = -2 = 0xFFFFFE
        let imm24: u32 = 0xFFFFFEu32; // -2 sign-extended 24-bit
        let word = 0xEA00_0000u32 | imm24; // B always
        let mut code = vec![0u8; 0x10]; // padding
        code.extend_from_slice(&word.to_le_bytes()); // at offset 0x10 = VA 0x1010
        code.extend_from_slice(&[0u8; 4]);
        let seg = make_exec_seg(0x1000, code);
        let binary = LoadedBinary::from_segments(Arch::Arm32, vec![seg]);
        let seg_idx = SegmentIndex::build(&binary.segments);
        let region = ScanRegion::new(&binary.segments[0], Va::new(0x1000), Va::new(0x1018));
        let xrefs = scan_arm32(&region, &seg_idx);
        assert_eq!(xrefs.len(), 1);
        assert_eq!(xrefs[0].from, Va::new(0x1010));
        assert_eq!(xrefs[0].to, Va::new(0x1010)); // 0x1010 + 8 + (-2*4) = 0x1010
        assert_eq!(xrefs[0].kind, XrefKind::Jump);
    }

    #[test]
    fn arm32_ldr_literal() {
        // LDR r0, [PC, #4] at 0x1000 -> target = 0x1000 + 8 + 4 = 0x100C
        // Encoding: cond=0xE, 0101_U001_1111_Rt_imm12 with U=1, Rt=0, imm12=4
        let word = 0xE59F_0004u32; // LDR r0, [PC, #+4]
        let mut code = word.to_le_bytes().to_vec();
        code.extend_from_slice(&[0u8; 0x10]);
        let seg = make_exec_seg(0x1000, code);
        let binary = LoadedBinary::from_segments(Arch::Arm32, vec![seg]);
        let seg_idx = SegmentIndex::build(&binary.segments);
        let region = ScanRegion::new(&binary.segments[0], Va::new(0x1000), Va::new(0x1014));
        let xrefs = scan_arm32(&region, &seg_idx);
        assert_eq!(xrefs.len(), 1);
        assert_eq!(xrefs[0].from, Va::new(0x1000));
        assert_eq!(xrefs[0].to, Va::new(0x100C));
        assert_eq!(xrefs[0].kind, XrefKind::DataRead);
    }

    // ── Thumb-2 tests ─────────────────────────────────────────────────────────

    fn seg_thumb(va: u64, halfwords: &[u16]) -> Segment {
        let bytes: Vec<u8> = halfwords
            .iter()
            .flat_map(|hw| hw.to_le_bytes())
            .collect();
        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        Segment {
            va: Va::new(va),
            data: unsafe { SegData::new_for_test(leaked) },
            executable: true,
            readable: true,
            writable: false,
            byte_scannable: false,
            mode: DecodeMode::Thumb,
            name: ".text".to_string(),
        }
    }

    #[test]
    fn thumb_b_t2_unconditional() {
        // B.N (T2) 16-bit at 0x1000: imm11 = 2, offset = +4, target = 0x1000+4+4 = 0x1008
        let hw = 0xE000u16 | 2; // B #4
        let seg = seg_thumb(0x1000, &[hw, 0, 0, 0, 0, 0, 0, 0]);
        let binary = LoadedBinary::from_segments(Arch::Arm32, vec![seg]);
        let seg_idx = SegmentIndex::build(&binary.segments);
        let region = ScanRegion::new(&binary.segments[0], Va::new(0x1000), Va::new(0x1010));
        let xrefs = scan_thumb(&region, &seg_idx);
        assert_eq!(xrefs.len(), 1);
        assert_eq!(xrefs[0].from, Va::new(0x1000));
        assert_eq!(xrefs[0].to, Va::new(0x1008));
        assert_eq!(xrefs[0].kind, XrefKind::Jump);
    }

    #[test]
    fn thumb_b_t1_conditional() {
        // B.EQ (cond=0) at 0x1000: imm8=4, offset=+8, target=0x1000+4+8=0x100C
        let hw = 0xD000u16 | 4; // BEQ #8
        let seg = seg_thumb(0x1000, &[hw, 0, 0, 0, 0, 0, 0, 0]);
        let binary = LoadedBinary::from_segments(Arch::Arm32, vec![seg]);
        let seg_idx = SegmentIndex::build(&binary.segments);
        let region = ScanRegion::new(&binary.segments[0], Va::new(0x1000), Va::new(0x1010));
        let xrefs = scan_thumb(&region, &seg_idx);
        assert_eq!(xrefs.len(), 1);
        assert_eq!(xrefs[0].from, Va::new(0x1000));
        assert_eq!(xrefs[0].to, Va::new(0x100C));
        assert_eq!(xrefs[0].kind, XrefKind::Jump);
    }

    #[test]
    fn thumb_bl_t1() {
        // BL forward: encode BL to target 0x1010 from pc 0x1000.
        // offset = 0x1010 - (0x1000 + 4) = 12 = 0xC
        // raw25 = S:I1:I2:imm10:imm11:0 = 0:0:0:0:6:0 => value 12
        //   => S=0, I1=0, I2=0, imm10=0, imm11=6
        // J1 = NOT(I1 EOR S) = NOT(0 EOR 0) = 1
        // J2 = NOT(I2 EOR S) = NOT(0 EOR 0) = 1
        // hw1 = 0xF000 | (S<<10) | imm10 = 0xF000
        // hw2 bits: 11 J1 1 J2 imm11
        //         = 11  1  1  1  00000000110 = 1111_1000_0000_0110 = 0xF806
        let hw1 = 0xF000u16;
        let hw2 = 0xF806u16; // J1=1, bit12=1, J2=1, imm11=6
        let seg = seg_thumb(0x1000, &[hw1, hw2, 0, 0, 0, 0, 0, 0, 0, 0]);
        let binary = LoadedBinary::from_segments(Arch::Arm32, vec![seg]);
        let seg_idx = SegmentIndex::build(&binary.segments);
        let region = ScanRegion::new(&binary.segments[0], Va::new(0x1000), Va::new(0x1014));
        let xrefs = scan_thumb(&region, &seg_idx);
        assert!(!xrefs.is_empty(), "expected at least one BL xref");
        let bl = xrefs.iter().find(|x| x.kind == XrefKind::Call).unwrap();
        assert_eq!(bl.from, Va::new(0x1000));
        assert_eq!(bl.to, Va::new(0x1010));
    }

    #[test]
    fn thumb_ldr_t1_literal() {
        // LDR T1 at 0x1000: `01001 Rt imm8` where Rt=0, imm8=1
        // target = Align(0x1000+4, 4) + 1*4 = 0x1004 + 4 = 0x1008
        let hw = 0x4801u16; // LDR r0, [PC, #4]
        let seg = seg_thumb(0x1000, &[hw, 0, 0, 0, 0, 0, 0, 0]);
        let binary = LoadedBinary::from_segments(Arch::Arm32, vec![seg]);
        let seg_idx = SegmentIndex::build(&binary.segments);
        let region = ScanRegion::new(&binary.segments[0], Va::new(0x1000), Va::new(0x1010));
        let xrefs = scan_thumb(&region, &seg_idx);
        assert_eq!(xrefs.len(), 1);
        assert_eq!(xrefs[0].from, Va::new(0x1000));
        assert_eq!(xrefs[0].to, Va::new(0x1008));
        assert_eq!(xrefs[0].kind, XrefKind::DataRead);
    }
}
