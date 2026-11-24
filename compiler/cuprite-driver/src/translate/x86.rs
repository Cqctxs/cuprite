//! Lowering x86 SIMD instructions with the host instruction's semantics.

use cuprite_kernel::{
    BinOp as KBinOp, Const, Intrinsic, Local as KLocal, Operand as KOperand, Rvalue as KRvalue,
    Ty as KTy, UnOp as KUnOp,
};
use rustc_middle::mir::{Operand, Place};

use super::types::held_ty;
use super::{unsigned_of, Frame, Result, Slot, Translator};

impl<'tcx> Translator<'tcx> {
    /// An x86 SIMD instruction, called through its LLVM intrinsic: what the
    /// instruction computes, from Intel's definitions, lane by lane. The
    /// floating-point environment is Rust's default (round to nearest even,
    /// exceptions masked). Instructions whose results differ between CPU
    /// models (`rcpps`, `rsqrtps`) are refused.
    pub(super) fn x86(
        &mut self,
        frame: &mut Frame<'_, 'tcx>,
        symbol: &str,
        args: &[rustc_span::Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
    ) -> Result<()> {
        let values = args.iter().map(|arg| self.operand_slot(frame, &arg.node)).collect::<Result<Vec<_>>>()?;
        let refused = |why: &str| format!("calls the x86 instruction `{symbol}`, which {why}");
        let unexpected = || refused("has unexpected arguments");
        let arg = |index: usize| values.get(index).cloned().ok_or_else(unexpected);
        let dest_ty = destination.ty(frame.body, self.tcx).ty;
        let name = symbol.strip_prefix("llvm.x86.").unwrap_or(symbol);
        let family = if name.starts_with("vcvt") { name } else { name.split_once('.').map_or(name, |(_, rest)| rest) };

        // Scalar conversions of lane 0.
        if let Some(kind) = ["cvtss2si", "cvttss2si", "cvtsd2si", "cvttsd2si"]
            .into_iter()
            .find(|kind| family == *kind || family == format!("{kind}64"))
        {
            let out = held_ty(dest_ty).ok_or_else(unexpected)?;
            let lane = *self.lane_numbers(&arg(0)?)?.first().ok_or_else(unexpected)?;
            let value = self.float_to_int_indefinite(lane, out, !kind.starts_with("cvtt"));
            let dest = self.place(frame, destination)?;
            return self.write_number(dest, KRvalue::Use(value.into()));
        }

        let result = self.fresh(dest_ty)?;
        let out: Vec<KLocal> = self
            .lanes(&result)?
            .into_iter()
            .map(|lane| match lane {
                Slot::Scalar(local) => Ok(local),
                _ => Err(unexpected()),
            })
            .collect::<Result<_>>()?;
        let out_ty = out.first().map(|&lane| self.b.ty(lane)).ok_or_else(unexpected)?;
        let immediate = |this: &mut Self, index: usize| -> Result<u64> {
            let operand = args.get(index).map(|arg| &arg.node).ok_or_else(unexpected)?;
            this.known(operand).map(|c| c.bits as u64).ok_or_else(|| refused("takes an immediate known only at run time"))
        };
        let lanes: Vec<KOperand> = match family {
            // Comparisons: a lane of all ones where the predicate holds.
            "cmp.ps" | "cmp.pd" | "cmp.ps.256" | "cmp.pd.256" | "cmp.ss" | "cmp.sd" => {
                let predicate = immediate(self, 2)?;
                let (a, b) = (self.lane_numbers(&arg(0)?)?, self.lane_numbers(&arg(1)?)?);
                let scalar = family.ends_with(".ss") || family.ends_with(".sd");
                let mut lanes = Vec::new();
                for (index, (&x, &y)) in a.iter().zip(&b).enumerate() {
                    if scalar && index > 0 {
                        lanes.push(x);
                        continue;
                    }
                    let holds = self.x86_compare(predicate, x, y);
                    lanes.push(self.mask(holds, out_ty).into());
                }
                lanes
            }
            // `a < b ? a : b`: the second operand when either is NaN or both
            // are zeros, unlike `f32::min`.
            "min.ps" | "min.pd" | "min.ps.256" | "min.pd.256" | "min.ss" | "min.sd" | "max.ps" | "max.pd"
            | "max.ps.256" | "max.pd.256" | "max.ss" | "max.sd" => {
                let op = if family.starts_with("min") { KBinOp::Lt } else { KBinOp::Gt };
                let scalar = family.ends_with(".ss") || family.ends_with(".sd");
                let (a, b) = (self.lane_numbers(&arg(0)?)?, self.lane_numbers(&arg(1)?)?);
                let mut lanes = Vec::new();
                for (index, (&x, &y)) in a.iter().zip(&b).enumerate() {
                    if scalar && index > 0 {
                        lanes.push(x);
                        continue;
                    }
                    let first = self.b.binary(op, x, y);
                    lanes.push(self.b.select(first, x, y).into());
                }
                lanes
            }
            // Float to `i32` lanes; out of range or NaN gives `i32::MIN`.
            "cvtps2dq" | "cvttps2dq" | "cvt.ps2dq.256" | "cvtt.ps2dq.256" | "cvtpd2dq" | "cvttpd2dq" => {
                let round = !family.starts_with("cvtt");
                let a = self.lane_numbers(&arg(0)?)?;
                let mut lanes = Vec::new();
                for index in 0..out.len() {
                    lanes.push(match a.get(index) {
                        Some(&x) => self.float_to_int_indefinite(x, out_ty, round).into(),
                        None => Const::int(out_ty, 0).into(),
                    });
                }
                lanes
            }
            // The high half of each 32-bit product.
            "pmulh.w" | "pmulhu.w" => {
                let wide = if family == "pmulh.w" { KTy::I32 } else { KTy::U32 };
                let (a, b) = (self.lane_numbers(&arg(0)?)?, self.lane_numbers(&arg(1)?)?);
                let mut lanes = Vec::new();
                for (x, y) in a.into_iter().zip(b) {
                    let (wx, wy) = (self.b.cast(x, wide), self.b.cast(y, wide));
                    let product = self.b.binary(KBinOp::Mul, wx, wy);
                    let high = self.b.binary(KBinOp::Shr, product, Const::int(KTy::U32, 16));
                    lanes.push(self.b.cast(high, out_ty).into());
                }
                lanes
            }
            // `(a + b + 1) >> 1`, without overflow.
            "pavg.b" | "pavg.w" => {
                let (a, b) = (self.lane_numbers(&arg(0)?)?, self.lane_numbers(&arg(1)?)?);
                let mut lanes = Vec::new();
                for (x, y) in a.into_iter().zip(b) {
                    let unsigned = unsigned_of(u64::from(out_ty.bytes()));
                    let (ux, uy) = (self.b.cast(x, unsigned), self.b.cast(y, unsigned));
                    let (wx, wy) = (self.b.cast(ux, KTy::U32), self.b.cast(uy, KTy::U32));
                    let sum = self.b.binary(KBinOp::Add, wx, wy);
                    let rounded = self.b.binary(KBinOp::Add, sum, Const::int(KTy::U32, 1));
                    let halved = self.b.binary(KBinOp::Shr, rounded, Const::int(KTy::U32, 1));
                    lanes.push(self.b.cast(halved, out_ty).into());
                }
                lanes
            }
            // Bits 16..1 of `(a * b >> 14) + 1`.
            "pmul.hr.sw.128" | "pmul.hr.sw" => {
                let (a, b) = (self.lane_numbers(&arg(0)?)?, self.lane_numbers(&arg(1)?)?);
                let mut lanes = Vec::new();
                for (x, y) in a.into_iter().zip(b) {
                    let (wx, wy) = (self.b.cast(x, KTy::I32), self.b.cast(y, KTy::I32));
                    let product = self.b.binary(KBinOp::Mul, wx, wy);
                    let shifted = self.b.binary(KBinOp::Shr, product, Const::int(KTy::U32, 14));
                    let rounded = self.b.binary(KBinOp::Add, shifted, Const::int(KTy::I32, 1));
                    let halved = self.b.binary(KBinOp::Shr, rounded, Const::int(KTy::U32, 1));
                    lanes.push(self.b.cast(halved, out_ty).into());
                }
                lanes
            }
            // Adjacent `i16` products, summed into `i32` (wrapping only for
            // `-32768 * -32768` twice, as the instruction does).
            "pmadd.wd" => {
                let (a, b) = (self.lane_numbers(&arg(0)?)?, self.lane_numbers(&arg(1)?)?);
                let mut lanes = Vec::new();
                for pair in 0..out.len() {
                    let mut sum: KOperand = Const::int(KTy::I32, 0).into();
                    for index in [2 * pair, 2 * pair + 1] {
                        let (x, y) = (*a.get(index).ok_or_else(unexpected)?, *b.get(index).ok_or_else(unexpected)?);
                        let (wx, wy) = (self.b.cast(x, KTy::I32), self.b.cast(y, KTy::I32));
                        let product = self.b.binary(KBinOp::Mul, wx, wy);
                        sum = self.b.binary(KBinOp::Add, sum, product).into();
                    }
                    lanes.push(sum);
                }
                lanes
            }
            // Narrowed with saturation: `a`'s lanes then `b`'s, per 128 bits.
            "packsswb.128" | "packssdw.128" | "packuswb.128" | "packusdw" | "packsswb" | "packssdw" | "packuswb" => {
                let (a, b) = (self.lane_numbers(&arg(0)?)?, self.lane_numbers(&arg(1)?)?);
                let per_block = a.len().min(128 / a.first().map_or(128, |&x| self.b.ty(x).bits() as usize));
                let (min, max) = if out_ty.is_signed() {
                    (-(1i128 << (out_ty.bits() - 1)), (1i128 << (out_ty.bits() - 1)) - 1)
                } else {
                    (0, (1i128 << out_ty.bits()) - 1)
                };
                let mut lanes = Vec::new();
                for block in 0..a.len() / per_block.max(1) {
                    for source in [&a, &b] {
                        for &x in &source[block * per_block..(block + 1) * per_block] {
                            let wide = self.b.cast(x, KTy::I64);
                            let below = self.b.binary(KBinOp::Lt, wide, Const::int(KTy::I64, min));
                            let above = self.b.binary(KBinOp::Gt, wide, Const::int(KTy::I64, max));
                            let upper = self.b.select(above, Const::int(KTy::I64, max), wide);
                            let clamped = self.b.select(below, Const::int(KTy::I64, min), upper);
                            lanes.push(self.b.cast(clamped, out_ty).into());
                        }
                    }
                }
                lanes
            }
            // Bytes of `a` chosen by `b`, within each 128 bits; zero where
            // `b`'s byte has its top bit set.
            "pshuf.b.128" | "pshuf.b" => {
                let a = self.lanes(&arg(0)?)?;
                let b = self.lane_numbers(&arg(1)?)?;
                let mut lanes = Vec::new();
                for (index, &selector) in b.iter().enumerate() {
                    let block = index / 16 * 16;
                    let options = a.get(block..block + 16).ok_or_else(unexpected)?.to_vec();
                    let ty = self.b.ty(selector);
                    let low = self.b.binary(KBinOp::And, selector, Const::int(ty, 15));
                    let at = self.b.cast(low, KTy::U64);
                    let Slot::Scalar(picked) = self.select_among(at, &options)? else { return Err(unexpected()) };
                    let top = self.b.binary(KBinOp::And, selector, Const::int(ty, 0x80));
                    let cleared = self.b.binary(KBinOp::Ne, top, Const::int(ty, 0));
                    lanes.push(self.b.select(cleared, Const::int(out_ty, 0), picked).into());
                }
                lanes
            }
            // Sums of absolute byte differences, per 8 bytes, into `u64`s.
            "psad.bw" => {
                let (a, b) = (self.lane_numbers(&arg(0)?)?, self.lane_numbers(&arg(1)?)?);
                let mut lanes = Vec::new();
                for group in 0..out.len() {
                    let mut sum: KOperand = Const::int(KTy::U64, 0).into();
                    for index in group * 8..group * 8 + 8 {
                        let (x, y) = (*a.get(index).ok_or_else(unexpected)?, *b.get(index).ok_or_else(unexpected)?);
                        let (wx, wy) = (self.b.cast(x, KTy::U64), self.b.cast(y, KTy::U64));
                        let larger = self.b.binary(KBinOp::Gt, wx, wy);
                        let (up, down) = (self.b.binary(KBinOp::Sub, wx, wy), self.b.binary(KBinOp::Sub, wy, wx));
                        let difference = self.b.select(larger, up, down);
                        sum = self.b.binary(KBinOp::Add, sum, difference).into();
                    }
                    lanes.push(self.b.cast(sum, out_ty).into());
                }
                lanes
            }
            // Shifts by the low 64 bits of a vector: past the width, logical
            // shifts give zero and arithmetic ones the sign.
            "psll.w" | "psll.d" | "psll.q" | "psrl.w" | "psrl.d" | "psrl.q" | "psra.w" | "psra.d" => {
                let a = self.lane_numbers(&arg(0)?)?;
                let counts = self.lane_numbers(&arg(1)?)?;
                let count_bits = counts.first().map_or(64, |&c| self.b.ty(c).bits());
                let mut count: KOperand = Const::int(KTy::U64, 0).into();
                for (index, &part) in counts.iter().take((64 / count_bits) as usize).enumerate() {
                    let unsigned = self.b.cast(part, unsigned_of(u64::from(count_bits / 8)));
                    let wide = self.b.cast(unsigned, KTy::U64);
                    let placed = self.b.binary(KBinOp::Shl, wide, Const::int(KTy::U32, (index as u32 * count_bits).into()));
                    count = self.b.binary(KBinOp::Or, count, placed).into();
                }
                let bits = u64::from(out_ty.bits());
                let past = self.b.binary(KBinOp::Ge, count, Const::int(KTy::U64, bits.into()));
                let arithmetic = family.starts_with("psra");
                let limited = self.b.select(past, Const::int(KTy::U64, (bits - 1).into()), count);
                let source: KOperand = if arithmetic { limited.into() } else { count };
                let amount = self.b.cast(source, KTy::U32);
                let mut lanes = Vec::new();
                for x in a {
                    let x = if arithmetic { x } else { self.b.cast(x, unsigned_of(bits / 8)).into() };
                    let op = if family.starts_with("psll") { KBinOp::Shl } else { KBinOp::Shr };
                    let shifted = self.b.binary(op, x, amount);
                    let shifted_ty = self.b.ty(shifted);
                    let value = if arithmetic { shifted } else { self.b.select(past, Const::int(shifted_ty, 0), shifted) };
                    lanes.push(self.b.cast(value, out_ty).into());
                }
                lanes
            }
            // `f32` to `f16` bits, rounding to nearest even (immediate 0, or 4
            // for the default floating-point environment).
            "vcvtps2ph.128" | "vcvtps2ph.256" => {
                let rounding = immediate(self, 1)?;
                if rounding != 0 && rounding != 4 {
                    return Err(refused("rounds other than to nearest even here"));
                }
                let a = self.lane_numbers(&arg(0)?)?;
                let mut lanes = Vec::new();
                for index in 0..out.len() {
                    lanes.push(match a.get(index) {
                        Some(&x) => {
                            let half = self.f32_to_f16_bits(x);
                            self.b.cast(half, out_ty).into()
                        }
                        None => Const::int(out_ty, 0).into(),
                    });
                }
                lanes
            }
            "rcp.ps" | "rsqrt.ps" | "rcp.ss" | "rsqrt.ss" | "rcp.ps.256" | "rsqrt.ps.256" => {
                return Err(refused("approximates, differently on each CPU model"));
            }
            _ => return Err(refused("has no exact GPU version yet")),
        };
        if lanes.len() != out.len() {
            return Err(unexpected());
        }
        for (lane, value) in out.into_iter().zip(lanes) {
            self.set(lane, KRvalue::Use(value));
        }
        self.store(frame, destination, result)
    }

    /// An x86 comparison predicate (`_CMP_*`, 0 to 31; 16 on repeat 0 to 15
    /// but signal on quiet NaNs, which changes no result).
    fn x86_compare(&mut self, predicate: u64, a: KOperand, b: KOperand) -> KOperand {
        let unordered = |this: &mut Self| {
            let (x, y) = (this.b.binary(KBinOp::Ne, a, a), this.b.binary(KBinOp::Ne, b, b));
            this.b.binary(KBinOp::Or, x, y)
        };
        let not = |this: &mut Self, value: KLocal| -> KOperand { this.b.unary(KUnOp::Not, value).into() };
        match predicate & 15 {
            0 => self.b.binary(KBinOp::Eq, a, b).into(),
            1 => self.b.binary(KBinOp::Lt, a, b).into(),
            2 => self.b.binary(KBinOp::Le, a, b).into(),
            3 => unordered(self).into(),
            4 => self.b.binary(KBinOp::Ne, a, b).into(),
            5 => {
                let lt = self.b.binary(KBinOp::Lt, a, b);
                not(self, lt)
            }
            6 => {
                let le = self.b.binary(KBinOp::Le, a, b);
                not(self, le)
            }
            7 => {
                let unordered = unordered(self);
                not(self, unordered)
            }
            8 => {
                let (eq, unordered) = (self.b.binary(KBinOp::Eq, a, b), unordered(self));
                self.b.binary(KBinOp::Or, eq, unordered).into()
            }
            9 => {
                let ge = self.b.binary(KBinOp::Ge, a, b);
                not(self, ge)
            }
            10 => {
                let gt = self.b.binary(KBinOp::Gt, a, b);
                not(self, gt)
            }
            11 => Const::bool(false).into(),
            12 => {
                let (ne, unordered) = (self.b.binary(KBinOp::Ne, a, b), unordered(self));
                let ordered = not(self, unordered);
                self.b.binary(KBinOp::And, ne, ordered).into()
            }
            13 => self.b.binary(KBinOp::Ge, a, b).into(),
            14 => self.b.binary(KBinOp::Gt, a, b).into(),
            _ => Const::bool(true).into(),
        }
    }

    /// All ones where `holds`, else zeros, as a lane of type `ty`.
    fn mask(&mut self, holds: KOperand, ty: KTy) -> KLocal {
        let int = if ty.is_float() { unsigned_of(u64::from(ty.bytes())) } else { ty };
        let bits = self.b.select(holds, Const::int(int, -1), Const::int(int, 0));
        if ty.is_float() { self.b.reinterpret(bits, ty) } else { bits }
    }

    /// A float to an integer as x86's conversions do: rounded to nearest
    /// even (or truncated), and the integer's minimum, the "integer
    /// indefinite", for NaN and values out of range.
    fn float_to_int_indefinite(&mut self, value: KOperand, out: KTy, round: bool) -> KLocal {
        let ty = self.b.ty(value);
        let rounded = self.b.call(if round { Intrinsic::RoundTiesEven } else { Intrinsic::Trunc }, &[value]);
        let limit = 2f64.powi(out.bits() as i32 - 1);
        let (low, high) = if ty == KTy::F32 {
            (Const::f32(-limit as f32), Const::f32(limit as f32))
        } else {
            (Const::f64(-limit), Const::f64(limit))
        };
        let above_low = self.b.binary(KBinOp::Ge, rounded, low);
        let below_high = self.b.binary(KBinOp::Lt, rounded, high);
        let valid = self.b.binary(KBinOp::And, above_low, below_high);
        let converted = self.b.cast(rounded, out);
        let indefinite = Const::int(out, -(1i128 << (out.bits() - 1)));
        self.b.select(valid, converted, indefinite)
    }

    /// An `f32`'s nearest `f16`, ties to even, as `u16` bits, as F16C's
    /// `vcvtps2ph` gives it: overflow to infinity, subnormal results, and
    /// a NaN quieted with its payload's top bits.
    fn f32_to_f16_bits(&mut self, value: KOperand) -> KLocal {
        let u32c = |v: i128| Const::int(KTy::U32, v);
        let i32c = |v: i128| Const::int(KTy::I32, v);
        let bits = self.b.reinterpret(value, KTy::U32);
        let shifted_sign = self.b.binary(KBinOp::Shr, bits, u32c(16));
        let sign = self.b.binary(KBinOp::And, shifted_sign, u32c(0x8000));
        let shifted_exponent = self.b.binary(KBinOp::Shr, bits, u32c(23));
        let exponent = self.b.binary(KBinOp::And, shifted_exponent, u32c(0xFF));
        let mantissa = self.b.binary(KBinOp::And, bits, u32c(0x7F_FFFF));
        // Infinity, or NaN with the quiet bit set.
        let payload = self.b.binary(KBinOp::Shr, mantissa, u32c(13));
        let quiet = self.b.binary(KBinOp::Or, payload, u32c(0x7E00));
        let has_payload = self.b.binary(KBinOp::Ne, mantissa, u32c(0));
        let special = self.b.select(has_payload, quiet, u32c(0x7C00));
        // The exponent rebased for `f16`.
        let signed_exponent = self.b.cast(exponent, KTy::I32);
        let rebased = self.b.binary(KBinOp::Sub, signed_exponent, i32c(127 - 15));
        // Normal results: round the mantissa's low 13 bits to nearest even;
        // a carry moves into the exponent, up to infinity.
        let high = self.b.cast(rebased, KTy::U32);
        let exponent_bits = self.b.binary(KBinOp::Shl, high, u32c(10));
        let normal_bits = self.b.binary(KBinOp::Or, exponent_bits, payload);
        let low = self.b.binary(KBinOp::And, mantissa, u32c(0x1FFF));
        let normal = self.round_even(normal_bits, low, u32c(0x1000));
        // Subnormal results: the full significand shifted down, rounded.
        let significand = self.b.binary(KBinOp::Or, mantissa, u32c(0x80_0000));
        let shift = self.b.binary(KBinOp::Sub, i32c(14), rebased);
        let capped = self.b.binary(KBinOp::Lt, shift, i32c(25));
        let shift = self.b.select(capped, shift, i32c(25));
        let shift = self.b.cast(shift, KTy::U32);
        let kept = self.b.binary(KBinOp::Shr, significand, shift);
        let one = self.b.binary(KBinOp::Shl, u32c(1), shift);
        let below = self.b.binary(KBinOp::Sub, one, u32c(1));
        let dropped = self.b.binary(KBinOp::And, significand, below);
        let half = self.b.binary(KBinOp::Shr, one, u32c(1));
        let subnormal = self.round_even(kept, dropped, half);
        let is_special = self.b.binary(KBinOp::Eq, exponent, u32c(0xFF));
        let overflows = self.b.binary(KBinOp::Ge, rebased, i32c(31));
        let is_small = self.b.binary(KBinOp::Le, rebased, i32c(0));
        let finite = self.b.select(is_small, subnormal, normal);
        let bounded = self.b.select(overflows, u32c(0x7C00), finite);
        let magnitude = self.b.select(is_special, special, bounded);
        let combined = self.b.binary(KBinOp::Or, sign, magnitude);
        self.b.cast(combined, KTy::U16)
    }

    /// `kept`, plus one where the `dropped` bits are above `half`, or equal
    /// to it with `kept` odd: rounding to nearest, ties to even.
    fn round_even(&mut self, kept: KLocal, dropped: KLocal, half: impl Into<KOperand>) -> KLocal {
        let half = half.into();
        let above = self.b.binary(KBinOp::Gt, dropped, half);
        let tie = self.b.binary(KBinOp::Eq, dropped, half);
        let odd_bit = self.b.binary(KBinOp::And, kept, Const::int(KTy::U32, 1));
        let odd = self.b.binary(KBinOp::Ne, odd_bit, Const::int(KTy::U32, 0));
        let tie_up = self.b.binary(KBinOp::And, tie, odd);
        let up = self.b.binary(KBinOp::Or, above, tie_up);
        let increment = self.b.cast(up, KTy::U32);
        self.b.binary(KBinOp::Add, kept, increment)
    }
}
