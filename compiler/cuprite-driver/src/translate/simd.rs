//! Portable SIMD operations on the lanes of a Rust vector.
//!
//! # SIMD
//!
//! A SIMD vector (`#[repr(simd)]`, a struct around an array) is held as its
//! lanes. The portable `simd_*` intrinsics apply the scalar operation to
//! each lane ([`Translator::simd`]). x86 instructions that `std::arch`
//! reaches through LLVM intrinsics are translated from Intel's definitions
//! ([`Translator::x86`]), in Rust's default floating-point environment;
//! ones whose results differ between CPU models (`rcpps`) are refused.
//!
//! A runtime CPU feature test (`is_x86_feature_detected!`, which ends in
//! `std_detect::detect::cache::test`) reads a kernel parameter: the
//! rewritten function calls the same test on the host before dispatching
//! and passes the answer, so the kernel takes the path the CPU would.

use cuprite_kernel::{
    BinOp as KBinOp, Const, Intrinsic, Local as KLocal, Operand as KOperand, Rvalue as KRvalue,
    Ty as KTy, UnOp as KUnOp,
};
use rustc_middle::mir::{Operand, Place};

use super::types::held_ty;
use super::{unsigned_of, Frame, Result, Slot, Translator};

impl<'tcx> Translator<'tcx> {
    /// The lanes of a SIMD vector: a `#[repr(simd)]` struct around an array,
    /// held as its elements.
    pub(super) fn lanes(&self, slot: &Slot) -> Result<Vec<Slot>> {
        match slot {
            Slot::Fields(fields) if fields.len() == 1 => match &fields[0] {
                Slot::Fields(lanes) => Ok(lanes.clone()),
                _ => Err("uses a SIMD vector whose lanes aren't an array".into()),
            },
            _ => Err("uses a SIMD vector whose lanes aren't an array".into()),
        }
    }

    /// The numbers in a vector's lanes.
    pub(super) fn lane_numbers(&mut self, slot: &Slot) -> Result<Vec<KOperand>> {
        let lanes = self.lanes(slot)?;
        lanes.into_iter().map(|lane| self.number(lane)).collect()
    }

    /// A `simd_*` intrinsic: the scalar operation on each lane, as the
    /// intrinsic's documentation defines it, into a new vector.
    pub(super) fn simd(
        &mut self,
        frame: &mut Frame<'_, 'tcx>,
        name: &str,
        args: &[rustc_span::Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
    ) -> Result<()> {
        let tcx = self.tcx;
        let values = args.iter().map(|arg| self.operand_slot(frame, &arg.node)).collect::<Result<Vec<_>>>()?;
        let unexpected = || format!("uses the intrinsic `{name}` with unexpected arguments");
        let arg = |index: usize| values.get(index).cloned().ok_or_else(unexpected);
        let dest_ty = destination.ty(frame.body, tcx).ty;
        // Intrinsics giving one number.
        let scalar: Option<KRvalue> = match name {
            "simd_extract" | "simd_extract_dyn" => {
                let lanes = self.lanes(&arg(0)?)?;
                let index = match self.known(&args[1].node) { Some(c) => c.into(), None => self.number(arg(1)?)? };
                let lane = match index {
                    KOperand::Const(c) => lanes.get(c.bits as usize).cloned().ok_or("extracts a lane that doesn't exist")?,
                    KOperand::Local(local) => Slot::Choice { index: self.b.copy(local), options: lanes },
                };
                Some(KRvalue::Use(self.number(lane)?))
            }
            "simd_reduce_add_ordered" | "simd_reduce_mul_ordered" => {
                let op = if name == "simd_reduce_add_ordered" { KBinOp::Add } else { KBinOp::Mul };
                let mut acc = self.number(arg(1)?)?;
                for lane in self.lane_numbers(&arg(0)?)? {
                    acc = self.b.binary(op, acc, lane).into();
                }
                Some(KRvalue::Use(acc))
            }
            "simd_reduce_and" | "simd_reduce_or" | "simd_reduce_xor" | "simd_reduce_max" | "simd_reduce_min" => {
                let lanes = self.lane_numbers(&arg(0)?)?;
                let (&first, rest) = lanes.split_first().ok_or_else(unexpected)?;
                if matches!(name, "simd_reduce_max" | "simd_reduce_min") && self.b.ty(first).is_float() {
                    return Err(format!("uses the intrinsic `{name}` on floats, whose NaN handling isn't settled"));
                }
                let mut acc = first;
                for &lane in rest {
                    acc = match name {
                        "simd_reduce_and" => self.b.binary(KBinOp::And, acc, lane).into(),
                        "simd_reduce_or" => self.b.binary(KBinOp::Or, acc, lane).into(),
                        "simd_reduce_xor" => self.b.binary(KBinOp::Xor, acc, lane).into(),
                        _ => {
                            let op = if name == "simd_reduce_max" { KBinOp::Gt } else { KBinOp::Lt };
                            let better = self.b.binary(op, lane, acc);
                            self.b.select(better, lane, acc).into()
                        }
                    };
                }
                Some(KRvalue::Use(acc))
            }
            "simd_reduce_all" | "simd_reduce_any" => {
                let op = if name == "simd_reduce_all" { KBinOp::And } else { KBinOp::Or };
                let mut acc: KOperand = Const::bool(name == "simd_reduce_all").into();
                for lane in self.lane_numbers(&arg(0)?)? {
                    let ty = self.b.ty(lane);
                    let set = self.b.binary(KBinOp::Ne, lane, Const::int(ty, 0));
                    acc = self.b.binary(op, acc, set).into();
                }
                Some(KRvalue::Use(acc))
            }
            // Bit `i` is lane `i`'s most significant bit.
            "simd_bitmask" => {
                let out = held_ty(dest_ty).filter(|ty| ty.is_int()).ok_or_else(|| {
                    format!("uses the intrinsic `simd_bitmask` giving `{dest_ty}`, which isn't an integer")
                })?;
                let mut bits: KOperand = Const::int(out, 0).into();
                for (index, lane) in self.lane_numbers(&arg(0)?)?.into_iter().enumerate() {
                    let ty = self.b.ty(lane);
                    let unsigned = self.b.cast(lane, unsigned_of(u64::from(ty.bytes())));
                    let top = self.b.binary(KBinOp::Shr, unsigned, Const::int(KTy::U32, i128::from(ty.bits() - 1)));
                    let bit = self.b.cast(top, out);
                    let placed = self.b.binary(KBinOp::Shl, bit, Const::int(KTy::U32, index as i128));
                    bits = self.b.binary(KBinOp::Or, bits, placed).into();
                }
                Some(KRvalue::Use(bits))
            }
            _ => None,
        };
        if let Some(rvalue) = scalar {
            let dest = self.place(frame, destination)?;
            return self.write_number(dest, rvalue);
        }
        // Intrinsics giving a vector, lane by lane.
        let result = self.fresh(dest_ty)?;
        let out: Vec<KLocal> = self
            .lanes(&result)?
            .into_iter()
            .map(|lane| match lane {
                Slot::Scalar(local) => Ok(local),
                _ => Err("uses a SIMD vector whose lanes aren't numbers".to_string()),
            })
            .collect::<Result<_>>()?;
        let binary = match name {
            "simd_add" => Some(KBinOp::Add),
            "simd_sub" => Some(KBinOp::Sub),
            "simd_mul" => Some(KBinOp::Mul),
            "simd_div" => Some(KBinOp::Div),
            "simd_rem" => Some(KBinOp::Rem),
            "simd_shl" => Some(KBinOp::Shl),
            "simd_shr" => Some(KBinOp::Shr),
            "simd_and" => Some(KBinOp::And),
            "simd_or" => Some(KBinOp::Or),
            "simd_xor" => Some(KBinOp::Xor),
            _ => None,
        };
        let compare = match name {
            "simd_eq" => Some(KBinOp::Eq),
            "simd_ne" => Some(KBinOp::Ne),
            "simd_lt" => Some(KBinOp::Lt),
            "simd_le" => Some(KBinOp::Le),
            "simd_gt" => Some(KBinOp::Gt),
            "simd_ge" => Some(KBinOp::Ge),
            _ => None,
        };
        let call = match name {
            "simd_fabs" => Some(Intrinsic::Abs),
            "simd_fsqrt" => Some(Intrinsic::Sqrt),
            "simd_floor" => Some(Intrinsic::Floor),
            "simd_ceil" => Some(Intrinsic::Ceil),
            "simd_trunc" => Some(Intrinsic::Trunc),
            "simd_round_ties_even" => Some(Intrinsic::RoundTiesEven),
            "simd_minimum_number_nsz" => Some(Intrinsic::Min),
            "simd_maximum_number_nsz" => Some(Intrinsic::Max),
            "simd_fma" => Some(Intrinsic::MulAdd),
            "simd_ctpop" => Some(Intrinsic::CountOnes),
            "simd_ctlz" => Some(Intrinsic::LeadingZeros),
            "simd_cttz" => Some(Intrinsic::TrailingZeros),
            _ => None,
        };
        let rvalues: Vec<KRvalue> = if let Some(op) = binary {
            let (a, b) = (self.lane_numbers(&arg(0)?)?, self.lane_numbers(&arg(1)?)?);
            if op == KBinOp::Rem && a.first().is_some_and(|&lane| self.b.ty(lane).is_float()) {
                return Err("uses `%` on float lanes, which has no exact GPU version yet".into());
            }
            a.into_iter().zip(b).map(|(a, b)| KRvalue::Binary(op, a, b)).collect()
        } else if let Some(op) = compare {
            // A lane of all ones where the comparison holds.
            let (a, b) = (self.lane_numbers(&arg(0)?)?, self.lane_numbers(&arg(1)?)?);
            let mut rvalues = Vec::new();
            for ((a, b), &lane) in a.into_iter().zip(b).zip(&out) {
                let ty = self.b.ty(lane);
                let holds = self.b.binary(op, a, b);
                rvalues.push(KRvalue::Select(holds.into(), Const::int(ty, -1).into(), Const::int(ty, 0).into()));
            }
            rvalues
        } else if let Some(intrinsic) = call {
            let operands = args.len();
            let columns = (0..operands).map(|index| self.lane_numbers(&arg(index)?)).collect::<Result<Vec<_>>>()?;
            let mut rvalues = Vec::new();
            for (index, &lane) in out.iter().enumerate() {
                let lane_args: Vec<KOperand> = columns.iter().map(|column| column[index]).collect();
                if intrinsic.counts_bits() {
                    let count = self.b.call(intrinsic, &lane_args);
                    rvalues.push(KRvalue::Cast(count.into(), self.b.ty(lane)));
                } else {
                    rvalues.push(KRvalue::Call(intrinsic, lane_args));
                }
            }
            rvalues
        } else {
            match name {
                "simd_neg" => self.lane_numbers(&arg(0)?)?.into_iter().map(|a| KRvalue::Unary(KUnOp::Neg, a)).collect(),
                // `as` on each lane; `simd_cast` is undefined where `as`
                // would saturate a float, so `as` is right for it too.
                "simd_cast" | "simd_as" => {
                    let a = self.lane_numbers(&arg(0)?)?;
                    a.into_iter().zip(&out).map(|(a, &lane)| KRvalue::Cast(a, self.b.ty(lane))).collect()
                }
                "simd_splat" => {
                    let value = self.number(arg(0)?)?;
                    out.iter().map(|_| KRvalue::Use(value)).collect()
                }
                "simd_select" => {
                    let (mask, a, b) =
                        (self.lane_numbers(&arg(0)?)?, self.lane_numbers(&arg(1)?)?, self.lane_numbers(&arg(2)?)?);
                    let mut rvalues = Vec::new();
                    for ((mask, a), b) in mask.into_iter().zip(a).zip(b) {
                        let ty = self.b.ty(mask);
                        let set = self.b.binary(KBinOp::Ne, mask, Const::int(ty, 0));
                        rvalues.push(KRvalue::Select(set.into(), a, b));
                    }
                    rvalues
                }
                "simd_select_bitmask" => {
                    let mask = self.number(arg(0)?)?;
                    let (a, b) = (self.lane_numbers(&arg(1)?)?, self.lane_numbers(&arg(2)?)?);
                    let ty = self.b.ty(mask);
                    let mut rvalues = Vec::new();
                    for (index, (a, b)) in a.into_iter().zip(b).enumerate() {
                        let shifted = self.b.binary(KBinOp::Shr, mask, Const::int(KTy::U32, index as i128));
                        let bit = self.b.binary(KBinOp::And, shifted, Const::int(ty, 1));
                        let set = self.b.binary(KBinOp::Ne, bit, Const::int(ty, 0));
                        rvalues.push(KRvalue::Select(set.into(), a, b));
                    }
                    rvalues
                }
                "simd_insert" | "simd_insert_dyn" => {
                    let lanes = self.lane_numbers(&arg(0)?)?;
                    let index = match self.known(&args[1].node) { Some(c) => c.into(), None => self.number(arg(1)?)? };
                    let value = self.number(arg(2)?)?;
                    let mut rvalues = Vec::new();
                    for (at, lane) in lanes.into_iter().enumerate() {
                        rvalues.push(match index {
                            KOperand::Const(c) => KRvalue::Use(if c.bits as usize == at { value } else { lane }),
                            KOperand::Local(local) => {
                                let here = self.chosen(local, at);
                                KRvalue::Select(here.into(), value, lane)
                            }
                        });
                    }
                    rvalues
                }
                // Lanes of both vectors, numbered on, picked by constant
                // indices.
                "simd_shuffle" => {
                    let mut all = self.lane_numbers(&arg(0)?)?;
                    all.extend(self.lane_numbers(&arg(1)?)?);
                    let Some(Operand::Constant(indices)) = args.get(2).map(|arg| &arg.node) else {
                        return Err("uses the intrinsic `simd_shuffle` with indices known only at run time".into());
                    };
                    let bytes = self.constant_bytes(indices)?;
                    let indices: Vec<usize> = bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|chunk| u32::from_le_bytes(*chunk) as usize)
                        .collect();
                    if indices.len() != out.len() {
                        return Err(unexpected());
                    }
                    indices
                        .into_iter()
                        .map(|index| all.get(index).map(|&lane| KRvalue::Use(lane)).ok_or_else(unexpected))
                        .collect::<Result<_>>()?
                }
                "simd_saturating_add" | "simd_saturating_sub" => {
                    let (a, b) = (self.lane_numbers(&arg(0)?)?, self.lane_numbers(&arg(1)?)?);
                    let mut rvalues = Vec::new();
                    for (a, b) in a.into_iter().zip(b) {
                        let saturated = self.saturating(name == "simd_saturating_add", a, b)?;
                        rvalues.push(KRvalue::Use(saturated.into()));
                    }
                    rvalues
                }
                "simd_bswap" | "simd_bitreverse" | "simd_round" => {
                    let mut rvalues = Vec::new();
                    for lane in self.lane_numbers(&arg(0)?)? {
                        let result = match name {
                            "simd_bswap" => self.byte_swap(lane),
                            "simd_bitreverse" => self.bit_reverse(lane),
                            _ => self.round_away(lane),
                        };
                        rvalues.push(KRvalue::Use(result.into()));
                    }
                    rvalues
                }
                "simd_carryless_mul" => {
                    let (a, b) = (self.lane_numbers(&arg(0)?)?, self.lane_numbers(&arg(1)?)?);
                    let mut rvalues = Vec::new();
                    for (a, b) in a.into_iter().zip(b) {
                        rvalues.push(KRvalue::Use(self.carryless(a, b).into()));
                    }
                    rvalues
                }
                // The shift is a vector of the lanes' type.
                "simd_funnel_shl" | "simd_funnel_shr" => {
                    let (a, b, shift) =
                        (self.lane_numbers(&arg(0)?)?, self.lane_numbers(&arg(1)?)?, self.lane_numbers(&arg(2)?)?);
                    let mut rvalues = Vec::new();
                    for ((a, b), shift) in a.into_iter().zip(b).zip(shift) {
                        rvalues.push(KRvalue::Use(self.funnel(name == "simd_funnel_shl", a, b, shift).into()));
                    }
                    rvalues
                }
                _ => return Err(format!("uses the intrinsic `{name}`{}", crate::coverage::reason(name))),
            }
        };
        if rvalues.len() != out.len() {
            return Err(unexpected());
        }
        for (lane, rvalue) in out.into_iter().zip(rvalues) {
            self.set(lane, rvalue);
        }
        self.store(frame, destination, result)
    }
}
