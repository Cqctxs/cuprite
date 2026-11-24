//! MIR assignments, scalar operations, and operand evaluation.

use cuprite_kernel::{
    BinOp as KBinOp, Const, Local as KLocal, Operand as KOperand, Rvalue as KRvalue, Ty as KTy,
    UnOp as KUnOp,
};
use rustc_middle::mir::{AggregateKind, BinOp, CastKind, Operand, Place, Rvalue, UnOp};

use super::fold::runtime_check;
use super::refusal::{aggregate_refusal, binary_refusal, cast_refusal, rvalue_refusal};
use super::types::{array_len, constant, held_ty};
use super::{int_min, is_ref, points_to_slice, Frame, Result, Slot, Translator};

impl<'tcx> Translator<'tcx> {
    pub(super) fn assign(&mut self, frame: &mut Frame<'_, 'tcx>, place: &Place<'tcx>, rvalue: &Rvalue<'tcx>) -> Result<()> {
        let ty = place.ty(frame.body, self.tcx).ty;
        if is_ref(ty) {
            let target = self.reference(frame, rvalue)?;
            return self.store(frame, place, target);
        }
        // Reinterpreting bits, of numbers or of plain data.
        if let Rvalue::Cast(CastKind::Transmute, operand, to) = rvalue {
            let value = self.transmute(frame, operand, *to)?;
            return self.store(frame, place, value);
        }
        if held_ty(ty).is_some() {
            let value = self.scalar_rvalue(frame, rvalue)?;
            let dest = self.place(frame, place)?;
            if let Slot::Bytes { numbers, offset } = &dest {
                let number = self.b.assign(value);
                return self.encode(numbers, *offset, &Slot::Scalar(number), ty);
            }
            return self.write_number(dest, value);
        }
        // Arithmetic with its overflow flag, which debug builds check.
        if let Rvalue::BinaryOp(op @ (BinOp::AddWithOverflow | BinOp::SubWithOverflow | BinOp::MulWithOverflow), operands) =
            rvalue
        {
            let (a, b) = (self.scalar_operand(frame, &operands.0)?, self.scalar_operand(frame, &operands.1)?);
            let (result, overflowed) = self.with_overflow(*op, a, b)?;
            return self.store(frame, place, Slot::Fields(vec![Slot::Scalar(result), Slot::Scalar(overflowed)]));
        }
        // `Ord::cmp` of numbers: `Less`, `Equal` or `Greater`, whose
        // discriminants are -1, 0 and 1.
        if let Rvalue::BinaryOp(BinOp::Cmp, operands) = rvalue {
            let (mut a, mut b) = (self.scalar_operand(frame, &operands.0)?, self.scalar_operand(frame, &operands.1)?);
            if self.b.ty(a) == KTy::Bool {
                a = self.b.cast(a, KTy::U8).into();
                b = self.b.cast(b, KTy::U8).into();
            }
            let Some((tag_ty, variants)) = self.enum_shape(ty)? else {
                return Err(binary_refusal(BinOp::Cmp));
            };
            let (less, greater) = (self.b.binary(KBinOp::Lt, a, b), self.b.binary(KBinOp::Gt, a, b));
            let above = self.b.select(greater, Const::int(tag_ty, 1), Const::int(tag_ty, 0));
            let tag = self.b.select(less, Const::int(tag_ty, -1), above);
            let value = Slot::Enum { tag, variants: vec![Slot::Unit; variants.len()] };
            return self.store(frame, place, value);
        }
        match rvalue {
            Rvalue::Use(operand, _) => {
                let value = self.operand_slot(frame, operand)?;
                self.store(frame, place, value)
            }
            Rvalue::CopyForDeref(source) => {
                let value = self.read_place(frame, source)?;
                self.store(frame, place, value)
            }
            // `[x; N]`: every element a copy of `x`.
            Rvalue::Repeat(operand, count) => {
                let value = self.operand_slot(frame, operand)?;
                let count = array_len(self.tcx, *count)?;
                let value = if count == 0 { Slot::Unit } else { Slot::Fields(vec![value; count]) };
                self.store(frame, place, value)
            }
            // An enum's variant: its discriminant and its fields, leaving the
            // other variants alone.
            Rvalue::Aggregate(kind, operands)
                if matches!(**kind, AggregateKind::Adt(adt, _, _, _, None) if self.tcx.adt_def(adt).is_enum()) =>
            {
                let AggregateKind::Adt(_, variant, ..) = **kind else { unreachable!("matched above") };
                let Some((_, shape)) = self.enum_shape(ty)? else { return Err(aggregate_refusal(kind)) };
                let tag = self.discriminant(ty, variant)?;
                let tag = self.b.copy(tag);
                let values: Vec<Slot> =
                    operands.iter().map(|operand| self.operand_slot(frame, operand)).collect::<Result<_>>()?;
                let mut variants = vec![Slot::Unit; shape.len()];
                if !values.is_empty() {
                    variants[variant.as_usize()] = Slot::Fields(values);
                }
                self.store(frame, place, Slot::Enum { tag, variants })
            }
            // A union, through one field: its bytes, the rest zero.
            Rvalue::Aggregate(kind, operands) if matches!(**kind, AggregateKind::Adt(_, _, _, _, Some(_))) => {
                let [operand] = &operands.raw[..] else { return Err(aggregate_refusal(kind)) };
                let field_ty = operand.ty(frame.body, self.tcx);
                let value = self.operand_slot(frame, operand)?;
                let Slot::Bytes { numbers, .. } = self.fresh(ty)? else { return Err(aggregate_refusal(kind)) };
                for &(_, number) in &numbers {
                    let zero = Const::int(self.b.ty(number), 0);
                    self.set(number, KRvalue::Use(zero.into()));
                }
                self.encode(&numbers, 0, &value, field_ty)?;
                self.store(frame, place, Slot::Bytes { numbers, offset: 0 })
            }
            Rvalue::Aggregate(kind, operands) => {
                match **kind {
                    AggregateKind::Tuple | AggregateKind::Array(_) => {}
                    AggregateKind::Adt(adt, _, _, _, None) if self.tcx.adt_def(adt).is_struct() => {}
                    // A closure is its captures: values, or references to
                    // what it captured by reference.
                    AggregateKind::Closure(..) => {}
                    _ => return Err(aggregate_refusal(kind)),
                }
                let values: Vec<Slot> =
                    operands.iter().map(|operand| self.operand_slot(frame, operand)).collect::<Result<_>>()?;
                let value = if values.is_empty() { Slot::Unit } else { Slot::Fields(values) };
                self.store(frame, place, value)
            }
            _ => Err(rvalue_refusal(rvalue)),
        }
    }

    fn scalar_rvalue(&mut self, frame: &mut Frame<'_, 'tcx>, rvalue: &Rvalue<'tcx>) -> Result<KRvalue> {
        Ok(match rvalue {
            Rvalue::Use(operand, _) => KRvalue::Use(self.scalar_operand(frame, operand)?),
            Rvalue::CopyForDeref(place) => match self.read_place(frame, place)? {
                Slot::Memory { address, ty } => KRvalue::Load(address.into(), ty),
                slot => KRvalue::Use(self.number(slot)?),
            },
            // Pointers compared by their positions.
            Rvalue::BinaryOp(op, operands) if is_ref(operands.0.ty(frame.body, self.tcx)) => {
                let op = match op {
                    BinOp::Eq => KBinOp::Eq,
                    BinOp::Ne => KBinOp::Ne,
                    BinOp::Lt => KBinOp::Lt,
                    BinOp::Le => KBinOp::Le,
                    BinOp::Gt => KBinOp::Gt,
                    BinOp::Ge => KBinOp::Ge,
                    other => return Err(binary_refusal(*other)),
                };
                let (a, b) = (self.operand_slot(frame, &operands.0)?, self.operand_slot(frame, &operands.1)?);
                match self.positions(&a, &b)? {
                    Some((a, b)) => KRvalue::Binary(op, a, b),
                    // The same target, at no position: equal.
                    None => KRvalue::Use(Const::bool(matches!(op, KBinOp::Eq | KBinOp::Le | KBinOp::Ge)).into()),
                }
            }
            Rvalue::BinaryOp(op, operands) => {
                let (a, b) = (self.scalar_operand(frame, &operands.0)?, self.scalar_operand(frame, &operands.1)?);
                let op = match op {
                    // The unchecked forms are undefined on overflow, so
                    // wrapping is correct for every defined use.
                    BinOp::Add | BinOp::AddUnchecked => KBinOp::Add,
                    BinOp::Sub | BinOp::SubUnchecked => KBinOp::Sub,
                    BinOp::Mul | BinOp::MulUnchecked => KBinOp::Mul,
                    // Divisors are nonzero: every assertion was proven.
                    BinOp::Div => KBinOp::Div,
                    BinOp::Rem if self.b.ty(a).is_float() => {
                        return Err("uses `%` on floats, which has no exact GPU version yet".into());
                    }
                    BinOp::Rem => KBinOp::Rem,
                    BinOp::BitAnd => KBinOp::And,
                    BinOp::BitOr => KBinOp::Or,
                    BinOp::BitXor => KBinOp::Xor,
                    BinOp::Shl | BinOp::ShlUnchecked => KBinOp::Shl,
                    BinOp::Shr | BinOp::ShrUnchecked => KBinOp::Shr,
                    BinOp::Eq => KBinOp::Eq,
                    BinOp::Ne => KBinOp::Ne,
                    BinOp::Lt => KBinOp::Lt,
                    BinOp::Le => KBinOp::Le,
                    BinOp::Gt => KBinOp::Gt,
                    BinOp::Ge => KBinOp::Ge,
                    other => return Err(binary_refusal(*other)),
                };
                KRvalue::Binary(op, a, b)
            }
            // An enum's discriminant.
            Rvalue::Discriminant(place) => {
                let slot = self.read_place(frame, place)?;
                match self.resolve(slot)? {
                    Slot::Enum { tag, .. } => KRvalue::Use(tag.into()),
                    _ => return Err(rvalue_refusal(rvalue)),
                }
            }
            // A read-only slice's length, or that of an array as a slice.
            Rvalue::UnaryOp(UnOp::PtrMetadata, operand) => match self.operand_slot(frame, operand)? {
                Slot::Ref(inner) => match *inner {
                    Slot::Slice { len, .. } => KRvalue::Use(len.into()),
                    _ if !points_to_slice(operand.ty(frame.body, self.tcx)) => {
                        return Err("reads the metadata of a pointer other than to a slice".into());
                    }
                    Slot::Fields(elements) => KRvalue::Use(Const::int(KTy::U64, elements.len() as i128).into()),
                    Slot::Unit => KRvalue::Use(Const::int(KTy::U64, 0).into()),
                    _ => return Err("reads the length of something other than a read-only slice".into()),
                },
                Slot::Slice { len, .. } => KRvalue::Use(len.into()),
                _ => return Err("reads the length of something other than a read-only slice".into()),
            },
            Rvalue::UnaryOp(op, operand) => {
                let a = self.scalar_operand(frame, operand)?;
                match op {
                    UnOp::Neg => KRvalue::Unary(KUnOp::Neg, a),
                    UnOp::Not => KRvalue::Unary(KUnOp::Not, a),
                    UnOp::PtrMetadata => return Err(rvalue_refusal(rvalue)),
                }
            }
            Rvalue::Cast(kind, operand, ty) => {
                let numeric = matches!(
                    kind,
                    CastKind::IntToInt | CastKind::IntToFloat | CastKind::FloatToInt | CastKind::FloatToFloat
                );
                let to = held_ty(*ty).filter(|_| numeric).ok_or_else(|| cast_refusal(*kind, *ty))?;
                KRvalue::Cast(self.scalar_operand(frame, operand)?, to)
            }
            Rvalue::Ref(..) | Rvalue::RawPtr(..) => {
                return Err("borrows something other than the current element".into());
            }
            _ => return Err(rvalue_refusal(rvalue)),
        })
    }

    pub(super) fn scalar_operand(&mut self, frame: &mut Frame<'_, 'tcx>, operand: &Operand<'tcx>) -> Result<KOperand> {
        match operand {
            Operand::Constant(c) => constant(self.tcx, c)
                .map(KOperand::Const)
                .ok_or_else(|| format!("uses an unsupported constant of type `{}`", c.ty())),
            Operand::Copy(place) | Operand::Move(place) => {
                let slot = self.read_place(frame, place)?;
                self.number(slot)
            }
            Operand::RuntimeChecks(checks) => Ok(Const::bool(runtime_check(self.tcx, *checks)).into()),
        }
    }

    /// `a op b` and whether it overflowed, as `AddWithOverflow` and its
    /// siblings compute them, for the overflow checks of debug builds.
    pub(super) fn with_overflow(&mut self, op: BinOp, a: KOperand, b: KOperand) -> Result<(KLocal, KLocal)> {
        let ty = self.b.ty(a);
        let kernel_op = match op {
            BinOp::AddWithOverflow => KBinOp::Add,
            BinOp::SubWithOverflow => KBinOp::Sub,
            BinOp::MulWithOverflow => KBinOp::Mul,
            other => return Err(binary_refusal(other)),
        };
        let result = self.b.binary(kernel_op, a, b);
        if ty.bits() < 64 {
            // Exact in 64 bits: overflowed if the result doesn't fit.
            let wide_ty = if ty.is_signed() { KTy::I64 } else { KTy::U64 };
            let (wa, wb) = (self.b.cast(a, wide_ty), self.b.cast(b, wide_ty));
            let wide = self.b.binary(kernel_op, wa, wb);
            let back = self.b.cast(result, wide_ty);
            let overflowed = self.b.binary(KBinOp::Ne, wide, back);
            return Ok((result, overflowed));
        }
        let zero = Const::int(ty, 0);
        let overflowed = match (op, ty.is_signed()) {
            (BinOp::AddWithOverflow, false) => self.b.binary(KBinOp::Lt, result, a),
            (BinOp::SubWithOverflow, false) => self.b.binary(KBinOp::Lt, a, b),
            // Signs: both operands share one the result lacks (add), or the
            // operands differ and the result differs from `a` (sub).
            (BinOp::AddWithOverflow, true) => {
                let (x, y) = (self.b.binary(KBinOp::Xor, a, result), self.b.binary(KBinOp::Xor, b, result));
                let both = self.b.binary(KBinOp::And, x, y);
                self.b.binary(KBinOp::Lt, both, zero)
            }
            (BinOp::SubWithOverflow, true) => {
                let (x, y) = (self.b.binary(KBinOp::Xor, a, b), self.b.binary(KBinOp::Xor, a, result));
                let both = self.b.binary(KBinOp::And, x, y);
                self.b.binary(KBinOp::Lt, both, zero)
            }
            // `result / b != a`, with the divisor kept away from 0 (and from
            // -1, whose case is `a == MIN`), where division isn't defined.
            (_, signed) => {
                let b_zero = self.b.binary(KBinOp::Eq, b, zero);
                let minus_one = Const::int(ty, -1);
                let b_minus_one = self.b.binary(KBinOp::Eq, b, minus_one);
                let unsafe_divisor = if signed { self.b.binary(KBinOp::Or, b_zero, b_minus_one) } else { b_zero };
                let divisor = self.b.select(unsafe_divisor, Const::int(ty, 1), b);
                let quotient = self.b.binary(KBinOp::Div, result, divisor);
                let general = self.b.binary(KBinOp::Ne, quotient, a);
                let min = int_min(ty);
                let a_min = self.b.binary(KBinOp::Eq, a, min);
                let special = if signed { self.b.select(b_minus_one, a_min, Const::bool(false)) } else { self.b.copy(Const::bool(false)) };
                let checked = self.b.select(unsafe_divisor, special, general);
                self.b.select(b_zero, Const::bool(false), checked)
            }
        };
        Ok((result, overflowed))
    }

    /// The slot of an operand of any type. Constants get new locals.
    pub(super) fn operand_slot(&mut self, frame: &mut Frame<'_, 'tcx>, operand: &Operand<'tcx>) -> Result<Slot> {
        match operand {
            Operand::Constant(c) => self.constant_slot(c),
            Operand::Copy(place) | Operand::Move(place) => match self.read_place(frame, place)? {
                Slot::Memory { address, ty } => Ok(Slot::Scalar(self.b.load(address, ty))),
                slot => self.resolve(slot),
            },
            Operand::RuntimeChecks(checks) => Ok(Slot::Scalar(self.b.copy(Const::bool(runtime_check(self.tcx, *checks))))),
        }
    }

    /// An operand's value if it is a constant number.
    pub(super) fn known(&self, operand: &Operand<'tcx>) -> Option<Const> {
        match operand {
            Operand::Constant(c) => constant(self.tcx, c),
            _ => None,
        }
    }
}
