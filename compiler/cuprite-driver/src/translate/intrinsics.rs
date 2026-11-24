//! Classification of library calls and intrinsics, and lowering of exact Rust intrinsics.
//!
//! # Exact intrinsics
//!
//! Intrinsics with one exact result are built from kernel operations in
//! [`Translator::exact`]: `round` (ties away from zero) from `trunc`,
//! `copysign` on the bits, IEEE `minimum`/`maximum` (NaN wins, `-0 < +0`),
//! `bitreverse` by mask swaps and a byte swap, funnel shifts, carry-less
//! products by unrolled shifts and xors, `carrying_mul_add` at double width,
//! and `float_to_int_unchecked` as a saturating cast (its inputs are in
//! range, so the result is the same). Hints are answered without effect:
//! `is_val_statically_known` is `false`, `black_box` is a copy, prefetches
//! do nothing. Whole values are moved slot by slot (`select_unpredictable`,
//! `typed_swap_nonoverlapping`) or compared in 8-byte chunks (`raw_eq`).
//! The `assert_*_valid` checks are asked of the compiler: a type that
//! passes emits nothing, and one that fails always panics, so it is refused.

use cuprite_kernel::{
    BinOp as KBinOp, Const, Intrinsic, Local as KLocal, Operand as KOperand, Rvalue as KRvalue,
    Ty as KTy,
};
use rustc_middle::mir::{BinOp, Operand, Place};
use rustc_middle::ty::{self, layout::ValidityRequirement, FloatTy, Ty, TyCtxt};
use rustc_span::def_id::DefId;

use super::fold::fold_operand;
use super::types::held_ty;
use super::{int_max, int_min, signed_of, unsigned_of, Frame, Result, Slot, Translator};

/// A library method with an exact kernel equivalent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Method {
    /// `wrapping_add`, `wrapping_sub`, `wrapping_mul`, `wrapping_shl` and
    /// `wrapping_shr`, which mask the shift amount as the kernel's do.
    Binary(KBinOp),
    /// `wrapping_neg`.
    Neg,
    Intrinsic(Intrinsic),
}

/// Recognizes inherent methods of the primitive numeric types, by their
/// definitions in `core` or `std`, never by printed paths.
pub(crate) fn method(tcx: TyCtxt<'_>, def_id: DefId) -> Option<Method> {
    let impl_id = tcx.inherent_impl_of_assoc(def_id)?;
    if !matches!(tcx.crate_name(impl_id.krate).as_str(), "core" | "std") {
        return None;
    }
    let self_ty = tcx.type_of(impl_id).instantiate_identity().skip_normalization();
    let name = tcx.item_name(def_id);
    let name = name.as_str();
    match self_ty.kind() {
        ty::Int(_) | ty::Uint(_) => Some(match name {
            "wrapping_add" => Method::Binary(KBinOp::Add),
            "wrapping_sub" => Method::Binary(KBinOp::Sub),
            "wrapping_mul" => Method::Binary(KBinOp::Mul),
            "wrapping_shl" => Method::Binary(KBinOp::Shl),
            "wrapping_shr" => Method::Binary(KBinOp::Shr),
            "wrapping_neg" => Method::Neg,
            "rotate_left" => Method::Intrinsic(Intrinsic::RotateLeft),
            "rotate_right" => Method::Intrinsic(Intrinsic::RotateRight),
            "leading_zeros" => Method::Intrinsic(Intrinsic::LeadingZeros),
            "trailing_zeros" => Method::Intrinsic(Intrinsic::TrailingZeros),
            "count_ones" => Method::Intrinsic(Intrinsic::CountOnes),
            _ => return None,
        }),
        ty::Float(FloatTy::F32 | FloatTy::F64) => Some(Method::Intrinsic(match name {
            "sqrt" => Intrinsic::Sqrt,
            "floor" => Intrinsic::Floor,
            "ceil" => Intrinsic::Ceil,
            "trunc" => Intrinsic::Trunc,
            "abs" => Intrinsic::Abs,
            "min" => Intrinsic::Min,
            "max" => Intrinsic::Max,
            "mul_add" => Intrinsic::MulAdd,
            _ => return None,
        })),
        _ => None,
    }
}

/// What a compiler intrinsic becomes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IntrinsicOp {
    Method(Method),
    /// `cold_path` and `assume`: nothing.
    Nothing,
    /// `transmute`: the argument's bits as the result's type.
    Transmute,
    /// A `simd_*` intrinsic, lane by lane (`Translator::simd`).
    Simd,
    /// `bswap`: an integer's bytes reversed.
    ByteSwap,
    /// An intrinsic built from several kernel operations
    /// (`Translator::exact`).
    Exact,
}

/// The `simd_*` intrinsics translated lane by lane.
const SIMD: [&str; 57] = [
    "simd_insert", "simd_extract", "simd_insert_dyn", "simd_extract_dyn", "simd_splat", "simd_add", "simd_sub",
    "simd_mul", "simd_div", "simd_rem", "simd_shl", "simd_shr", "simd_and", "simd_or", "simd_xor", "simd_cast",
    "simd_as", "simd_neg", "simd_fabs", "simd_minimum_number_nsz", "simd_maximum_number_nsz", "simd_eq", "simd_ne",
    "simd_lt", "simd_le", "simd_gt", "simd_ge", "simd_shuffle", "simd_saturating_add", "simd_saturating_sub",
    "simd_reduce_add_ordered", "simd_reduce_mul_ordered", "simd_reduce_all", "simd_reduce_any", "simd_reduce_max",
    "simd_reduce_min", "simd_reduce_and", "simd_reduce_or", "simd_reduce_xor", "simd_bitmask", "simd_select",
    "simd_select_bitmask", "simd_bswap", "simd_ctlz", "simd_ctpop", "simd_cttz", "simd_ceil", "simd_floor",
    "simd_trunc", "simd_fsqrt", "simd_fma", "simd_round", "simd_round_ties_even", "simd_bitreverse",
    "simd_funnel_shl", "simd_funnel_shr", "simd_carryless_mul",
];

/// The intrinsics library code reaches after inlining, by name: exactly
/// those `coverage::INTRINSICS` lists as supported.
pub(crate) fn intrinsic(name: &str) -> Option<IntrinsicOp> {
    use Intrinsic::*;
    let call = |i| Some(IntrinsicOp::Method(Method::Intrinsic(i)));
    let binary = |op| Some(IntrinsicOp::Method(Method::Binary(op)));
    match name {
        "sqrtf32" | "sqrtf64" => call(Sqrt),
        "floorf32" | "floorf64" => call(Floor),
        "ceilf32" | "ceilf64" => call(Ceil),
        "truncf32" | "truncf64" => call(Trunc),
        "fabs" => call(Abs),
        "fmaf32" | "fmaf64" => call(MulAdd),
        // `f32::min` and `max`: the other operand when one is NaN, either
        // zero when both are zeros. (`minimumf32` propagates NaN instead,
        // which the kernel doesn't provide yet.)
        "minimum_number_nsz_f32" | "minimum_number_nsz_f64" => call(Min),
        "maximum_number_nsz_f32" | "maximum_number_nsz_f64" => call(Max),
        "rotate_left" => call(RotateLeft),
        "rotate_right" => call(RotateRight),
        // The `_nonzero` forms are undefined at zero, so the defined counts
        // are correct for every legal use.
        "ctlz" | "ctlz_nonzero" => call(LeadingZeros),
        "cttz" | "cttz_nonzero" => call(TrailingZeros),
        "ctpop" => call(CountOnes),
        // Overflow, a zero divisor and oversized shifts are undefined for
        // the unchecked forms, so the wrapping, masking kernel operations
        // are correct for every defined use.
        "wrapping_add" | "unchecked_add" => binary(KBinOp::Add),
        "wrapping_sub" | "unchecked_sub" => binary(KBinOp::Sub),
        "wrapping_mul" | "unchecked_mul" => binary(KBinOp::Mul),
        "unchecked_div" | "exact_div" => binary(KBinOp::Div),
        "unchecked_rem" => binary(KBinOp::Rem),
        "unchecked_shl" => binary(KBinOp::Shl),
        "unchecked_shr" => binary(KBinOp::Shr),
        "round_ties_even_f32" | "round_ties_even_f64" => call(RoundTiesEven),
        // The operands of a disjoint `or` share no set bits.
        "disjoint_bitor" => binary(KBinOp::Or),
        "cold_path" | "assume" => Some(IntrinsicOp::Nothing),
        name if name.starts_with("prefetch_") => Some(IntrinsicOp::Nothing),
        "transmute" => Some(IntrinsicOp::Transmute),
        "bswap" => Some(IntrinsicOp::ByteSwap),
        name if SIMD.contains(&name) => Some(IntrinsicOp::Simd),
        name if EXACT.contains(&name) => Some(IntrinsicOp::Exact),
        _ => None,
    }
}

/// The intrinsics built from several kernel operations
/// (`Translator::exact`).
const EXACT: [&str; 31] = [
    "float_to_int_unchecked", "is_val_statically_known", "integer_min", "integer_max", "copysignf32",
    "copysignf64", "saturating_add", "saturating_sub", "roundf32", "roundf64", "minimumf32", "minimumf64",
    "maximumf32", "maximumf64", "bitreverse", "unchecked_funnel_shl", "unchecked_funnel_shr", "carryless_mul",
    "carrying_mul_add", "black_box", "select_unpredictable", "raw_eq", "typed_swap_nonoverlapping",
    "assert_inhabited", "assert_zero_valid", "assert_mem_uninitialized_valid", "write_bytes",
    "size_of_val", "align_of_val", "ptr_offset_from", "ptr_offset_from_unsigned",
];

impl<'tcx> Translator<'tcx> {
    /// An intrinsic built from several kernel operations, as its
    /// documentation defines it.
    pub(super) fn exact(
        &mut self,
        frame: &mut Frame<'_, 'tcx>,
        name: &str,
        generic_args: ty::GenericArgsRef<'tcx>,
        args: &[rustc_span::Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
    ) -> Result<()> {
        let tcx = self.tcx;
        let unexpected = || format!("uses the intrinsic `{name}` with unexpected arguments");
        let dest_ty = destination.ty(frame.body, tcx).ty;
        // The value a pointer or reference argument points to.
        let target = |slot: Slot| match slot {
            Slot::Ref(target) => Ok(*target),
            _ => Err(format!("uses the intrinsic `{name}` through a pointer to data of another type")),
        };
        // Intrinsics over whole values.
        match (name, args) {
            // rustc's checks that `mem::zeroed`, `MaybeUninit::assume_init`
            // and the like are valid for their type. A type that fails
            // panics on the CPU every time.
            ("assert_inhabited" | "assert_zero_valid" | "assert_mem_uninitialized_valid", []) => {
                let requirement = match name {
                    "assert_inhabited" => ValidityRequirement::Inhabited,
                    "assert_zero_valid" => ValidityRequirement::Zero,
                    _ => ValidityRequirement::UninitMitigated0x01Fill,
                };
                let ty = generic_args.type_at(0);
                return match tcx.check_validity_requirement((requirement, self.typing_env().as_query_input(ty))) {
                    Ok(true) => Ok(()),
                    _ => Err(format!("uses the intrinsic `{name}` for `{ty}`, which always panics")),
                };
            }
            // `count` values filled with a byte, as `mem::zeroed` does: one
            // value, or none.
            ("write_bytes", [dst, byte, count]) => {
                let pointee = generic_args.type_at(0);
                let count = fold_operand(self.tcx, frame.body, &count.node);
                let dst = target(self.operand_slot(frame, &dst.node)?)?;
                return match count {
                    Some(0) => Ok(()),
                    Some(1) => {
                        let byte = self.scalar_operand(frame, &byte.node)?;
                        let byte = self.b.copy(byte);
                        let size = self.size(pointee)?;
                        let bytes_ty = Ty::new_array(tcx, tcx.types.u8, size);
                        let value =
                            if size == 0 { Slot::Unit } else { Slot::Fields(vec![Slot::Scalar(byte); size as usize]) };
                        match dst {
                            Slot::Bytes { numbers, offset } => self.encode(&numbers, offset, &value, bytes_ty),
                            dst => self.decode(&dst, pointee, 0, &value, bytes_ty),
                        }
                    }
                    _ => Err(format!("uses the intrinsic `{name}` to fill a number of values known only at run time")),
                };
            }
            // Its contract allows `false` for any value.
            ("is_val_statically_known", [_]) => {
                let dest = self.place(frame, destination)?;
                return self.write_number(dest, KRvalue::Use(Const::bool(false).into()));
            }
            // The size of what a pointer points to: a constant for a sized
            // type, the length times the element size for a slice.
            ("align_of_val", [_]) if generic_args.type_at(0).is_sized(tcx, self.typing_env()) => {
                let align = self.layout(generic_args.type_at(0))?.align.abi.bytes();
                let dest = self.place(frame, destination)?;
                return self.write_number(dest, KRvalue::Use(Const::int(KTy::U64, align.into()).into()));
            }
            ("size_of_val", [pointer]) => {
                let pointee = generic_args.type_at(0);
                let value: KOperand = match pointee.kind() {
                    _ if pointee.is_sized(tcx, self.typing_env()) => Const::int(KTy::U64, self.size(pointee)?.into()).into(),
                    ty::Slice(element) => {
                        let size = Const::int(KTy::U64, self.size(*element)?.into());
                        match target(self.operand_slot(frame, &pointer.node)?)? {
                            Slot::Unit => Const::int(KTy::U64, 0).into(),
                            Slot::Fields(elements) => Const::int(KTy::U64, (elements.len() as u64 * self.size(*element)?).into()).into(),
                            Slot::Slice { len, .. } => self.b.binary(KBinOp::Mul, len, size).into(),
                            _ => return Err(unexpected()),
                        }
                    }
                    _ => return Err(format!("uses the intrinsic `{name}` for `{pointee}`")),
                };
                let dest = self.place(frame, destination)?;
                return self.write_number(dest, KRvalue::Use(value));
            }
            // The distance between two pointers into one array or slice, in
            // elements.
            ("ptr_offset_from" | "ptr_offset_from_unsigned", [a, b]) => {
                let pointee = generic_args.type_at(0);
                let (a, b) = (self.operand_slot(frame, &a.node)?, self.operand_slot(frame, &b.node)?);
                let distance: KOperand = match self.positions(&a, &b)? {
                    Some((a, b)) => {
                        let elements = self.b.binary(KBinOp::Sub, a, b);
                        // Bytes are counted in bytes; the others in elements.
                        match (&a, &b) {
                            (KOperand::Const(_), KOperand::Const(_)) => {
                                self.b.binary(KBinOp::Div, elements, Const::int(KTy::U64, self.size(pointee)?.into())).into()
                            }
                            _ => elements.into(),
                        }
                    }
                    None => Const::int(KTy::U64, 0).into(),
                };
                let dest = self.place(frame, destination)?;
                return self.write_number(dest, KRvalue::Cast(distance, held_ty(dest_ty).ok_or_else(unexpected)?));
            }
            ("black_box", [value]) => {
                let value = self.operand_slot(frame, &value.node)?;
                return self.store(frame, destination, value);
            }
            ("select_unpredictable", [condition, yes, no]) => {
                let condition = self.scalar_operand(frame, &condition.node)?;
                let condition = self.b.copy(condition);
                let yes = self.operand_slot(frame, &yes.node)?;
                let no = self.operand_slot(frame, &no.node)?;
                let result = self.snapshot(&no)?;
                self.copy_where(condition, &result, &yes)?;
                return self.store(frame, destination, result);
            }
            ("typed_swap_nonoverlapping", [x, y]) => {
                let x = target(self.operand_slot(frame, &x.node)?)?;
                let y = target(self.operand_slot(frame, &y.node)?)?;
                let old = self.snapshot(&x)?;
                if self.copy(&x, &y)? != x || self.copy(&y, &old)? != y {
                    return Err("swaps values holding references".into());
                }
                return Ok(());
            }
            // Compares the bytes, which its contract requires to be
            // initialized: a float compares by its bits.
            ("raw_eq", [a, b]) => {
                let ty = generic_args.type_at(0);
                let a = target(self.operand_slot(frame, &a.node)?)?;
                let b = target(self.operand_slot(frame, &b.node)?)?;
                let size = self.size(ty)?;
                let mut equal: KOperand = Const::bool(true).into();
                for offset in (0..size).step_by(8) {
                    let width = (size - offset).min(8);
                    let x = self.gather(&a, ty, 0, offset, width)?;
                    let y = self.gather(&b, ty, 0, offset, width)?;
                    let same = self.b.binary(KBinOp::Eq, x, y);
                    equal = self.b.binary(KBinOp::And, equal, same).into();
                }
                let dest = self.place(frame, destination)?;
                return self.write_number(dest, KRvalue::Use(equal));
            }
            _ => {}
        }
        let values = args.iter().map(|arg| self.scalar_operand(frame, &arg.node)).collect::<Result<Vec<_>>>()?;
        let result: KOperand = match (name, values.as_slice()) {
            // Undefined where `as` would saturate, so `as` is right for
            // every defined use.
            ("float_to_int_unchecked", &[value]) => {
                let to = held_ty(dest_ty).filter(|ty| ty.is_int()).ok_or_else(unexpected)?;
                self.b.cast(value, to).into()
            }
            ("integer_min" | "integer_max", &[a, b]) => {
                let op = if name == "integer_min" { KBinOp::Lt } else { KBinOp::Gt };
                let first = self.b.binary(op, a, b);
                self.b.select(first, a, b).into()
            }
            ("copysignf32" | "copysignf64", &[magnitude, sign]) => self.copysign(magnitude, sign).into(),
            ("saturating_add" | "saturating_sub", &[a, b]) => self.saturating(name == "saturating_add", a, b)?.into(),
            ("roundf32" | "roundf64", &[value]) => self.round_away(value).into(),
            ("minimumf32" | "minimumf64" | "maximumf32" | "maximumf64", &[a, b]) => {
                self.float_extreme(name.starts_with("minimum"), a, b).into()
            }
            ("bitreverse", &[value]) => self.bit_reverse(value).into(),
            ("unchecked_funnel_shl" | "unchecked_funnel_shr", &[a, b, shift]) => {
                self.funnel(name == "unchecked_funnel_shl", a, b, shift).into()
            }
            ("carryless_mul", &[a, b]) => self.carryless(a, b).into(),
            // `a * b + c + d` at double width: the low half (unsigned) and
            // the high half, as a tuple.
            ("carrying_mul_add", &[a, b, c, d]) => {
                let ty = self.b.ty(a);
                if ty.bits() > 64 {
                    return Err(format!("uses the intrinsic `{name}` on `{ty:?}`, which needs 256-bit integers"));
                }
                let bytes = u64::from(ty.bytes());
                let wide = if ty.is_signed() { signed_of(2 * bytes) } else { unsigned_of(2 * bytes) };
                let [a, b, c, d] = [a, b, c, d].map(|operand| self.b.cast(operand, wide));
                let product = self.b.binary(KBinOp::Mul, a, b);
                let sum = self.b.binary(KBinOp::Add, product, c);
                let sum = self.b.binary(KBinOp::Add, sum, d);
                let high_bits = self.b.binary(KBinOp::Shr, sum, Const::int(KTy::U32, i128::from(ty.bits())));
                let result = self.fresh(dest_ty)?;
                let Slot::Fields(halves) = &result else { return Err(unexpected()) };
                let [Slot::Scalar(low), Slot::Scalar(high)] = halves.as_slice() else { return Err(unexpected()) };
                let (low, high) = (*low, *high);
                self.set(low, KRvalue::Cast(sum.into(), unsigned_of(bytes)));
                self.set(high, KRvalue::Cast(high_bits.into(), ty));
                return self.store(frame, destination, result);
            }
            _ => return Err(unexpected()),
        };
        let dest = self.place(frame, destination)?;
        self.write_number(dest, KRvalue::Use(result))
    }

    /// The float `magnitude` with the sign bit of `sign`.
    fn copysign(&mut self, magnitude: KOperand, sign: KOperand) -> KLocal {
        let ty = self.b.ty(magnitude);
        let bits_ty = unsigned_of(u64::from(ty.bytes()));
        let sign_bit = 1u128 << (ty.bits() - 1);
        let (m, s) = (self.b.reinterpret(magnitude, bits_ty), self.b.reinterpret(sign, bits_ty));
        let m = self.b.binary(KBinOp::And, m, Const { ty: bits_ty, bits: sign_bit - 1 });
        let s = self.b.binary(KBinOp::And, s, Const { ty: bits_ty, bits: sign_bit });
        let bits = self.b.binary(KBinOp::Or, m, s);
        self.b.reinterpret(bits, ty)
    }

    /// `f32::round`: to the nearest integer, ties away from zero. The
    /// fraction `value - trunc(value)` is exact, and where it is nonzero
    /// the value is below 2^52, so adding one is exact too. NaN and the
    /// infinities keep `trunc`'s result (their fraction is NaN).
    pub(super) fn round_away(&mut self, value: KOperand) -> KLocal {
        let ty = self.b.ty(value);
        let (half, one) = if ty == KTy::F32 { (Const::f32(0.5), Const::f32(1.0)) } else { (Const::f64(0.5), Const::f64(1.0)) };
        let truncated = self.b.call(Intrinsic::Trunc, &[value]);
        let fraction = self.b.binary(KBinOp::Sub, value, truncated);
        let size = self.b.call(Intrinsic::Abs, &[fraction.into()]);
        let far = self.b.binary(KBinOp::Ge, size, half);
        let step = self.copysign(one.into(), value);
        let away = self.b.binary(KBinOp::Add, truncated, step);
        self.b.select(far, away, truncated)
    }

    /// `f32::minimum` or `maximum`: NaN if either operand is, and `-0.0`
    /// below `0.0`.
    fn float_extreme(&mut self, minimum: bool, a: KOperand, b: KOperand) -> KLocal {
        let ty = self.b.ty(a);
        let bits_ty = unsigned_of(u64::from(ty.bytes()));
        let a_nan = self.b.binary(KBinOp::Ne, a, a);
        let b_nan = self.b.binary(KBinOp::Ne, b, b);
        let nan = self.b.binary(KBinOp::Or, a_nan, b_nan);
        let propagated = self.b.binary(KBinOp::Add, a, b);
        // Equal operands differ only in a zero's sign: `minimum` takes the
        // sign bit if either has it, `maximum` only if both do.
        let (x, y) = (self.b.reinterpret(a, bits_ty), self.b.reinterpret(b, bits_ty));
        let merged = self.b.binary(if minimum { KBinOp::Or } else { KBinOp::And }, x, y);
        let equal = self.b.reinterpret(merged, ty);
        let (first, second) = if minimum { (KBinOp::Lt, KBinOp::Gt) } else { (KBinOp::Gt, KBinOp::Lt) };
        let a_wins = self.b.binary(first, a, b);
        let b_wins = self.b.binary(second, a, b);
        let picked = self.b.select(b_wins, b, equal);
        let picked = self.b.select(a_wins, a, picked);
        self.b.select(nan, propagated, picked)
    }

    /// An integer with its bits in reverse order: neighbors swapped in
    /// pairs, then pairs, then nibbles, then the bytes reversed.
    pub(super) fn bit_reverse(&mut self, value: KOperand) -> KLocal {
        let ty = self.b.ty(value);
        let bytes = u64::from(ty.bytes());
        let unsigned = unsigned_of(bytes);
        let width_mask = u128::MAX >> (128 - ty.bits());
        let mut bits: KOperand = self.b.cast(value, unsigned).into();
        for (shift, pattern) in [(1, 0x55u8), (2, 0x33), (4, 0x0F)] {
            let mask = Const { ty: unsigned, bits: u128::from_le_bytes([pattern; 16]) & width_mask };
            let amount = Const::int(KTy::U32, shift);
            let low = self.b.binary(KBinOp::And, bits, mask);
            let up = self.b.binary(KBinOp::Shl, low, amount);
            let high = self.b.binary(KBinOp::Shr, bits, amount);
            let down = self.b.binary(KBinOp::And, high, mask);
            bits = self.b.binary(KBinOp::Or, up, down).into();
        }
        let swapped = self.byte_swap(bits);
        self.b.cast(swapped, ty)
    }

    /// `a` and `b` concatenated (`a` high), shifted left by `shift`, the
    /// high half (`funnel_shl`); or shifted right, the low half
    /// (`funnel_shr`). The shift is below the width, as the contract
    /// requires; at zero the result is `a` or `b`, which the kernel's
    /// masked shifts can't give.
    pub(super) fn funnel(&mut self, left: bool, a: KOperand, b: KOperand, shift: KOperand) -> KLocal {
        let ty = self.b.ty(a);
        let unsigned = unsigned_of(u64::from(ty.bytes()));
        let (a, b) = (self.b.cast(a, unsigned), self.b.cast(b, unsigned));
        let shift = self.b.cast(shift, KTy::U32);
        let rest = self.b.binary(KBinOp::Sub, Const::int(KTy::U32, i128::from(ty.bits())), shift);
        let (a_shift, b_shift, unshifted) = if left { (shift, rest, a) } else { (rest, shift, b) };
        let high = self.b.binary(KBinOp::Shl, a, a_shift);
        let low = self.b.binary(KBinOp::Shr, b, b_shift);
        let joined = self.b.binary(KBinOp::Or, high, low);
        let zero = self.b.binary(KBinOp::Eq, shift, Const::int(KTy::U32, 0));
        let bits = self.b.select(zero, unshifted, joined);
        self.b.cast(bits, ty)
    }

    /// The low half of the carry-less product: `a << i` xored in for each
    /// set bit `i` of `b`.
    pub(super) fn carryless(&mut self, a: KOperand, b: KOperand) -> KLocal {
        let ty = self.b.ty(a);
        let unsigned = unsigned_of(u64::from(ty.bytes()));
        let (a, b) = (self.b.cast(a, unsigned), self.b.cast(b, unsigned));
        let zero = Const::int(unsigned, 0);
        let mut product: KOperand = zero.into();
        for index in 0..ty.bits() {
            let amount = Const::int(KTy::U32, i128::from(index));
            let shifted_b = self.b.binary(KBinOp::Shr, b, amount);
            let bit = self.b.binary(KBinOp::And, shifted_b, Const::int(unsigned, 1));
            let set = self.b.binary(KBinOp::Ne, bit, zero);
            let shifted_a = self.b.binary(KBinOp::Shl, a, amount);
            let term = self.b.select(set, shifted_a, zero);
            product = self.b.binary(KBinOp::Xor, product, term).into();
        }
        self.b.cast(product, ty)
    }

    /// `a + b` or `a - b` of integers, clamped to the type's range.
    pub(super) fn saturating(&mut self, add: bool, a: KOperand, b: KOperand) -> Result<KLocal> {
        let ty = self.b.ty(a);
        if !ty.is_int() {
            return Err("saturates something other than integers".into());
        }
        let op = if add { KBinOp::Add } else { KBinOp::Sub };
        let bits = ty.bits();
        if bits < 64 {
            // Exact in 64 bits, then clamped.
            let (wa, wb) = (self.b.cast(a, KTy::I64), self.b.cast(b, KTy::I64));
            let wide = self.b.binary(op, wa, wb);
            let (min, max) = if ty.is_signed() {
                (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
            } else {
                (0, (1i128 << bits) - 1)
            };
            let (min, max) = (Const::int(KTy::I64, min), Const::int(KTy::I64, max));
            let below = self.b.binary(KBinOp::Lt, wide, min);
            let above = self.b.binary(KBinOp::Gt, wide, max);
            let upper = self.b.select(above, max, wide);
            let clamped = self.b.select(below, min, upper);
            return Ok(self.b.cast(clamped, ty));
        }
        if !ty.is_signed() {
            let result = self.b.binary(op, a, b);
            return Ok(if add {
                let wrapped = self.b.binary(KBinOp::Lt, result, a);
                self.b.select(wrapped, Const::int(ty, -1), result)
            } else {
                let wrapped = self.b.binary(KBinOp::Lt, a, b);
                self.b.select(wrapped, Const::int(ty, 0), result)
            });
        }
        // Signed 64 and 128 bits: on overflow, the result saturates toward
        // `a`'s sign.
        let checked = if add { BinOp::AddWithOverflow } else { BinOp::SubWithOverflow };
        let (result, overflowed) = self.with_overflow(checked, a, b)?;
        let negative = self.b.binary(KBinOp::Lt, a, Const::int(ty, 0));
        let limit = self.b.select(negative, int_min(ty), int_max(ty));
        Ok(self.b.select(overflowed, limit, result))
    }

    /// An integer with its bytes in reverse order.
    pub(super) fn byte_swap(&mut self, value: KOperand) -> KLocal {
        let ty = self.b.ty(value);
        let bytes = u64::from(ty.bytes());
        if bytes == 1 {
            return self.b.copy(value);
        }
        let unsigned = unsigned_of(bytes);
        let bits = self.b.cast(value, unsigned);
        let mut swapped: KOperand = Const::int(unsigned, 0).into();
        for index in 0..bytes {
            let shifted = self.b.binary(KBinOp::Shr, bits, Const::int(KTy::U32, (8 * index).into()));
            let byte = self.b.binary(KBinOp::And, shifted, Const::int(unsigned, 0xFF));
            let placed = self.b.binary(KBinOp::Shl, byte, Const::int(KTy::U32, (8 * (bytes - 1 - index)).into()));
            swapped = self.b.binary(KBinOp::Or, swapped, placed).into();
        }
        self.b.cast(swapped, ty)
    }
}
