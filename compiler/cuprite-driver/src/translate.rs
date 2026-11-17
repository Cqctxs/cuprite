//! Translation of a loop body's MIR into a kernel element function
//! (`cuprite_kernel::Function`): what one iteration does to its element.
//!
//! The function takes the element's value and the enclosing function's
//! scalar parameters, and returns the element's new value.
//!
//! # Slots
//!
//! Every MIR value lives in a [`Slot`]: a number in a kernel local, a struct
//! or tuple as one slot per field, or a reference as the slot it points to.
//! Memory is never modelled. Reading through a reference reads the slot it
//! names; writing through it writes that slot. The loop's element is a
//! kernel local like any other, and its reference points to it. Copying a
//! struct copies its fields.
//!
//! # Calls
//!
//! A call to a function with available MIR is inlined: its arguments bind
//! its parameters (a reference parameter aliases the argument's slot), its
//! optimized MIR is instantiated with the call's types and translated into
//! the same kernel, and its return value is copied to the destination.
//! Trait methods are resolved to their implementations, so generic code
//! from other crates (`glam`, `core`'s operator impls) works with the types
//! it is called with. Library methods in [`method`] and intrinsics in
//! [`intrinsic`] become kernel operations directly.
//!
//! Closures are values too: a closure is its captures, a struct whose
//! by-reference captures are references, so they alias what they captured.
//! A reference inside a value is bound like a reference local. Calls
//! through the `Fn` traits use the "rust-call" convention, which passes the
//! arguments as one tuple; it is spread into the closure body's parameters.
//! The bodies rustc builds for calling an `Fn` closure through `FnOnce`, or
//! a function item through the `Fn` traits, are inlined the same way.
//!
//! # Arrays
//!
//! An array is held as its elements, like a tuple, up to [`MAX_ARRAY`] of
//! them. A constant index names an element. An index known only at run time
//! names a [`Slot::Choice`]: reading it selects among every element,
//! writing it writes the element chosen and keeps the others. MIR checks the
//! index first, and the check runs on the GPU too. An array as a slice
//! keeps its elements, so its length is known. Constant tables (`&[f32; 5]`,
//! `&[u8]`) and read-only statics of plain data become elements the same
//! way.
//!
//! `transmute` (and `to_bits`, `from_le_bytes`...) reinterprets plain data
//! by rustc's layout: each number of the result gathers the bytes it covers
//! from the source's numbers.
//!
//! # Enums
//!
//! An enum is held as its discriminant's value and every variant's fields
//! ([`Slot::Enum`]); only the active variant's fields mean anything.
//! Building a variant sets the discriminant and that variant's fields;
//! `match` reads the discriminant and switches on it, as MIR does. Layout
//! matters only where bytes do: constant enums and `transmute` read and
//! write the tag or niche (`Option<NonZero<u32>>`'s `None` is a zero) as
//! rustc's layout stores it. `Ord::cmp` of numbers gives `Ordering`'s
//! discriminant directly. A `char` is held as its `u32` code, and a pattern
//! type (`NonZero`'s field) as the number it restricts.
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
//!
//! # Pointer casts
//!
//! A cast from `*const T` to `*const U` is a view of the same bytes: where
//! `U`'s numbers are `T`'s locals at the same offsets and types, the view
//! aliases them. Otherwise it is [`Slot::Bytes`]: reading through it decodes
//! the bytes as the type read, as `transmute` would, and writing through it
//! is refused.
//!
//! Supported values: integers, floats, `bool` and `char`, structs, tuples,
//! arrays, enums and SIMD vectors of them, closures, with references to any
//! of those.
//! Slices other than read-only parameters and constants, unions, raw
//! pointer arithmetic, coroutines and statics that can change are refused
//! with a reason.
//!
//! Panics must be provable away: [`assert_holds`] accepts an assertion only
//! when its condition folds to a constant that passes, as for a division by
//! a nonzero constant, in the loop body and in every function it calls.

use std::collections::{HashMap, HashSet};

use cuprite_kernel::{
    builder::Builder, BinOp as KBinOp, BlockId, Const, Function, Intrinsic, Local as KLocal, Operand as KOperand,
    Rvalue as KRvalue, Ty as KTy, UnOp as KUnOp,
};
use rustc_middle::{
    mir::{
        interpret::{GlobalAlloc, Scalar},
        AggregateKind, BasicBlock, BinOp, Body, CastKind, ConstOperand, ConstValue, Local, Operand, Place,
        ProjectionElem, Rvalue, StatementKind, TerminatorKind, UnOp, RETURN_PLACE, START_BLOCK,
    },
    ty::{self, FloatTy, Instance, InstanceKind, IntTy, ShimKind, Ty, TyCtxt, TypingEnv, UintTy},
};
use rustc_abi::{ExternAbi, TagEncoding, VariantIdx, Variants};
use rustc_middle::{mir::NonDivergingIntrinsic, ty::adjustment::PointerCoercion};
use rustc_span::def_id::DefId;

type Result<T> = std::result::Result<T, String>;

/// The kernel type of a MIR scalar type, on a 64-bit target.
pub(crate) fn kernel_ty(ty: Ty<'_>) -> Option<KTy> {
    Some(match ty.kind() {
        ty::Bool => KTy::Bool,
        ty::Int(IntTy::I8) => KTy::I8,
        ty::Int(IntTy::I16) => KTy::I16,
        ty::Int(IntTy::I32) => KTy::I32,
        ty::Int(IntTy::I64 | IntTy::Isize) => KTy::I64,
        ty::Uint(UintTy::U8) => KTy::U8,
        ty::Uint(UintTy::U16) => KTy::U16,
        ty::Uint(UintTy::U32) => KTy::U32,
        ty::Uint(UintTy::U64 | UintTy::Usize) => KTy::U64,
        ty::Float(FloatTy::F32) => KTy::F32,
        ty::Float(FloatTy::F64) => KTy::F64,
        _ => return None,
    })
}

/// The kernel type a number is held in inside a kernel: as `kernel_ty`,
/// 128-bit integers (as pairs of halves, by `cuprite_kernel::wide`), and
/// a `char` as its `u32` code, a pattern type (`NonZero`'s field) as the
/// type it restricts. Only valid values ever reach these, so the kernel
/// computes on them as the CPU does. Parameters and slices take only
/// `kernel_ty`'s types.
pub(crate) fn held_ty(ty: Ty<'_>) -> Option<KTy> {
    match ty.kind() {
        ty::Int(IntTy::I128) => Some(KTy::I128),
        ty::Uint(UintTy::U128) => Some(KTy::U128),
        ty::Char => Some(KTy::U32),
        ty::Pat(inner, _) => held_ty(*inner),
        _ => kernel_ty(ty),
    }
}

/// A scalar constant of a supported type, evaluating it if it's an
/// unevaluated constant such as an associated `const`.
pub(crate) fn constant<'tcx>(tcx: TyCtxt<'tcx>, constant: &ConstOperand<'tcx>) -> Option<Const> {
    let ty = held_ty(constant.ty())?;
    let scalar = match constant.const_.try_to_scalar_int() {
        Some(scalar) => scalar,
        None => match constant.const_.eval(tcx, TypingEnv::fully_monomorphized(), constant.span).ok()? {
            ConstValue::Scalar(Scalar::Int(scalar)) => scalar,
            _ => return None,
        },
    };
    let bits = scalar.to_bits(scalar.size());
    Some(Const { ty, bits })
}

/// How many elements an array the kernel holds in locals may have. A run-time
/// index chooses among all of them, so each access costs one operation per
/// element.
const MAX_ARRAY: u64 = 1024;

/// The number of elements of an array type.
pub(crate) fn array_len<'tcx>(tcx: TyCtxt<'tcx>, len: ty::Const<'tcx>) -> Result<usize> {
    let len = tcx
        .normalize_erasing_regions(TypingEnv::fully_monomorphized(), ty::Unnormalized::new(len))
        .try_to_target_usize(tcx)
        .ok_or("uses an array whose length isn't known")?;
    if len > MAX_ARRAY {
        return Err(format!("uses an array of {len} elements, more than the {MAX_ARRAY} a kernel holds in locals"));
    }
    Ok(len as usize)
}

/// The field types of a struct, tuple or closure (its captures), or an
/// array's elements, with the generic arguments filled in. A function item
/// has none. Errors for anything else that isn't a number.
pub(crate) fn fields_of<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Result<Vec<Ty<'tcx>>> {
    match ty.kind() {
        ty::Tuple(fields) => Ok(fields.to_vec()),
        // An array is held as its elements, like a tuple of them.
        ty::Array(element, len) => Ok(vec![*element; array_len(tcx, *len)?]),
        // Captured by reference, a capture's type is the reference.
        ty::Closure(_, args) => Ok(args.as_closure().upvar_tys().to_vec()),
        ty::FnDef(..) => Ok(Vec::new()),
        ty::Adt(adt, args) if adt.is_struct() && !adt.is_box() => Ok(adt
            .non_enum_variant()
            .fields
            .iter()
            .map(|field| tcx.normalize_erasing_regions(TypingEnv::fully_monomorphized(), field.ty(tcx, args)))
            .collect()),
        ty::Never => Ok(Vec::new()),
        _ => Err(type_refusal(ty)),
    }
}

/// A CPU feature test, `std_detect::detect::cache::test(bit)`, which
/// `is_x86_feature_detected!` calls: the kernel takes the host's answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Feature {
    pub(crate) test: DefId,
    pub(crate) bit: u32,
}

/// One number passed to the kernel as an argument: read from a parameter of
/// plain data, or the host's answer to a CPU feature test.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Leaf {
    pub(crate) param: Local,
    /// The parameter is a shared reference, read through first.
    pub(crate) deref: bool,
    /// Field indices from the parameter (or what it points to) to the
    /// number; empty for a number itself.
    pub(crate) fields: Vec<usize>,
    pub(crate) ty: KTy,
    /// For a feature test's answer, the test; `param`, `deref` and `fields`
    /// are then unused.
    pub(crate) feature: Option<Feature>,
}

impl Leaf {
    /// The answer to a feature test, as a `bool`.
    pub(crate) fn feature(feature: Feature) -> Self {
        Leaf { param: Local::from_u32(0), deref: false, fields: Vec::new(), ty: KTy::Bool, feature: Some(feature) }
    }

    /// The type the kernel receives it as: narrow integers and `bool` are
    /// widened to what a launch can pass.
    pub(crate) fn passed(&self) -> KTy {
        match self.ty {
            KTy::Bool | KTy::U8 | KTy::U16 => KTy::U32,
            KTy::I8 | KTy::I16 => KTy::I32,
            other => other,
        }
    }
}

/// How many numbers a parameter may pass.
const MAX_LEAVES: usize = 256;

/// The numbers in a parameter of plain data: a number, a struct or tuple
/// of numbers, or a shared reference to one. `None` if it is something
/// else (with the reason).
pub(crate) fn leaves<'tcx>(tcx: TyCtxt<'tcx>, param: Local, ty: Ty<'tcx>) -> Result<Vec<Leaf>> {
    fn walk<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>, path: &mut Vec<usize>, out: &mut Vec<(Vec<usize>, KTy)>) -> Result<()> {
        if let Some(kernel) = kernel_ty(ty) {
            out.push((path.clone(), kernel));
            return Ok(());
        }
        // A vector's lanes can't be named by field projections on the CPU side.
        if matches!(ty.kind(), ty::Adt(adt, _) if adt.repr().simd()) {
            return Err(format!("is the SIMD vector `{ty}`"));
        }
        for (index, field) in fields_of(tcx, ty)?.into_iter().enumerate() {
            path.push(index);
            walk(tcx, field, path, out)?;
            path.pop();
        }
        if out.len() > MAX_LEAVES {
            return Err(format!("holds more than {MAX_LEAVES} numbers"));
        }
        Ok(())
    }
    let (deref, ty) = match ty.kind() {
        ty::Ref(_, inner, ty::Mutability::Not) => (true, *inner),
        ty::Ref(..) => return Err("is a mutable reference".into()),
        _ => (false, ty),
    };
    let mut found = Vec::new();
    walk(tcx, ty, &mut Vec::new(), &mut found)?;
    Ok(found.into_iter().map(|(fields, ty)| Leaf { param, deref, fields, ty, feature: None }).collect())
}

/// What a `for` loop's iterator yields: the element, a value of a
/// read-only input slice, or a pair of those, from `zip`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Lane {
    /// `&mut` to the current element of the slice parameter.
    Element,
    /// `&` to the current element of this input slice parameter.
    Input(Local),
    Pair(Box<Lane>, Box<Lane>),
}

impl Lane {
    /// The input slices, in the order the tuple names them.
    pub(crate) fn inputs(&self, out: &mut Vec<Local>) {
        match self {
            Lane::Element => {}
            Lane::Input(local) => out.push(*local),
            Lane::Pair(a, b) => {
                a.inputs(out);
                b.inputs(out);
            }
        }
    }

    pub(crate) fn elements(&self) -> usize {
        match self {
            Lane::Element => 1,
            Lane::Input(_) => 0,
            Lane::Pair(a, b) => a.elements() + b.elements(),
        }
    }
}

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
}

/// The `simd_*` intrinsics translated lane by lane.
const SIMD: [&str; 51] = [
    "simd_insert", "simd_extract", "simd_insert_dyn", "simd_extract_dyn", "simd_splat", "simd_add", "simd_sub",
    "simd_mul", "simd_div", "simd_rem", "simd_shl", "simd_shr", "simd_and", "simd_or", "simd_xor", "simd_cast",
    "simd_as", "simd_neg", "simd_fabs", "simd_minimum_number_nsz", "simd_maximum_number_nsz", "simd_eq", "simd_ne",
    "simd_lt", "simd_le", "simd_gt", "simd_ge", "simd_shuffle", "simd_saturating_add", "simd_saturating_sub",
    "simd_reduce_add_ordered", "simd_reduce_mul_ordered", "simd_reduce_all", "simd_reduce_any", "simd_reduce_max",
    "simd_reduce_min", "simd_reduce_and", "simd_reduce_or", "simd_reduce_xor", "simd_bitmask", "simd_select",
    "simd_select_bitmask", "simd_bswap", "simd_ctlz", "simd_ctpop", "simd_cttz", "simd_ceil", "simd_floor",
    "simd_trunc", "simd_fsqrt", "simd_fma",
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
        "cold_path" | "assume" => Some(IntrinsicOp::Nothing),
        "transmute" => Some(IntrinsicOp::Transmute),
        "bswap" => Some(IntrinsicOp::ByteSwap),
        name if SIMD.contains(&name) => Some(IntrinsicOp::Simd),
        _ => None,
    }
}

/// True if the assertion ending `block` always passes: its condition folds
/// to the expected constant.
pub(crate) fn assert_holds(body: &Body<'_>, block: BasicBlock) -> bool {
    let TerminatorKind::Assert { cond, expected, .. } = &body.basic_blocks[block].terminator().kind else {
        return false;
    };
    Folder { body }.operand(cond, 0) == Some(i128::from(*expected))
}

/// Folds integer and `bool` expressions over constants, following locals
/// assigned exactly once.
struct Folder<'a, 'tcx> {
    body: &'a Body<'tcx>,
}

impl<'tcx> Folder<'_, 'tcx> {
    const DEPTH: usize = 16;

    fn operand(&self, operand: &Operand<'tcx>, depth: usize) -> Option<i128> {
        match operand {
            Operand::Constant(c) => {
                let scalar = c.const_.try_to_scalar_int()?;
                match c.ty().kind() {
                    ty::Bool => Some(i128::from(scalar.try_to_bool().ok()?)),
                    ty::Int(_) => Some(scalar.to_int(scalar.size())),
                    ty::Uint(_) => i128::try_from(scalar.to_uint(scalar.size())).ok(),
                    _ => None,
                }
            }
            Operand::Copy(place) | Operand::Move(place) if depth < Self::DEPTH => {
                let local = place.as_local()?;
                self.definition(local).and_then(|rvalue| self.rvalue(rvalue, depth + 1))
            }
            _ => None,
        }
    }

    fn is_bool(&self, operand: &Operand<'tcx>) -> bool {
        match operand {
            Operand::Constant(c) => c.ty().is_bool(),
            Operand::Copy(place) | Operand::Move(place) => {
                place.projection.is_empty() && self.body.local_decls[place.local].ty.is_bool()
            }
            Operand::RuntimeChecks(_) => false,
        }
    }

    /// The one statement assigning `local`, if nothing else does.
    fn definition(&self, local: Local) -> Option<&Rvalue<'tcx>> {
        let mut found = None;
        for data in self.body.basic_blocks.iter() {
            for statement in &data.statements {
                if let StatementKind::Assign(assign) = &statement.kind {
                    if assign.0.local == local {
                        if found.is_some() || !assign.0.projection.is_empty() {
                            return None;
                        }
                        found = Some(&assign.1);
                    }
                }
            }
            if let TerminatorKind::Call { destination, .. } = &data.terminator().kind {
                if destination.local == local {
                    return None;
                }
            }
        }
        found
    }

    fn rvalue(&self, rvalue: &Rvalue<'tcx>, depth: usize) -> Option<i128> {
        match rvalue {
            Rvalue::Use(operand, _) => self.operand(operand, depth),
            Rvalue::UnaryOp(UnOp::Not, operand) if self.is_bool(operand) => {
                self.operand(operand, depth).map(|value| 1 - value)
            }
            // Shift checks compare the amount after casting it to `u32`.
            Rvalue::Cast(CastKind::IntToInt, operand, ty) => {
                let value = self.operand(operand, depth)?;
                let (signed, bits) = match ty.kind() {
                    ty::Int(int) => (true, int.bit_width().unwrap_or(64)),
                    ty::Uint(uint) => (false, uint.bit_width().unwrap_or(64)),
                    _ => return None,
                };
                let fits = if signed {
                    let half = 1i128 << (bits - 1);
                    (-half..half).contains(&value)
                } else {
                    (0..1i128 << bits).contains(&value)
                };
                fits.then_some(value)
            }
            Rvalue::BinaryOp(op, operands) => {
                let (a, b) = (self.operand(&operands.0, depth), self.operand(&operands.1, depth));
                match op {
                    // A known `false` decides `&`, a known `true` decides `|`.
                    BinOp::BitAnd if a == Some(0) || b == Some(0) => Some(0),
                    BinOp::BitOr if a == Some(1) || b == Some(1) => Some(1),
                    _ => {
                        let (a, b) = (a?, b?);
                        Some(i128::from(match op {
                            BinOp::Eq => a == b,
                            BinOp::Ne => a != b,
                            BinOp::Lt => a < b,
                            BinOp::Le => a <= b,
                            BinOp::Gt => a > b,
                            BinOp::Ge => a >= b,
                            BinOp::BitAnd => a != 0 && b != 0,
                            BinOp::BitOr => a != 0 || b != 0,
                            _ => return None,
                        }))
                    }
                }
            }
            _ => None,
        }
    }
}

/// What the function around a loop reads and writes, around the
/// translation.
pub(crate) struct Loop<'a, 'tcx> {
    pub(crate) iteration: Iteration<'a, 'tcx>,
    /// What the iterator yields.
    pub(crate) lane: &'a Lane,
    /// The input slices `lane` reads, in order, and their element types.
    pub(crate) inputs: &'a [(Local, KTy)],
    /// The read-only slices the code indexes, and their element types.
    pub(crate) tables: &'a [(Local, KTy)],
    /// The numbers the kernel receives from the other parameters.
    pub(crate) leaves: &'a [Leaf],
    /// Parameters only the iterator may use: the slices.
    pub(crate) opaque: &'a [Local],
    /// The calls that build the iterator, which the kernel skips.
    pub(crate) chain: &'a HashSet<BasicBlock>,
    /// The locals the iterator passes through, which the kernel skips.
    pub(crate) plumbing: &'a HashSet<Local>,
}

/// How the function runs one iteration.
#[derive(Clone, Copy)]
pub(crate) enum Iteration<'a, 'tcx> {
    /// A `for` loop.
    For {
        /// `next`'s result: `(option as Some).0` is what the iterator yields.
        option: Place<'tcx>,
        /// The first block of the body.
        start: BasicBlock,
        /// True for the loop header's blocks, where an iteration ends.
        is_header: &'a dyn Fn(BasicBlock) -> bool,
    },
    /// `for_each(iterator, function)`, ending `block`: an iteration calls
    /// `function` with what the iterator yields.
    ForEach { block: BasicBlock, function: &'a Operand<'tcx> },
}

impl<'tcx> Loop<'_, 'tcx> {
    fn is_header(&self, block: BasicBlock) -> bool {
        match &self.iteration {
            Iteration::For { is_header, .. } => is_header(block),
            Iteration::ForEach { .. } => false,
        }
    }

    /// The block whose call is the iteration, for `for_each`.
    fn each_block(&self) -> Option<BasicBlock> {
        match &self.iteration {
            Iteration::For { .. } => None,
            Iteration::ForEach { block, .. } => Some(*block),
        }
    }
}

/// A translated function around its loop.
pub(crate) struct Translated {
    pub(crate) function: Function,
    /// The blocks the kernel runs: the setup before the loop and its body.
    pub(crate) blocks: HashSet<BasicBlock>,
    /// Whether the body writes the element at all.
    pub(crate) stores: bool,
    /// The kernel's scalar arguments: the loop's leaves.
    pub(crate) leaves: Vec<Leaf>,
    /// The CPU feature tests the code makes, in order; each must be among
    /// the leaves for the kernel to read the host's answer.
    pub(crate) features: Vec<Feature>,
}

/// Translates the code before the loop and the loop body. The element
/// function computes the setup again for each element, which is pure and
/// usually folds away, then runs one iteration.
pub(crate) fn translate<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    name: &str,
    element: KTy,
    shape: &Loop<'_, 'tcx>,
) -> Result<Translated> {
    let (loop_blocks, setup) = match &shape.iteration {
        Iteration::For { start, .. } => {
            let loop_blocks = loop_region(body, shape, *start)?;
            let setup = setup_region(body, shape, &loop_blocks)?;
            check_carried(body, &setup, &loop_blocks, shape)?;
            (loop_blocks, setup)
        }
        // The setup ends with the `for_each` call, which is the iteration.
        // The closure can't carry values between elements: `each` accepts
        // only `Fn` closures, which can't change what they capture.
        Iteration::ForEach { block, .. } => {
            let mut setup = setup_region(body, shape, &HashSet::new())?;
            setup.insert(*block);
            (HashSet::new(), setup)
        }
    };

    let mut params = vec![element];
    params.extend(shape.inputs.iter().map(|&(_, ty)| ty));
    for _ in shape.tables {
        params.extend([KTy::Ptr, KTy::U64]);
    }
    params.extend(shape.leaves.iter().map(Leaf::passed));
    let (mut b, kernel_params) = Builder::new(name, &params);
    let value = b.copy(kernel_params[0]);
    let mut translator = Translator {
        tcx,
        b,
        element: value,
        stores: false,
        stack: Vec::new(),
        written: HashSet::new(),
        features: Vec::new(),
        wanted: Vec::new(),
    };

    let mut slots = HashMap::new();
    for &local in shape.opaque {
        slots.insert(local, Slot::Opaque);
    }
    // The values of the input slices this iteration reads.
    let inputs: HashMap<Local, KLocal> =
        shape.inputs.iter().zip(&kernel_params[1..]).map(|(&(local, _), &param)| (local, param)).collect();
    // The other parameters, rebuilt from their numbers.
    // The tables, as their address and length.
    let table_params = &kernel_params[1 + shape.inputs.len()..1 + shape.inputs.len() + 2 * shape.tables.len()];
    for (&(local, element), pair) in shape.tables.iter().zip(table_params.as_chunks::<2>().0) {
        let slice = Slot::Slice { pointer: pair[0], len: pair[1], element };
        slots.insert(local, Slot::Ref(Box::new(slice)));
    }
    let leaf_params = &kernel_params[1 + shape.inputs.len() + 2 * shape.tables.len()..];
    let mut by_param: Vec<(Local, Vec<(&Leaf, KLocal)>)> = Vec::new();
    for (leaf, &param) in shape.leaves.iter().zip(leaf_params) {
        if let Some(feature) = leaf.feature {
            translator.features.push((feature, param));
            continue;
        }
        match by_param.iter_mut().find(|(local, _)| *local == leaf.param) {
            Some((_, leaves)) => leaves.push((leaf, param)),
            None => by_param.push((leaf.param, vec![(leaf, param)])),
        }
    }
    for param in (2..=body.arg_count).map(Local::from_usize) {
        if slots.contains_key(&param) {
            continue;
        }
        let ty = body.local_decls[param].ty;
        let pointee = match ty.kind() {
            ty::Ref(_, inner, _) => *inner,
            _ => ty,
        };
        let tree = translator.fresh(pointee)?;
        for (leaf, kernel) in by_param.iter().find(|(local, _)| *local == param).map_or(&[][..], |(_, l)| l) {
            let mut slot = &tree;
            for &index in &leaf.fields {
                let Slot::Fields(fields) = slot else { unreachable!("leaves follow the type") };
                slot = &fields[index];
            }
            let Slot::Scalar(local) = slot else { unreachable!("leaves are numbers") };
            let rvalue = if leaf.ty == KTy::Bool {
                KRvalue::Binary(KBinOp::Ne, (*kernel).into(), Const::int(KTy::U32, 0).into())
            } else if leaf.ty != leaf.passed() {
                KRvalue::Cast((*kernel).into(), leaf.ty)
            } else {
                KRvalue::Use((*kernel).into())
            };
            translator.b.set(*local, rvalue);
        }
        let slot = if is_ref(ty) { Slot::Ref(Box::new(tree)) } else { tree };
        slots.insert(param, slot);
    }
    let item = lane_slot(shape.lane, value, &inputs);
    if let Iteration::For { option, .. } = &shape.iteration {
        slots.insert(option.local, Slot::Some(Box::new(item.clone())));
    }

    let mut blocks: HashSet<BasicBlock> = setup.union(&loop_blocks).copied().collect();
    let mut frame = Frame {
        body,
        slots,
        blocks: HashMap::new(),
        exit: BlockId(0),
        loop_rules: Some(LoopRules {
            shape,
            body: loop_blocks.clone(),
            assigned_outside: assigned_outside(body, &blocks),
            item,
        }),
    };
    frame.exit = translator.b.block();
    translator.run_frame(&mut frame, &ordered(body, &blocks), START_BLOCK)?;
    let stores = translator.stores;
    let features = std::mem::take(&mut translator.wanted);
    let function = translator
        .b
        .finish_function(Some(value))
        .map_err(|invalid| format!("translated into an invalid kernel ({invalid})"))?;
    blocks.retain(|block| !shape.chain.contains(block));
    Ok(Translated { function, blocks, stores, leaves: shape.leaves.to_vec(), features })
}

/// Whether `ty` is element data: built from numbers, through structs,
/// tuples, arrays, enums, closures and references, without slices, strings,
/// trait objects, raw pointers or function pointers. The coverage census
/// translates functions over element data, and skips the rest (loops over
/// slices, text, heap containers), with the reason.
pub(crate) fn element_data<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>, depth: usize) -> Result<()> {
    if depth > 16 {
        return Err(format!("nests `{ty}` too deeply"));
    }
    let walk = |ty: Ty<'tcx>| element_data(tcx, ty, depth + 1);
    match ty.kind() {
        ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::Never | ty::FnDef(..) => Ok(()),
        ty::Ref(_, inner, _) => walk(*inner),
        ty::Array(inner, _) => walk(*inner),
        // A pattern type (`NonZero`, `NonNull`) holds its underlying type.
        ty::Pat(inner, _) => walk(*inner),
        ty::Tuple(fields) => fields.iter().try_for_each(walk),
        ty::Closure(_, args) => args.as_closure().upvar_tys().iter().try_for_each(walk),
        ty::Adt(adt, _) if adt.is_box() => Err("holds a `Box`".into()),
        ty::Adt(adt, args) => adt
            .all_fields()
            .map(|field| tcx.normalize_erasing_regions(TypingEnv::fully_monomorphized(), field.ty(tcx, args)))
            .try_for_each(walk)
            .map_err(|reason| if depth == 0 { format!("takes or returns `{ty}`, which {reason}") } else { reason }),
        ty::Slice(_) | ty::Str => Err(format!("holds a slice or string (`{ty}`)")),
        ty::RawPtr(..) => Err(format!("holds a raw pointer (`{ty}`)")),
        ty::FnPtr(..) => Err(format!("holds a function pointer (`{ty}`)")),
        ty::Dynamic(..) => Err(format!("holds a trait object (`{ty}`)")),
        _ => Err(format!("holds `{ty}`")),
    }
}

/// Translates a whole non-generic function or closure as a kernel function
/// of its parameters, for the coverage census: `Ok` if the translator
/// handles everything it does (with every call inlined), else the first
/// refusal.
pub(crate) fn check_function<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId, body: &Body<'tcx>) -> Result<()> {
    let (mut b, _) = Builder::new("coverage", &[]);
    let element = b.local(KTy::U32);
    let mut translator = Translator { tcx, b, element, stores: false, stack: vec![def_id], written: HashSet::new(), features: Vec::new(), wanted: Vec::new() };
    let mut slots = HashMap::new();
    for param in (1..=body.arg_count).map(Local::from_usize) {
        let ty = body.local_decls[param].ty;
        slots.insert(param, translator.fresh_deep(ty, 0)?);
    }
    let mut frame = Frame { body, slots, blocks: HashMap::new(), exit: BlockId(0), loop_rules: None };
    frame.exit = translator.b.block();
    translator.run_frame(&mut frame, &ordered(body, &function_region(body)), START_BLOCK)?;
    translator.b.switch_to(frame.exit);
    translator.local_slot(&mut frame, RETURN_PLACE)?;
    translator
        .b
        .finish_function(None)
        .map(|_| ())
        .map_err(|invalid| format!("translated into an invalid kernel ({invalid})"))
}

/// The slot for what the iterator yields: references to the element and
/// to this iteration's input values, paired as `zip` pairs them.
fn lane_slot(lane: &Lane, element: KLocal, inputs: &HashMap<Local, KLocal>) -> Slot {
    match lane {
        Lane::Element => Slot::Ref(Box::new(Slot::Scalar(element))),
        Lane::Input(local) => Slot::Ref(Box::new(Slot::Scalar(inputs[local]))),
        Lane::Pair(a, b) => Slot::Fields(vec![lane_slot(a, element, inputs), lane_slot(b, element, inputs)]),
    }
}

/// The loop body's blocks: everything reachable from its start without
/// passing the header. Leaving the loop any other way (`break`, `return`)
/// makes later elements depend on earlier ones, so it is refused.
fn loop_region(body: &Body<'_>, shape: &Loop<'_, '_>, start: BasicBlock) -> Result<HashSet<BasicBlock>> {
    let mut region = HashSet::new();
    let mut pending = vec![start];
    while let Some(block) = pending.pop() {
        if shape.is_header(block) || !region.insert(block) {
            continue;
        }
        let data = &body.basic_blocks[block];
        if data.is_cleanup {
            return Err("has an unsupported loop shape".into());
        }
        match &data.terminator().kind {
            TerminatorKind::Return => return Err("leaves the loop early".into()),
            kind => pending.extend(successors(kind)),
        }
    }
    Ok(region)
}

/// The blocks before the loop: everything reachable from the start without
/// passing the header, or reaching the `for_each` call.
fn setup_region(body: &Body<'_>, shape: &Loop<'_, '_>, loop_blocks: &HashSet<BasicBlock>) -> Result<HashSet<BasicBlock>> {
    let mut region = HashSet::new();
    let mut pending = vec![START_BLOCK];
    while let Some(block) = pending.pop() {
        if shape.is_header(block) || shape.each_block() == Some(block) || !region.insert(block) {
            continue;
        }
        if loop_blocks.contains(&block) {
            return Err("has an unsupported loop shape".into());
        }
        pending.extend(successors(&body.basic_blocks[block].terminator().kind));
    }
    Ok(region)
}

/// Each thread runs the setup and one iteration, so a value one iteration
/// leaves for the next (a counter, a running total) can't be carried:
/// refuse any local the body assigns that is also assigned before the loop
/// or in its header.
fn check_carried(
    body: &Body<'_>,
    setup: &HashSet<BasicBlock>,
    loop_blocks: &HashSet<BasicBlock>,
    shape: &Loop<'_, '_>,
) -> Result<()> {
    let assigned_in = |blocks: &dyn Fn(BasicBlock) -> bool| {
        let mut assigned = HashSet::new();
        for (block, data) in body.basic_blocks.iter_enumerated() {
            if !blocks(block) || data.is_cleanup {
                continue;
            }
            for statement in &data.statements {
                if let StatementKind::Assign(assign) = &statement.kind {
                    assigned.insert(assign.0.local);
                }
            }
            if let TerminatorKind::Call { destination, .. } = &data.terminator().kind {
                assigned.insert(destination.local);
            }
        }
        assigned
    };
    let inside = assigned_in(&|block| loop_blocks.contains(&block));
    let outside = assigned_in(&|block| !loop_blocks.contains(&block) && (setup.contains(&block) || shape.is_header(block)));
    let carried = inside.intersection(&outside).any(|&local| {
        let ty = body.local_decls[local].ty;
        local != RETURN_PLACE && !ty.is_unit() && !shape.plumbing.contains(&local)
    });
    if carried {
        return Err("carries a value from one element to the next".into());
    }
    Ok(())
}

/// Every block a function body runs, from its start, ignoring unwinding.
fn function_region(body: &Body<'_>) -> HashSet<BasicBlock> {
    let mut region = HashSet::new();
    let mut pending = vec![START_BLOCK];
    while let Some(block) = pending.pop() {
        if region.insert(block) {
            pending.extend(successors(&body.basic_blocks[block].terminator().kind));
        }
    }
    region
}

/// Where a terminator continues when nothing unwinds.
fn successors(kind: &TerminatorKind<'_>) -> Vec<BasicBlock> {
    match kind {
        TerminatorKind::Goto { target, .. }
        | TerminatorKind::FalseEdge { real_target: target, .. }
        | TerminatorKind::FalseUnwind { real_target: target, .. }
        | TerminatorKind::Assert { target, .. }
        | TerminatorKind::Drop { target, .. }
        | TerminatorKind::Call { target: Some(target), .. } => vec![*target],
        TerminatorKind::SwitchInt { targets, .. } => targets.all_targets().to_vec(),
        _ => Vec::new(),
    }
}

/// `region` in reverse postorder, so a block comes after the blocks that
/// dominate it, where its values and references are set.
fn ordered(body: &Body<'_>, region: &HashSet<BasicBlock>) -> Vec<BasicBlock> {
    body.basic_blocks.reverse_postorder().iter().copied().filter(|block| region.contains(block)).collect()
}

/// Locals assigned outside the body, by statements or calls.
fn assigned_outside(body: &Body<'_>, region: &HashSet<BasicBlock>) -> HashSet<Local> {
    let mut assigned = HashSet::new();
    for (block, data) in body.basic_blocks.iter_enumerated() {
        if region.contains(&block) {
            continue;
        }
        for statement in &data.statements {
            if let StatementKind::Assign(assign) = &statement.kind {
                assigned.insert(assign.0.local);
            }
        }
        if let TerminatorKind::Call { destination, .. } = &data.terminator().kind {
            assigned.insert(destination.local);
        }
    }
    assigned
}

/// Where a MIR value lives in the kernel.
#[derive(Clone, Debug, PartialEq)]
enum Slot {
    Scalar(KLocal),
    /// A struct's or tuple's fields, in declaration order.
    Fields(Vec<Slot>),
    /// A value with no data, such as `()`.
    Unit,
    /// A reference: the slot it points to.
    Ref(Box<Slot>),
    /// A reference not assigned yet.
    Unbound,
    /// The loop's `Option<_>` from `next`, known to be `Some`.
    Some(Box<Slot>),
    /// A value only the iterator may use, such as the slice.
    Opaque,
    /// A read-only slice in GPU memory: its address and length.
    Slice { pointer: KLocal, len: KLocal, element: KTy },
    /// A number in a read-only slice, at an address.
    Memory { address: KLocal, ty: KTy },
    /// An array element chosen at run time: the option `index` names.
    /// Reading it selects among all options; writing it writes the one
    /// chosen. `index` holds the index as it was when the place was
    /// formed, and is below the number of options: MIR checks it first.
    Choice { index: KLocal, options: Vec<Slot> },
    /// An enum: its discriminant's value, and each variant's fields
    /// (`Fields`, or `Unit` for none), whatever variant is active. Only the
    /// active variant's fields mean anything. As a value being stored, a
    /// `Unit` variant is one it doesn't write.
    Enum { tag: KLocal, variants: Vec<Slot> },
    /// The bytes of other data seen as a value of another type, through a
    /// pointer cast whose numbers don't line up (`read_unaligned`,
    /// `bytemuck`): the data's numbers with their offsets, and where this
    /// value starts among them. Reading it decodes the bytes as the type
    /// read, as `transmute` would; writing through it is refused.
    Bytes { numbers: Vec<(u64, KLocal)>, offset: u64 },
}

impl Slot {
    fn scalars(&self, out: &mut Vec<KLocal>) {
        match self {
            Slot::Scalar(local) => out.push(*local),
            Slot::Fields(fields) => fields.iter().for_each(|field| field.scalars(out)),
            Slot::Enum { tag, variants } => {
                out.push(*tag);
                variants.iter().for_each(|variant| variant.scalars(out));
            }
            Slot::Bytes { numbers, .. } => out.extend(numbers.iter().map(|&(_, local)| local)),
            Slot::Choice { index, options } => {
                out.push(*index);
                options.iter().for_each(|option| option.scalars(out));
            }
            Slot::Ref(inner) | Slot::Some(inner) => inner.scalars(out),
            Slot::Slice { pointer, len, .. } => out.extend([*pointer, *len]),
            Slot::Memory { address, .. } => out.push(*address),
            Slot::Unit | Slot::Unbound | Slot::Opaque => {}
        }
    }
}

/// One projection of a place, with any run-time index already read.
enum Step {
    Deref,
    Field(usize),
    Downcast(u32),
    Index(KLocal),
    Constant { offset: u64, from_end: bool },
    Subslice { from: u64, to: u64, from_end: bool },
}

/// The body-specific rules of the loop body, the outermost frame.
struct LoopRules<'a, 'tcx> {
    shape: &'a Loop<'a, 'tcx>,
    /// The loop body; the other blocks translated are the setup.
    body: HashSet<BasicBlock>,
    /// Locals assigned in blocks the kernel doesn't run.
    assigned_outside: HashSet<Local>,
    /// What the iterator yields this iteration.
    item: Slot,
}

/// One body being translated: the loop body, or a function it calls.
struct Frame<'a, 'tcx> {
    body: &'a Body<'tcx>,
    slots: HashMap<Local, Slot>,
    blocks: HashMap<BasicBlock, BlockId>,
    /// Where `return`, or the end of an iteration, continues.
    exit: BlockId,
    loop_rules: Option<LoopRules<'a, 'tcx>>,
}

struct Translator<'tcx> {
    tcx: TyCtxt<'tcx>,
    b: Builder,
    /// The element's current value.
    element: KLocal,
    stores: bool,
    /// The functions being inlined, innermost last.
    stack: Vec<DefId>,
    /// Every local written, for `each`.
    written: HashSet<KLocal>,
    /// The kernel parameters holding the host's answers to feature tests.
    features: Vec<(Feature, KLocal)>,
    /// The feature tests the code makes, in order.
    wanted: Vec<Feature>,
}

/// How deep inlining may go.
const MAX_DEPTH: usize = 24;

fn is_ref(ty: Ty<'_>) -> bool {
    matches!(ty.kind(), ty::Ref(..) | ty::RawPtr(..))
}

/// `std_detect::detect::cache::test`, which every runtime CPU feature
/// check reaches.
pub(crate) fn is_feature_test(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    tcx.crate_name(def_id.krate).as_str() == "std_detect"
        && tcx.item_name(def_id).as_str() == "test"
        && tcx.opt_parent(def_id).is_some_and(|parent| tcx.opt_item_name(parent).is_some_and(|name| name.as_str() == "cache"))
}

fn points_to_array(ty: Ty<'_>) -> bool {
    ty.builtin_deref(true).is_some_and(|pointee| matches!(pointee.kind(), ty::Array(..)))
}

fn points_to_slice(ty: Ty<'_>) -> bool {
    ty.builtin_deref(true).is_some_and(|pointee| matches!(pointee.kind(), ty::Slice(_)))
}

impl<'tcx> Translator<'tcx> {
    fn typing_env(&self) -> TypingEnv<'tcx> {
        TypingEnv::fully_monomorphized()
    }

    /// Translates `region` (in order), entering at `start` from the current
    /// block. Leaves the builder in the last block translated.
    fn run_frame(&mut self, frame: &mut Frame<'_, 'tcx>, region: &[BasicBlock], start: BasicBlock) -> Result<()> {
        for &block in region {
            let id = self.b.block();
            frame.blocks.insert(block, id);
        }
        self.b.goto(frame.blocks[&start]);
        let doomed = doomed(frame, region);
        for &block in region {
            self.b.switch_to(frame.blocks[&block]);
            if doomed.contains(&block) {
                // Only a panic (or a call that never returns) can follow:
                // what the block computes only feeds its message, which
                // the CPU makes when it runs the code again.
                self.b.fail();
            } else {
                self.block(frame, block)?;
            }
        }
        Ok(())
    }

    /// Where a jump from `from` to `block` goes. In the loop's frame, the
    /// header starts the body when reached from the setup, and ends the
    /// iteration when reached from the body.
    fn target(&self, frame: &Frame<'_, 'tcx>, from: BasicBlock, block: BasicBlock) -> Result<BlockId> {
        if let Some(rules) = &frame.loop_rules {
            if let Iteration::For { start, is_header, .. } = &rules.shape.iteration {
                if is_header(block) {
                    if rules.body.contains(&from) {
                        return Ok(frame.exit);
                    }
                    return frame.blocks.get(start).copied().ok_or_else(|| "has an unsupported loop shape".into());
                }
            }
        }
        frame.blocks.get(&block).copied().ok_or_else(|| "has an unsupported loop shape".into())
    }

    fn set(&mut self, local: KLocal, rvalue: KRvalue) {
        if local == self.element {
            self.stores = true;
        }
        self.written.insert(local);
        self.b.set(local, rvalue);
    }

    fn block(&mut self, frame: &mut Frame<'_, 'tcx>, block: BasicBlock) -> Result<()> {
        let data = &frame.body.basic_blocks[block];
        let plumbing = frame.loop_rules.as_ref().map(|rules| rules.shape.plumbing);
        for statement in &data.statements {
            match &statement.kind {
                // The iterator's own values: only `next` uses them.
                StatementKind::Assign(assign) if plumbing.is_some_and(|p| p.contains(&assign.0.local)) => {}
                StatementKind::Assign(assign) => self.assign(frame, &assign.0, &assign.1)?,
                StatementKind::Intrinsic(intrinsic) => match &**intrinsic {
                    NonDivergingIntrinsic::Assume(_) => {}
                    NonDivergingIntrinsic::CopyNonOverlapping(_) => {
                        return Err("copies memory (`copy_nonoverlapping`)".into());
                    }
                },
                StatementKind::SetDiscriminant { place, variant_index } => {
                    let ty = place.ty(frame.body, self.tcx).ty;
                    let value = self.discriminant(ty, *variant_index)?;
                    match self.place(frame, place)? {
                        Slot::Enum { tag, .. } => self.set(tag, KRvalue::Use(value.into())),
                        _ => return Err("sets the discriminant of something other than an enum".into()),
                    }
                }
                // Storage markers, borrow-checker annotations and no-ops.
                StatementKind::FakeRead(_)
                | StatementKind::StorageLive(_)
                | StatementKind::StorageDead(_)
                | StatementKind::PlaceMention(_)
                | StatementKind::AscribeUserType(..)
                | StatementKind::Coverage(_)
                | StatementKind::ConstEvalCounter
                | StatementKind::Nop
                | StatementKind::BackwardIncompatibleDropHint { .. } => {}
            }
        }
        // In `for_each`'s block: its function, and what the iterator yields.
        let each = frame.loop_rules.as_ref().and_then(|rules| match &rules.shape.iteration {
            Iteration::ForEach { block: each_block, function } if *each_block == block => {
                Some((*function, rules.item.clone()))
            }
            _ => None,
        });
        match &data.terminator().kind {
            TerminatorKind::Goto { target, .. }
            | TerminatorKind::FalseEdge { real_target: target, .. }
            | TerminatorKind::FalseUnwind { real_target: target, .. } => {
                let target = self.target(frame, block, *target)?;
                self.b.goto(target);
            }
            TerminatorKind::SwitchInt { discr, targets } => {
                let value = self.scalar_operand(frame, discr)?;
                let ty = self.b.ty(value);
                for (case, target) in targets.iter() {
                    let target = self.target(frame, block, target)?;
                    let next = self.b.block();
                    let case = Const { ty, bits: case };
                    let matches = if ty == KTy::Bool && case.bits == 0 {
                        self.b.unary(KUnOp::Not, value)
                    } else if ty == KTy::Bool {
                        self.b.copy(value)
                    } else {
                        self.b.binary(KBinOp::Eq, value, case)
                    };
                    self.b.branch(matches, target, next);
                    self.b.switch_to(next);
                }
                let otherwise = self.target(frame, block, targets.otherwise())?;
                self.b.goto(otherwise);
            }
            // A check the CPU makes: proven ones cost nothing; the others run,
            // and a failure stops the thread and sends the call back to the
            // CPU, which panics there.
            TerminatorKind::Assert { cond, expected, target, .. } => {
                let target = self.target(frame, block, *target)?;
                if assert_holds(frame.body, block) {
                    self.b.goto(target);
                } else {
                    let condition = self.scalar_operand(frame, cond)?;
                    let failed = self.b.block();
                    if *expected {
                        self.b.branch(condition, target, failed);
                    } else {
                        self.b.branch(condition, failed, target);
                    }
                    self.b.switch_to(failed);
                    self.b.fail();
                }
            }
            TerminatorKind::Return => self.b.goto(frame.exit),
            // Never reached in a correct program; if it were, the CPU decides.
            TerminatorKind::Unreachable => self.b.fail(),
            // `for_each`'s call is the iteration: call its function with
            // what the iterator yields, and end.
            TerminatorKind::Call { .. } if each.is_some() => {
                let (function, item) = each.expect("checked by the guard");
                self.each(frame, function, item)?;
                self.b.goto(frame.exit);
            }
            TerminatorKind::Call { func, args, destination, target: Some(target), .. } => {
                // Calls that build the loop's iterator are skipped.
                if !frame.loop_rules.as_ref().is_some_and(|rules| rules.shape.chain.contains(&block)) {
                    self.call(frame, func, args, destination)?;
                }
                let target = self.target(frame, block, *target)?;
                self.b.goto(target);
            }
            // A call that never returns, such as a panic: the CPU makes it.
            TerminatorKind::Call { target: None, .. } => self.b.fail(),
            // Dropping a type without drop glue does nothing; generic code
            // is full of such drops once instantiated with numbers.
            TerminatorKind::Drop { place, target, .. } => {
                if place.ty(frame.body, self.tcx).ty.needs_drop(self.tcx, self.typing_env()) {
                    return Err("runs a destructor".into());
                }
                let target = self.target(frame, block, *target)?;
                self.b.goto(target);
            }
            // Only cleanup blocks, which the region never includes, unwind.
            TerminatorKind::UnwindResume | TerminatorKind::UnwindTerminate(_) => {
                return Err("unwinds".into());
            }
            TerminatorKind::TailCall { .. } => return Err("makes a tail call (`become`)".into()),
            TerminatorKind::Yield { .. } | TerminatorKind::CoroutineDrop => return Err("is a coroutine".into()),
            TerminatorKind::InlineAsm { .. } => return Err("uses inline assembly".into()),
        }
        Ok(())
    }

    fn call(
        &mut self,
        frame: &mut Frame<'_, 'tcx>,
        func: &Operand<'tcx>,
        args: &[rustc_span::Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
    ) -> Result<()> {
        let tcx = self.tcx;
        // The callee is in the operand's type: a constant, or a function
        // item held in a local.
        let func_ty = func.ty(frame.body, tcx);
        let ty::FnDef(def_id, generic_args) = *func_ty.kind() else {
            return Err("calls a function pointer".into());
        };
        let generic_args = generic_args.no_bound_vars().ok_or("calls a function with bound generic arguments")?;
        // A "rust-call" call passes its arguments as one tuple, as calls to
        // closures through the `Fn` traits do.
        let rust_call = func_ty.fn_sig(tcx).abi() == ExternAbi::RustCall;
        let path = || tcx.def_path_str_with_args(def_id, generic_args);
        // A CPU feature test: the host's answer, a kernel parameter. (While
        // the parameters are being found, an unknown value, so that both
        // paths are translated.)
        if is_feature_test(tcx, def_id) {
            let [bit] = args else { return Err("tests a CPU feature with unexpected arguments".into()) };
            let KOperand::Const(bit) = self.scalar_operand(frame, &bit.node)? else {
                return Err("tests a CPU feature chosen at run time".into());
            };
            let feature = Feature { test: def_id, bit: bit.bits as u32 };
            if !self.wanted.contains(&feature) {
                self.wanted.push(feature);
            }
            let answer = match self.features.iter().find(|(known, _)| *known == feature) {
                Some(&(_, param)) => KRvalue::Binary(KBinOp::Ne, param.into(), Const::int(KTy::U32, 0).into()),
                None => KRvalue::Use(self.b.local(KTy::Bool).into()),
            };
            let dest = self.place(frame, destination)?;
            return self.write_number(dest, answer);
        }
        // An x86 instruction, declared as an LLVM intrinsic.
        if tcx.is_foreign_item(def_id) {
            let symbol = tcx.codegen_fn_attrs(def_id).symbol_name;
            if let Some(symbol) = symbol.filter(|symbol| symbol.as_str().starts_with("llvm.x86.")) {
                return self.x86(frame, symbol.as_str(), args, destination);
            }
        }
        if let Some(method) = method(tcx, def_id) {
            return self.operation(frame, IntrinsicOp::Method(method), args, destination);
        }
        let instance = Instance::try_resolve(tcx, self.typing_env(), def_id, generic_args)
            .ok()
            .flatten()
            .ok_or_else(|| format!("calls `{}`, which can't be resolved to one function", path()))?;
        match instance.def {
            InstanceKind::Intrinsic(intrinsic_id) => {
                let name = tcx.item_name(intrinsic_id);
                let op = intrinsic(name.as_str())
                    .ok_or_else(|| format!("uses the intrinsic `{name}`{}", crate::coverage::reason(name.as_str())))?;
                if op == IntrinsicOp::Simd {
                    return self.simd(frame, name.as_str(), args, destination);
                }
                self.operation(frame, op, args, destination)
            }
            InstanceKind::Item(item) if tcx.is_mir_available(item) => {
                self.inline(frame, instance, args, rust_call, destination, &path())
            }
            InstanceKind::Item(_) => Err(format!("calls `{}`, whose MIR isn't available", path())),
            // Calling an `Fn` closure through `FnOnce`, or a function item
            // through the `Fn` traits: rustc builds these bodies itself.
            InstanceKind::Shim(ShimKind::ClosureOnce { .. } | ShimKind::FnPtr(..)) => {
                self.inline(frame, instance, args, rust_call, destination, &path())
            }
            _ => Err(format!("calls `{}`, which has no GPU summary", path())),
        }
    }

    /// A library method or intrinsic, as a kernel operation.
    fn operation(
        &mut self,
        frame: &mut Frame<'_, 'tcx>,
        op: IntrinsicOp,
        args: &[rustc_span::Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
    ) -> Result<()> {
        if op == IntrinsicOp::Nothing {
            return Ok(());
        }
        if op == IntrinsicOp::ByteSwap {
            let [arg] = args else {
                return Err("calls a library function with unexpected arguments".into());
            };
            let value = self.scalar_operand(frame, &arg.node)?;
            let swapped = self.byte_swap(value);
            let dest = self.place(frame, destination)?;
            return self.write_number(dest, KRvalue::Use(swapped.into()));
        }
        if op == IntrinsicOp::Transmute {
            let [arg] = args else {
                return Err("calls a library function with unexpected arguments".into());
            };
            let to = destination.ty(frame.body, self.tcx).ty;
            let value = self.transmute(frame, &arg.node, to)?;
            return self.store(frame, destination, value);
        }
        let args = args.iter().map(|arg| self.scalar_operand(frame, &arg.node)).collect::<Result<Vec<_>>>()?;
        let rvalue = match (op, args.as_slice()) {
            (IntrinsicOp::Method(Method::Binary(op)), [a, b]) => KRvalue::Binary(op, *a, *b),
            (IntrinsicOp::Method(Method::Neg), [a]) => KRvalue::Unary(KUnOp::Neg, *a),
            (IntrinsicOp::Method(Method::Intrinsic(intrinsic)), _) => KRvalue::Call(intrinsic, args.clone()),
            _ => return Err("calls a library function with unexpected arguments".into()),
        };
        let dest = self.place(frame, destination)?;
        self.write_number(dest, rvalue)
    }

    /// One iteration of `for_each`: calls `function` with `item` through
    /// `FnMut::call_mut`, as `for_each` does. Each thread has its own copy
    /// of the closure, so one that changes what it captured (a counter, a
    /// running total) would carry a value from one element to the next: it
    /// is refused when the call writes any local the captures reach. The
    /// closure's kind can't decide this: passed to `for_each`, a closure
    /// takes the kind of its `FnMut` bound, whatever its body does.
    fn each(&mut self, frame: &mut Frame<'_, 'tcx>, function: &Operand<'tcx>, item: Slot) -> Result<()> {
        let tcx = self.tcx;
        let function_ty = function.ty(frame.body, tcx);
        // The arguments, as the tuple `FnMut::call_mut` takes.
        let arguments = match function_ty.kind() {
            // A closure's signature takes its arguments as one tuple.
            ty::Closure(_, args) => tcx.instantiate_bound_regions_with_erased(args.as_closure().sig()).inputs()[0],
            ty::FnDef(..) => Ty::new_tup(tcx, tcx.instantiate_bound_regions_with_erased(function_ty.fn_sig(tcx)).inputs()),
            _ => return Err("passes `for_each` something other than a closure or function".into()),
        };
        let fn_mut_trait = tcx.lang_items().fn_mut_trait().ok_or("has no `FnMut` trait")?;
        let call = tcx
            .associated_item_def_ids(fn_mut_trait)
            .iter()
            .copied()
            .find(|&item| tcx.item_name(item).as_str() == "call_mut")
            .ok_or("has no `FnMut::call_mut`")?;
        let generic_args = tcx.mk_args(&[function_ty.into(), arguments.into()]);
        let path = tcx.def_path_str_with_args(call, generic_args);
        let instance = Instance::try_resolve(tcx, self.typing_env(), call, generic_args)
            .ok()
            .flatten()
            .ok_or_else(|| format!("calls `{path}`, which can't be resolved to one function"))?;
        if !matches!(
            instance.def,
            InstanceKind::Item(_) | InstanceKind::Shim(ShimKind::ClosureOnce { .. } | ShimKind::FnPtr(..))
        ) || matches!(instance.def, InstanceKind::Item(item) if !tcx.is_mir_available(item))
        {
            return Err(format!("calls `{path}`, whose MIR isn't available"));
        }
        let receiver = self.operand_slot(frame, function)?;
        let mut captured = Vec::new();
        receiver.scalars(&mut captured);
        let outer = std::mem::take(&mut self.written);
        let values = vec![Slot::Ref(Box::new(receiver)), Slot::Fields(vec![item])];
        let called = self.inline_values(instance, values, true, &path);
        let written = std::mem::replace(&mut self.written, outer);
        self.written.extend(&written);
        called?;
        if captured.iter().any(|local| written.contains(local)) {
            return Err(
                "passes `for_each` a closure that changes what it captures, which carries a value from one element to the next"
                    .into(),
            );
        }
        Ok(())
    }

    /// Translates a call to `instance` in place. With `rust_call`, the last
    /// argument is a tuple of the callee's remaining arguments.
    fn inline(
        &mut self,
        frame: &mut Frame<'_, 'tcx>,
        instance: Instance<'tcx>,
        args: &[rustc_span::Spanned<Operand<'tcx>>],
        rust_call: bool,
        destination: &Place<'tcx>,
        path: &str,
    ) -> Result<()> {
        // Arguments are read in the caller's frame, before the call.
        let values = args.iter().map(|arg| self.operand_slot(frame, &arg.node)).collect::<Result<Vec<_>>>()?;
        let result = self.inline_values(instance, values, rust_call, path)?;
        self.store(frame, destination, result)
    }

    /// `inline`, with the arguments' slots; returns the result's slot.
    fn inline_values(&mut self, instance: Instance<'tcx>, mut values: Vec<Slot>, rust_call: bool, path: &str) -> Result<Slot> {
        let tcx = self.tcx;
        let def_id = instance.def_id();
        if self.stack.contains(&def_id) {
            return Err(format!("calls `{path}` recursively"));
        }
        if self.stack.len() >= MAX_DEPTH {
            return Err(format!("calls `{path}` through more than {MAX_DEPTH} nested functions"));
        }
        let body = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            self.typing_env(),
            ty::EarlyBinder::bind(tcx, tcx.instance_mir(instance.def).clone()),
        );
        let within = |reason: String| format!("calls `{path}`, which {reason}");
        let region = function_region(&body);
        // A closure's body takes the tuple's fields as separate parameters;
        // a body with `spread_arg` (a shim) takes the tuple itself, as MIR
        // passes it. rustc's inliner does the same.
        if rust_call && body.spread_arg.is_none() {
            match values.pop() {
                Some(Slot::Fields(fields)) => values.extend(fields),
                Some(Slot::Unit) => {}
                _ => return Err(within("takes arguments the kernel can't spread".into())),
            }
        }
        if values.len() != body.arg_count {
            return Err(within("is called with a different number of arguments".into()));
        }
        let mut slots = HashMap::new();
        for (index, value) in values.into_iter().enumerate() {
            let param = Local::from_usize(index + 1);
            let slot = if is_ref(body.local_decls[param].ty) {
                value
            } else {
                let fresh = self.fresh(body.local_decls[param].ty).map_err(within)?;
                self.copy(&fresh, &value)?
            };
            slots.insert(param, slot);
        }
        let mut callee = Frame {
            body: &body,
            slots,
            blocks: HashMap::new(),
            exit: self.b.block(),
            loop_rules: None,
        };
        self.stack.push(def_id);
        self.run_frame(&mut callee, &ordered(&body, &region), START_BLOCK).map_err(within)?;
        self.stack.pop();
        self.b.switch_to(callee.exit);
        self.local_slot(&mut callee, RETURN_PLACE).map_err(within)
    }

    /// Like `fresh`, with every reference inside bound to a fresh value of
    /// its own: a parameter of a function translated on its own.
    fn fresh_deep(&mut self, ty: Ty<'tcx>, depth: usize) -> Result<Slot> {
        if depth > 16 {
            return Err(format!("nests `{ty}` too deeply"));
        }
        if let Some(pointee) = ty.builtin_deref(true) {
            return Ok(Slot::Ref(Box::new(self.fresh_deep(pointee, depth + 1)?)));
        }
        if held_ty(ty).is_some() {
            return self.fresh(ty);
        }
        if let Some((tag, variants)) = self.enum_shape(ty)? {
            let tag = self.b.local(tag);
            let variants = variants
                .into_iter()
                .map(|fields| self.fresh_fields_deep(fields, depth))
                .collect::<Result<_>>()?;
            return Ok(Slot::Enum { tag, variants });
        }
        let fields = self.fields(ty)?;
        self.fresh_fields_deep(fields, depth)
    }

    fn fresh_fields_deep(&mut self, fields: Vec<Ty<'tcx>>, depth: usize) -> Result<Slot> {
        if fields.is_empty() {
            return Ok(Slot::Unit);
        }
        Ok(Slot::Fields(fields.into_iter().map(|field| self.fresh_deep(field, depth + 1)).collect::<Result<_>>()?))
    }

    /// The kernel type of an enum's discriminant and each variant's field
    /// types, if `ty` is an enum.
    fn enum_shape(&self, ty: Ty<'tcx>) -> Result<Option<(KTy, Vec<Vec<Ty<'tcx>>>)>> {
        let ty::Adt(adt, args) = ty.kind() else {
            return Ok(None);
        };
        if !adt.is_enum() {
            return Ok(None);
        }
        let tcx = self.tcx;
        let discriminant = ty.discriminant_ty(tcx);
        let tag = kernel_ty(discriminant)
            .ok_or_else(|| format!("uses the enum `{ty}`, whose discriminant is a `{discriminant}`"))?;
        let variants = adt
            .variants()
            .iter()
            .map(|variant| {
                variant
                    .fields
                    .iter()
                    .map(|field| tcx.normalize_erasing_regions(self.typing_env(), field.ty(tcx, args)))
                    .collect()
            })
            .collect();
        Ok(Some((tag, variants)))
    }

    /// The value of the discriminant of `ty`'s variant `variant`.
    fn discriminant(&self, ty: Ty<'tcx>, variant: VariantIdx) -> Result<Const> {
        let tag = kernel_ty(ty.discriminant_ty(self.tcx)).ok_or_else(|| format!("uses the enum `{ty}`"))?;
        let value = ty
            .discriminant_for_variant(self.tcx, variant)
            .ok_or_else(|| format!("uses `{ty}`, which has no variant {}", variant.as_usize()))?;
        Ok(Const::int(tag, value.val as i128))
    }

    /// The slot for a fresh value of type `ty`.
    fn fresh(&mut self, ty: Ty<'tcx>) -> Result<Slot> {
        if let Some(kernel) = held_ty(ty) {
            return Ok(Slot::Scalar(self.b.local(kernel)));
        }
        if is_ref(ty) {
            return Ok(Slot::Unbound);
        }
        if let Some((tag, variants)) = self.enum_shape(ty)? {
            let tag = self.b.local(tag);
            let variants = variants.into_iter().map(|fields| self.fresh_fields(fields)).collect::<Result<_>>()?;
            return Ok(Slot::Enum { tag, variants });
        }
        let fields = self.fields(ty)?;
        self.fresh_fields(fields)
    }

    fn fresh_fields(&mut self, fields: Vec<Ty<'tcx>>) -> Result<Slot> {
        if fields.is_empty() {
            return Ok(Slot::Unit);
        }
        Ok(Slot::Fields(fields.into_iter().map(|field| self.fresh(field)).collect::<Result<_>>()?))
    }

    fn fields(&self, ty: Ty<'tcx>) -> Result<Vec<Ty<'tcx>>> {
        fields_of(self.tcx, ty)
    }

    /// The slot of a local, created on first use.
    fn local_slot(&mut self, frame: &mut Frame<'_, 'tcx>, local: Local) -> Result<Slot> {
        if let Some(slot) = frame.slots.get(&local) {
            return Ok(slot.clone());
        }
        if let Some(rules) = &frame.loop_rules {
            if (1..=frame.body.arg_count).contains(&local.as_usize()) {
                return Err("uses a parameter a kernel can't take".into());
            }
            if rules.assigned_outside.contains(&local) {
                return Err("uses a value computed outside the loop body".into());
            }
        }
        let slot = self.fresh(frame.body.local_decls[local].ty)?;
        frame.slots.insert(local, slot.clone());
        Ok(slot)
    }

    /// The slot a place names.
    fn place(&mut self, frame: &mut Frame<'_, 'tcx>, place: &Place<'tcx>) -> Result<Slot> {
        let mut slot = self.local_slot(frame, place.local)?;
        for (depth, element) in place.projection.iter().enumerate() {
            if let Slot::Bytes { numbers, offset } = &slot {
                let ty = Place::ty_from(place.local, &place.projection[..depth], frame.body, self.tcx).ty;
                let layout = self.layout(ty)?;
                let at = match element {
                    ProjectionElem::Field(field, _) => layout.fields.offset(field.as_usize()).bytes(),
                    ProjectionElem::ConstantIndex { offset: index, from_end: false, .. } => layout.fields.offset(index as usize).bytes(),
                    _ => return Err(format!("uses `{ty}` seen through a pointer to other data")),
                };
                slot = Slot::Bytes { numbers: numbers.clone(), offset: offset + at };
                continue;
            }
            if slot == Slot::Opaque {
                return Err("uses a slice parameter other than through the loop's iterator".into());
            }
            let step = match element {
                ProjectionElem::Deref => Step::Deref,
                ProjectionElem::Field(field, _) => Step::Field(field.as_usize()),
                ProjectionElem::Downcast(_, variant) => Step::Downcast(variant.as_u32()),
                // MIR checks the index before it gets here, and the check
                // runs on the GPU too. The index is kept as it is now: a
                // reference to the element must not follow later changes.
                ProjectionElem::Index(index) => {
                    let index = self.scalar_operand(frame, &Operand::Copy(Place::from(index)))?;
                    Step::Index(self.b.copy(index))
                }
                ProjectionElem::ConstantIndex { offset, from_end, .. } => Step::Constant { offset, from_end },
                ProjectionElem::Subslice { from, to, from_end } => Step::Subslice { from, to, from_end },
                element => return Err(projection_refusal(&element)),
            };
            slot = self.step(slot, &step)?;
        }
        Ok(slot)
    }

    /// The value a place holds, as read: bytes seen through a pointer cast
    /// are decoded as the place's type.
    fn read_place(&mut self, frame: &mut Frame<'_, 'tcx>, place: &Place<'tcx>) -> Result<Slot> {
        let slot = self.place(frame, place)?;
        if !matches!(slot, Slot::Bytes { .. }) {
            return Ok(slot);
        }
        let ty = place.ty(frame.body, self.tcx).ty;
        let value = self.fresh(ty)?;
        self.decode(&value, ty, 0, &slot, ty)?;
        Ok(value)
    }

    /// The part of `slot` that `step` names.
    fn step(&mut self, slot: Slot, step: &Step) -> Result<Slot> {
        Ok(match (step, slot) {
            // A step into an element chosen at run time is that step into
            // whichever element is chosen.
            (_, Slot::Choice { index, options }) => {
                let options: Vec<Slot> =
                    options.into_iter().map(|option| self.step(option, step)).collect::<Result<_>>()?;
                // Reading one would read every slice at the index, and only
                // the chosen one's bounds were checked.
                if options.iter().any(|option| matches!(option, Slot::Slice { .. } | Slot::Memory { .. })) {
                    return Err("chooses between slices at run time".into());
                }
                Slot::Choice { index, options }
            }
            (Step::Deref, Slot::Ref(inner)) => *inner,
            (Step::Deref, _) => return Err("reads through a reference it can't follow".into()),
            (Step::Field(field), Slot::Fields(fields)) => {
                fields.get(*field).cloned().ok_or("uses a field that doesn't exist")?
            }
            (Step::Downcast(1), Slot::Some(inner)) => Slot::Fields(vec![*inner]),
            (Step::Downcast(variant), Slot::Enum { variants, .. }) => {
                variants.get(*variant as usize).cloned().ok_or("uses a variant that doesn't exist")?
            }
            (Step::Downcast(_), _) => return Err("uses an enum".into()),
            // An array, held as its elements.
            (Step::Index(index), Slot::Fields(elements)) => Slot::Choice { index: *index, options: elements },
            (Step::Index(_), Slot::Unit) => return Err("indexes an empty array".into()),
            (&Step::Constant { offset, from_end }, Slot::Fields(elements)) => {
                let at = if from_end { elements.len().checked_sub(offset as usize) } else { Some(offset as usize) };
                at.and_then(|at| elements.get(at).cloned()).ok_or("indexes past the end of an array")?
            }
            (&Step::Subslice { from, to, from_end }, Slot::Fields(elements)) => {
                // An array's subslice ends at `to`; a slice's `to` elements
                // before its end.
                let end = if from_end { elements.len().checked_sub(to as usize) } else { Some(to as usize) };
                let part = end
                    .and_then(|end| elements.get(from as usize..end))
                    .ok_or("takes a subslice past the end of an array")?;
                if part.is_empty() { Slot::Unit } else { Slot::Fields(part.to_vec()) }
            }
            (&Step::Subslice { from, to, from_end: true }, Slot::Unit) if from == 0 && to == 0 => Slot::Unit,
            // A read-only slice in memory.
            (Step::Index(index), Slot::Slice { pointer, element, .. }) => {
                Slot::Memory { address: self.b.offset(pointer, *index, element), ty: element }
            }
            (&Step::Constant { offset, from_end }, Slot::Slice { pointer, len, element }) => {
                let offset = Const::int(KTy::U64, offset.into());
                let index: KOperand = if from_end { self.b.binary(KBinOp::Sub, len, offset).into() } else { offset.into() };
                Slot::Memory { address: self.b.offset(pointer, index, element), ty: element }
            }
            (&Step::Subslice { from, to, from_end: true }, Slot::Slice { pointer, len, element }) => {
                let start = self.b.offset(pointer, Const::int(KTy::U64, from.into()), element);
                let trimmed = self.b.binary(KBinOp::Sub, len, Const::int(KTy::U64, (from + to).into()));
                Slot::Slice { pointer: start, len: trimmed, element }
            }
            (Step::Field(_), _) => return Err("uses a field of a value the kernel can't hold".into()),
            (Step::Index(_) | Step::Constant { .. }, _) => return Err("indexes something other than an array or slice".into()),
            (Step::Subslice { .. }, _) => return Err("takes a subslice of something other than an array or slice".into()),
        })
    }

    /// Binds a reference local to the slot it points to. A reference
    /// assigned twice must point to the same slot: choosing at run time
    /// between targets would need memory.
    fn bind(&mut self, frame: &mut Frame<'_, 'tcx>, local: Local, value: Slot) -> Result<()> {
        if !matches!(value, Slot::Ref(_)) {
            return Err("uses a reference it can't follow".into());
        }
        if let Some(rules) = &frame.loop_rules {
            if rules.assigned_outside.contains(&local) {
                return Err("uses a value computed outside the loop body".into());
            }
        }
        match frame.slots.get(&local) {
            None | Some(Slot::Unbound) => {
                frame.slots.insert(local, value);
                Ok(())
            }
            Some(existing) if *existing == value => Ok(()),
            Some(_) => Err("chooses between references at run time".into()),
        }
    }

    /// Writes `value` to `place`: copies a value, or binds a reference.
    fn store(&mut self, frame: &mut Frame<'_, 'tcx>, place: &Place<'tcx>, value: Slot) -> Result<()> {
        if place.projection.is_empty() && is_ref(frame.body.local_decls[place.local].ty) {
            return self.bind(frame, place.local, value);
        }
        let dest = self.place(frame, place)?;
        let written = self.copy(&dest, &value)?;
        if written == dest {
            return Ok(());
        }
        // The value binds references inside the local, such as a closure's
        // captures. The local's slot records them; a field can't be updated
        // inside a larger slot.
        if !place.projection.is_empty() {
            return Err("stores a reference inside part of a value".into());
        }
        if let Some(rules) = &frame.loop_rules {
            if rules.assigned_outside.contains(&place.local) {
                return Err("uses a value computed outside the loop body".into());
            }
        }
        frame.slots.insert(place.local, written);
        Ok(())
    }

    /// Copies `src` into `dest`, field by field. If they share kernel
    /// locals (`a = (a.1, a.0)`), through temporaries. Returns the slot
    /// `dest` names afterwards, with the references inside it bound.
    fn copy(&mut self, dest: &Slot, src: &Slot) -> Result<Slot> {
        let (mut written, mut read) = (Vec::new(), Vec::new());
        dest.scalars(&mut written);
        src.scalars(&mut read);
        if dest != src && written.iter().any(|local| read.contains(local)) && !matches!(dest, Slot::Scalar(_)) {
            let temporary = self.snapshot(src)?;
            return self.copy_fields(dest, &temporary);
        }
        self.copy_fields(dest, src)
    }

    /// Numbers are copied. A reference inside the value is bound, as
    /// reference locals are (`bind`): one not bound yet takes the source's
    /// target, and one already bound must have the same target.
    fn copy_fields(&mut self, dest: &Slot, src: &Slot) -> Result<Slot> {
        match (dest, src) {
            (_, Slot::Choice { .. }) => {
                let src = self.resolve(src.clone())?;
                return self.copy_fields(dest, &src);
            }
            // Writes the option chosen; the others keep their values.
            (Slot::Choice { index, options }, _) => {
                for (at, option) in options.iter().enumerate() {
                    let chosen = self.chosen(*index, at);
                    self.copy_where(chosen, option, src)?;
                }
            }
            (Slot::Scalar(d), Slot::Scalar(s)) => {
                if d != s {
                    self.set(*d, KRvalue::Use((*s).into()));
                }
            }
            (Slot::Fields(d), Slot::Fields(s)) if d.len() == s.len() => {
                let fields = d.iter().zip(s).map(|(d, s)| self.copy_fields(d, s)).collect::<Result<_>>()?;
                return Ok(Slot::Fields(fields));
            }
            // The discriminant, and each variant the value writes.
            (Slot::Enum { tag: d, variants: dv }, Slot::Enum { tag: s, variants: sv }) if dv.len() == sv.len() => {
                if d != s {
                    self.set(*d, KRvalue::Use((*s).into()));
                }
                let variants = dv
                    .iter()
                    .zip(sv)
                    .map(|(d, s)| if *s == Slot::Unit { Ok(d.clone()) } else { self.copy_fields(d, s) })
                    .collect::<Result<_>>()?;
                return Ok(Slot::Enum { tag: *d, variants });
            }
            (Slot::Unit, Slot::Unit) | (Slot::Unbound, Slot::Unbound) => {}
            (Slot::Unbound, Slot::Ref(_)) => return Ok(src.clone()),
            (Slot::Ref(d), Slot::Ref(s)) if d == s => {}
            (Slot::Bytes { .. }, _) => return Err("writes through a pointer to data of another type".into()),
            (Slot::Ref(_), Slot::Ref(_)) => return Err("chooses between references at run time".into()),
            _ => return Err("copies between values of different shapes".into()),
        }
        Ok(dest.clone())
    }

    /// Copies `src` into `dest` where `condition` holds, and leaves `dest`
    /// as it is elsewhere. References can't be written conditionally.
    fn copy_where(&mut self, condition: KLocal, dest: &Slot, src: &Slot) -> Result<()> {
        match (dest, src) {
            (Slot::Scalar(d), Slot::Scalar(s)) => {
                if d != s {
                    self.set(*d, KRvalue::Select(condition.into(), (*s).into(), (*d).into()));
                }
            }
            (Slot::Fields(d), Slot::Fields(s)) if d.len() == s.len() => {
                for (d, s) in d.iter().zip(s) {
                    self.copy_where(condition, d, s)?;
                }
            }
            (Slot::Choice { index, options }, _) => {
                for (at, option) in options.iter().enumerate() {
                    let chosen = self.chosen(*index, at);
                    let both = self.b.binary(KBinOp::And, condition, chosen);
                    self.copy_where(both, option, src)?;
                }
            }
            (Slot::Enum { tag: d, variants: dv }, Slot::Enum { tag: s, variants: sv }) if dv.len() == sv.len() => {
                if d != s {
                    self.set(*d, KRvalue::Select(condition.into(), (*s).into(), (*d).into()));
                }
                for (d, s) in dv.iter().zip(sv) {
                    if *s != Slot::Unit {
                        self.copy_where(condition, d, s)?;
                    }
                }
            }
            (Slot::Unit, Slot::Unit) | (Slot::Unbound, Slot::Unbound) => {}
            (Slot::Ref(d), Slot::Ref(s)) if d == s => {}
            (Slot::Ref(_) | Slot::Unbound, Slot::Ref(_)) => return Err("chooses between references at run time".into()),
            _ => return Err("copies between values of different shapes".into()),
        }
        Ok(())
    }

    /// Whether `index` chooses option `at`.
    fn chosen(&mut self, index: KLocal, at: usize) -> KLocal {
        let ty = self.b.ty(index);
        self.b.binary(KBinOp::Eq, index, Const::int(ty, at as i128))
    }

    /// The value a slot holds, with an element chosen at run time read into
    /// new locals.
    fn resolve(&mut self, slot: Slot) -> Result<Slot> {
        let Slot::Choice { index, options } = slot else {
            return Ok(slot);
        };
        let options = options.into_iter().map(|option| self.resolve(option)).collect::<Result<Vec<_>>>()?;
        self.select_among(index, &options)
    }

    /// The option `index` chooses, number by number. A reference chosen at
    /// run time points to the chosen target.
    fn select_among(&mut self, index: KLocal, options: &[Slot]) -> Result<Slot> {
        let shapes = || "copies between values of different shapes".to_string();
        let Some(first) = options.first() else {
            return Err("indexes an empty array".into());
        };
        if options.iter().all(|option| option == first) {
            return Ok(first.clone());
        }
        Ok(match first {
            Slot::Scalar(_) => {
                let numbers = options
                    .iter()
                    .map(|option| match option {
                        Slot::Scalar(local) => Ok(*local),
                        _ => Err(shapes()),
                    })
                    .collect::<Result<Vec<_>>>()?;
                let value = self.b.copy(numbers[0]);
                for (at, number) in numbers.iter().enumerate().skip(1) {
                    let chosen = self.chosen(index, at);
                    self.b.set(value, KRvalue::Select(chosen.into(), (*number).into(), value.into()));
                }
                Slot::Scalar(value)
            }
            Slot::Fields(fields) => {
                let mut chosen = Vec::with_capacity(fields.len());
                for field in 0..fields.len() {
                    let column = options
                        .iter()
                        .map(|option| match option {
                            Slot::Fields(f) if f.len() == fields.len() => Ok(f[field].clone()),
                            _ => Err(shapes()),
                        })
                        .collect::<Result<Vec<_>>>()?;
                    chosen.push(self.select_among(index, &column)?);
                }
                Slot::Fields(chosen)
            }
            Slot::Enum { variants, .. } => {
                let mut tags = Vec::with_capacity(options.len());
                let mut columns = vec![Vec::with_capacity(options.len()); variants.len()];
                for option in options {
                    let Slot::Enum { tag, variants: v } = option else { return Err(shapes()) };
                    if v.len() != variants.len() {
                        return Err(shapes());
                    }
                    tags.push(Slot::Scalar(*tag));
                    for (column, variant) in columns.iter_mut().zip(v) {
                        column.push(variant.clone());
                    }
                }
                let Slot::Scalar(tag) = self.select_among(index, &tags)? else { return Err(shapes()) };
                let variants = columns.iter().map(|column| self.select_among(index, column)).collect::<Result<_>>()?;
                Slot::Enum { tag, variants }
            }
            Slot::Ref(_) => {
                let targets = options
                    .iter()
                    .map(|option| match option {
                        Slot::Ref(target) if !matches!(**target, Slot::Slice { .. } | Slot::Memory { .. }) => {
                            Ok((**target).clone())
                        }
                        Slot::Ref(_) => Err("chooses between slices at run time".to_string()),
                        _ => Err(shapes()),
                    })
                    .collect::<Result<Vec<_>>>()?;
                Slot::Ref(Box::new(Slot::Choice { index, options: targets }))
            }
            Slot::Unbound => return Err("chooses between references at run time".into()),
            Slot::Slice { .. } | Slot::Memory { .. } => return Err("chooses between slices at run time".into()),
            Slot::Unit | Slot::Some(_) | Slot::Opaque | Slot::Choice { .. } | Slot::Bytes { .. } => return Err(shapes()),
        })
    }

    /// The number a slot holds, read from memory or chosen at run time if
    /// need be.
    fn number(&mut self, slot: Slot) -> Result<KOperand> {
        match self.resolve(slot)? {
            Slot::Scalar(local) => Ok(local.into()),
            Slot::Memory { address, ty } => Ok(self.b.load(address, ty).into()),
            _ => Err("uses a value as a number that isn't one".into()),
        }
    }

    /// Writes a number to the slot `place` names.
    fn write_number(&mut self, dest: Slot, rvalue: KRvalue) -> Result<()> {
        match dest {
            Slot::Scalar(local) => self.set(local, rvalue),
            Slot::Choice { .. } => {
                // Typed like the options; validation checks the rvalue.
                fn number(slot: &Slot) -> Option<KLocal> {
                    match slot {
                        Slot::Scalar(local) => Some(*local),
                        Slot::Choice { options, .. } => options.first().and_then(number),
                        _ => None,
                    }
                }
                let ty = number(&dest).map(|number| self.b.ty(number)).ok_or("writes a number into something else")?;
                let value = self.b.local(ty);
                self.b.set(value, rvalue);
                self.copy_fields(&dest, &Slot::Scalar(value))?;
            }
            Slot::Bytes { .. } => return Err("writes through a pointer to data of another type".into()),
            Slot::Memory { .. } => return Err("writes to a read-only slice".into()),
            _ => return Err("writes a number into something else".into()),
        }
        Ok(())
    }

    /// A copy of `src`'s numbers in new locals.
    fn snapshot(&mut self, src: &Slot) -> Result<Slot> {
        Ok(match src {
            Slot::Scalar(local) => Slot::Scalar(self.b.copy(*local)),
            Slot::Fields(fields) => Slot::Fields(fields.iter().map(|field| self.snapshot(field)).collect::<Result<_>>()?),
            Slot::Choice { .. } => self.resolve(src.clone())?,
            Slot::Enum { tag, variants } => Slot::Enum {
                tag: self.b.copy(*tag),
                variants: variants.iter().map(|variant| self.snapshot(variant)).collect::<Result<_>>()?,
            },
            other => other.clone(),
        })
    }

    fn assign(&mut self, frame: &mut Frame<'_, 'tcx>, place: &Place<'tcx>, rvalue: &Rvalue<'tcx>) -> Result<()> {
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

    /// The slot a reference-valued rvalue points to, as `Slot::Ref`.
    fn reference(&mut self, frame: &mut Frame<'_, 'tcx>, rvalue: &Rvalue<'tcx>) -> Result<Slot> {
        match rvalue {
            Rvalue::Ref(_, _, place) | Rvalue::RawPtr(_, place) => Ok(Slot::Ref(Box::new(self.place(frame, place)?))),
            Rvalue::Use(operand, _) => self.operand_slot(frame, operand),
            Rvalue::CopyForDeref(place) => self.place(frame, place),
            // An array as a slice: the same elements, whose length the
            // slot knows.
            Rvalue::Cast(CastKind::PointerCoercion(PointerCoercion::Unsize, _), operand, ty)
                if points_to_array(operand.ty(frame.body, self.tcx)) && points_to_slice(*ty) =>
            {
                self.operand_slot(frame, operand)
            }
            // A pointer to the same bytes as another type: a view sharing
            // the numbers they hold.
            Rvalue::Cast(
                kind @ (CastKind::PtrToPtr | CastKind::PointerCoercion(PointerCoercion::MutToConstPointer, _)),
                operand,
                ty,
            ) => {
                let from = operand.ty(frame.body, self.tcx).builtin_deref(true);
                let (Some(from), Some(to)) = (from, ty.builtin_deref(true)) else {
                    return Err(cast_refusal(*kind, *ty));
                };
                let Slot::Ref(target) = self.operand_slot(frame, operand)? else {
                    return Err(cast_refusal(*kind, *ty));
                };
                if from == to {
                    return Ok(Slot::Ref(target));
                }
                if !from.is_sized(self.tcx, self.typing_env()) || !to.is_sized(self.tcx, self.typing_env()) {
                    return Err(cast_refusal(*kind, *ty));
                }
                // Bytes stay bytes, whatever type they are seen as.
                if let Slot::Bytes { .. } = *target {
                    return Ok(Slot::Ref(target));
                }
                let mut numbers = Vec::new();
                self.numbers_of(&target, from, 0, &mut numbers).map_err(|_| cast_refusal(*kind, *ty))?;
                // Numbers that line up alias; others are read as bytes.
                let view = self.view(to, 0, &numbers).unwrap_or(Slot::Bytes { numbers, offset: 0 });
                Ok(Slot::Ref(Box::new(view)))
            }
            Rvalue::Cast(kind, _, ty) => Err(cast_refusal(*kind, *ty)),
            _ => Err("uses a reference it can't follow".into()),
        }
    }

    /// The numbers a value of type `ty` holds, with their offsets in its
    /// layout from `base`.
    fn numbers_of(&self, slot: &Slot, ty: Ty<'tcx>, base: u64, out: &mut Vec<(u64, KLocal)>) -> Result<()> {
        match slot {
            Slot::Scalar(local) => out.push((base, *local)),
            Slot::Fields(fields) => {
                let layout = self.layout(ty)?;
                for (index, (field, field_ty)) in fields.iter().zip(self.fields(ty)?).enumerate() {
                    self.numbers_of(field, field_ty, base + layout.fields.offset(index).bytes(), out)?;
                }
            }
            Slot::Unit => {}
            _ => return Err(format!("views `{ty}` as other data")),
        }
        Ok(())
    }

    /// A value of type `ty` at `base` made of the given numbers: each of its
    /// numbers must be one of them, at its offset and of its type, so reading
    /// and writing through it reads and writes them.
    fn view(&self, ty: Ty<'tcx>, base: u64, numbers: &[(u64, KLocal)]) -> Result<Slot> {
        if let Some(kernel) = held_ty(ty) {
            return numbers
                .iter()
                .find(|&&(at, local)| at == base && self.b.ty(local) == kernel)
                .map(|&(_, local)| Slot::Scalar(local))
                .ok_or_else(|| format!("views data as `{ty}`, whose numbers lie elsewhere"));
        }
        if is_ref(ty) || self.enum_shape(ty)?.is_some() {
            return Err(format!("views data as `{ty}`"));
        }
        let layout = self.layout(ty)?;
        let fields = self.fields(ty)?;
        if fields.is_empty() {
            return Ok(Slot::Unit);
        }
        let parts = fields
            .into_iter()
            .enumerate()
            .map(|(index, field)| self.view(field, base + layout.fields.offset(index).bytes(), numbers))
            .collect::<Result<_>>()?;
        Ok(Slot::Fields(parts))
    }

    fn scalar_rvalue(&mut self, frame: &mut Frame<'_, 'tcx>, rvalue: &Rvalue<'tcx>) -> Result<KRvalue> {
        Ok(match rvalue {
            Rvalue::Use(operand, _) => KRvalue::Use(self.scalar_operand(frame, operand)?),
            Rvalue::CopyForDeref(place) => match self.read_place(frame, place)? {
                Slot::Memory { address, ty } => KRvalue::Load(address.into(), ty),
                slot => KRvalue::Use(self.number(slot)?),
            },
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

    fn scalar_operand(&mut self, frame: &mut Frame<'_, 'tcx>, operand: &Operand<'tcx>) -> Result<KOperand> {
        match operand {
            Operand::Constant(c) => constant(self.tcx, c)
                .map(KOperand::Const)
                .ok_or_else(|| format!("uses an unsupported constant of type `{}`", c.ty())),
            Operand::Copy(place) | Operand::Move(place) => {
                let slot = self.read_place(frame, place)?;
                self.number(slot)
            }
            Operand::RuntimeChecks(checks) => Ok(self.runtime_check(*checks).into()),
        }
    }

    /// Whether this build makes a kind of run-time check, as the CPU code
    /// does.
    fn runtime_check(&self, checks: rustc_middle::mir::RuntimeChecks) -> Const {
        use rustc_middle::mir::RuntimeChecks;
        let session = self.tcx.sess;
        Const::bool(match checks {
            RuntimeChecks::OverflowChecks => session.overflow_checks(),
            RuntimeChecks::UbChecks => session.ub_checks(),
            RuntimeChecks::ContractChecks => session.contract_checks(),
        })
    }

    /// `a op b` and whether it overflowed, as `AddWithOverflow` and its
    /// siblings compute them, for the overflow checks of debug builds.
    fn with_overflow(&mut self, op: BinOp, a: KOperand, b: KOperand) -> Result<(KLocal, KLocal)> {
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
    fn operand_slot(&mut self, frame: &mut Frame<'_, 'tcx>, operand: &Operand<'tcx>) -> Result<Slot> {
        match operand {
            Operand::Constant(c) => self.constant_slot(c),
            Operand::Copy(place) | Operand::Move(place) => match self.read_place(frame, place)? {
                Slot::Memory { address, ty } => Ok(Slot::Scalar(self.b.load(address, ty))),
                slot => self.resolve(slot),
            },
            Operand::RuntimeChecks(checks) => Ok(Slot::Scalar(self.b.copy(self.runtime_check(*checks)))),
        }
    }

    /// A constant of any supported type, in new locals.
    fn constant_slot(&mut self, c: &ConstOperand<'tcx>) -> Result<Slot> {
        let ty = c.ty();
        if let Some(scalar) = constant(self.tcx, c) {
            return Ok(Slot::Scalar(self.b.copy(scalar)));
        }
        if is_ref(ty) {
            return self.constant_ref(c);
        }
        let slot = self.fresh(ty)?;
        if slot == Slot::Unit {
            return Ok(slot);
        }
        let bytes = self.constant_bytes(c)?;
        self.fill(&slot, ty, &bytes, 0)?;
        Ok(slot)
    }

    /// A constant's bytes, laid out as rustc lays out its type.
    fn constant_bytes(&self, c: &ConstOperand<'tcx>) -> Result<Vec<u8>> {
        let ty = c.ty();
        let tcx = self.tcx;
        let value = c
            .const_
            .eval(tcx, self.typing_env(), c.span)
            .map_err(|_| format!("uses a constant of type `{ty}` that can't be evaluated"))?;
        let layout = tcx
            .layout_of(self.typing_env().as_query_input(ty))
            .map_err(|_| format!("uses a constant of type `{ty}` without a layout"))?;
        let size = layout.size.bytes_usize();
        let bytes: Vec<u8> = match value {
            ConstValue::Scalar(Scalar::Int(int)) => int.to_bits(int.size()).to_le_bytes()[..size].to_vec(),
            // A value of no size, made of empty fields.
            ConstValue::ZeroSized => Vec::new(),
            ConstValue::Indirect { alloc_id, offset } => match tcx.global_alloc(alloc_id) {
                GlobalAlloc::Memory(alloc) => {
                    let start = offset.bytes_usize();
                    alloc.inner().inspect_with_uninit_and_ptr_outside_interpreter(start..start + size).to_vec()
                }
                _ => return Err(format!("uses a constant of type `{ty}` that isn't plain data")),
            },
            _ => return Err(format!("uses a constant of type `{ty}` that isn't plain data")),
        };
        Ok(bytes)
    }

    /// A reference to constant data, such as a table (`&[f32; 5]`, `&[u8]`)
    /// or a read-only static: a reference to new locals holding its value.
    /// A static that can change (interior mutability, `static mut`) is
    /// refused.
    fn constant_ref(&mut self, c: &ConstOperand<'tcx>) -> Result<Slot> {
        let tcx = self.tcx;
        let ty = c.ty();
        let refusal = || format!("uses a reference to constant data (`{ty}`)");
        let pointee = ty.builtin_deref(true).ok_or_else(refusal)?;
        let value = c.const_.eval(tcx, self.typing_env(), c.span).map_err(|_| refusal())?;
        let (alloc_id, offset, len) = match value {
            ConstValue::Scalar(Scalar::Ptr(pointer, _)) => {
                let (provenance, offset) = pointer.prov_and_relative_offset();
                (provenance.alloc_id(), offset.bytes_usize(), None)
            }
            ConstValue::Slice { alloc_id, meta } => (alloc_id, 0, Some(meta)),
            _ => return Err(refusal()),
        };
        let alloc = match tcx.global_alloc(alloc_id) {
            GlobalAlloc::Memory(alloc) => alloc,
            GlobalAlloc::Static(def_id) => {
                let static_ty = tcx.type_of(def_id).instantiate_identity().skip_normalization();
                if tcx.static_mutability(def_id) != Some(rustc_hir::Mutability::Not)
                    || !static_ty.is_freeze(tcx, self.typing_env())
                {
                    return Err(format!("reads a static that can change (`{}`)", tcx.def_path_str(def_id)));
                }
                tcx.eval_static_initializer(def_id).map_err(|_| refusal())?
            }
            _ => return Err(refusal()),
        };
        let bytes = alloc.inner().inspect_with_uninit_and_ptr_outside_interpreter(0..alloc.inner().size().bytes_usize());
        let target = match (len, pointee.kind()) {
            (None, _) => {
                let slot = self.fresh(pointee)?;
                self.fill(&slot, pointee, bytes, offset)?;
                slot
            }
            // A slice: its elements, one after another.
            (Some(len), ty::Slice(element)) => {
                if len > MAX_ARRAY {
                    return Err(format!("uses a slice of {len} elements, more than the {MAX_ARRAY} a kernel holds in locals"));
                }
                let stride = self.size(*element)? as usize;
                let mut elements = Vec::with_capacity(len as usize);
                for index in 0..len as usize {
                    let slot = self.fresh(*element)?;
                    self.fill(&slot, *element, bytes, offset + index * stride)?;
                    elements.push(slot);
                }
                if elements.is_empty() { Slot::Unit } else { Slot::Fields(elements) }
            }
            (Some(_), _) => return Err(refusal()),
        };
        Ok(Slot::Ref(Box::new(target)))
    }

    /// Sets a slot's locals from a constant's bytes, laid out as rustc lays
    /// out `ty`, little-endian.
    fn fill(&mut self, slot: &Slot, ty: Ty<'tcx>, bytes: &[u8], base: usize) -> Result<()> {
        match slot {
            Slot::Scalar(local) => {
                let kernel = held_ty(ty).ok_or("uses a constant of an unexpected type")?;
                let size = kernel.bytes() as usize;
                let mut buffer = [0u8; 16];
                buffer[..size].copy_from_slice(bytes.get(base..base + size).ok_or("reads past a constant")?);
                let bits = u128::from_le_bytes(buffer);
                self.b.set(*local, KRvalue::Use(Const { ty: kernel, bits }.into()));
                Ok(())
            }
            Slot::Fields(fields) => {
                let layout = self
                    .tcx
                    .layout_of(self.typing_env().as_query_input(ty))
                    .map_err(|_| format!("uses a constant of type `{ty}` without a layout"))?;
                let types = self.fields(ty)?;
                for (index, (field, field_ty)) in fields.iter().zip(types).enumerate() {
                    self.fill(field, field_ty, bytes, base + layout.fields.offset(index).bytes_usize())?;
                }
                Ok(())
            }
            Slot::Enum { tag, variants } => {
                let (variant, offsets) = self.constant_variant(ty, bytes, base)?;
                let discriminant = self.discriminant(ty, variant)?;
                self.b.set(*tag, KRvalue::Use(discriminant.into()));
                let Some((_, shape)) = self.enum_shape(ty)? else {
                    return Err(format!("uses a constant of type `{ty}` that isn't plain data"));
                };
                if let Slot::Fields(fields) = &variants[variant.as_usize()] {
                    for ((field, field_ty), offset) in fields.iter().zip(&shape[variant.as_usize()]).zip(offsets) {
                        self.fill(field, *field_ty, bytes, base + offset)?;
                    }
                }
                Ok(())
            }
            Slot::Unit => Ok(()),
            _ => Err("uses a constant holding a reference".into()),
        }
    }

    /// Which variant a constant enum at `base` in `bytes` holds, and where
    /// its fields are, from rustc's layout: a tag holding the discriminant,
    /// or a niche, a field's invalid values standing for other variants.
    fn constant_variant(&self, ty: Ty<'tcx>, bytes: &[u8], base: usize) -> Result<(VariantIdx, Vec<usize>)> {
        let tcx = self.tcx;
        let layout = tcx
            .layout_of(self.typing_env().as_query_input(ty))
            .map_err(|_| format!("uses a constant of type `{ty}` without a layout"))?;
        let ty::Adt(adt, _) = ty.kind() else {
            return Err(format!("uses a constant of type `{ty}` that isn't plain data"));
        };
        match &layout.variants {
            Variants::Empty => Err(format!("uses a constant of the uninhabited type `{ty}`")),
            Variants::Single { index } => {
                let count = adt.variant(*index).fields.len();
                Ok((*index, (0..count).map(|field| layout.fields.offset(field).bytes_usize()).collect()))
            }
            Variants::Multiple { tag, tag_encoding, tag_field, variants } => {
                let size = tag.size(&tcx).bytes_usize();
                let at = base + layout.fields.offset(tag_field.as_usize()).bytes_usize();
                let mut buffer = [0u8; 16];
                buffer[..size].copy_from_slice(bytes.get(at..at + size).ok_or("reads past a constant")?);
                let raw = u128::from_le_bytes(buffer);
                let mask = if size >= 16 { u128::MAX } else { (1u128 << (8 * size)) - 1 };
                let variant = match tag_encoding {
                    TagEncoding::Direct => adt
                        .discriminants(tcx)
                        .find(|(_, discriminant)| discriminant.val & mask == raw)
                        .map(|(variant, _)| variant)
                        .ok_or_else(|| format!("uses a constant of type `{ty}` with no valid variant"))?,
                    TagEncoding::Niche { untagged_variant, niche_variants, niche_start } => {
                        let relative = raw.wrapping_sub(*niche_start) & mask;
                        let count = u128::from(niche_variants.last.as_u32() - niche_variants.start.as_u32());
                        if relative <= count {
                            VariantIdx::from_u32(niche_variants.start.as_u32() + relative as u32)
                        } else {
                            *untagged_variant
                        }
                    }
                };
                let offsets = variants[variant].field_offsets.iter().map(|offset| offset.bytes_usize()).collect();
                Ok((variant, offsets))
            }
        }
    }

    /// `operand`'s bits as a value of type `to`, as `transmute` keeps them.
    /// Both are plain data of one size, laid out as rustc lays them out,
    /// little-endian; each number of the result gathers the bytes it covers
    /// from the source's numbers. Bytes no number covers are padding, which
    /// a defined transmute never reads into a number. Enums are read and
    /// written through their tags or niches, as their layout stores them.
    fn transmute(&mut self, frame: &mut Frame<'_, 'tcx>, operand: &Operand<'tcx>, to: Ty<'tcx>) -> Result<Slot> {
        let from = operand.ty(frame.body, self.tcx);
        if is_ref(from) || is_ref(to) {
            return Err(cast_refusal(CastKind::Transmute, to));
        }
        let (from_size, to_size) = (self.size(from)?, self.size(to)?);
        if from_size != to_size {
            return Err(format!("reinterprets `{from}` as `{to}`, of another size"));
        }
        let value = self.operand_slot(frame, operand)?;
        let result = self.fresh(to)?;
        self.decode(&result, to, 0, &value, from)?;
        Ok(result)
    }

    /// Sets `dest`, of type `ty`, from the bytes at `base` of `src`, a value
    /// of type `src_ty`.
    fn decode(&mut self, dest: &Slot, ty: Ty<'tcx>, base: u64, src: &Slot, src_ty: Ty<'tcx>) -> Result<()> {
        match dest {
            Slot::Scalar(local) => {
                let kernel = self.b.ty(*local);
                let width = u64::from(kernel.bytes());
                let rvalue = match src {
                    // The same bytes: one number reinterpreted.
                    Slot::Scalar(number) if base == 0 && self.b.ty(*number).bytes() == kernel.bytes() => {
                        self.same_width((*number).into(), kernel)
                    }
                    Slot::Bytes { numbers, offset } if numbers.iter().any(|&(at, number)| at == offset + base && self.b.ty(number).bytes() == kernel.bytes()) => {
                        let &(_, number) = numbers.iter().find(|&&(at, number)| at == offset + base && self.b.ty(number).bytes() == kernel.bytes()).expect("found above");
                        self.same_width(number.into(), kernel)
                    }
                    // 128 bits: two halves of 64.
                    _ if width == 16 => {
                        let low = self.gather(src, src_ty, 0, base, 8)?;
                        let high = self.gather(src, src_ty, 0, base + 8, 8)?;
                        let (low, high) = (self.b.cast(low, KTy::U128), self.b.cast(high, KTy::U128));
                        let high = self.b.binary(KBinOp::Shl, high, Const::int(KTy::U32, 64));
                        let bits = self.b.binary(KBinOp::Or, low, high);
                        self.same_width(bits.into(), kernel)
                    }
                    _ => {
                        let bits = self.gather(src, src_ty, 0, base, width)?;
                        let narrow = self.b.cast(bits, unsigned_of(width));
                        self.same_width(narrow.into(), kernel)
                    }
                };
                self.set(*local, rvalue);
            }
            Slot::Fields(fields) => {
                let layout = self.layout(ty)?;
                let types = self.fields(ty)?;
                for (index, (field, field_ty)) in fields.iter().zip(types).enumerate() {
                    self.decode(field, field_ty, base + layout.fields.offset(index).bytes(), src, src_ty)?;
                }
            }
            Slot::Unit => {}
            Slot::Enum { tag, variants } => {
                let layout = self.layout(ty)?;
                let Some((tag_ty, shape)) = self.enum_shape(ty)? else {
                    return Err(format!("reinterprets bits as `{ty}`"));
                };
                for (index, variant) in variants.iter().enumerate() {
                    let (Slot::Fields(fields), Some(offsets)) = (variant, variant_offsets(&layout, index)) else {
                        continue;
                    };
                    for ((field, field_ty), offset) in fields.iter().zip(&shape[index]).zip(offsets) {
                        self.decode(field, *field_ty, base + offset, src, src_ty)?;
                    }
                }
                let discriminant: KOperand = match &layout.variants {
                    Variants::Empty => return Err(format!("reinterprets bits as the uninhabited `{ty}`")),
                    Variants::Single { index } => self.discriminant(ty, *index)?.into(),
                    Variants::Multiple { tag: scalar, tag_encoding, tag_field, .. } => {
                        let size = scalar.size(&self.tcx).bytes();
                        if size > 8 {
                            return Err(format!("reinterprets bits as `{ty}`, whose tag is wider than 64 bits"));
                        }
                        let at = base + layout.fields.offset(tag_field.as_usize()).bytes();
                        let raw = self.gather(src, src_ty, 0, at, size)?;
                        match tag_encoding {
                            // The tag is the discriminant, in its width and
                            // signedness.
                            TagEncoding::Direct => {
                                let narrow = self.b.cast(raw, unsigned_of(size));
                                let signed = matches!(scalar.primitive(), rustc_abi::Primitive::Int(_, true));
                                let narrow = if signed { self.b.cast(narrow, signed_of(size)) } else { narrow };
                                self.b.cast(narrow, tag_ty).into()
                            }
                            // Tag values from `niche_start` on stand for the
                            // niche variants in order; others for the
                            // untagged variant.
                            TagEncoding::Niche { untagged_variant, niche_variants, niche_start } => {
                                let mask = if size == 8 { u64::MAX } else { (1u64 << (8 * size)) - 1 };
                                let start = Const::int(KTy::U64, i128::from(*niche_start as u64));
                                let shifted = self.b.binary(KBinOp::Sub, raw, start);
                                let relative = self.b.binary(KBinOp::And, shifted, Const::int(KTy::U64, mask.into()));
                                let mut chosen: KOperand = self.discriminant(ty, *untagged_variant)?.into();
                                let (first, last) = (niche_variants.start.as_u32(), niche_variants.last.as_u32());
                                for variant in first..=last {
                                    let step = Const::int(KTy::U64, (variant - first).into());
                                    let is = self.b.binary(KBinOp::Eq, relative, step);
                                    let value = self.discriminant(ty, VariantIdx::from_u32(variant))?;
                                    chosen = self.b.select(is, value, chosen).into();
                                }
                                chosen
                            }
                        }
                    }
                };
                self.set(*tag, KRvalue::Use(discriminant));
            }
            _ => return Err(format!("reinterprets bits as `{ty}`, which holds a reference")),
        }
        Ok(())
    }

    /// The bits of the bytes `offset..offset + width` (at most 8) of `src`,
    /// a value of type `ty` laid out from `base`, zero-extended to a `u64`.
    fn gather(&mut self, src: &Slot, ty: Ty<'tcx>, base: u64, offset: u64, width: u64) -> Result<KOperand> {
        let zero = KOperand::from(Const::int(KTy::U64, 0));
        Ok(match src {
            Slot::Scalar(number) => self.number_bits(*number, i128::from(base), offset, width).unwrap_or(zero),
            Slot::Fields(fields) => {
                let layout = self.layout(ty)?;
                let types = self.fields(ty)?;
                let mut bits = zero;
                for (index, (field, field_ty)) in fields.iter().zip(types).enumerate() {
                    let at = base + layout.fields.offset(index).bytes();
                    if at >= offset + width || offset >= at + self.size(field_ty)? {
                        continue;
                    }
                    let part = self.gather(field, field_ty, at, offset, width)?;
                    bits = self.or(bits, part);
                }
                bits
            }
            // Bytes seen through a pointer: the data's numbers, placed where
            // they fall in the value read.
            Slot::Bytes { numbers, offset: start } => {
                let mut bits = zero;
                for &(at, number) in numbers {
                    let position = i128::from(base) + i128::from(at) - i128::from(*start);
                    if let Some(placed) = self.number_bits(number, position, offset, width) {
                        bits = self.or(bits, placed);
                    }
                }
                bits
            }
            Slot::Unit => zero,
            // Each variant's bytes, with its tag, chosen by the discriminant.
            Slot::Enum { tag, variants } => {
                let layout = self.layout(ty)?;
                let Some((_, shape)) = self.enum_shape(ty)? else {
                    return Err(format!("reinterprets the bits of `{ty}`"));
                };
                let mut chosen = zero;
                for (index, variant) in variants.iter().enumerate() {
                    let Some(offsets) = variant_offsets(&layout, index) else { continue };
                    let mut bits = zero;
                    if let Slot::Fields(fields) = variant {
                        for ((field, field_ty), field_offset) in fields.iter().zip(&shape[index]).zip(offsets) {
                            let at = base + field_offset;
                            if at >= offset + width || offset >= at + self.size(*field_ty)? {
                                continue;
                            }
                            let part = self.gather(field, *field_ty, at, offset, width)?;
                            bits = self.or(bits, part);
                        }
                    }
                    if let Variants::Multiple { tag: scalar, tag_encoding, tag_field, .. } = &layout.variants {
                        let size = scalar.size(&self.tcx).bytes();
                        let at = base + layout.fields.offset(tag_field.as_usize()).bytes();
                        let stored = match tag_encoding {
                            TagEncoding::Direct => Some(self.discriminant(ty, VariantIdx::from_usize(index))?.bits as u64),
                            TagEncoding::Niche { untagged_variant, niche_variants, niche_start } => {
                                let (first, last) = (niche_variants.start.as_usize(), niche_variants.last.as_usize());
                                (index != untagged_variant.as_usize() && (first..=last).contains(&index))
                                    .then(|| (*niche_start as u64).wrapping_add((index - first) as u64))
                            }
                        };
                        if let Some(stored) = stored.filter(|_| at < offset + width && offset < at + size) {
                            let mask = if size >= 8 { u64::MAX } else { (1u64 << (8 * size)) - 1 };
                            let constant = KOperand::from(Const::int(KTy::U64, (stored & mask).into()));
                            let placed = self.place_bits(constant, at, offset);
                            bits = self.or(bits, placed.into());
                        }
                    }
                    let discriminant = self.discriminant(ty, VariantIdx::from_usize(index))?;
                    let is = self.b.binary(KBinOp::Eq, *tag, discriminant);
                    chosen = self.b.select(is, bits, chosen).into();
                }
                chosen
            }
            _ => return Err(format!("reinterprets the bits of `{ty}`, which holds a reference")),
        })
    }

    /// The bits a number at `position` gives the bytes `offset..offset +
    /// width` (at most 8), as a `u64`, if it overlaps them. A 128-bit number
    /// gives its two halves.
    fn number_bits(&mut self, number: KLocal, position: i128, offset: u64, width: u64) -> Option<KOperand> {
        let ty = self.b.ty(number);
        let size = i128::from(ty.bytes());
        let (offset, end) = (i128::from(offset), i128::from(offset + width));
        if position >= end || offset >= position + size {
            return None;
        }
        let halves: Vec<(i128, KLocal)> = if size == 16 {
            let unsigned = self.b.cast(number, KTy::U128);
            let low = self.b.cast(unsigned, KTy::U64);
            let shifted = self.b.binary(KBinOp::Shr, unsigned, Const::int(KTy::U32, 64));
            let high = self.b.cast(shifted, KTy::U64);
            vec![(position, low), (position + 8, high)]
        } else {
            vec![(position, self.unsigned_bits(number))]
        };
        let mut bits: Option<KOperand> = None;
        for (at, half) in halves {
            if at >= end || offset >= at + 8.min(size) {
                continue;
            }
            let placed = if at >= offset {
                self.b.binary(KBinOp::Shl, half, Const::int(KTy::U32, 8 * (at - offset)))
            } else {
                self.b.binary(KBinOp::Shr, half, Const::int(KTy::U32, 8 * (offset - at)))
            };
            bits = Some(match bits {
                Some(previous) => self.b.binary(KBinOp::Or, previous, placed).into(),
                None => placed.into(),
            });
        }
        bits
    }

    /// `bits`, of a value at `at`, moved to where they fall in a range
    /// starting at `offset`.
    fn place_bits(&mut self, bits: KOperand, at: u64, offset: u64) -> KLocal {
        if at >= offset {
            self.b.binary(KBinOp::Shl, bits, Const::int(KTy::U32, (8 * (at - offset)).into()))
        } else {
            self.b.binary(KBinOp::Shr, bits, Const::int(KTy::U32, (8 * (offset - at)).into()))
        }
    }

    fn or(&mut self, a: KOperand, b: KOperand) -> KOperand {
        if a == KOperand::from(Const::int(KTy::U64, 0)) {
            b
        } else {
            self.b.binary(KBinOp::Or, a, b).into()
        }
    }

    fn layout(&self, ty: Ty<'tcx>) -> Result<rustc_middle::ty::layout::TyAndLayout<'tcx>> {
        self.tcx.layout_of(self.typing_env().as_query_input(ty)).map_err(|_| format!("uses `{ty}`, which has no layout"))
    }

    /// A number as another type of its width, keeping its bits. A `bool`'s
    /// byte is 0 or 1, and only those bytes are valid `bool`s.
    fn same_width(&mut self, value: KOperand, to: KTy) -> KRvalue {
        let from = self.b.ty(value);
        if from == to {
            KRvalue::Use(value)
        } else if to == KTy::Bool {
            KRvalue::Binary(KBinOp::Ne, value, Const::int(from, 0).into())
        } else if from == KTy::Bool || from.is_int() && to.is_int() {
            KRvalue::Cast(value, to)
        } else {
            KRvalue::Reinterpret(value, to)
        }
    }

    /// An operand's value if it is a constant number.
    fn known(&self, operand: &Operand<'tcx>) -> Option<Const> {
        match operand {
            Operand::Constant(c) => constant(self.tcx, c),
            _ => None,
        }
    }

    /// The lanes of a SIMD vector: a `#[repr(simd)]` struct around an array,
    /// held as its elements.
    fn lanes(&self, slot: &Slot) -> Result<Vec<Slot>> {
        match slot {
            Slot::Fields(fields) if fields.len() == 1 => match &fields[0] {
                Slot::Fields(lanes) => Ok(lanes.clone()),
                _ => Err("uses a SIMD vector whose lanes aren't an array".into()),
            },
            _ => Err("uses a SIMD vector whose lanes aren't an array".into()),
        }
    }

    /// The numbers in a vector's lanes.
    fn lane_numbers(&mut self, slot: &Slot) -> Result<Vec<KOperand>> {
        let lanes = self.lanes(slot)?;
        lanes.into_iter().map(|lane| self.number(lane)).collect()
    }

    /// A `simd_*` intrinsic: the scalar operation on each lane, as the
    /// intrinsic's documentation defines it, into a new vector.
    fn simd(
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
                "simd_bswap" => {
                    let mut rvalues = Vec::new();
                    for lane in self.lane_numbers(&arg(0)?)? {
                        let swapped = self.byte_swap(lane);
                        rvalues.push(KRvalue::Use(swapped.into()));
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

    /// `a + b` or `a - b` of integers, clamped to the type's range.
    fn saturating(&mut self, add: bool, a: KOperand, b: KOperand) -> Result<KLocal> {
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
        // Signed 64 bits: on overflow, the result saturates toward `a`'s sign.
        let checked = if add { BinOp::AddWithOverflow } else { BinOp::SubWithOverflow };
        let (result, overflowed) = self.with_overflow(checked, a, b)?;
        let negative = self.b.binary(KBinOp::Lt, a, Const::int(ty, 0));
        let limit = self.b.select(negative, int_min(ty), int_max(ty));
        Ok(self.b.select(overflowed, limit, result))
    }

    /// An integer with its bytes in reverse order.
    fn byte_swap(&mut self, value: KOperand) -> KLocal {
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

    /// An x86 SIMD instruction, called through its LLVM intrinsic: what the
    /// instruction computes, from Intel's definitions, lane by lane. The
    /// floating-point environment is Rust's default (round to nearest even,
    /// exceptions masked). Instructions whose results differ between CPU
    /// models (`rcpps`, `rsqrtps`) are refused.
    fn x86(
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

    /// A number's bits, zero-extended to a `u64`.
    fn unsigned_bits(&mut self, number: KLocal) -> KLocal {
        let ty = self.b.ty(number);
        let same = self.same_width(number.into(), unsigned_of(u64::from(ty.bytes())));
        let bits = self.b.assign(same);
        self.b.cast(bits, KTy::U64)
    }

    fn size(&self, ty: Ty<'tcx>) -> Result<u64> {
        self.tcx
            .layout_of(self.typing_env().as_query_input(ty))
            .map(|layout| layout.size.bytes())
            .map_err(|_| format!("uses `{ty}`, which has no layout"))
    }
}

// Refusals. Each function below matches every variant of a MIR enum, with
// no wildcard: a toolchain that adds a variant fails to compile until it is
// classified here (and in `docs/coverage.md`). Translation handles the
// supported variants before reaching these.

/// Why an operation can't be translated where it appears.
fn rvalue_refusal(rvalue: &Rvalue<'_>) -> String {
    match rvalue {
        Rvalue::Use(..) | Rvalue::CopyForDeref(_) | Rvalue::WrapUnsafeBinder(..) => {
            "copies a value the kernel can't hold".into()
        }
        Rvalue::Repeat(..) => "builds an array (`[x; N]`)".into(),
        Rvalue::Ref(..) | Rvalue::RawPtr(..) => "borrows something it can't follow".into(),
        Rvalue::Reborrow(ty, ..) => format!("reborrows a `{ty}` (an experimental feature)"),
        Rvalue::ThreadLocalRef(_) => "uses a thread-local".into(),
        Rvalue::Cast(kind, _, ty) => cast_refusal(*kind, *ty),
        Rvalue::BinaryOp(op, _) => binary_refusal(*op),
        Rvalue::UnaryOp(op, _) => match op {
            UnOp::PtrMetadata => "reads the metadata of a pointer other than a read-only slice".into(),
            UnOp::Not | UnOp::Neg => format!("uses `{op:?}` on a value the kernel can't hold"),
        },
        Rvalue::Discriminant(_) => "uses an enum (reads a discriminant)".into(),
        Rvalue::Aggregate(kind, _) => aggregate_refusal(kind),
    }
}

fn binary_refusal(op: BinOp) -> String {
    match op {
        BinOp::Cmp => "compares three ways (`Ord::cmp`), which gives an enum".into(),
        BinOp::Offset => "offsets a pointer".into(),
        BinOp::Add
        | BinOp::AddUnchecked
        | BinOp::AddWithOverflow
        | BinOp::Sub
        | BinOp::SubUnchecked
        | BinOp::SubWithOverflow
        | BinOp::Mul
        | BinOp::MulUnchecked
        | BinOp::MulWithOverflow
        | BinOp::Div
        | BinOp::Rem
        | BinOp::BitXor
        | BinOp::BitAnd
        | BinOp::BitOr
        | BinOp::Shl
        | BinOp::ShlUnchecked
        | BinOp::Shr
        | BinOp::ShrUnchecked
        | BinOp::Eq
        | BinOp::Lt
        | BinOp::Le
        | BinOp::Ne
        | BinOp::Ge
        | BinOp::Gt => format!("uses `{op:?}` on values the kernel can't hold"),
    }
}

/// Why a cast to `ty` can't be translated.
fn cast_refusal(kind: CastKind, ty: Ty<'_>) -> String {
    match kind {
        CastKind::IntToInt | CastKind::FloatToInt | CastKind::FloatToFloat | CastKind::IntToFloat => {
            format!("casts to `{ty}`, a type the kernel doesn't have")
        }
        CastKind::Transmute => format!("reinterprets bits as `{ty}` (a transmute)"),
        CastKind::Subtype => format!("changes a value's type to the subtype `{ty}`"),
        CastKind::PtrToPtr => format!("converts a pointer to `{ty}`"),
        CastKind::PointerExposeProvenance | CastKind::PointerWithExposedProvenance => {
            "uses a pointer as an address".into()
        }
        CastKind::FnPtrToPtr => "uses a function pointer as an address".into(),
        CastKind::BoxDerefTransmute => "reads through a `Box`".into(),
        CastKind::PointerCoercion(coercion, _) => match coercion {
            PointerCoercion::ReifyFnPointer(..) | PointerCoercion::UnsafeFnPointer | PointerCoercion::ClosureFnPointer(..) => {
                format!("makes a function pointer (`{ty}`)")
            }
            PointerCoercion::MutToConstPointer => format!("converts a pointer to `{ty}`"),
            PointerCoercion::ArrayToPointer => "turns an array into a pointer".into(),
            PointerCoercion::Unsize => format!("unsizes a pointer to `{ty}`"),
        },
    }
}

fn aggregate_refusal(kind: &AggregateKind<'_>) -> String {
    match kind {
        AggregateKind::Array(_) => "builds an array".into(),
        AggregateKind::Tuple | AggregateKind::Closure(..) => "builds a value the kernel can't hold".into(),
        AggregateKind::Adt(_, _, _, _, Some(_)) => "builds a union".into(),
        AggregateKind::Adt(..) => "builds an enum".into(),
        AggregateKind::Coroutine(..) | AggregateKind::CoroutineClosure(..) => "builds a coroutine".into(),
        AggregateKind::RawPtr(..) => "builds a raw pointer from parts".into(),
    }
}

/// Why a place projection can't be followed.
fn projection_refusal<V, T>(element: &ProjectionElem<V, T>) -> String {
    match element {
        ProjectionElem::Deref => "reads through a reference it can't follow".into(),
        ProjectionElem::Field(..) => "uses a field of a value the kernel can't hold".into(),
        ProjectionElem::Index(_) | ProjectionElem::ConstantIndex { .. } => "indexes an array".into(),
        ProjectionElem::Subslice { .. } => "takes a subslice".into(),
        ProjectionElem::Downcast(..) => "uses an enum".into(),
        ProjectionElem::OpaqueCast(_) => "casts an opaque type".into(),
        ProjectionElem::UnwrapUnsafeBinder(_) => "unwraps an unsafe binder".into(),
        ProjectionElem::PhantomDeref => "reborrows through a user type (an experimental feature)".into(),
    }
}

/// Why a value of type `ty` can't be held in kernel locals.
fn type_refusal(ty: Ty<'_>) -> String {
    match ty.kind() {
        ty::Bool | ty::Int(_) | ty::Uint(_) | ty::Float(_) => format!("uses `{ty}`, a number the kernel doesn't have"),
        ty::Char => "uses `char`".into(),
        ty::Adt(adt, _) if adt.is_box() => format!("uses a `Box` (`{ty}`)"),
        ty::Adt(adt, _) if adt.is_enum() => format!("uses the enum `{ty}`"),
        ty::Adt(adt, _) if adt.is_union() => format!("uses the union `{ty}`"),
        ty::Adt(adt, _) if adt.repr().simd() => format!("uses the SIMD vector `{ty}`"),
        ty::Adt(..) => format!("uses a value of type `{ty}`"),
        ty::Array(..) => format!("uses an array (`{ty}`)"),
        ty::Slice(_) | ty::Str => format!("uses an unsized value (`{ty}`)"),
        ty::RawPtr(..) | ty::Ref(..) => format!("uses a pointer it can't follow (`{ty}`)"),
        ty::FnPtr(..) => format!("uses a function pointer (`{ty}`)"),
        ty::Dynamic(..) => format!("uses a trait object (`{ty}`)"),
        ty::Closure(..) | ty::FnDef(..) | ty::Tuple(_) | ty::Never => format!("uses a value of type `{ty}`"),
        ty::Coroutine(..) | ty::CoroutineClosure(..) | ty::CoroutineWitness(..) => format!("uses a coroutine (`{ty}`)"),
        ty::Foreign(_) => format!("uses an extern type (`{ty}`)"),
        ty::Pat(..) => format!("uses a pattern type (`{ty}`)"),
        ty::UnsafeBinder(_) => format!("uses an unsafe binder (`{ty}`)"),
        ty::Alias(..) | ty::Param(_) | ty::Bound(..) | ty::Placeholder(_) | ty::Infer(_) | ty::Error(_) => {
            format!("uses the unresolved type `{ty}`")
        }
    }
}


/// The blocks of `region` that can never finish normally: every path from
/// them ends in a call that never returns (a panic), or never ends. A frame
/// finishes at `return`; the loop's frame also at the end of an iteration.
fn doomed(frame: &Frame<'_, '_>, region: &[BasicBlock]) -> HashSet<BasicBlock> {
    let body = frame.body;
    let mut finishing: HashSet<BasicBlock> = HashSet::new();
    loop {
        let before = finishing.len();
        for &block in region {
            if finishing.contains(&block) {
                continue;
            }
            let kind = &body.basic_blocks[block].terminator().kind;
            // The `for_each` call is an iteration, which ends there.
            let each = frame.loop_rules.as_ref().is_some_and(|rules| rules.shape.each_block() == Some(block));
            let finishes = matches!(kind, TerminatorKind::Return)
                || each
                || successors(kind).into_iter().any(|next| match frame.loop_rules.as_ref().map(|r| (r, &r.shape.iteration)) {
                    Some((rules, Iteration::For { start, is_header, .. })) if is_header(next) => {
                        // From the body the iteration ends; from the setup
                        // the body starts.
                        rules.body.contains(&block) || finishing.contains(start)
                    }
                    _ => finishing.contains(&next),
                });
            if finishes {
                finishing.insert(block);
            }
        }
        if finishing.len() == before {
            break;
        }
    }
    region.iter().copied().filter(|block| !finishing.contains(block)).collect()
}

/// The unsigned kernel integer of `bytes` bytes.
fn unsigned_of(bytes: u64) -> KTy {
    match bytes {
        1 => KTy::U8,
        2 => KTy::U16,
        4 => KTy::U32,
        8 => KTy::U64,
        _ => KTy::U128,
    }
}

/// The signed kernel integer of `bytes` bytes.
fn signed_of(bytes: u64) -> KTy {
    match bytes {
        1 => KTy::I8,
        2 => KTy::I16,
        4 => KTy::I32,
        8 => KTy::I64,
        _ => KTy::I128,
    }
}

/// Where variant `index`'s fields are in an enum's layout, if the variant
/// can exist.
fn variant_offsets(layout: &rustc_middle::ty::layout::TyAndLayout<'_>, index: usize) -> Option<Vec<u64>> {
    match &layout.variants {
        Variants::Empty => None,
        Variants::Single { index: only } => (only.as_usize() == index)
            .then(|| (0..layout.fields.count()).map(|field| layout.fields.offset(field).bytes()).collect()),
        Variants::Multiple { variants, .. } => {
            Some(variants[VariantIdx::from_usize(index)].field_offsets.iter().map(|offset| offset.bytes()).collect())
        }
    }
}

/// The smallest value of a signed integer type.
fn int_min(ty: KTy) -> Const {
    Const { ty, bits: 1u128 << (ty.bits() - 1) }
}

/// The largest value of a signed integer type.
fn int_max(ty: KTy) -> Const {
    Const { ty, bits: (1u128 << (ty.bits() - 1)) - 1 }
}
