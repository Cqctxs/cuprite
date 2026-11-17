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
//! Supported values: integers, floats, `bool` and `char`, structs, tuples,
//! arrays and enums of them, closures, with references to any of those.
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

/// The kernel type a number is held in inside a kernel: as `kernel_ty`, and
/// a `char` as its `u32` code, a pattern type (`NonZero`'s field) as the
/// type it restricts. Only valid values ever reach these, so the kernel
/// computes on them as the CPU does. Parameters and slices take only
/// `kernel_ty`'s types.
pub(crate) fn held_ty(ty: Ty<'_>) -> Option<KTy> {
    match ty.kind() {
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
    let bits = u64::try_from(scalar.to_bits(scalar.size())).ok()?;
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
        ty::Adt(adt, args) if adt.is_struct() && !adt.repr().simd() && !adt.is_box() => Ok(adt
            .non_enum_variant()
            .fields
            .iter()
            .map(|field| tcx.normalize_erasing_regions(TypingEnv::fully_monomorphized(), field.ty(tcx, args)))
            .collect()),
        ty::Never => Ok(Vec::new()),
        _ => Err(type_refusal(ty)),
    }
}

/// One number read from a parameter of plain data, passed to the kernel
/// as an argument.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Leaf {
    pub(crate) param: Local,
    /// The parameter is a shared reference, read through first.
    pub(crate) deref: bool,
    /// Field indices from the parameter (or what it points to) to the
    /// number; empty for a number itself.
    pub(crate) fields: Vec<usize>,
    pub(crate) ty: KTy,
}

impl Leaf {
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
    Ok(found.into_iter().map(|(fields, ty)| Leaf { param, deref, fields, ty }).collect())
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
        ty::Int(IntTy::I128) | ty::Uint(UintTy::U128) => None,
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
}

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
    let function = translator
        .b
        .finish_function(Some(value))
        .map_err(|invalid| format!("translated into an invalid kernel ({invalid})"))?;
    blocks.retain(|block| !shape.chain.contains(block));
    Ok(Translated {
        function,
        blocks,
        stores,
    })
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
    let mut translator = Translator { tcx, b, element, stores: false, stack: vec![def_id], written: HashSet::new() };
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
}

/// How deep inlining may go.
const MAX_DEPTH: usize = 24;

fn is_ref(ty: Ty<'_>) -> bool {
    matches!(ty.kind(), ty::Ref(..) | ty::RawPtr(..))
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
                    let case = Const { ty, bits: case as u64 };
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
        for element in place.projection.iter() {
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
            Slot::Unit | Slot::Some(_) | Slot::Opaque | Slot::Choice { .. } => return Err(shapes()),
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
                let value = self.place(frame, source)?;
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
            Rvalue::Cast(kind, _, ty) => Err(cast_refusal(*kind, *ty)),
            _ => Err("uses a reference it can't follow".into()),
        }
    }

    fn scalar_rvalue(&mut self, frame: &mut Frame<'_, 'tcx>, rvalue: &Rvalue<'tcx>) -> Result<KRvalue> {
        Ok(match rvalue {
            Rvalue::Use(operand, _) => KRvalue::Use(self.scalar_operand(frame, operand)?),
            Rvalue::CopyForDeref(place) => match self.place(frame, place)? {
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
                let slot = self.place(frame, place)?;
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
                let slot = self.place(frame, place)?;
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
                let min = Const::int(ty, i128::from(i64::MIN));
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
            Operand::Copy(place) | Operand::Move(place) => match self.place(frame, place)? {
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
        self.fill(&slot, ty, &bytes, 0)?;
        Ok(slot)
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
                let mut buffer = [0u8; 8];
                buffer[..size].copy_from_slice(bytes.get(base..base + size).ok_or("reads past a constant")?);
                let bits = u64::from_le_bytes(buffer);
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
            Slot::Scalar(number) => {
                let size = u64::from(self.b.ty(*number).bytes());
                if base >= offset + width || offset >= base + size {
                    return Ok(zero);
                }
                let wide = self.unsigned_bits(*number);
                self.place_bits(wide.into(), base, offset).into()
            }
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
                            TagEncoding::Direct => Some(self.discriminant(ty, VariantIdx::from_usize(index))?.bits),
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
        _ => KTy::U64,
    }
}

/// The signed kernel integer of `bytes` bytes.
fn signed_of(bytes: u64) -> KTy {
    match bytes {
        1 => KTy::I8,
        2 => KTy::I16,
        4 => KTy::I32,
        _ => KTy::I64,
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
