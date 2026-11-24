//! Rust scalar types, plain-data parameters, and slice element layouts.
//!
//! # Slices of structs, and several outputs
//!
//! A map's slices may hold numbers, or structs, tuples and arrays of them
//! with no padding ([`slice()`]). Each slice's element is a [`Layout`]: its
//! numbers' offsets, in the order the translator nests them. The
//! element function takes every number of each written slice's element,
//! then of each read slice's, rebuilds the elements as values of their
//! types, runs the loop body on them, and returns the written slices'
//! numbers. The kernel loads and stores them at their offsets
//! (`shapes::in_place_map`). Other `&mut` slices zipped with the first are
//! written the same way; the runtime commits all of them or none.

use cuprite_kernel::{layout::Layout, Const, Ty as KTy};
use rustc_middle::mir::{interpret::Scalar, ConstOperand, ConstValue, Local};
use rustc_middle::ty::{self, FloatTy, IntTy, Ty, TyCtxt, TypingEnv, UintTy};
use rustc_span::def_id::DefId;

use super::refusal::type_refusal;
use super::Result;

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
pub(super) const MAX_ARRAY: u64 = 1024;

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

/// A slice parameter a map iterates, and where the numbers of its element
/// sit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Slice {
    pub(crate) param: Local,
    pub(crate) layout: Layout,
    /// For each of the layout's numbers, the field indices from the element
    /// to it, as `Translator::fresh` nests its slots.
    pub(crate) paths: Vec<Vec<usize>>,
}

/// The element type of a slice reference.
pub(crate) fn slice_element(ty: Ty<'_>) -> Option<Ty<'_>> {
    match ty.kind() {
        ty::Ref(_, inner, _) => match inner.kind() {
            ty::Slice(element) => Some(*element),
            _ => None,
        },
        _ => None,
    }
}

/// A slice parameter's element layout: a number, or a struct, tuple or
/// array of numbers with no padding, whose bytes can move between the CPU
/// and the GPU as they are. The numbers are in the order `fresh` nests
/// them.
pub(crate) fn slice<'tcx>(tcx: TyCtxt<'tcx>, param: Local, element: Ty<'tcx>) -> Result<Slice> {
    fn walk<'tcx>(
        tcx: TyCtxt<'tcx>,
        ty: Ty<'tcx>,
        offset: u64,
        path: &mut Vec<usize>,
        out: &mut Vec<(u64, KTy, Vec<usize>)>,
    ) -> Result<()> {
        if let Some(kernel) = kernel_ty(ty).filter(|kernel| *kernel != KTy::Bool) {
            out.push((offset, kernel, path.clone()));
            return Ok(());
        }
        let aggregate = match ty.kind() {
            ty::Tuple(_) | ty::Array(..) => true,
            ty::Adt(adt, _) => adt.is_struct() && !adt.is_box() && !adt.repr().simd(),
            _ => false,
        };
        if !aggregate {
            return Err(format!("holds `{ty}`"));
        }
        let layout = tcx
            .layout_of(TypingEnv::fully_monomorphized().as_query_input(ty))
            .map_err(|_| format!("holds `{ty}`, which has no layout"))?;
        for (index, field) in fields_of(tcx, ty)?.into_iter().enumerate() {
            path.push(index);
            walk(tcx, field, offset + layout.fields.offset(index).bytes(), path, out)?;
            path.pop();
            if out.len() > MAX_LEAVES {
                return Err(format!("holds more than {MAX_LEAVES} numbers"));
            }
        }
        Ok(())
    }
    let refusal = |reason: String| format!("iterates over `[{element}]`, whose element {reason}");
    let mut found = Vec::new();
    walk(tcx, element, 0, &mut Vec::new(), &mut found).map_err(refusal)?;
    let size = tcx
        .layout_of(TypingEnv::fully_monomorphized().as_query_input(element))
        .map_err(|_| refusal("has no layout".into()))?
        .size
        .bytes();
    let to_u32 = |value: u64| u32::try_from(value).map_err(|_| refusal("is too large".into()));
    let mut leaves = Vec::new();
    let mut paths = Vec::new();
    for (offset, ty, path) in found {
        leaves.push((to_u32(offset)?, ty));
        paths.push(path);
    }
    let layout = Layout { size: to_u32(size)?, leaves };
    if layout.leaves.is_empty() {
        return Err(refusal("holds no numbers".into()));
    }
    // Every number is aligned in a Rust value; what can fail is coverage.
    layout.validate().map_err(|_| refusal("has padding".into()))?;
    Ok(Slice { param, layout, paths })
}
