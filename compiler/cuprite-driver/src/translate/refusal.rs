//! Reasons for declining MIR constructs. Exhaustive matches flag new variants on toolchain upgrades.

use rustc_middle::mir::{AggregateKind, BinOp, CastKind, ProjectionElem, Rvalue, UnOp};
use rustc_middle::ty::{self, adjustment::PointerCoercion, Ty};

/// Why an operation can't be translated where it appears.
pub(super) fn rvalue_refusal(rvalue: &Rvalue<'_>) -> String {
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

pub(super) fn binary_refusal(op: BinOp) -> String {
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
pub(super) fn cast_refusal(kind: CastKind, ty: Ty<'_>) -> String {
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

pub(super) fn aggregate_refusal(kind: &AggregateKind<'_>) -> String {
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
pub(super) fn projection_refusal<V, T>(element: &ProjectionElem<V, T>) -> String {
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
pub(super) fn type_refusal(ty: Ty<'_>) -> String {
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
