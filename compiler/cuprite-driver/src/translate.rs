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
//! Supported values: integers, floats and `bool`, and structs and tuples of
//! them, with references to any of those. Enums, arrays, slices, raw
//! pointer arithmetic, closures and statics are refused with a reason.
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
        AggregateKind, AssertKind, BasicBlock, BinOp, Body, CastKind, ConstOperand, ConstValue, Local, Operand, Place,
        ProjectionElem, Rvalue, StatementKind, TerminatorKind, UnOp, RETURN_PLACE, START_BLOCK,
    },
    ty::{self, FloatTy, Instance, InstanceKind, IntTy, Ty, TyCtxt, TypingEnv, UintTy},
};
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

/// A scalar constant of a supported type, evaluating it if it's an
/// unevaluated constant such as an associated `const`.
pub(crate) fn constant<'tcx>(tcx: TyCtxt<'tcx>, constant: &ConstOperand<'tcx>) -> Option<Const> {
    let ty = kernel_ty(constant.ty())?;
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
enum IntrinsicOp {
    Method(Method),
    /// `likely` and `unlikely`: their argument.
    Identity,
    /// `cold_path` and `assume`: nothing.
    Nothing,
}

/// The intrinsics library code reaches after inlining, by name.
fn intrinsic(name: &str) -> Option<IntrinsicOp> {
    use Intrinsic::*;
    let call = |i| Some(IntrinsicOp::Method(Method::Intrinsic(i)));
    let binary = |op| Some(IntrinsicOp::Method(Method::Binary(op)));
    match name {
        "sqrtf32" | "sqrtf64" => call(Sqrt),
        "floorf32" | "floorf64" => call(Floor),
        "ceilf32" | "ceilf64" => call(Ceil),
        "truncf32" | "truncf64" => call(Trunc),
        "fabsf32" | "fabsf64" => call(Abs),
        "fmaf32" | "fmaf64" => call(MulAdd),
        "minnumf32" | "minnumf64" => call(Min),
        "maxnumf32" | "maxnumf64" => call(Max),
        "rotate_left" => call(RotateLeft),
        "rotate_right" => call(RotateRight),
        "ctlz" => call(LeadingZeros),
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
        "likely" | "unlikely" => Some(IntrinsicOp::Identity),
        "cold_path" | "assume" => Some(IntrinsicOp::Nothing),
        _ => None,
    }
}

fn describe_assert<O>(msg: &AssertKind<O>) -> String {
    match msg {
        AssertKind::BoundsCheck { .. } => "index out of bounds".into(),
        AssertKind::Overflow(op, ..) => format!("`{op:?}` can overflow"),
        AssertKind::OverflowNeg(_) => "negation can overflow".into(),
        AssertKind::DivisionByZero(_) => "division by zero".into(),
        AssertKind::RemainderByZero(_) => "remainder by zero".into(),
        _ => "an assertion".into(),
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

/// What a loop body reads and writes, around the translation.
pub(crate) struct Loop<'a, 'tcx> {
    /// `next`'s result: `(option as Some).0` is the element reference.
    pub(crate) option: Place<'tcx>,
    /// The first block of the body.
    pub(crate) start: BasicBlock,
    /// True for the loop header's blocks, where an iteration ends.
    pub(crate) is_header: &'a dyn Fn(BasicBlock) -> bool,
    /// Locals assigned once, before the loop, from a constant.
    pub(crate) constants: &'a HashMap<Local, Const>,
    /// The enclosing function's parameters the body may read, in the order
    /// the element function takes them after the element.
    pub(crate) scalars: &'a [Local],
}

/// A translated loop body.
pub(crate) struct Translated {
    pub(crate) function: Function,
    /// The body's blocks.
    pub(crate) blocks: HashSet<BasicBlock>,
    /// Whether the body writes the element at all.
    pub(crate) stores: bool,
}

/// Translates the loop body.
pub(crate) fn translate<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    name: &str,
    element: KTy,
    shape: &Loop<'_, 'tcx>,
) -> Result<Translated> {
    let region = loop_region(body, shape)?;
    let mut params = vec![element];
    for &local in shape.scalars {
        params.push(kernel_ty(body.local_decls[local].ty).ok_or("takes a parameter a kernel can't use")?);
    }
    let (mut b, kernel_params) = Builder::new(name, &params);
    let value = b.copy(kernel_params[0]);
    let mut translator = Translator {
        tcx,
        b,
        element: value,
        stores: false,
        stack: Vec::new(),
    };
    let mut slots: HashMap<Local, Slot> =
        shape.scalars.iter().zip(&kernel_params[1..]).map(|(&local, &kernel)| (local, Slot::Scalar(kernel))).collect();
    // `next` returned `Some(&mut element)`.
    slots.insert(shape.option.local, Slot::Some(Box::new(Slot::Ref(Box::new(Slot::Scalar(value))))));
    let mut frame = Frame {
        body,
        slots,
        blocks: HashMap::new(),
        exit: BlockId(0),
        loop_rules: Some(LoopRules {
            shape,
            assigned_outside: assigned_outside(body, &region),
        }),
    };
    frame.exit = translator.b.block();
    translator.run_frame(&mut frame, &ordered(body, &region), shape.start)?;
    let stores = translator.stores;
    let function = translator
        .b
        .finish_function(Some(value))
        .map_err(|invalid| format!("translated into an invalid kernel ({invalid})"))?;
    Ok(Translated {
        function,
        blocks: region,
        stores,
    })
}

/// The loop body's blocks: everything reachable from its start without
/// passing the header. Leaving the loop any other way (`break`, `return`)
/// makes later elements depend on earlier ones, so it is refused.
fn loop_region(body: &Body<'_>, shape: &Loop<'_, '_>) -> Result<HashSet<BasicBlock>> {
    let mut region = HashSet::new();
    let mut pending = vec![shape.start];
    while let Some(block) = pending.pop() {
        if (shape.is_header)(block) || !region.insert(block) {
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
    /// The loop's `Option<&mut T>` from `next`, known to be `Some`.
    Some(Box<Slot>),
}

impl Slot {
    fn scalars(&self, out: &mut Vec<KLocal>) {
        match self {
            Slot::Scalar(local) => out.push(*local),
            Slot::Fields(fields) => fields.iter().for_each(|field| field.scalars(out)),
            Slot::Ref(inner) | Slot::Some(inner) => inner.scalars(out),
            Slot::Unit | Slot::Unbound => {}
        }
    }
}

/// The body-specific rules of the loop body, the outermost frame.
struct LoopRules<'a, 'tcx> {
    shape: &'a Loop<'a, 'tcx>,
    assigned_outside: HashSet<Local>,
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
}

/// How deep inlining may go.
const MAX_DEPTH: usize = 24;

fn is_ref(ty: Ty<'_>) -> bool {
    matches!(ty.kind(), ty::Ref(..) | ty::RawPtr(..))
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
        for &block in region {
            self.b.switch_to(frame.blocks[&block]);
            self.block(frame, block)?;
        }
        Ok(())
    }

    fn target(&self, frame: &Frame<'_, 'tcx>, block: BasicBlock) -> Result<BlockId> {
        if frame.loop_rules.as_ref().is_some_and(|rules| (rules.shape.is_header)(block)) {
            return Ok(frame.exit);
        }
        frame.blocks.get(&block).copied().ok_or_else(|| "has an unsupported loop shape".into())
    }

    fn set(&mut self, local: KLocal, rvalue: KRvalue) {
        if local == self.element {
            self.stores = true;
        }
        self.b.set(local, rvalue);
    }

    fn block(&mut self, frame: &mut Frame<'_, 'tcx>, block: BasicBlock) -> Result<()> {
        let data = &frame.body.basic_blocks[block];
        for statement in &data.statements {
            match &statement.kind {
                StatementKind::Assign(assign) => self.assign(frame, &assign.0, &assign.1)?,
                StatementKind::Intrinsic(intrinsic) => {
                    if !matches!(**intrinsic, rustc_middle::mir::NonDivergingIntrinsic::Assume(_)) {
                        return Err("copies memory".into());
                    }
                }
                StatementKind::SetDiscriminant { .. } => return Err("uses an enum".into()),
                // Storage markers, borrow-checker annotations and no-ops.
                _ => {}
            }
        }
        match &data.terminator().kind {
            TerminatorKind::Goto { target, .. }
            | TerminatorKind::FalseEdge { real_target: target, .. }
            | TerminatorKind::FalseUnwind { real_target: target, .. }
            // Only assertions that always pass get this far.
            | TerminatorKind::Assert { target, .. } => {
                let target = self.target(frame, *target)?;
                self.b.goto(target);
            }
            TerminatorKind::SwitchInt { discr, targets } => {
                let value = self.scalar_operand(frame, discr)?;
                let ty = self.b.ty(value);
                for (case, target) in targets.iter() {
                    let target = self.target(frame, target)?;
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
                let otherwise = self.target(frame, targets.otherwise())?;
                self.b.goto(otherwise);
            }
            TerminatorKind::Return | TerminatorKind::Unreachable => self.b.goto(frame.exit),
            TerminatorKind::Call { func, args, destination, target: Some(target), .. } => {
                self.call(frame, func, args, destination)?;
                let target = self.target(frame, *target)?;
                self.b.goto(target);
            }
            TerminatorKind::Call { target: None, .. } => return Err("may panic or never return".into()),
            TerminatorKind::Drop { .. } => return Err("runs a destructor".into()),
            other => return Err(format!("contains unsupported control flow `{other:?}`")),
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
        let (def_id, generic_args) = func.const_fn_def().ok_or("calls a function pointer or closure value")?;
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
                let op = intrinsic(name.as_str()).ok_or_else(|| format!("uses the intrinsic `{name}`"))?;
                self.operation(frame, op, args, destination)
            }
            InstanceKind::Item(item) if tcx.is_mir_available(item) => {
                self.inline(frame, instance, args, destination, &path())
            }
            InstanceKind::Item(_) => Err(format!("calls `{}`, whose MIR isn't available", path())),
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
        let args = args.iter().map(|arg| self.scalar_operand(frame, &arg.node)).collect::<Result<Vec<_>>>()?;
        let rvalue = match (op, args.as_slice()) {
            (IntrinsicOp::Method(Method::Binary(op)), [a, b]) => KRvalue::Binary(op, *a, *b),
            (IntrinsicOp::Method(Method::Neg), [a]) => KRvalue::Unary(KUnOp::Neg, *a),
            (IntrinsicOp::Method(Method::Intrinsic(intrinsic)), _) => KRvalue::Call(intrinsic, args.clone()),
            (IntrinsicOp::Identity, [a]) => KRvalue::Use(*a),
            _ => return Err("calls a library function with unexpected arguments".into()),
        };
        let Slot::Scalar(local) = self.place(frame, destination)? else {
            return Err("returns a library result into something other than a number".into());
        };
        self.set(local, rvalue);
        Ok(())
    }

    /// Translates a call to `instance` in place.
    fn inline(
        &mut self,
        frame: &mut Frame<'_, 'tcx>,
        instance: Instance<'tcx>,
        args: &[rustc_span::Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        path: &str,
    ) -> Result<()> {
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
        if body.spread_arg.is_some() {
            return Err(within("takes its arguments as a tuple".into()));
        }
        let region = function_region(&body);
        for &block in &region {
            if let TerminatorKind::Assert { msg, .. } = &body.basic_blocks[block].terminator().kind {
                if !assert_holds(&body, block) {
                    return Err(within(format!("may panic: {}", describe_assert(&**msg))));
                }
            }
        }
        // Arguments are read in the caller's frame, before the call.
        let mut slots = HashMap::new();
        for (index, arg) in args.iter().enumerate() {
            let param = Local::from_usize(index + 1);
            let value = self.operand_slot(frame, &arg.node)?;
            let slot = if is_ref(body.local_decls[param].ty) {
                value
            } else {
                let slot = self.fresh(body.local_decls[param].ty).map_err(within)?;
                self.copy(&slot, &value)?;
                slot
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
        let result = self.local_slot(&mut callee, RETURN_PLACE).map_err(within)?;
        self.store(frame, destination, result)
    }

    /// The slot for a fresh value of type `ty`.
    fn fresh(&mut self, ty: Ty<'tcx>) -> Result<Slot> {
        if let Some(kernel) = kernel_ty(ty) {
            return Ok(Slot::Scalar(self.b.local(kernel)));
        }
        if is_ref(ty) {
            return Ok(Slot::Unbound);
        }
        let fields = self.fields(ty)?;
        if fields.is_empty() {
            return Ok(Slot::Unit);
        }
        Ok(Slot::Fields(fields.into_iter().map(|field| self.fresh(field)).collect::<Result<_>>()?))
    }

    /// The field types of a struct or tuple.
    fn fields(&self, ty: Ty<'tcx>) -> Result<Vec<Ty<'tcx>>> {
        let tcx = self.tcx;
        match ty.kind() {
            ty::Tuple(fields) => Ok(fields.to_vec()),
            ty::Adt(adt, args) if adt.is_struct() && !adt.repr().simd() => Ok(adt
                .non_enum_variant()
                .fields
                .iter()
                .map(|field| tcx.normalize_erasing_regions(self.typing_env(), field.ty(tcx, args)))
                .collect()),
            ty::Adt(adt, _) if adt.is_enum() => Err(format!("uses the enum `{ty}`")),
            ty::Never => Ok(Vec::new()),
            _ => Err(format!("uses a value of type `{ty}`")),
        }
    }

    /// The slot of a local, created on first use.
    fn local_slot(&mut self, frame: &mut Frame<'_, 'tcx>, local: Local) -> Result<Slot> {
        if let Some(slot) = frame.slots.get(&local) {
            return Ok(slot.clone());
        }
        if let Some(rules) = &frame.loop_rules {
            if let Some(&constant) = rules.shape.constants.get(&local) {
                // A fresh copy each time: it is read wherever it is used.
                let copy = self.b.copy(constant);
                return Ok(Slot::Scalar(copy));
            }
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
            slot = match (element, slot) {
                (ProjectionElem::Deref, Slot::Ref(inner)) => *inner,
                (ProjectionElem::Deref, _) => return Err("reads through a reference it can't follow".into()),
                (ProjectionElem::Field(field, _), Slot::Fields(fields)) => {
                    fields.get(field.as_usize()).cloned().ok_or("uses a field that doesn't exist")?
                }
                (ProjectionElem::Downcast(_, variant), Slot::Some(inner)) if variant.as_u32() == 1 => {
                    Slot::Fields(vec![*inner])
                }
                (ProjectionElem::Downcast(..), _) => return Err("uses an enum".into()),
                (ProjectionElem::Index(_) | ProjectionElem::ConstantIndex { .. } | ProjectionElem::Subslice { .. }, _) => {
                    return Err("indexes an array or slice".into());
                }
                _ => return Err(format!("uses a place the kernel can't follow (`{place:?}`)")),
            };
        }
        Ok(slot)
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
        self.copy(&dest, &value)
    }

    /// Copies `src` into `dest`, field by field. If they share kernel
    /// locals (`a = (a.1, a.0)`), through temporaries.
    fn copy(&mut self, dest: &Slot, src: &Slot) -> Result<()> {
        let (mut written, mut read) = (Vec::new(), Vec::new());
        dest.scalars(&mut written);
        src.scalars(&mut read);
        if dest != src && written.iter().any(|local| read.contains(local)) && !matches!(dest, Slot::Scalar(_)) {
            let temporary = self.snapshot(src)?;
            return self.copy_fields(dest, &temporary);
        }
        self.copy_fields(dest, src)
    }

    fn copy_fields(&mut self, dest: &Slot, src: &Slot) -> Result<()> {
        match (dest, src) {
            (Slot::Scalar(d), Slot::Scalar(s)) => {
                if d != s {
                    self.set(*d, KRvalue::Use((*s).into()));
                }
                Ok(())
            }
            (Slot::Fields(d), Slot::Fields(s)) if d.len() == s.len() => {
                d.iter().zip(s).try_for_each(|(d, s)| self.copy_fields(d, s))
            }
            (Slot::Unit, Slot::Unit) => Ok(()),
            (Slot::Ref(d), Slot::Ref(s)) if d == s => Ok(()),
            (Slot::Ref(_) | Slot::Unbound, Slot::Ref(_)) => Err("stores a reference inside a value".into()),
            _ => Err("copies between values of different shapes".into()),
        }
    }

    /// A copy of `src`'s numbers in new locals.
    fn snapshot(&mut self, src: &Slot) -> Result<Slot> {
        Ok(match src {
            Slot::Scalar(local) => Slot::Scalar(self.b.copy(*local)),
            Slot::Fields(fields) => Slot::Fields(fields.iter().map(|field| self.snapshot(field)).collect::<Result<_>>()?),
            other => other.clone(),
        })
    }

    fn assign(&mut self, frame: &mut Frame<'_, 'tcx>, place: &Place<'tcx>, rvalue: &Rvalue<'tcx>) -> Result<()> {
        let ty = place.ty(frame.body, self.tcx).ty;
        if is_ref(ty) {
            let target = self.reference(frame, rvalue)?;
            return self.store(frame, place, target);
        }
        if kernel_ty(ty).is_some() {
            let value = self.scalar_rvalue(frame, rvalue)?;
            let Slot::Scalar(local) = self.place(frame, place)? else {
                return Err("writes a number into something else".into());
            };
            self.set(local, value);
            return Ok(());
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
            Rvalue::Aggregate(kind, operands) => {
                match **kind {
                    AggregateKind::Tuple => {}
                    AggregateKind::Adt(adt, _, _, _, None) if self.tcx.adt_def(adt).is_struct() => {}
                    AggregateKind::Adt(..) => return Err("builds an enum or union".into()),
                    AggregateKind::Array(_) => return Err("builds an array".into()),
                    _ => return Err("builds a closure or coroutine".into()),
                }
                let values: Vec<Slot> =
                    operands.iter().map(|operand| self.operand_slot(frame, operand)).collect::<Result<_>>()?;
                let value = if values.is_empty() { Slot::Unit } else { Slot::Fields(values) };
                self.store(frame, place, value)
            }
            _ => Err("uses an unsupported operation".into()),
        }
    }

    /// The slot a reference-valued rvalue points to, as `Slot::Ref`.
    fn reference(&mut self, frame: &mut Frame<'_, 'tcx>, rvalue: &Rvalue<'tcx>) -> Result<Slot> {
        match rvalue {
            Rvalue::Ref(_, _, place) | Rvalue::RawPtr(_, place) => Ok(Slot::Ref(Box::new(self.place(frame, place)?))),
            Rvalue::Use(operand, _) => self.operand_slot(frame, operand),
            Rvalue::CopyForDeref(place) => self.place(frame, place),
            Rvalue::Cast(..) => Err("converts a pointer".into()),
            _ => Err("uses a reference it can't follow".into()),
        }
    }

    fn scalar_rvalue(&mut self, frame: &mut Frame<'_, 'tcx>, rvalue: &Rvalue<'tcx>) -> Result<KRvalue> {
        Ok(match rvalue {
            Rvalue::Use(operand, _) => KRvalue::Use(self.scalar_operand(frame, operand)?),
            Rvalue::CopyForDeref(place) => match self.place(frame, place)? {
                Slot::Scalar(local) => KRvalue::Use(local.into()),
                _ => return Err("reads something other than a number".into()),
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
                    other => return Err(format!("uses `{other:?}`, which is not supported")),
                };
                KRvalue::Binary(op, a, b)
            }
            Rvalue::UnaryOp(op, operand) => {
                let a = self.scalar_operand(frame, operand)?;
                match op {
                    UnOp::Neg => KRvalue::Unary(KUnOp::Neg, a),
                    UnOp::Not => KRvalue::Unary(KUnOp::Not, a),
                    other => return Err(format!("uses `{other:?}`, which is not supported")),
                }
            }
            Rvalue::Cast(kind, operand, ty) => {
                let numeric = matches!(
                    kind,
                    CastKind::IntToInt | CastKind::IntToFloat | CastKind::FloatToInt | CastKind::FloatToFloat
                );
                let to = kernel_ty(*ty).filter(|_| numeric).ok_or_else(|| format!("casts to `{ty}`"))?;
                KRvalue::Cast(self.scalar_operand(frame, operand)?, to)
            }
            Rvalue::Ref(..) | Rvalue::RawPtr(..) => {
                return Err("borrows something other than the current element".into());
            }
            Rvalue::Discriminant(_) => return Err("uses an enum".into()),
            _ => return Err("uses an unsupported operation".into()),
        })
    }

    fn scalar_operand(&mut self, frame: &mut Frame<'_, 'tcx>, operand: &Operand<'tcx>) -> Result<KOperand> {
        match operand {
            Operand::Constant(c) => constant(self.tcx, c)
                .map(KOperand::Const)
                .ok_or_else(|| format!("uses an unsupported constant of type `{}`", c.ty())),
            Operand::Copy(place) | Operand::Move(place) => {
                if let Some(rules) = &frame.loop_rules {
                    if place.projection.is_empty() {
                        if let Some(&c) = rules.shape.constants.get(&place.local) {
                            return Ok(c.into());
                        }
                    }
                }
                match self.place(frame, place)? {
                    Slot::Scalar(local) => Ok(local.into()),
                    _ => Err("uses a value as a number that isn't one".into()),
                }
            }
            Operand::RuntimeChecks(_) => Err("reads a runtime-check flag".into()),
        }
    }

    /// The slot of an operand of any type. Constants get new locals.
    fn operand_slot(&mut self, frame: &mut Frame<'_, 'tcx>, operand: &Operand<'tcx>) -> Result<Slot> {
        match operand {
            Operand::Constant(c) => self.constant_slot(c),
            Operand::Copy(place) | Operand::Move(place) => self.place(frame, place),
            Operand::RuntimeChecks(_) => Err("reads a runtime-check flag".into()),
        }
    }

    /// A constant of any supported type, in new locals.
    fn constant_slot(&mut self, c: &ConstOperand<'tcx>) -> Result<Slot> {
        let ty = c.ty();
        if let Some(scalar) = constant(self.tcx, c) {
            return Ok(Slot::Scalar(self.b.copy(scalar)));
        }
        if is_ref(ty) {
            return Err(format!("uses a reference to constant data (`{ty}`)"));
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

    /// Sets a slot's locals from a constant's bytes, laid out as rustc lays
    /// out `ty`, little-endian.
    fn fill(&mut self, slot: &Slot, ty: Ty<'tcx>, bytes: &[u8], base: usize) -> Result<()> {
        match slot {
            Slot::Scalar(local) => {
                let kernel = kernel_ty(ty).ok_or("uses a constant of an unexpected type")?;
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
            Slot::Unit => Ok(()),
            _ => Err("uses a constant holding a reference".into()),
        }
    }
}
