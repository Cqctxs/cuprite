//! Translation of a loop body's MIR into a kernel element function
//! (`cuprite_kernel::Function`): what one iteration does to its element.
//!
//! The function takes the element's value and the enclosing function's
//! scalar parameters, and returns the element's new value. Inside, the
//! body's scalar locals become kernel locals, its blocks kernel blocks, and
//! every jump back to the loop header returns.
//!
//! Supported: integer, float and `bool` locals and constants; arithmetic,
//! bit operations, shifts and comparisons; `as` casts between them;
//! branches, `match` on integers and inner `while` loops; and the library
//! methods in [`method`]. Anything else is refused with a reason.
//!
//! Panics must be provable away: [`assert_holds`] accepts an assertion only
//! when its condition folds to a constant that passes, as for a division by
//! a nonzero constant. Plain `+` and `*` are only reached without overflow
//! checks, where they wrap, as the kernel's operations do.

use std::collections::{HashMap, HashSet};

use cuprite_kernel::{
    builder::Builder, BinOp as KBinOp, BlockId, Const, Function, Intrinsic, Local as KLocal, Operand as KOperand,
    Rvalue as KRvalue, Ty as KTy, UnOp as KUnOp,
};
use rustc_middle::{
    mir::{
        BasicBlock, BinOp, Body, CastKind, ConstOperand, Local, Operand, Place, ProjectionElem, Rvalue, StatementKind,
        TerminatorKind, UnOp,
    },
    ty::{self, FloatTy, IntTy, Ty, TyCtxt, UintTy},
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

/// A scalar constant of a supported type.
pub(crate) fn constant(constant: &ConstOperand<'_>) -> Option<Const> {
    let ty = kernel_ty(constant.ty())?;
    let scalar = constant.const_.try_to_scalar_int()?;
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
    let region = region(body, shape)?;
    let mut params = vec![element];
    for &local in shape.scalars {
        params.push(kernel_ty(body.local_decls[local].ty).ok_or("takes a parameter a kernel can't use")?);
    }
    let (mut b, kernel_params) = Builder::new(name, &params);
    let value = b.copy(kernel_params[0]);
    let mut translator = Translator {
        tcx,
        body,
        shape,
        b,
        element: value,
        locals: shape.scalars.iter().copied().zip(kernel_params[1..].iter().copied()).collect(),
        refs: HashSet::new(),
        blocks: HashMap::new(),
        exit: BlockId(0),
        assigned_outside: assigned_outside(body, &region),
        region: &region,
        stores: false,
    };
    translator.run()?;
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

/// The body's blocks: everything reachable from its start without passing
/// the header. Leaving the loop any other way (`break`, `return`) makes
/// later elements depend on earlier ones, so it is refused.
fn region(body: &Body<'_>, shape: &Loop<'_, '_>) -> Result<HashSet<BasicBlock>> {
    let mut region = HashSet::new();
    let mut pending = vec![shape.start];
    while let Some(block) = pending.pop() {
        if (shape.is_header)(block) || !region.insert(block) {
            continue;
        }
        let terminator = body.basic_blocks[block].terminator();
        match &terminator.kind {
            TerminatorKind::Return => return Err("leaves the loop early".into()),
            TerminatorKind::Goto { target, .. }
            | TerminatorKind::FalseEdge { real_target: target, .. }
            | TerminatorKind::FalseUnwind { real_target: target, .. }
            | TerminatorKind::Assert { target, .. }
            | TerminatorKind::Call { target: Some(target), .. } => pending.push(*target),
            TerminatorKind::SwitchInt { targets, .. } => pending.extend(targets.all_targets()),
            TerminatorKind::Unreachable => {}
            other => return Err(format!("contains unsupported control flow `{other:?}` in the loop body")),
        }
    }
    // The loop's exit is reached only through the header.
    for &block in &region {
        let data = &body.basic_blocks[block];
        if data.is_cleanup {
            return Err("has an unsupported loop shape".into());
        }
    }
    Ok(region)
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

struct Translator<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    shape: &'a Loop<'a, 'tcx>,
    b: Builder,
    /// The element's current value.
    element: KLocal,
    /// MIR locals and the kernel locals holding them.
    locals: HashMap<Local, KLocal>,
    /// Locals holding a reference to the element.
    refs: HashSet<Local>,
    blocks: HashMap<BasicBlock, BlockId>,
    /// Ends an iteration: returns the element.
    exit: BlockId,
    assigned_outside: HashSet<Local>,
    region: &'a HashSet<BasicBlock>,
    stores: bool,
}

impl<'tcx> Translator<'_, 'tcx> {
    fn run(&mut self) -> Result<()> {
        self.blocks()
    }

    fn blocks(&mut self) -> Result<()> {
        let mut order: Vec<BasicBlock> = self.region.iter().copied().collect();
        order.sort();
        for &block in &order {
            let id = self.b.block();
            self.blocks.insert(block, id);
        }
        self.exit = self.b.block();
        let start = self.blocks[&self.shape.start];
        self.b.goto(start);
        for block in order {
            self.b.switch_to(self.blocks[&block]);
            self.block(block)?;
        }
        Ok(())
    }

    fn target(&self, block: BasicBlock) -> Result<BlockId> {
        if (self.shape.is_header)(block) {
            return Ok(self.exit);
        }
        self.blocks.get(&block).copied().ok_or_else(|| "has an unsupported loop shape".into())
    }

    fn block(&mut self, block: BasicBlock) -> Result<()> {
        let data = &self.body.basic_blocks[block];
        for statement in &data.statements {
            let StatementKind::Assign(assign) = &statement.kind else {
                continue; // Storage markers and other no-ops.
            };
            self.assign(&assign.0, &assign.1)?;
        }
        match &data.terminator().kind {
            TerminatorKind::Goto { target, .. }
            | TerminatorKind::FalseEdge { real_target: target, .. }
            | TerminatorKind::FalseUnwind { real_target: target, .. }
            // Only assertions that always pass get this far.
            | TerminatorKind::Assert { target, .. } => {
                let target = self.target(*target)?;
                self.b.goto(target);
            }
            TerminatorKind::SwitchInt { discr, targets } => {
                let value = self.operand(discr)?;
                let ty = self.b.ty(value);
                for (case, target) in targets.iter() {
                    let target = self.target(target)?;
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
                let otherwise = self.target(targets.otherwise())?;
                self.b.goto(otherwise);
            }
            TerminatorKind::Call { func, args, destination, target: Some(target), .. } => {
                let (def_id, _) = func.const_fn_def().ok_or("calls a function value")?;
                let method = method(self.tcx, def_id).ok_or("calls a function with no GPU summary")?;
                let args = args.iter().map(|arg| self.operand(&arg.node)).collect::<Result<Vec<_>>>()?;
                let rvalue = match (method, args.as_slice()) {
                    (Method::Binary(op), [a, b]) => KRvalue::Binary(op, *a, *b),
                    (Method::Neg, [a]) => KRvalue::Unary(KUnOp::Neg, *a),
                    (Method::Intrinsic(intrinsic), _) => KRvalue::Call(intrinsic, args.clone()),
                    _ => return Err("calls a library method with unexpected arguments".into()),
                };
                let local = self.scalar_destination(destination)?;
                self.b.set(local, rvalue);
                let target = self.target(*target)?;
                self.b.goto(target);
            }
            // Never reached; ending the iteration keeps the kernel well formed.
            TerminatorKind::Unreachable => self.b.goto(self.exit),
            _ => return Err("has an unsupported loop shape".into()),
        }
        Ok(())
    }

    fn is_element_ref(&self, ty: Ty<'tcx>) -> bool {
        matches!(ty.kind(), ty::Ref(..))
    }

    /// The kernel local for a scalar MIR local the body assigns.
    fn scalar_destination(&mut self, place: &Place<'tcx>) -> Result<KLocal> {
        let local = place.as_local().ok_or("writes memory other than the current element")?;
        self.local(local)
    }

    fn local(&mut self, local: Local) -> Result<KLocal> {
        if let Some(&kernel) = self.locals.get(&local) {
            return Ok(kernel);
        }
        let ty = self.body.local_decls[local].ty;
        let kernel_ty = kernel_ty(ty).ok_or_else(|| format!("uses a value of type `{ty}`"))?;
        let kernel = self.b.local(kernel_ty);
        self.locals.insert(local, kernel);
        Ok(kernel)
    }

    fn assign(&mut self, place: &Place<'tcx>, rvalue: &Rvalue<'tcx>) -> Result<()> {
        // `*element = value`.
        if let [ProjectionElem::Deref] = place.projection.as_slice() {
            if !self.refs.contains(&place.local) {
                return Err("writes memory other than the current element".into());
            }
            let value = self.rvalue(rvalue)?;
            self.b.set(self.element, value);
            self.stores = true;
            return Ok(());
        }
        let local = place.as_local().ok_or("writes memory other than the current element")?;
        let ty = self.body.local_decls[local].ty;
        if ty.is_unit() {
            return Ok(());
        }
        if self.is_element_ref(ty) {
            // A reference may only be the element's, or a reborrow of it.
            let aliases = match rvalue {
                Rvalue::Use(Operand::Copy(source) | Operand::Move(source), _) => {
                    self.is_option_element(source)
                        || source.as_local().is_some_and(|source| self.refs.contains(&source))
                }
                Rvalue::Ref(_, _, source) => {
                    matches!(source.projection.as_slice(), [ProjectionElem::Deref])
                        && self.refs.contains(&source.local)
                }
                _ => false,
            };
            if !aliases {
                return Err("borrows something other than the current element".into());
            }
            self.refs.insert(local);
            return Ok(());
        }
        let value = self.rvalue(rvalue)?;
        let kernel = self.local(local)?;
        self.b.set(kernel, value);
        Ok(())
    }

    fn is_option_element(&self, place: &Place<'tcx>) -> bool {
        place.local == self.shape.option.local
            && matches!(place.projection.as_slice(),
                [ProjectionElem::Downcast(_, variant), ProjectionElem::Field(field, _)]
                    if variant.as_u32() == 1 && field.as_u32() == 0)
    }

    fn rvalue(&mut self, rvalue: &Rvalue<'tcx>) -> Result<KRvalue> {
        Ok(match rvalue {
            Rvalue::Use(operand, _) => KRvalue::Use(self.operand(operand)?),
            Rvalue::BinaryOp(op, operands) => {
                let (a, b) = (self.operand(&operands.0)?, self.operand(&operands.1)?);
                let op = match op {
                    BinOp::Add => KBinOp::Add,
                    BinOp::Sub => KBinOp::Sub,
                    BinOp::Mul => KBinOp::Mul,
                    // Divisors are nonzero: every assertion was proven.
                    BinOp::Div => KBinOp::Div,
                    BinOp::Rem if self.b.ty(a).is_float() => {
                        return Err("uses `%` on floats, which has no exact GPU version yet".into());
                    }
                    BinOp::Rem => KBinOp::Rem,
                    BinOp::BitAnd => KBinOp::And,
                    BinOp::BitOr => KBinOp::Or,
                    BinOp::BitXor => KBinOp::Xor,
                    BinOp::Shl => KBinOp::Shl,
                    BinOp::Shr => KBinOp::Shr,
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
                let a = self.operand(operand)?;
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
                KRvalue::Cast(self.operand(operand)?, to)
            }
            Rvalue::Ref(..) | Rvalue::RawPtr(..) => {
                return Err("borrows something other than the current element".into());
            }
            _ => return Err("uses an unsupported operation".into()),
        })
    }

    fn operand(&mut self, operand: &Operand<'tcx>) -> Result<KOperand> {
        let place = match operand {
            Operand::Copy(place) | Operand::Move(place) => place,
            Operand::Constant(c) => {
                return constant(c)
                    .map(KOperand::Const)
                    .ok_or_else(|| format!("uses an unsupported constant of type `{}`", c.ty()));
            }
            Operand::RuntimeChecks(_) => return Err("reads a runtime-check flag".into()),
        };
        match place.projection.as_slice() {
            [ProjectionElem::Deref] if self.refs.contains(&place.local) => Ok(self.element.into()),
            [] => {
                let local = place.local;
                if let Some(&c) = self.shape.constants.get(&local) {
                    return Ok(c.into());
                }
                let parameter = self.shape.scalars.contains(&local);
                if !parameter && self.assigned_outside.contains(&local) {
                    return Err("uses a value computed outside the loop body".into());
                }
                if !parameter && (1..=self.body.arg_count).contains(&local.as_usize()) {
                    return Err("uses a parameter a kernel can't take".into());
                }
                Ok(self.local(local)?.into())
            }
            _ => Err("reads memory other than the current element".into()),
        }
    }
}
