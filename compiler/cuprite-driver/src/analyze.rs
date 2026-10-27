//! M2 recognizer: decides from built MIR whether a function is an in-place
//! multiply-add map over a `u32` slice, and which one.
//!
//! The accepted shape is a function `fn(&mut [u32])` whose body is a single
//! `for` loop over the slice (directly or through `iter_mut()`) in which
//! each element is replaced by a combination of wrapping `+`, `-` and `*`
//! with constants. The combination must reduce to `x * mul + add`. Plain
//! operators count only where they cannot panic, that is, without overflow
//! checks. Everything else is rejected with a reason.
//!
//! Library functions are recognized by their definitions (lang items, and
//! inherent methods of `core`'s `u32` and `[T]`), never by printed names.
//! Built MIR is read before inlining, so those calls are still visible; the
//! borrow checker has accepted the function, so the loop's element
//! reference is the only access to the slice while the loop runs.

use std::{
    collections::{HashMap, HashSet},
    fmt, fs,
    path::Path,
    sync::Mutex,
};

use rustc_attr_ir::lang_items::LangItem;
use rustc_hir::def::DefKind;
use rustc_middle::{
    mir::{
        AssertKind, BasicBlock, BinOp, Body, Local, Operand, Place, ProjectionElem, Rvalue,
        StatementKind, TerminatorKind, RETURN_PLACE,
    },
    ty::{self, Ty, TyCtxt},
};
use rustc_span::def_id::{LocalDefId, LOCAL_CRATE};

pub(crate) const REPORT_VAR: &str = "CUPRITE_REPORT";
pub(crate) const AUTO_VAR: &str = "CUPRITE_AUTO";

/// `x * mul + add`, wrapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Affine {
    pub(crate) mul: u32,
    pub(crate) add: u32,
}

impl Affine {
    const IDENTITY: Self = Self { mul: 1, add: 0 };

    fn constant(value: u32) -> Self {
        Self { mul: 0, add: value }
    }

    fn add(self, other: Self) -> Self {
        Self {
            mul: self.mul.wrapping_add(other.mul),
            add: self.add.wrapping_add(other.add),
        }
    }

    fn sub(self, other: Self) -> Self {
        Self {
            mul: self.mul.wrapping_sub(other.mul),
            add: self.add.wrapping_sub(other.add),
        }
    }

    /// Defined only when one side is a constant; otherwise the element
    /// would be multiplied by itself.
    fn mul(self, other: Self) -> Option<Self> {
        let (scale, map) = match (self.mul, other.mul) {
            (0, _) => (self.add, other),
            (_, 0) => (other.add, self),
            _ => return None,
        };
        Some(Self {
            mul: map.mul.wrapping_mul(scale),
            add: map.add.wrapping_mul(scale),
        })
    }
}

impl fmt::Display for Affine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "x * {} + {}", self.mul, self.add)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Verdict {
    Candidate(Affine),
    Rejected(String),
}

/// Verdicts for the crate being compiled, recorded as each function's MIR
/// is built. Query providers are plain function pointers, so this is global.
static VERDICTS: Mutex<Option<HashMap<LocalDefId, Verdict>>> = Mutex::new(None);

pub(crate) fn record<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId, body: &Body<'tcx>) {
    if !matches!(tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn) {
        return;
    }
    let verdict = match Recognizer::new(tcx, def_id, body).recognize() {
        Ok(map) => Verdict::Candidate(map),
        Err(reason) => Verdict::Rejected(reason),
    };
    VERDICTS
        .lock()
        .unwrap()
        .get_or_insert_default()
        .insert(def_id, verdict);
}

pub(crate) fn verdict(def_id: LocalDefId) -> Option<Verdict> {
    VERDICTS.lock().unwrap().as_ref()?.get(&def_id).cloned()
}

/// `<dir>/<crate>.txt`: one line per function, sorted by path.
pub(crate) fn write_report(tcx: TyCtxt<'_>, dir: &Path) {
    let krate = tcx.crate_name(LOCAL_CRATE);
    let mut lines: Vec<String> = tcx
        .hir_body_owners()
        .filter_map(|def_id| {
            let line = match verdict(def_id)? {
                Verdict::Candidate(map) => format!("candidate: {map}"),
                Verdict::Rejected(reason) => format!("rejected: {reason}"),
            };
            Some(format!("{krate}::{}: {line}", tcx.def_path_str(def_id)))
        })
        .collect();
    lines.sort();
    fs::create_dir_all(dir).expect("cannot create the report directory");
    fs::write(dir.join(format!("{krate}.txt")), lines.join("\n") + "\n")
        .expect("cannot write the report");
}

/// A library call the recognizer has a summary for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Known {
    /// `IntoIterator::into_iter`; which receivers are allowed is checked
    /// where the iterator is traced.
    IntoIter,
    /// `<[u32]>::iter_mut`.
    SliceIterMut,
    /// `Iterator::next`.
    Next,
    /// `u32::wrapping_add`, `wrapping_sub` or `wrapping_mul`.
    Wrapping(BinOp),
}

/// What a local holds while the loop body is evaluated.
#[derive(Clone, Copy, Debug)]
enum Value {
    /// The `&mut u32` to the current element.
    Element,
    /// A `u32` as a function of the element's original value.
    Map(Affine),
    /// `()`, which the body may assign but never compute with.
    Unit,
}

struct Recognizer<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
    body: &'a Body<'tcx>,
}

type Result<T> = std::result::Result<T, String>;

impl<'a, 'tcx> Recognizer<'a, 'tcx> {
    fn new(tcx: TyCtxt<'tcx>, def_id: LocalDefId, body: &'a Body<'tcx>) -> Self {
        Self { tcx, def_id, body }
    }

    fn recognize(&self) -> Result<Affine> {
        self.check_signature()?;
        let calls = self.check_effects()?;

        let mut nexts = calls.iter().filter(|(_, known)| *known == Known::Next);
        let (Some(&(next_block, _)), None) = (nexts.next(), nexts.next()) else {
            return Err("is not a single `for` loop over the slice".into());
        };
        let TerminatorKind::Call {
            args,
            destination,
            target: Some(after_next),
            ..
        } = &self.body.basic_blocks[next_block].terminator().kind
        else {
            unreachable!("`check_effects` recorded a call");
        };
        let iterator_chain = self.iterator_chain(&args[0].node)?;
        let (some, none) = self.split_on_next(*after_next, *destination)?;

        let mut walked = HashSet::new();
        let map = self.evaluate_body(some, next_block, *destination, &mut walked)?;

        // Everything the loop body did not evaluate is the iterator setup,
        // the loop header and the exit; none of it may compute or write.
        let allowed: HashSet<BasicBlock> =
            iterator_chain.iter().copied().chain([next_block]).collect();
        for &(block, known) in &calls {
            if !walked.contains(&block) && !allowed.contains(&block) {
                return Err(format!(
                    "uses `{}` outside the loop body",
                    self.callee(block)
                ));
            }
            if walked.contains(&block) && !matches!(known, Known::Wrapping(_)) {
                return Err("iterates inside the loop body".into());
            }
        }
        self.check_outside_loop(&walked, *after_next, next_block, none)?;

        if map == Affine::IDENTITY {
            return Err("does not change the slice".into());
        }
        Ok(map)
    }

    fn check_signature(&self) -> Result<()> {
        if self.tcx.generics_of(self.def_id).count() != 0 {
            return Err("is generic".into());
        }
        let params = self.body.arg_count;
        if params != 1 {
            return Err(format!(
                "takes {params} parameters; only `fn(&mut [u32])` is supported"
            ));
        }
        let param = self.body.local_decls[Local::from_usize(1)].ty;
        if !self.is_mut_u32_slice(param) {
            return Err(format!("takes `{param}`, not `&mut [u32]`"));
        }
        let output = self.body.local_decls[RETURN_PLACE].ty;
        if !output.is_unit() {
            return Err(format!("returns `{output}`"));
        }
        Ok(())
    }

    fn is_mut_u32_slice(&self, ty: Ty<'tcx>) -> bool {
        matches!(ty.kind(), ty::Ref(_, inner, ty::Mutability::Mut)
            if matches!(inner.kind(), ty::Slice(element) if *element == self.tcx.types.u32))
    }

    /// Rejects anything that may panic, run a destructor, call unknown code,
    /// or loop other than the one loop; returns the known calls.
    fn check_effects(&self) -> Result<Vec<(BasicBlock, Known)>> {
        let mut calls = Vec::new();
        let mut loops = 0;
        for (block, data) in self.body.basic_blocks.iter_enumerated() {
            for statement in &data.statements {
                match &statement.kind {
                    StatementKind::Assign(_)
                    | StatementKind::FakeRead(_)
                    | StatementKind::StorageLive(_)
                    | StatementKind::StorageDead(_)
                    | StatementKind::PlaceMention(_)
                    | StatementKind::AscribeUserType(..)
                    | StatementKind::Coverage(_)
                    | StatementKind::ConstEvalCounter
                    | StatementKind::BackwardIncompatibleDropHint { .. }
                    | StatementKind::Nop => {}
                    other => return Err(format!("contains unsupported MIR `{other:?}`")),
                }
            }
            match &data.terminator().kind {
                TerminatorKind::Call { func, .. } => calls.push((block, self.classify(func)?)),
                TerminatorKind::Assert { msg, .. } => {
                    return Err(format!("may panic: {}", describe_assert(msg)));
                }
                TerminatorKind::Drop { .. } => return Err("runs a destructor".into()),
                TerminatorKind::FalseUnwind { .. } => {
                    loops += 1;
                    if loops > 1 {
                        return Err("contains more than one loop".into());
                    }
                }
                TerminatorKind::Goto { .. }
                | TerminatorKind::SwitchInt { .. }
                | TerminatorKind::Return
                | TerminatorKind::UnwindResume
                | TerminatorKind::Unreachable
                | TerminatorKind::FalseEdge { .. } => {}
                other => return Err(format!("contains unsupported control flow `{other:?}`")),
            }
        }
        Ok(calls)
    }

    fn classify(&self, func: &Operand<'tcx>) -> Result<Known> {
        let tcx = self.tcx;
        let Some((def_id, args)) = func.const_fn_def() else {
            return Err("calls a function pointer or closure value".into());
        };
        if tcx.is_lang_item(def_id, LangItem::IntoIterIntoIter) {
            return Ok(Known::IntoIter);
        }
        if tcx.is_lang_item(def_id, LangItem::IteratorNext) {
            return Ok(Known::Next);
        }
        if let Some(impl_id) = tcx.inherent_impl_of_assoc(def_id) {
            if tcx.crate_name(impl_id.krate).as_str() == "core" {
                let self_ty = tcx
                    .type_of(impl_id)
                    .instantiate_identity()
                    .skip_normalization();
                let name = tcx.item_name(def_id);
                if self_ty == tcx.types.u32 {
                    match name.as_str() {
                        "wrapping_add" => return Ok(Known::Wrapping(BinOp::Add)),
                        "wrapping_sub" => return Ok(Known::Wrapping(BinOp::Sub)),
                        "wrapping_mul" => return Ok(Known::Wrapping(BinOp::Mul)),
                        _ => {}
                    }
                }
                if matches!(self_ty.kind(), ty::Slice(_))
                    && name.as_str() == "iter_mut"
                    && args.type_at(0) == tcx.types.u32
                {
                    return Ok(Known::SliceIterMut);
                }
            }
        }
        Err(format!(
            "calls `{}`, which has no GPU summary",
            tcx.def_path_str_with_args(def_id, args)
        ))
    }

    fn callee(&self, block: BasicBlock) -> String {
        match &self.body.basic_blocks[block].terminator().kind {
            TerminatorKind::Call { func, .. } => match func.const_fn_def() {
                Some((def_id, args)) => self.tcx.def_path_str_with_args(def_id, args),
                None => "a function value".into(),
            },
            _ => "a call".into(),
        }
    }

    /// Every way `local` is assigned: statement right-hand sides and the
    /// blocks of calls returning into it.
    fn definitions(&self, local: Local) -> (Vec<&'a Rvalue<'tcx>>, Vec<BasicBlock>) {
        let mut assigned = Vec::new();
        let mut returned = Vec::new();
        for (block, data) in self.body.basic_blocks.iter_enumerated() {
            for statement in &data.statements {
                if let StatementKind::Assign(assign) = &statement.kind {
                    if assign.0.as_local() == Some(local) {
                        assigned.push(&assign.1);
                    }
                }
            }
            if let TerminatorKind::Call { destination, .. } = &data.terminator().kind {
                if destination.as_local() == Some(local) {
                    returned.push(block);
                }
            }
        }
        (assigned, returned)
    }

    /// Follows moves, copies and reborrows back to a parameter or a call.
    fn origin(&self, mut local: Local) -> Result<Origin> {
        for _ in 0..self.body.local_decls.len() {
            if local == Local::from_usize(1) {
                return Ok(Origin::Param);
            }
            let next = match self.definitions(local) {
                (assigned, returned) if assigned.is_empty() && returned.len() == 1 => {
                    return Ok(Origin::Call(returned[0]));
                }
                (assigned, returned) if assigned.len() == 1 && returned.is_empty() => {
                    match assigned[0] {
                        Rvalue::Use(Operand::Move(place) | Operand::Copy(place), _) => {
                            place.as_local()
                        }
                        Rvalue::Ref(_, _, place) => match place.projection.as_slice() {
                            [] | [ProjectionElem::Deref] => Some(place.local),
                            _ => None,
                        },
                        _ => None,
                    }
                }
                _ => None,
            };
            local = next.ok_or("iterates over something other than the slice parameter")?;
        }
        Err("iterates over something other than the slice parameter".into())
    }

    fn operand_local(operand: &Operand<'tcx>) -> Option<Local> {
        match operand {
            Operand::Move(place) | Operand::Copy(place) => place.as_local(),
            Operand::Constant(_) | Operand::RuntimeChecks(_) => None,
        }
    }

    /// Checks that `next`'s receiver is an iterator over the slice
    /// parameter, made by `into_iter` on `&mut [u32]` or by `iter_mut`.
    /// Returns the blocks of those constructor calls.
    fn iterator_chain(&self, receiver: &Operand<'tcx>) -> Result<Vec<BasicBlock>> {
        let not_slice = || "iterates over something other than the slice parameter".to_string();
        let mut chain = Vec::new();
        let mut current = Self::operand_local(receiver).ok_or_else(not_slice)?;
        loop {
            let Origin::Call(block) = self.origin(current)? else {
                return Err(not_slice());
            };
            let TerminatorKind::Call { func, args, .. } =
                &self.body.basic_blocks[block].terminator().kind
            else {
                return Err(not_slice());
            };
            chain.push(block);
            let source = Self::operand_local(&args[0].node).ok_or_else(not_slice)?;
            match (self.classify(func)?, self.origin(source)?) {
                (Known::IntoIter, Origin::Param) | (Known::SliceIterMut, Origin::Param) => {
                    return Ok(chain);
                }
                // `iter_mut()` followed by the `for` loop's identity
                // `into_iter`.
                (Known::IntoIter, Origin::Call(_)) if chain.len() == 1 => current = source,
                _ => return Err(not_slice()),
            }
        }
    }

    /// The block after `next` switches on `Some`/`None`; returns the
    /// `Some` (loop body) and `None` (exit) targets.
    fn split_on_next(
        &self,
        block: BasicBlock,
        option: Place<'tcx>,
    ) -> Result<(BasicBlock, BasicBlock)> {
        let data = &self.body.basic_blocks[block];
        let TerminatorKind::SwitchInt { discr, targets } = &data.terminator().kind else {
            return Err("has an unsupported loop shape".into());
        };
        let discriminant = Self::operand_local(discr).ok_or("has an unsupported loop shape")?;
        let reads_option = data.statements.iter().any(|statement| {
            matches!(&statement.kind, StatementKind::Assign(assign)
                if assign.0.as_local() == Some(discriminant)
                    && matches!(&assign.1, Rvalue::Discriminant(place) if *place == option))
        });
        let explicit: HashMap<u128, BasicBlock> = targets.iter().collect();
        let (Some(&some), Some(&none)) = (explicit.get(&1), explicit.get(&0)) else {
            return Err("has an unsupported loop shape".into());
        };
        if !reads_option {
            return Err("has an unsupported loop shape".into());
        }
        Ok((some, none))
    }

    /// True for the loop header: `next`'s block, or an empty block that
    /// only jumps there.
    fn is_header(&self, block: BasicBlock, next_block: BasicBlock) -> bool {
        if block == next_block {
            return true;
        }
        let data = &self.body.basic_blocks[block];
        data.statements.is_empty()
            && match data.terminator().kind {
                TerminatorKind::FalseUnwind { real_target, .. } => real_target == next_block,
                TerminatorKind::Goto { target, .. } => target == next_block,
                _ => false,
            }
    }

    /// Evaluates the loop body symbolically, from the `Some` branch back to
    /// the header, and returns the map applied to the element.
    fn evaluate_body(
        &self,
        start: BasicBlock,
        next_block: BasicBlock,
        option: Place<'tcx>,
        walked: &mut HashSet<BasicBlock>,
    ) -> Result<Affine> {
        let constants = self.constants_before_loop();
        let mut values: HashMap<Local, Value> = HashMap::new();
        let mut element = Affine::IDENTITY;
        let mut block = start;
        loop {
            if !walked.insert(block) {
                return Err("has an unsupported loop shape".into());
            }
            let data = &self.body.basic_blocks[block];
            let env = Env {
                option,
                constants: &constants,
                u32_ty: self.tcx.types.u32,
            };
            for statement in &data.statements {
                let StatementKind::Assign(assign) = &statement.kind else {
                    continue; // Storage markers and other no-ops.
                };
                let (place, rvalue) = &**assign;
                let value = env.rvalue(rvalue, &values, element)?;
                if let Some(local) = place.as_local() {
                    values.insert(local, value);
                } else if let ([ProjectionElem::Deref], Some(Value::Element)) =
                    (place.projection.as_slice(), values.get(&place.local))
                {
                    let Value::Map(map) = value else {
                        return Err("stores something other than a `u32`".into());
                    };
                    element = map;
                } else {
                    return Err("writes memory other than the current element".into());
                }
            }
            block = match &data.terminator().kind {
                TerminatorKind::Call {
                    func,
                    args,
                    destination,
                    target: Some(target),
                    ..
                } => {
                    let Known::Wrapping(op) = self.classify(func)? else {
                        return Err("iterates inside the loop body".into());
                    };
                    let lhs = env.operand(&args[0].node, &values, element)?;
                    let rhs = env.operand(&args[1].node, &values, element)?;
                    let local = destination
                        .as_local()
                        .ok_or("writes memory other than the current element")?;
                    values.insert(local, Value::Map(binary(op, lhs, rhs)?));
                    *target
                }
                TerminatorKind::Goto { target, .. } if self.is_header(*target, next_block) => {
                    return Ok(element);
                }
                TerminatorKind::Goto { target, .. } => *target,
                TerminatorKind::SwitchInt { .. } => {
                    return Err("branches inside the loop body".into())
                }
                _ => return Err("has an unsupported loop shape".into()),
            };
        }
    }

    /// Locals assigned once, from a `u32` constant, and never borrowed.
    fn constants_before_loop(&self) -> HashMap<Local, u32> {
        let mut assigned: HashMap<Local, Vec<&Rvalue<'tcx>>> = HashMap::new();
        let mut borrowed = HashSet::new();
        for data in self.body.basic_blocks.iter() {
            for statement in &data.statements {
                if let StatementKind::Assign(assign) = &statement.kind {
                    if let Some(local) = assign.0.as_local() {
                        assigned.entry(local).or_default().push(&assign.1);
                    }
                    if let Rvalue::Ref(_, _, place) | Rvalue::RawPtr(_, place) = &assign.1 {
                        borrowed.insert(place.local);
                    }
                }
            }
            if let TerminatorKind::Call { destination, .. } = &data.terminator().kind {
                if let Some(local) = destination.as_local() {
                    assigned.entry(local).or_default().clear();
                    borrowed.insert(local); // Never a constant.
                }
            }
        }
        assigned
            .into_iter()
            .filter(|(local, _)| !borrowed.contains(local))
            .filter_map(|(local, rvalues)| match rvalues.as_slice() {
                [Rvalue::Use(Operand::Constant(constant), _)] => {
                    Some((local, self.u32_constant(constant)?))
                }
                _ => None,
            })
            .collect()
    }

    fn u32_constant(&self, constant: &rustc_middle::mir::ConstOperand<'tcx>) -> Option<u32> {
        if constant.ty() != self.tcx.types.u32 {
            return None;
        }
        Some(constant.const_.try_to_scalar_int()?.to_u32())
    }

    /// The setup before the loop and the exit after it may only move
    /// values between locals: no writes through references and no branches
    /// other than the loop's own.
    fn check_outside_loop(
        &self,
        walked: &HashSet<BasicBlock>,
        switch_block: BasicBlock,
        next_block: BasicBlock,
        exit: BasicBlock,
    ) -> Result<()> {
        for (block, data) in self.body.basic_blocks.iter_enumerated() {
            if walked.contains(&block) || data.is_cleanup {
                continue;
            }
            for statement in &data.statements {
                if let StatementKind::Assign(assign) = &statement.kind {
                    if assign.0.is_indirect() {
                        return Err("writes through a reference outside the loop body".into());
                    }
                    if let Rvalue::BinaryOp(..) | Rvalue::UnaryOp(..) = assign.1 {
                        return Err("computes values outside the loop body".into());
                    }
                }
            }
            match &data.terminator().kind {
                TerminatorKind::SwitchInt { .. } if block != switch_block => {
                    return Err("branches outside the loop".into());
                }
                TerminatorKind::FalseUnwind { real_target, .. } if *real_target != next_block => {
                    return Err("has an unsupported loop shape".into());
                }
                _ => {}
            }
        }
        // The exit must return without doing anything else.
        let mut block = exit;
        for _ in 0..self.body.basic_blocks.len() {
            match self.body.basic_blocks[block].terminator().kind {
                TerminatorKind::Return => return Ok(()),
                TerminatorKind::Goto { target, .. }
                | TerminatorKind::FalseEdge {
                    real_target: target,
                    ..
                } => block = target,
                _ => break,
            }
        }
        Err("does more than return after the loop".into())
    }
}

enum Origin {
    Param,
    Call(BasicBlock),
}

/// Evaluation context for the loop body.
struct Env<'c, 'tcx> {
    /// `next`'s result: `(option as Some).0` is the element reference.
    option: Place<'tcx>,
    constants: &'c HashMap<Local, u32>,
    u32_ty: Ty<'tcx>,
}

impl<'tcx> Env<'_, 'tcx> {
    fn operand(
        &self,
        operand: &Operand<'tcx>,
        values: &HashMap<Local, Value>,
        element: Affine,
    ) -> Result<Value> {
        let place = match operand {
            Operand::Move(place) | Operand::Copy(place) => place,
            Operand::Constant(constant) => {
                if constant.ty().is_unit() {
                    return Ok(Value::Unit);
                }
                let scalar = constant
                    .const_
                    .try_to_scalar_int()
                    .filter(|_| constant.ty() == self.u32_ty);
                return match scalar.map(|scalar| scalar.to_u32()) {
                    Some(value) => Ok(Value::Map(Affine::constant(value))),
                    None => Err(format!(
                        "uses an unsupported constant of type `{}`",
                        constant.ty()
                    )),
                };
            }
            Operand::RuntimeChecks(_) => return Err("reads a runtime-check flag".into()),
        };
        if place.local == self.option.local {
            return match place.projection.as_slice() {
                [ProjectionElem::Downcast(_, variant), ProjectionElem::Field(field, _)]
                    if variant.as_u32() == 1 && field.as_u32() == 0 =>
                {
                    Ok(Value::Element)
                }
                _ => Err("has an unsupported loop shape".into()),
            };
        }
        match (place.projection.as_slice(), values.get(&place.local)) {
            ([], Some(value)) => Ok(*value),
            ([ProjectionElem::Deref], Some(Value::Element)) => Ok(Value::Map(element)),
            ([], None) => match self.constants.get(&place.local) {
                Some(&value) => Ok(Value::Map(Affine::constant(value))),
                None => Err("uses a value computed outside the loop body".into()),
            },
            _ => Err("reads memory other than the current element".into()),
        }
    }

    fn rvalue(
        &self,
        rvalue: &Rvalue<'tcx>,
        values: &HashMap<Local, Value>,
        element: Affine,
    ) -> Result<Value> {
        match rvalue {
            Rvalue::Use(operand, _) => self.operand(operand, values, element),
            Rvalue::BinaryOp(op, operands) => {
                let lhs = self.operand(&operands.0, values, element)?;
                let rhs = self.operand(&operands.1, values, element)?;
                Ok(Value::Map(binary(*op, lhs, rhs)?))
            }
            Rvalue::Ref(_, _, place) => {
                match (place.projection.as_slice(), values.get(&place.local)) {
                    ([ProjectionElem::Deref], Some(Value::Element)) => Ok(Value::Element),
                    _ => Err("borrows something other than the current element".into()),
                }
            }
            _ => Err("uses an unsupported operation".into()),
        }
    }
}

/// Wrapping `+`, `-` or `*` of two maps. Plain operators only reach here
/// without overflow checks, where they wrap.
fn binary(op: BinOp, lhs: Value, rhs: Value) -> Result<Affine> {
    let (Value::Map(lhs), Value::Map(rhs)) = (lhs, rhs) else {
        return Err("computes with something other than a `u32`".into());
    };
    match op {
        BinOp::Add => Ok(lhs.add(rhs)),
        BinOp::Sub => Ok(lhs.sub(rhs)),
        BinOp::Mul => lhs
            .mul(rhs)
            .ok_or_else(|| "is not a multiply-add map (multiplies the element by itself)".into()),
        other => Err(format!("uses `{other:?}`, which is not supported")),
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

#[cfg(test)]
mod tests {
    use super::*;

    const X: Affine = Affine::IDENTITY;

    #[test]
    fn maps_compose_with_wrapping_arithmetic() {
        let three = Affine::constant(3);
        let seven = Affine::constant(7);
        assert_eq!(X.mul(three).unwrap().add(seven), Affine { mul: 3, add: 7 });
        assert_eq!(X.add(X), Affine { mul: 2, add: 0 });
        assert_eq!(
            seven.sub(X),
            Affine {
                mul: u32::MAX,
                add: 7
            }
        );
        let big = Affine::constant(u32::MAX);
        assert_eq!(
            X.add(seven).mul(big).unwrap(),
            Affine {
                mul: u32::MAX,
                add: 7u32.wrapping_mul(u32::MAX)
            }
        );
    }

    #[test]
    fn the_element_times_itself_is_not_a_map() {
        assert_eq!(X.mul(X), None);
        assert_eq!(X.add(Affine::constant(1)).mul(X), None);
    }

    /// Every composition must agree with evaluating the operations
    /// directly, including where they wrap.
    #[test]
    fn composition_matches_direct_evaluation() {
        let values = [0, 1, 2, 12345, u32::MAX - 1, u32::MAX];
        for &a in &values {
            for &b in &values {
                let map = X
                    .mul(Affine::constant(a))
                    .unwrap()
                    .sub(Affine::constant(b))
                    .add(X);
                for &x in &values {
                    let direct = x.wrapping_mul(a).wrapping_sub(b).wrapping_add(x);
                    assert_eq!(x.wrapping_mul(map.mul).wrapping_add(map.add), direct);
                }
            }
        }
    }
}
