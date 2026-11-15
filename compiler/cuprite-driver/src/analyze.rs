//! Recognizer: decides from built MIR whether a function is an in-place map
//! over a slice of numbers (M2), or a counted loop that repeats such maps
//! (M3), and translates what the map does to each element into a kernel
//! element function (`translate`).
//!
//! - **Maps.** A function `fn(&mut [T], scalars...)`, with `T` a primitive
//!   number and each scalar an `i32`, `u32`, `i64`, `u64`, `usize`,
//!   `isize`, `f32` or `f64`, whose body is a single `for` loop over the
//!   slice (directly or through `iter_mut()`). The loop body may compute
//!   anything `translate` supports from the element, the scalars and
//!   constants, and write the element. It may not leave the loop early.
//! - **Repeated maps.** A function `fn(&mut [T])` or `fn(&mut [T], u32)`
//!   whose body is a single `for _ in 0..n` loop, with `n` a constant or the
//!   `u32` parameter, whose body only calls maps of this crate on the whole
//!   slice. Maps change each element independently, so the calls compose
//!   per element, and the kernel loops `n` times per element.
//!
//! Everything else is rejected with a reason. Neither rule allows a
//! function to reassign its parameters, because the rewritten body reads
//! them on entry. Nothing may panic, unless the assertion provably passes.
//!
//! Library functions are recognized by their definitions (lang items, and
//! inherent methods of the primitive types and `[T]`), never by printed
//! names. Built MIR is read before inlining, so those calls are still
//! visible; the borrow checker has accepted the function, so the loop's
//! element reference is the only access to the slice while the loop runs.

use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
    sync::Mutex,
};

use cuprite_kernel::{Const, Function, Ty as KTy};
use rustc_attr_ir::lang_items::LangItem;
use rustc_hir::def::DefKind;
use rustc_middle::{
    mir::{
        AggregateKind, AssertKind, BasicBlock, Body, BorrowKind, Local, Operand, Place, ProjectionElem, Rvalue,
        StatementKind, TerminatorKind, RETURN_PLACE,
    },
    ty::{self, Ty, TyCtxt},
};
use rustc_span::def_id::{LocalDefId, LOCAL_CRATE};

use crate::{
    replace::Passes,
    translate::{self, Method},
};

/// The slice parameter.
const SLICE: Local = Local::from_u32(1);
/// The pass-count parameter of a repeated map.
const COUNT: Local = Local::from_u32(2);

pub(crate) const REPORT_VAR: &str = "CUPRITE_REPORT";
pub(crate) const AUTO_VAR: &str = "CUPRITE_AUTO";

/// One map: what it does to each element of its slice.
#[derive(Clone, Debug)]
pub(crate) struct Map {
    pub(crate) element: KTy,
    /// Takes the element, then the values of `scalars`.
    pub(crate) function: Function,
    /// The function's parameters after the slice, in order.
    pub(crate) scalars: Vec<Local>,
}

#[derive(Clone, Debug)]
pub(crate) enum Verdict {
    /// One pass of a map.
    Candidate(Map),
    /// Maps applied in order, all of that `passes` times, which can stay on
    /// the GPU throughout. None of them reads scalars.
    Repeated(Vec<Map>, Passes),
    Rejected(String),
}

/// What recognition found for one function.
enum Recorded {
    Done(Verdict),
    /// A counted loop around calls to functions of this crate. Callees may
    /// be built after their callers, so it is resolved once every body has
    /// been built: after analysis, when the report is written and optimized
    /// MIR is requested.
    Region {
        element: KTy,
        calls: Vec<LocalDefId>,
        passes: Passes,
    },
}

/// Results for the crate being compiled, recorded as each function's MIR is
/// built. Query providers are plain function pointers, so this is global.
static RECORDED: Mutex<Option<HashMap<LocalDefId, Recorded>>> = Mutex::new(None);

pub(crate) fn record<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId, body: &Body<'tcx>) {
    if !matches!(tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn) {
        return;
    }
    let recorded = Recognizer::new(tcx, def_id, body)
        .recognize()
        .unwrap_or_else(|reason| Recorded::Done(Verdict::Rejected(reason)));
    RECORDED
        .lock()
        .unwrap()
        .get_or_insert_default()
        .insert(def_id, recorded);
}

/// The verdict for `def_id`. Call only after analysis, when every body has
/// been built.
pub(crate) fn verdict(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Option<Verdict> {
    let recorded = RECORDED.lock().unwrap();
    let recorded = recorded.as_ref()?;
    let (element, calls, passes) = match recorded.get(&def_id)? {
        Recorded::Done(verdict) => return Some(verdict.clone()),
        Recorded::Region { element, calls, passes } => (*element, calls, *passes),
    };
    let mut stages = Vec::new();
    for callee in calls {
        // Only single maps over the same slice, called with just the slice:
        // a callee that is itself a region would need its own pass count.
        let not_a_map = || {
            Some(Verdict::Rejected(format!(
                "calls `{}`, which is not a recognized map",
                tcx.def_path_str(*callee)
            )))
        };
        let Some(Recorded::Done(Verdict::Candidate(map))) = recorded.get(callee) else {
            return not_a_map();
        };
        if map.element != element || !map.scalars.is_empty() {
            return not_a_map();
        }
        stages.push(map.clone());
    }
    Some(if stages.is_empty() || passes == Passes::Const(0) {
        Verdict::Rejected("does not change the slice".into())
    } else {
        Verdict::Repeated(stages, passes)
    })
}

/// Describes a map for the report: "map over `[f32]` reading `scale`".
fn describe_map(tcx: TyCtxt<'_>, def_id: LocalDefId, map: &Map) -> String {
    let mut text = format!("map over `[{}]`", map.element.name());
    if !map.scalars.is_empty() {
        let names: Vec<String> = map.scalars.iter().map(|&local| parameter(tcx, def_id, local)).collect();
        text += &format!(" reading {}", names.join(", "));
    }
    text
}

/// A parameter's name for messages, such as "`passes`".
fn parameter(tcx: TyCtxt<'_>, def_id: LocalDefId, local: Local) -> String {
    match tcx.fn_arg_idents(def_id).get(local.as_usize() - 1) {
        Some(Some(ident)) => format!("`{ident}`"),
        _ => format!("parameter {}", local.as_usize()),
    }
}

/// `<dir>/<crate>.txt`: one line per function, sorted by path.
pub(crate) fn write_report(tcx: TyCtxt<'_>, dir: &Path) {
    let krate = tcx.crate_name(LOCAL_CRATE);
    let mut lines: Vec<String> = tcx
        .hir_body_owners()
        .filter_map(|def_id| {
            let line = match verdict(tcx, def_id)? {
                Verdict::Candidate(map) => format!("candidate: {}", describe_map(tcx, def_id, &map)),
                Verdict::Repeated(stages, passes) => {
                    let passes = match passes {
                        Passes::Const(passes) => passes.to_string(),
                        Passes::Arg(index) => parameter(tcx, def_id, Local::from_usize(index)),
                    };
                    let krate = format!("{}::", tcx.crate_name(LOCAL_CRATE));
                    let names: Vec<String> = stages
                        .iter()
                        .map(|map| format!("`{}`", map.function.kernel.name.trim_start_matches(&krate)))
                        .collect();
                    format!(
                        "candidate: {} over `[{}]`, repeated {passes} times",
                        names.join(" then "),
                        stages[0].element.name()
                    )
                }
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
    /// `<[T]>::iter_mut`.
    SliceIterMut,
    /// `Iterator::next`.
    Next,
    /// A numeric method with a kernel equivalent (`translate::method`).
    Method(Method),
    /// A non-generic, non-trait function of this crate, which only a
    /// repeated map may call, and only if the callee is a map.
    Local(LocalDefId),
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

    fn recognize(&self) -> Result<Recorded> {
        if self.counts_passes() {
            self.recognize_region()
        } else {
            let map = self.recognize_map()?;
            Ok(Recorded::Done(Verdict::Candidate(map)))
        }
    }

    fn recognize_map(&self) -> Result<Map> {
        let (element, scalars) = self.check_signature(false)?;
        self.check_parameters_unchanged()?;
        let calls = self.check_effects(false)?;
        let header = self.single_loop(&calls, "is not a single `for` loop over the slice")?;
        let iterator_chain = self.iterator_chain(header.receiver)?;

        let constants = self.constants_before_loop();
        let is_header = |block| self.is_header(block, header.next_block);
        let shape = translate::Loop {
            option: header.option,
            start: header.some,
            is_header: &is_header,
            constants: &constants,
            scalars: &scalars,
        };
        let name = format!("{}::{}", self.tcx.crate_name(LOCAL_CRATE), self.tcx.def_path_str(self.def_id));
        let translated = translate::translate(self.tcx, self.body, &name, element, &shape)?;
        self.check_rest(&calls, &translated.blocks, &iterator_chain, &header)?;
        if !translated.stores {
            return Err("does not change the slice".into());
        }
        Ok(Map {
            element,
            function: translated.function,
            scalars,
        })
    }

    /// A counted loop whose body calls functions of this crate on the
    /// slice. Whether those are maps is decided later (`verdict`).
    fn recognize_region(&self) -> Result<Recorded> {
        let (element, _) = self.check_signature(true)?;
        self.check_parameters_unchanged()?;
        let calls = self.check_effects(true)?;
        let header = self.single_loop(&calls, "is not a single counted loop")?;
        let (range_setup, passes) = self.pass_count(header.receiver)?;

        let mut walked = HashSet::new();
        let callees = self.region_calls(header.some, header.next_block, &mut walked)?;
        self.check_rest(&calls, &walked, &[range_setup], &header)?;
        Ok(Recorded::Region {
            element,
            calls: callees,
            passes,
        })
    }

    /// True if the function loops over an integer `Range`, which makes it a
    /// candidate for the repeated-map rule.
    fn counts_passes(&self) -> bool {
        let tcx = self.tcx;
        let is_integer_range = |ty: Ty<'tcx>| {
            matches!(ty.kind(), ty::Adt(adt, args)
                if tcx.is_lang_item(adt.did(), LangItem::Range)
                    && args.type_at(0).is_integral())
        };
        self.body.basic_blocks.iter().any(|data| {
            matches!(&data.terminator().kind, TerminatorKind::Call { func, .. }
            if func.const_fn_def().is_some_and(|(def_id, args)| {
                tcx.is_lang_item(def_id, LangItem::IteratorNext)
                    && is_integer_range(args.type_at(0))
            }))
        })
    }

    /// The one `next` call, which must be a `for` loop's.
    fn single_loop(
        &self,
        calls: &[(BasicBlock, Known)],
        not_a_loop: &str,
    ) -> Result<LoopHeader<'a, 'tcx>> {
        let nexts: Vec<BasicBlock> =
            calls.iter().filter(|(_, known)| *known == Known::Next).map(|&(block, _)| block).collect();
        let next_block = match nexts[..] {
            [block] => block,
            [] => return Err(not_a_loop.into()),
            _ => return Err("contains more than one loop".into()),
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
        let (some, none) = self.split_on_next(*after_next, *destination)?;
        Ok(LoopHeader {
            next_block,
            receiver: &args[0].node,
            option: *destination,
            after_next: *after_next,
            some,
            none,
        })
    }

    /// Everything the loop body walk did not visit is the iterator setup,
    /// the loop header and the exit; none of it may call, compute or write.
    fn check_rest(
        &self,
        calls: &[(BasicBlock, Known)],
        walked: &HashSet<BasicBlock>,
        setup: &[BasicBlock],
        header: &LoopHeader<'_, 'tcx>,
    ) -> Result<()> {
        let allowed: HashSet<BasicBlock> =
            setup.iter().copied().chain([header.next_block]).collect();
        for &(block, known) in calls {
            if !walked.contains(&block) && !allowed.contains(&block) {
                return Err(format!(
                    "uses `{}` outside the loop body",
                    self.callee(block)
                ));
            }
            if walked.contains(&block) && !matches!(known, Known::Method(_) | Known::Local(_)) {
                return Err("iterates inside the loop body".into());
            }
        }
        self.check_outside_loop(walked, header.after_next, header.next_block, header.none)
    }

    /// `fn(&mut [T], scalars...) -> ()` with `T` a primitive number, or with
    /// `counted`, `fn(&mut [T])` or `fn(&mut [T], u32)`, the second being the
    /// pass count. Returns `T` and the scalar parameters.
    fn check_signature(&self, counted: bool) -> Result<(KTy, Vec<Local>)> {
        if self.tcx.generics_of(self.def_id).count() != 0 {
            return Err("is generic".into());
        }
        let params = self.body.arg_count;
        if params == 0 {
            return Err("takes no parameters; the first must be a `&mut` slice of numbers".into());
        }
        if counted && params > 2 {
            return Err(format!(
                "takes {params} parameters; a repeated map takes the slice and at most a `u32` pass count"
            ));
        }
        let param = self.body.local_decls[SLICE].ty;
        let element = Self::mut_slice_element(param)
            .ok_or_else(|| format!("takes `{param}`, not a `&mut` slice of numbers"))?;
        if counted && params == 2 {
            let count = self.body.local_decls[COUNT].ty;
            if count != self.tcx.types.u32 {
                return Err(format!("takes `{count}` as its pass count, not `u32`"));
            }
        }
        let scalars: Vec<Local> = if counted { Vec::new() } else { (2..=params).map(Local::from_usize).collect() };
        for &local in &scalars {
            let ty = self.body.local_decls[local].ty;
            if !translate::kernel_ty(ty).is_some_and(|ty| ty.is_passable()) {
                return Err(format!(
                    "takes `{ty}` as {}, which a kernel can't take",
                    parameter(self.tcx, self.def_id, local)
                ));
            }
        }
        let output = self.body.local_decls[RETURN_PLACE].ty;
        if !output.is_unit() {
            return Err(format!("returns `{output}`"));
        }
        Ok((element, scalars))
    }

    /// `T` for `&mut [T]` with `T` a number.
    fn mut_slice_element(ty: Ty<'tcx>) -> Option<KTy> {
        match ty.kind() {
            ty::Ref(_, inner, ty::Mutability::Mut) => match inner.kind() {
                ty::Slice(element) => translate::kernel_ty(*element).filter(|ty| *ty != KTy::Bool),
                _ => None,
            },
            _ => None,
        }
    }

    fn is_parameter(&self, local: Local) -> bool {
        (1..=self.body.arg_count).contains(&local.as_usize())
    }

    /// The rewritten body passes the parameters' values on entry to the
    /// runtime, so the original body must not assign them, or borrow them
    /// in a way that could.
    fn check_parameters_unchanged(&self) -> Result<()> {
        let changed = |local: Local| {
            Err(format!(
                "reassigns its parameter {}",
                parameter(self.tcx, self.def_id, local)
            ))
        };
        for data in self.body.basic_blocks.iter() {
            for statement in &data.statements {
                let StatementKind::Assign(assign) = &statement.kind else {
                    continue;
                };
                if let Some(local) = assign.0.as_local().filter(|&l| self.is_parameter(l)) {
                    return changed(local);
                }
                match &assign.1 {
                    Rvalue::Ref(_, BorrowKind::Mut { .. }, place) | Rvalue::RawPtr(_, place)
                        if place.projection.is_empty() && self.is_parameter(place.local) =>
                    {
                        return changed(place.local);
                    }
                    _ => {}
                }
            }
            if let TerminatorKind::Call { destination, .. } = &data.terminator().kind {
                if let Some(local) = destination.as_local().filter(|&l| self.is_parameter(l)) {
                    return changed(local);
                }
            }
        }
        Ok(())
    }

    /// Rejects anything that may panic, run a destructor or call unknown
    /// code; returns the known calls. An assertion is allowed only if it
    /// provably passes. Calls to functions of this crate are allowed only
    /// with `allow_local`.
    fn check_effects(&self, allow_local: bool) -> Result<Vec<(BasicBlock, Known)>> {
        let mut calls = Vec::new();
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
                TerminatorKind::Call { func, .. } => {
                    let known = self.classify(func)?;
                    if let (Known::Local(_), false) = (known, allow_local) {
                        return Err(self.no_summary(func));
                    }
                    calls.push((block, known));
                }
                TerminatorKind::Assert { msg, .. } => {
                    if !translate::assert_holds(self.body, block) {
                        return Err(format!("may panic: {}", describe_assert(msg)));
                    }
                }
                TerminatorKind::Drop { .. } => return Err("runs a destructor".into()),
                TerminatorKind::Goto { .. }
                | TerminatorKind::SwitchInt { .. }
                | TerminatorKind::Return
                | TerminatorKind::UnwindResume
                | TerminatorKind::Unreachable
                // Loops: the `for` loop, and `while` loops inside its body.
                | TerminatorKind::FalseUnwind { .. }
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
        if let Some(method) = translate::method(tcx, def_id) {
            return Ok(Known::Method(method));
        }
        if let Some(impl_id) = tcx.inherent_impl_of_assoc(def_id) {
            let self_ty = tcx.type_of(impl_id).instantiate_identity().skip_normalization();
            if tcx.crate_name(impl_id.krate).as_str() == "core"
                && matches!(self_ty.kind(), ty::Slice(_))
                && tcx.item_name(def_id).as_str() == "iter_mut"
            {
                return Ok(Known::SliceIterMut);
            }
        }
        // Trait methods are excluded: the call names the trait's item, and
        // the body that runs is the impl's.
        if let Some(local) = def_id.as_local() {
            if args.is_empty()
                && tcx.trait_of_assoc(def_id).is_none()
                && matches!(tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn)
            {
                return Ok(Known::Local(local));
            }
        }
        Err(self.no_summary(func))
    }

    fn no_summary(&self, func: &Operand<'tcx>) -> String {
        match func.const_fn_def() {
            Some((def_id, args)) => format!(
                "calls `{}`, which has no GPU summary",
                self.tcx.def_path_str_with_args(def_id, args)
            ),
            None => "calls a function pointer or closure value".into(),
        }
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
    /// Parameters are never reassigned (`check_parameters_unchanged`), so
    /// `Param` is the value passed in.
    fn origin(&self, mut local: Local) -> Option<Origin> {
        for _ in 0..self.body.local_decls.len() {
            if self.is_parameter(local) {
                return Some(Origin::Param(local));
            }
            let next = match self.definitions(local) {
                (assigned, returned) if assigned.is_empty() && returned.len() == 1 => {
                    return Some(Origin::Call(returned[0]));
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
            local = next?;
        }
        None
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
            let Some(Origin::Call(block)) = self.origin(current) else {
                return Err(not_slice());
            };
            let TerminatorKind::Call { func, args, .. } =
                &self.body.basic_blocks[block].terminator().kind
            else {
                return Err(not_slice());
            };
            chain.push(block);
            let source = Self::operand_local(&args[0].node).ok_or_else(not_slice)?;
            match (
                self.classify(func)?,
                self.origin(source).ok_or_else(not_slice)?,
            ) {
                (Known::IntoIter | Known::SliceIterMut, Origin::Param(SLICE)) => {
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

    /// `next`'s receiver must be `(a..b).into_iter()` with integer constants
    /// `a` and `b`, or `(0..n).into_iter()` with `n` the pass-count
    /// parameter. Returns the `into_iter` block and the number of passes.
    fn pass_count(&self, receiver: &Operand<'tcx>) -> Result<(BasicBlock, Passes)> {
        let unsupported = || "has an unsupported counted loop".to_string();
        let iterator = Self::operand_local(receiver).ok_or_else(unsupported)?;
        let Some(Origin::Call(block)) = self.origin(iterator) else {
            return Err(unsupported());
        };
        let TerminatorKind::Call { func, args, .. } =
            &self.body.basic_blocks[block].terminator().kind
        else {
            return Err(unsupported());
        };
        if self.classify(func)? != Known::IntoIter {
            return Err(unsupported());
        }
        let range = Self::operand_local(&args[0].node).ok_or_else(unsupported)?;
        let (assigned, returned) = self.definitions(range);
        let ([Rvalue::Aggregate(kind, fields)], []) = (assigned.as_slice(), returned.as_slice())
        else {
            return Err(unsupported());
        };
        let AggregateKind::Adt(adt, ..) = **kind else {
            return Err(unsupported());
        };
        let [start, end] = fields.raw.as_slice() else {
            return Err(unsupported());
        };
        if !self.tcx.is_lang_item(adt, LangItem::Range) {
            return Err(unsupported());
        }
        let constant = |operand: &Operand<'tcx>| match operand {
            Operand::Constant(constant) => integer_constant(constant),
            _ => None,
        };
        let Some(start) = constant(start) else {
            return Err("counts passes from a value that is not a constant".into());
        };
        if let Some(end) = constant(end) {
            // `Range` iterates `end - start` times, or not at all.
            let passes = end
                .checked_sub(start)
                .and_then(|passes| u32::try_from(passes.max(0)).ok())
                .ok_or("counts more passes than fit in `u32`")?;
            return Ok((block, Passes::Const(passes)));
        }
        if start != 0 {
            return Err(format!("counts passes from {start}, not 0"));
        }
        let origin = Self::operand_local(end).and_then(|local| self.origin(local));
        match origin {
            Some(Origin::Param(COUNT)) => Ok((block, Passes::Arg(COUNT.as_usize()))),
            _ => Err(
                "counts passes up to a value other than a constant or the `u32` parameter".into(),
            ),
        }
    }

    /// Walks a counted loop's body back to the header. It may only call
    /// functions of this crate, each on the whole slice; returns them in
    /// order.
    fn region_calls(
        &self,
        start: BasicBlock,
        next_block: BasicBlock,
        walked: &mut HashSet<BasicBlock>,
    ) -> Result<Vec<LocalDefId>> {
        let mut callees = Vec::new();
        let mut block = start;
        loop {
            if !walked.insert(block) {
                return Err("has an unsupported loop shape".into());
            }
            let data = &self.body.basic_blocks[block];
            for statement in &data.statements {
                let StatementKind::Assign(assign) = &statement.kind else {
                    continue; // Storage markers and other no-ops.
                };
                // Moves and borrows into locals have no effect by themselves:
                // only the calls below can use them, and a branch is
                // rejected.
                if assign.0.as_local().is_none() {
                    return Err("writes memory in the loop body other than through calls".into());
                }
                if !matches!(assign.1, Rvalue::Use(..) | Rvalue::Ref(..)) {
                    return Err("computes values in the loop body".into());
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
                    let Known::Local(callee) = self.classify(func)? else {
                        return Err("computes values in the loop body".into());
                    };
                    let name = self.tcx.def_path_str(callee);
                    let [arg] = args.as_ref() else {
                        return Err(format!("calls `{name}`, which is not a recognized map"));
                    };
                    let origin = Self::operand_local(&arg.node).and_then(|l| self.origin(l));
                    if !matches!(origin, Some(Origin::Param(SLICE))) {
                        return Err(format!("passes something other than the slice to `{name}`"));
                    }
                    if destination.as_local().is_none() {
                        return Err(
                            "writes memory in the loop body other than through calls".into()
                        );
                    }
                    callees.push(callee);
                    *target
                }
                TerminatorKind::Goto { target, .. } if self.is_header(*target, next_block) => {
                    return Ok(callees);
                }
                TerminatorKind::Goto { target, .. } => *target,
                TerminatorKind::SwitchInt { .. } => {
                    return Err("branches inside the loop body".into())
                }
                _ => return Err("has an unsupported loop shape".into()),
            };
        }
    }

    /// Locals assigned once, from a scalar constant, and never borrowed.
    fn constants_before_loop(&self) -> HashMap<Local, Const> {
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
                [Rvalue::Use(Operand::Constant(constant), _)] => Some((local, translate::constant(constant)?)),
                _ => None,
            })
            .collect()
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
    Param(Local),
    Call(BasicBlock),
}

/// A `for` loop's `next` call and where it leads.
struct LoopHeader<'a, 'tcx> {
    next_block: BasicBlock,
    /// `next`'s argument, the `&mut` iterator.
    receiver: &'a Operand<'tcx>,
    /// `next`'s result.
    option: Place<'tcx>,
    /// The block that switches on `option`.
    after_next: BasicBlock,
    /// The loop body's first block.
    some: BasicBlock,
    /// The exit.
    none: BasicBlock,
}

/// The value of an integer constant of any width and sign, if it fits in
/// `i128`.
fn integer_constant(constant: &rustc_middle::mir::ConstOperand<'_>) -> Option<i128> {
    let scalar = constant.const_.try_to_scalar_int()?;
    match constant.ty().kind() {
        ty::Int(_) => Some(scalar.to_int(scalar.size())),
        ty::Uint(_) => i128::try_from(scalar.to_uint(scalar.size())).ok(),
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

