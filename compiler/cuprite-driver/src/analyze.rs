//! Recognizer: decides from built MIR whether a function is an in-place map
//! over slices (M2), or a counted loop that repeats such maps (M3), and
//! translates what the map does to each element into a kernel element
//! function (`translate`).
//!
//! - **Maps.** A function `fn(&mut [T], params...)` whose body is a single
//!   `for` loop over the slice (directly or through `iter_mut()`), or a
//!   single `for_each` call on such an iterator, sequential or Rayon's
//!   (`par_iter_mut()`), whose closure is the loop body. `T` is a number,
//!   or a struct, tuple or array of numbers with no padding
//!   (`translate::slice`), so its bytes can move to the GPU as they are.
//!   `enumerate()` pairs a supported iterator's item with its zero-based index.
//!   The loop may `zip` the slice with other `&mut` slices, which it then
//!   writes too, and with read-only slices; it may index read-only slices
//!   of numbers; and other parameters are plain data. The loop body may
//!   compute anything `translate` supports from those and constants, and
//!   write the elements. It may not leave the loop early, and a `for_each`
//!   closure may not change what it captures.
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

use cuprite_kernel::{Function, Ty as KTy};
use rustc_attr_ir::lang_items::LangItem;
use rustc_hir::def::DefKind;
use rustc_middle::{
    mir::{
        AggregateKind, BasicBlock, Body, BorrowKind, Local, Operand, Place, ProjectionElem, Rvalue,
        StatementKind, TerminatorKind, RETURN_PLACE,
    },
    ty::{self, Ty, TyCtxt},
};
use rustc_span::def_id::{LocalDefId, LOCAL_CRATE};

use crate::{
    replace::Passes,
    translate::{self, Lane, Leaf, Slice},
};

/// A map's parameters.
struct Params {
    /// The first parameter, the slice it changes.
    first: Slice,
    /// The other `&mut` slices, which the loop may zip with and write.
    outputs: HashMap<Local, Slice>,
    /// Read-only slices, which the loop may zip with or index.
    slices: HashMap<Local, Slice>,
    /// The numbers in every other parameter.
    leaves: Vec<Leaf>,
}

/// The slice parameter.
const SLICE: Local = Local::from_u32(1);
/// The pass-count parameter of a repeated map.
const COUNT: Local = Local::from_u32(2);

pub(crate) const REPORT_VAR: &str = "CUPRITE_REPORT";
pub(crate) const AUTO_VAR: &str = "CUPRITE_AUTO";

/// One map: what it does to each element of its slices.
#[derive(Clone, Debug)]
pub(crate) struct Map {
    /// The slices it writes: the first parameter, then any other `&mut`
    /// slices zipped with it.
    pub(crate) outputs: Vec<Slice>,
    /// Takes the numbers of each output's element, then of each input's,
    /// then the numbers `scalars` reads (as `Leaf::passed`), and returns
    /// the outputs' new numbers.
    pub(crate) function: Function,
    /// Read-only slice parameters zipped with the slice.
    pub(crate) inputs: Vec<Slice>,
    /// Read-only slice parameters the code indexes, and their element types.
    pub(crate) tables: Vec<(Local, KTy)>,
    /// Numbers read from the function's other parameters, in order.
    pub(crate) scalars: Vec<Leaf>,
}

#[derive(Clone, Debug)]
pub(crate) enum Verdict {
    /// One pass of a map.
    Candidate(Map),
    /// Maps applied in order, all of that `passes` times, which can stay on
    /// the GPU throughout. None of them reads inputs or scalars.
    Repeated(Vec<Map>, Passes),
    Rejected(String),
}

/// What is known about one function.
enum Recorded {
    /// Its built MIR, until it is recognized after analysis. Translation
    /// inlines the functions a loop body calls, from their optimized MIR,
    /// which is only safe to request once every body has been built: asking
    /// for it while building could reach this function again, a cycle.
    Saved(SavedBody),
    /// Being recognized; a request now comes from recursion and finds none.
    InProgress,
    Done(Verdict),
}

/// A function's built MIR, kept past the point where rustc frees it.
///
/// MIR refers to the compiler's arenas (`'tcx`). The copy is allocated in
/// those arenas, and the driver compiles one crate per process and reads
/// these only while compiling it, inside the same session, so the arenas
/// outlive every use. The lifetime is erased only to store the copy here.
struct SavedBody(&'static Body<'static>);

// SAFETY: the driver compiles on one thread at a time; the mutex around
// `RECORDED` only satisfies the static's bounds.
unsafe impl Send for SavedBody {}

impl SavedBody {
    fn save<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> Self {
        let saved: &'tcx Body<'tcx> = tcx.arena.alloc(body.clone());
        // SAFETY: see the type's documentation; `get` restores `'tcx`.
        Self(unsafe { std::mem::transmute::<&'tcx Body<'tcx>, &'static Body<'static>>(saved) })
    }

    fn get<'tcx>(&self, _tcx: TyCtxt<'tcx>) -> &'tcx Body<'tcx> {
        // SAFETY: the body was saved from this `tcx`'s session, which is
        // the only one the process runs.
        unsafe { std::mem::transmute::<&'static Body<'static>, &'tcx Body<'tcx>>(self.0) }
    }
}

/// Results for the crate being compiled, recorded as each function's MIR is
/// built. Query providers are plain function pointers, so this is global.
static RECORDED: Mutex<Option<HashMap<LocalDefId, Recorded>>> = Mutex::new(None);

pub(crate) fn record<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId, body: &Body<'tcx>) {
    if !matches!(tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn) {
        return;
    }
    // Only functions over a `&mut` slice can be candidates; refuse the rest
    // now, with the signature's reason.
    let takes_slice = body.arg_count >= 1 && Recognizer::is_mut_slice(body.local_decls[SLICE].ty);
    let recorded = if takes_slice {
        Recorded::Saved(SavedBody::save(tcx, body))
    } else {
        let reason = Recognizer::new(tcx, def_id, body)
            .check_signature(false)
            .err()
            .unwrap_or_else(|| "is not a candidate".into());
        Recorded::Done(Verdict::Rejected(reason))
    };
    RECORDED
        .lock()
        .unwrap()
        .get_or_insert_default()
        .insert(def_id, recorded);
}

/// The verdict for `def_id`. Call only after analysis, when every body has
/// been built.
///
/// Recognizes the function the first time it is asked, from its saved MIR.
/// The lock is released meanwhile: translation requests other functions'
/// optimized MIR, which asks for their verdicts.
pub(crate) fn verdict(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Option<Verdict> {
    let saved = {
        let mut recorded = RECORDED.lock().unwrap();
        let entry = recorded.as_mut()?.get_mut(&def_id)?;
        match entry {
            Recorded::Done(verdict) => return Some(verdict.clone()),
            Recorded::InProgress => return None,
            Recorded::Saved(_) => match std::mem::replace(entry, Recorded::InProgress) {
                Recorded::Saved(saved) => saved,
                _ => unreachable!("matched above"),
            },
        }
    };
    let verdict = Recognizer::new(tcx, def_id, saved.get(tcx))
        .recognize()
        .unwrap_or_else(Verdict::Rejected);
    RECORDED
        .lock()
        .unwrap()
        .get_or_insert_default()
        .insert(def_id, Recorded::Done(verdict.clone()));
    Some(verdict)
}

/// Composes a region's callees, each a map over the same slice called with
/// just the slice: a callee that is itself a region would need its own
/// pass count.
fn compose(tcx: TyCtxt<'_>, slice: &Slice, calls: &[LocalDefId], passes: Passes) -> Verdict {
    let mut stages = Vec::new();
    for &callee in calls {
        match verdict(tcx, callee) {
            Some(Verdict::Candidate(map))
                if matches!(&map.outputs[..], [output] if output.layout == slice.layout)
                    && map.scalars.is_empty()
                    && map.inputs.is_empty()
                    && map.tables.is_empty() =>
            {
                stages.push(map)
            }
            _ => {
                return Verdict::Rejected(format!(
                    "calls `{}`, which is not a recognized map",
                    tcx.def_path_str(callee)
                ));
            }
        }
    }
    if stages.is_empty() || passes == Passes::Const(0) {
        Verdict::Rejected("does not change the slice".into())
    } else {
        Verdict::Repeated(stages, passes)
    }
}

/// The element type of a function's first parameter, a slice, as Rust
/// prints it.
fn element_name(tcx: TyCtxt<'_>, def_id: LocalDefId) -> String {
    let sig = tcx.fn_sig(def_id).instantiate_identity().skip_binder();
    sig.inputs().first().and_then(|&ty| translate::slice_element(ty)).map_or_else(String::new, |ty| ty.to_string())
}

/// Describes a map for the report, such as "map over `[f32]` zipped with
/// `weights` reading `scale`", or "map over `[f64]` also writing `ys`".
fn describe_map(tcx: TyCtxt<'_>, def_id: LocalDefId, map: &Map) -> String {
    let mut text = format!("map over `[{}]`", element_name(tcx, def_id));
    let names = |locals: Vec<Local>| -> String {
        let mut seen = Vec::new();
        for local in locals {
            if !seen.contains(&local) {
                seen.push(local);
            }
        }
        seen.iter().map(|&local| parameter(tcx, def_id, local)).collect::<Vec<_>>().join(", ")
    };
    if map.outputs.len() > 1 {
        text += &format!(" also writing {}", names(map.outputs[1..].iter().map(|slice| slice.param).collect()));
    }
    if !map.inputs.is_empty() {
        text += &format!(" zipped with {}", names(map.inputs.iter().map(|slice| slice.param).collect()));
    }
    if !map.tables.is_empty() {
        text += &format!(" indexing {}", names(map.tables.iter().map(|&(local, _)| local).collect()));
    }
    let read: Vec<Local> = map.scalars.iter().filter(|leaf| leaf.feature.is_none()).map(|leaf| leaf.param).collect();
    if !read.is_empty() {
        text += &format!(" reading {}", names(read));
    }
    if map.scalars.iter().any(|leaf| leaf.feature.is_some()) {
        text += " with the host's CPU features";
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
                        element_name(tcx, def_id)
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
    /// `<[T]>::iter`.
    SliceIter,
    /// `Iterator::zip`.
    Zip,
    /// `Iterator::enumerate` or Rayon's `IndexedParallelIterator::enumerate`.
    Enumerate,
    /// `Iterator::next`.
    Next,
    /// `Iterator::for_each`.
    ForEach,
    /// A non-generic, non-trait function of this crate, which a repeated
    /// map may call, if the callee is a map.
    Local(LocalDefId),
    /// Any other function, which a map's loop body may call if `translate`
    /// can inline it.
    Other,
}

impl Known {
    /// Builds or runs an iterator.
    fn iterates(self) -> bool {
        matches!(
            self,
            Known::IntoIter | Known::SliceIterMut | Known::SliceIter | Known::Zip | Known::Enumerate | Known::Next | Known::ForEach
        )
    }
}

/// Rayon's methods that mean the same as the sequential ones, by trait and
/// method name. Indexed parallel slices preserve order through zip and enumerate.
const RAYON: [(&str, &str, Known); 5] = [
    ("IntoParallelRefMutIterator", "par_iter_mut", Known::SliceIterMut),
    ("IntoParallelRefIterator", "par_iter", Known::SliceIter),
    ("IndexedParallelIterator", "zip", Known::Zip),
    ("IndexedParallelIterator", "enumerate", Known::Enumerate),
    ("ParallelIterator", "for_each", Known::ForEach),
];

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

    fn recognize(&self) -> Result<Verdict> {
        // A loop over a range that calls maps repeats them; a range loop
        // inside a loop over the slice is part of that loop's body.
        if self.counts_passes() && !self.iterates_slice() {
            self.recognize_region()
        } else {
            Ok(Verdict::Candidate(self.recognize_map()?))
        }
    }

    /// A map: a `for` loop over the slice, or a `for_each` call on it.
    fn recognize_map(&self) -> Result<Map> {
        let params = self.check_signature(false)?;
        self.check_parameters_unchanged()?;
        let calls = self.check_effects(false)?;
        let has_next = calls.iter().any(|(_, known)| *known == Known::Next);
        let eaches: Vec<BasicBlock> =
            calls.iter().filter(|(_, known)| *known == Known::ForEach).map(|&(block, _)| block).collect();
        if !has_next && !eaches.is_empty() {
            return self.recognize_each(params, &calls, &eaches);
        }
        let (header, traced) = self.outer_loop(&calls, &params)?;
        let is_header = |block| self.is_header(block, header.next_block);
        let iteration = translate::Iteration::For { option: header.option, start: header.some, is_header: &is_header };
        let translated = self.translate_map(&params, &traced, iteration)?;
        self.check_rest(&calls, &translated.blocks, &traced.chain, &header)?;
        self.finish_map(traced, translated)
    }

    /// `iterator.for_each(function)`: the iteration calls `function` with
    /// what the iterator yields, as `Iterator::for_each` or Rayon's
    /// `ParallelIterator::for_each` does. Nothing may follow it.
    fn recognize_each(
        &self,
        params: Params,
        calls: &[(BasicBlock, Known)],
        eaches: &[BasicBlock],
    ) -> Result<Map> {
        let block = match eaches {
            [block] => *block,
            _ => return Err("contains more than one loop".into()),
        };
        let TerminatorKind::Call { args, target: Some(after), .. } = &self.body.basic_blocks[block].terminator().kind else {
            return Err("has an unsupported loop shape".into());
        };
        let [iterator, function] = &args[..] else {
            return Err("has an unsupported loop shape".into());
        };
        let receiver = Self::operand_local(&iterator.node).ok_or_else(Self::not_slice)?;
        let traced = self.trace(receiver, &params)?;
        let iteration = translate::Iteration::ForEach { block, function: &function.node };
        let translated = self.translate_map(&params, &traced, iteration)?;
        // Every other call builds the iterator, or is in the setup.
        for &(call, known) in calls {
            if call == block || traced.chain.contains(&call) {
                continue;
            }
            if known.iterates() {
                return Err("iterates outside the loop".into());
            }
            if !translated.blocks.contains(&call) {
                return Err(format!("uses `{}` after the loop", self.callee(call)));
            }
        }
        self.check_exit(*after)?;
        self.finish_map(traced, translated)
    }

    fn not_slice() -> String {
        "iterates over something other than the slice parameter".to_string()
    }

    /// What the loop's iterator yields, and the slices it and the code read
    /// and write.
    fn trace(&self, receiver: Local, params: &Params) -> Result<Traced> {
        let mut chain = HashSet::new();
        let mut plumbing = HashSet::from([receiver]);
        let lane = self.lanes(receiver, params, &mut chain, &mut plumbing)?;
        let mut output_locals = Vec::new();
        lane.outputs(&mut output_locals);
        if output_locals.iter().filter(|&&local| local == SLICE).count() != 1 {
            return Err(Self::not_slice());
        }
        // Every other `&mut` slice is written element by element, alongside
        // the first.
        let mut others: Vec<Local> = params.outputs.keys().copied().collect();
        others.sort();
        if let Some(&unused) = others.iter().find(|local| !output_locals.contains(local)) {
            return Err(format!(
                "takes {} as a `&mut` slice without iterating it alongside the first",
                parameter(self.tcx, self.def_id, unused)
            ));
        }
        let outputs: Vec<Slice> = std::iter::once(params.first.clone())
            .chain(output_locals.iter().filter(|&&local| local != SLICE).map(|local| params.outputs[local].clone()))
            .collect();
        let mut input_locals = Vec::new();
        lane.inputs(&mut input_locals);
        let inputs: Vec<Slice> = input_locals.iter().map(|local| params.slices[local].clone()).collect();
        // The other slices can be indexed, with bounds checks.
        let mut tables = Vec::new();
        for (&local, slice) in params.slices.iter().filter(|(local, _)| !input_locals.contains(local)) {
            let Some(element) = slice.layout.as_number() else {
                return Err(format!(
                    "takes {}, a slice of `{}`, without zipping it; only slices of numbers can be indexed",
                    parameter(self.tcx, self.def_id, local),
                    translate::slice_element(self.body.local_decls[local].ty).expect("a slice parameter")
                ));
            };
            tables.push((local, element));
        }
        tables.sort_by_key(|&(local, _)| local);
        let opaque: Vec<Local> = output_locals.iter().chain(&input_locals).copied().collect();
        Ok(Traced { lane, chain, plumbing, outputs, inputs, tables, opaque })
    }

    fn translate_map(
        &self,
        params: &Params,
        traced: &Traced,
        iteration: translate::Iteration<'_, 'tcx>,
    ) -> Result<translate::Translated> {
        let name = format!("{}::{}", self.tcx.crate_name(LOCAL_CRATE), self.tcx.def_path_str(self.def_id));
        // The host's answers to CPU feature tests are leaves too: found by
        // translating, then passed by translating again with them.
        let mut leaves = params.leaves.clone();
        for _ in 0..2 {
            let shape = translate::Loop {
                iteration,
                lane: &traced.lane,
                outputs: &traced.outputs,
                inputs: &traced.inputs,
                tables: &traced.tables,
                leaves: &leaves,
                opaque: &traced.opaque,
                chain: &traced.chain,
                plumbing: &traced.plumbing,
            };
            let translated = translate::translate(self.tcx, self.body, &name, &shape)?;
            let missing: Vec<translate::Feature> = translated
                .features
                .iter()
                .copied()
                .filter(|&feature| !leaves.iter().any(|leaf| leaf.feature == Some(feature)))
                .collect();
            if missing.is_empty() {
                return Ok(translated);
            }
            leaves.extend(missing.into_iter().map(Leaf::feature));
        }
        Err("tests different CPU features each time it is translated".into())
    }

    fn finish_map(&self, traced: Traced, translated: translate::Translated) -> Result<Map> {
        if !translated.stores {
            return Err("does not change the slice".into());
        }
        Ok(Map {
            outputs: traced.outputs,
            function: translated.function,
            inputs: traced.inputs,
            tables: traced.tables,
            scalars: translated.leaves,
        })
    }

    /// A counted loop whose body calls maps of this crate on the slice.
    fn recognize_region(&self) -> Result<Verdict> {
        let params = self.check_signature(true)?;
        self.check_parameters_unchanged()?;
        let calls = self.check_effects(true)?;
        let header = self.single_loop(&calls, "is not a single counted loop")?;
        let (range_setup, passes) = self.pass_count(header.receiver)?;

        let mut walked = HashSet::new();
        let callees = self.region_calls(header.some, header.next_block, &mut walked)?;
        self.check_region_rest(&calls, &walked, &[range_setup], &header)?;
        Ok(compose(self.tcx, &params.first, &callees, passes))
    }

    /// True if the function loops over an integer `Range`, which makes it a
    /// candidate for the repeated-map rule.
    /// Whether a `for` loop iterates over the slice parameter.
    fn iterates_slice(&self) -> bool {
        let (Ok(params), Ok(calls)) = (self.check_signature(false), self.check_effects(false)) else {
            return false;
        };
        self.outer_loop(&calls, &params).is_ok()
    }

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

    /// The `for` loop over the slice parameter, and its iterator traced. Other
    /// `for` loops may run inside its body, as calls `translate` inlines;
    /// with several loops, the outer one is the one whose iterator traces
    /// back to the slice.
    fn outer_loop(&self, calls: &[(BasicBlock, Known)], params: &Params) -> Result<(LoopHeader<'a, 'tcx>, Traced)> {
        let nexts: Vec<BasicBlock> =
            calls.iter().filter(|(_, known)| *known == Known::Next).map(|&(block, _)| block).collect();
        let mut outer = Vec::new();
        let mut first_error = None;
        for &next in &nexts {
            let header = self.single_loop(&[(next, Known::Next)], "is not a single `for` loop over the slice")?;
            let traced = Self::operand_local(header.receiver)
                .ok_or_else(Self::not_slice)
                .and_then(|receiver| self.trace(receiver, params));
            match traced {
                Ok(traced) => outer.push((header, traced)),
                Err(error) => first_error = first_error.or(Some(error)),
            }
        }
        match (outer.len(), nexts.len()) {
            (_, 0) => Err("is not a single `for` loop over the slice".into()),
            (1, _) => Ok(outer.pop().expect("one loop")),
            (0, 1) => Err(first_error.expect("the loop's error")),
            _ => Err("contains more than one loop".into()),
        }
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

    /// A map: the iterator's calls must all build the loop's iterator, and
    /// nothing may follow the loop. `translated` holds the setup before the
    /// loop and the body, which the kernel runs; every other call is part of
    /// the iterator or the loop header.
    fn check_rest(
        &self,
        calls: &[(BasicBlock, Known)],
        translated: &HashSet<BasicBlock>,
        chain: &HashSet<BasicBlock>,
        header: &LoopHeader<'_, 'tcx>,
    ) -> Result<()> {
        for &(block, known) in calls {
            // Loops inside the body are calls `translate` inlined; one over
            // a slice parameter fails there.
            let inner = translated.contains(&block) && !chain.contains(&block);
            if known.iterates() && !inner && !chain.contains(&block) && block != header.next_block {
                return Err("iterates inside the loop body".into());
            }
            if !translated.contains(&block) && !chain.contains(&block) && block != header.next_block {
                return Err(format!("uses `{}` after the loop", self.callee(block)));
            }
        }
        self.check_exit(header.none)
    }

    /// A region: everything the loop body walk did not visit is the
    /// iterator setup, the loop header and the exit; none of it may call,
    /// compute or write.
    fn check_region_rest(
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
            if walked.contains(&block) && matches!(known, Known::IntoIter | Known::SliceIterMut | Known::Next) {
                return Err("iterates inside the loop body".into());
            }
        }
        self.check_outside_loop(walked, header.after_next, header.next_block, header.none)
    }

    /// `fn(&mut [T], params...) -> ()` with `T` a number or a struct, tuple
    /// or array of numbers with no padding (`translate::slice`), each other
    /// parameter a slice of such elements or plain data; or with `counted`,
    /// `fn(&mut [T])` or `fn(&mut [T], u32)`, the second being the pass
    /// count.
    fn check_signature(&self, counted: bool) -> Result<Params> {
        if self.tcx.generics_of(self.def_id).count() != 0 {
            return Err("is generic".into());
        }
        let params = self.body.arg_count;
        if params == 0 {
            return Err("takes no parameters; the first must be a `&mut` slice".into());
        }
        if counted && params > 2 {
            return Err(format!(
                "takes {params} parameters; a repeated map takes the slice and at most a `u32` pass count"
            ));
        }
        let param = self.body.local_decls[SLICE].ty;
        if !Self::is_mut_slice(param) {
            return Err(format!("takes `{param}`, not a `&mut` slice"));
        }
        let first = translate::slice(self.tcx, SLICE, translate::slice_element(param).expect("a slice"))?;
        if counted && params == 2 {
            let count = self.body.local_decls[COUNT].ty;
            if count != self.tcx.types.u32 {
                return Err(format!("takes `{count}` as its pass count, not `u32`"));
            }
        }
        let mut found = Params { first, outputs: HashMap::new(), slices: HashMap::new(), leaves: Vec::new() };
        if !counted {
            for local in (2..=params).map(Local::from_usize) {
                let ty = self.body.local_decls[local].ty;
                let cannot = || format!("takes `{ty}` as {}, which a kernel can't take", parameter(self.tcx, self.def_id, local));
                if let Some(element) = translate::slice_element(ty) {
                    let slice = translate::slice(self.tcx, local, element).map_err(|_| cannot())?;
                    match Self::is_mut_slice(ty) {
                        true => found.outputs.insert(local, slice),
                        false => found.slices.insert(local, slice),
                    };
                    continue;
                }
                let leaves = translate::leaves(self.tcx, local, ty).map_err(|_| cannot())?;
                found.leaves.extend(leaves);
            }
        }
        let output = self.body.local_decls[RETURN_PLACE].ty;
        if !output.is_unit() {
            return Err(format!("returns `{output}`"));
        }
        Ok(found)
    }

    /// Whether `ty` is `&mut [T]`.
    fn is_mut_slice(ty: Ty<'tcx>) -> bool {
        matches!(ty.kind(), ty::Ref(_, inner, ty::Mutability::Mut) if matches!(inner.kind(), ty::Slice(_)))
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

    /// Rejects anything that may run a destructor or call unknown code;
    /// returns the known calls. Calls to functions of this crate are
    /// allowed only with `allow_local`.
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
                    // A map's calls are inlined by `translate`; a region may
                    // only call maps of this crate.
                    if allow_local && known == Known::Other {
                        return Err(self.no_summary(func));
                    }
                    calls.push((block, known));
                }
                // Checks run on the GPU too; one that fails there sends the
                // call back to the CPU, which panics as before.
                TerminatorKind::Assert { .. } => {}
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
        if let Some(impl_id) = tcx.inherent_impl_of_assoc(def_id) {
            let self_ty = tcx.type_of(impl_id).instantiate_identity().skip_normalization();
            if tcx.crate_name(impl_id.krate).as_str() == "core" && matches!(self_ty.kind(), ty::Slice(_)) {
                match tcx.item_name(def_id).as_str() {
                    "iter_mut" => return Ok(Known::SliceIterMut),
                    "iter" => return Ok(Known::SliceIter),
                    _ => {}
                }
            }
        }
        if let Some(trait_id) = tcx.trait_of_assoc(def_id) {
            let method = tcx.item_name(def_id);
            if tcx.is_lang_item(trait_id, LangItem::Iterator) {
                match method.as_str() {
                    "zip" => return Ok(Known::Zip),
                    "enumerate" => return Ok(Known::Enumerate),
                    "for_each" => return Ok(Known::ForEach),
                    _ => {}
                }
            }
            if tcx.crate_name(trait_id.krate).as_str() == "rayon" {
                let trait_name = tcx.item_name(trait_id);
                if let Some(&(_, _, known)) =
                    RAYON.iter().find(|(t, m, _)| *t == trait_name.as_str() && *m == method.as_str())
                {
                    return Ok(known);
                }
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
        Ok(Known::Other)
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
    fn origin(&self, local: Local) -> Option<Origin> {
        self.origin_through(local, &mut HashSet::new())
    }

    /// `origin`, recording every local passed through in `through`.
    fn origin_through(&self, mut local: Local, through: &mut HashSet<Local>) -> Option<Origin> {
        for _ in 0..self.body.local_decls.len() {
            through.insert(local);
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

    /// What an iterator yields, from the calls that built it: `into_iter`
    /// or `iter_mut` on a `&mut` slice parameter, `iter` or a plain `&[T]`
    /// on an input slice parameter, `zip`, and `enumerate`. Records the calls' blocks in
    /// `chain` and every local the iterator passes through in `plumbing`.
    fn lanes(
        &self,
        iterator: Local,
        params: &Params,
        chain: &mut HashSet<BasicBlock>,
        plumbing: &mut HashSet<Local>,
    ) -> Result<Lane> {
        let slices = &params.slices;
        let written = |param: Local| param == SLICE || params.outputs.contains_key(&param);
        let not_slice = || "iterates over something other than the slice parameter".to_string();
        let Some(Origin::Call(block)) = self.origin_through(iterator, plumbing) else {
            return Err(not_slice());
        };
        let TerminatorKind::Call { func, args, destination, .. } = &self.body.basic_blocks[block].terminator().kind
        else {
            return Err(not_slice());
        };
        chain.insert(block);
        plumbing.insert(destination.local);
        let argument = |index: usize| args.get(index).and_then(|arg| Self::operand_local(&arg.node)).ok_or_else(not_slice);
        match self.classify(func)? {
            known @ (Known::IntoIter | Known::SliceIterMut) => {
                let source = argument(0)?;
                match self.origin_through(source, plumbing) {
                    Some(Origin::Param(param)) if written(param) => Ok(Lane::Output(param)),
                    // The `for` loop's identity `into_iter` on an iterator.
                    Some(Origin::Call(_)) if known == Known::IntoIter => self.lanes(source, params, chain, plumbing),
                    _ => Err(not_slice()),
                }
            }
            // An input slice's `iter()`, which may lead a `zip`.
            Known::SliceIter => match self.origin_through(argument(0)?, plumbing) {
                Some(Origin::Param(param)) if slices.contains_key(&param) => Ok(Lane::Input(param)),
                _ => Err(not_slice()),
            },
            Known::Zip => {
                let first = self.lanes(argument(0)?, params, chain, plumbing)?;
                let second = self.zipped(argument(1)?, params, chain, plumbing)?;
                Ok(Lane::Pair(Box::new(first), Box::new(second)))
            }
            Known::Enumerate => {
                let item = self.lanes(argument(0)?, params, chain, plumbing)?;
                Ok(Lane::Pair(Box::new(Lane::Index), Box::new(item)))
            }
            _ => Err(not_slice()),
        }
    }

    /// What `zip` pairs with: an input slice, as `&[T]` or `.iter()`,
    /// another `&mut` slice, as `&mut [T]` or `.iter_mut()`, or another
    /// iterator.
    fn zipped(
        &self,
        source: Local,
        params: &Params,
        chain: &mut HashSet<BasicBlock>,
        plumbing: &mut HashSet<Local>,
    ) -> Result<Lane> {
        let slices = &params.slices;
        match self.origin_through(source, plumbing) {
            Some(Origin::Param(param)) if slices.contains_key(&param) => Ok(Lane::Input(param)),
            Some(Origin::Param(param)) if params.outputs.contains_key(&param) => Ok(Lane::Output(param)),
            Some(Origin::Call(block)) => {
                let TerminatorKind::Call { func, args, destination, .. } =
                    &self.body.basic_blocks[block].terminator().kind
                else {
                    unreachable!("an origin call is a call");
                };
                if self.classify(func)? == Known::SliceIter {
                    let receiver = args.first().and_then(|arg| Self::operand_local(&arg.node));
                    if let Some(Origin::Param(param)) = receiver.and_then(|r| self.origin_through(r, plumbing)) {
                        if slices.contains_key(&param) {
                            chain.insert(block);
                            plumbing.insert(destination.local);
                            return Ok(Lane::Input(param));
                        }
                    }
                }
                self.lanes(source, params, chain, plumbing)
            }
            _ => Err("zips with something other than a slice parameter".into()),
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
        self.check_exit(exit)
    }

    /// The exit must return without doing anything else: the GPU path
    /// skips it.
    fn check_exit(&self, exit: BasicBlock) -> Result<()> {
        let mut block = exit;
        for _ in 0..self.body.basic_blocks.len() {
            let data = &self.body.basic_blocks[block];
            let writes = data.statements.iter().any(|statement| {
                matches!(&statement.kind, StatementKind::Assign(assign) if assign.0.is_indirect())
            });
            if writes {
                break;
            }
            match data.terminator().kind {
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

/// A map's iterator, traced back to the slices.
struct Traced {
    lane: Lane,
    /// The calls that build the iterator.
    chain: HashSet<BasicBlock>,
    /// The locals the iterator passes through.
    plumbing: HashSet<Local>,
    /// The slices the iterator writes, the first parameter first.
    outputs: Vec<Slice>,
    /// The input slices the iterator zips with, in order.
    inputs: Vec<Slice>,
    /// The other slices, which the code may index.
    tables: Vec<(Local, KTy)>,
    /// The slices only the iterator may use.
    opaque: Vec<Local>,
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

