//! Translate one iteration of a recognized MIR loop into a kernel element function.
//! The entry point [`translate`] binds slice elements and parameters, then translates
//! the setup and loop body. [`check_function`] uses the same translator for the census.
//!
//! A [`Frame`] holds one MIR body's bindings and control flow. Inlined calls get
//! their own frame and share the kernel builder. [`Slot`] describes where a value
//! lives, including references to other values and reads from GPU memory.
//!
//! MIR assertions that cannot be folded run as kernel checks. A failed check
//! discards the GPU result and leaves the original CPU body to reproduce the panic.
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

mod fold;
mod intrinsics;
mod layout;
mod pointers;
mod refusal;
mod regions;
mod rvalues;
mod simd;
mod slots;
mod types;
mod x86;

pub(crate) use intrinsics::intrinsic;
pub(crate) use types::{fields_of, leaves, slice, slice_element, Feature, Leaf, Slice};

use cuprite_kernel::{
    builder::Builder, BinOp as KBinOp, BlockId, Const, Function, Local as KLocal,
    Operand as KOperand, Rvalue as KRvalue, Ty as KTy, UnOp as KUnOp,
};
use rustc_abi::{ExternAbi, VariantIdx, Variants};
use rustc_middle::mir::{
    BasicBlock, Body, Local, NonDivergingIntrinsic, Operand, Place, Rvalue, StatementKind,
    TerminatorKind, RETURN_PLACE, START_BLOCK,
};
use rustc_middle::ty::{self, Instance, InstanceKind, ShimKind, Ty, TyCtxt, TypingEnv};
use rustc_span::def_id::DefId;
use std::collections::{HashMap, HashSet};

use fold::assert_holds;
use intrinsics::{method, IntrinsicOp, Method};
use regions::{
    assigned_outside, check_carried, constant_target, doomed, function_region, loop_region,
    ordered, setup_region,
};
use slots::Slot;

type Result<T> = std::result::Result<T, String>;

/// What a `for` loop's iterator yields: an element of a slice the loop
/// writes, a value of a read-only input slice, or a pair of those, from
/// `zip`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Lane {
    /// `&mut` to the current element of this `&mut` slice parameter.
    Output(Local),
    /// `&` to the current element of this input slice parameter.
    Input(Local),
    Pair(Box<Lane>, Box<Lane>),
}

impl Lane {
    /// The input slices, in the order the tuple names them.
    pub(crate) fn inputs(&self, out: &mut Vec<Local>) {
        match self {
            Lane::Output(_) => {}
            Lane::Input(local) => out.push(*local),
            Lane::Pair(a, b) => {
                a.inputs(out);
                b.inputs(out);
            }
        }
    }

    /// The slices written, in the order the tuple names them.
    pub(crate) fn outputs(&self, out: &mut Vec<Local>) {
        match self {
            Lane::Output(local) => out.push(*local),
            Lane::Input(_) => {}
            Lane::Pair(a, b) => {
                a.outputs(out);
                b.outputs(out);
            }
        }
    }
}

/// What the function around a loop reads and writes, around the
/// translation.
pub(crate) struct Loop<'a, 'tcx> {
    pub(crate) iteration: Iteration<'a, 'tcx>,
    /// What the iterator yields.
    pub(crate) lane: &'a Lane,
    /// The slices `lane` writes, the function's first parameter first.
    pub(crate) outputs: &'a [Slice],
    /// The input slices `lane` reads, in order.
    pub(crate) inputs: &'a [Slice],
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

    // Each written and read slice's element numbers, then the tables, then
    // the other parameters' numbers.
    let iterated: Vec<&Slice> = shape.outputs.iter().chain(shape.inputs).collect();
    let mut params: Vec<KTy> = iterated.iter().flat_map(|slice| slice.layout.types()).collect();
    let element_param_count = params.len();
    for _ in shape.tables {
        params.extend([KTy::Ptr, KTy::U64]);
    }
    params.extend(shape.leaves.iter().map(Leaf::passed));
    let (builder, kernel_params) = Builder::new(name, &params);
    let mut translator = Translator::new(tcx, builder);

    let mut slots = HashMap::new();
    for &local in shape.opaque {
        slots.insert(local, Slot::Opaque);
    }
    // This iteration's element of each iterated slice, as a value of its
    // type, its numbers copied from the parameters. The written slices'
    // numbers are the function's results.
    let mut elements: HashMap<Local, Slot> = HashMap::new();
    let mut results = Vec::new();
    let (element_params, remaining) = kernel_params.split_at(element_param_count);
    let (table_params, leaf_params) = remaining.split_at(2 * shape.tables.len());
    let mut next_param = element_params.iter();
    for (position, slice) in iterated.iter().enumerate() {
        let element = slice_element(body.local_decls[slice.param].ty).expect("an iterated slice");
        let tree = translator.fresh(element)?;
        for path in &slice.paths {
            let local = tree.scalar_at(path);
            let argument = *next_param.next().expect("a parameter per number");
            translator.b.set(local, KRvalue::Use(argument.into()));
            if position < shape.outputs.len() {
                results.push(local);
            }
        }
        elements.insert(slice.param, tree);
    }
    translator.elements = results.iter().copied().collect();
    // The tables, as their address and length.
    for (&(local, element), pair) in shape.tables.iter().zip(table_params.as_chunks::<2>().0) {
        let slice = Slot::Slice { pointer: pair[0], len: pair[1], element };
        slots.insert(local, Slot::Ref(Box::new(slice)));
    }
    translator.bind_scalar_parameters(body, shape.leaves, leaf_params, &mut slots)?;
    let item = lane_slot(shape.lane, &elements);
    if let Iteration::For { option, .. } = &shape.iteration {
        slots.insert(option.local, Slot::Some(Box::new(item.clone())));
    }

    let mut blocks: HashSet<BasicBlock> = setup.union(&loop_blocks).copied().collect();
    let mut frame = Frame {
        body,
        slots,
        blocks: HashMap::new(),
        exit: translator.b.block(),
        loop_rules: Some(LoopRules {
            shape,
            body: loop_blocks,
            assigned_outside: assigned_outside(body, &blocks),
            item,
        }),
    };
    translator.run_frame(&mut frame, &ordered(body, &blocks), START_BLOCK)?;
    let stores = translator.stores;
    let features = std::mem::take(&mut translator.wanted);
    let function = translator
        .b
        .finish_function(results)
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
    let (builder, _) = Builder::new("coverage", &[]);
    let mut translator = Translator::new(tcx, builder);
    translator.stack.push(def_id);
    let mut slots = HashMap::new();
    for param in (1..=body.arg_count).map(Local::from_usize) {
        let ty = body.local_decls[param].ty;
        slots.insert(param, translator.fresh_deep(ty, 0)?);
    }
    let mut frame = Frame { body, slots, blocks: HashMap::new(), exit: translator.b.block(), loop_rules: None };
    translator.run_frame(&mut frame, &ordered(body, &function_region(body, tcx)), START_BLOCK)?;
    translator.b.switch_to(frame.exit);
    translator.local_slot(&mut frame, RETURN_PLACE)?;
    translator
        .b
        .finish_function(None)
        .map(|_| ())
        .map_err(|invalid| format!("translated into an invalid kernel ({invalid})"))
}

/// The slot for what the iterator yields: references to this iteration's
/// element of each slice, paired as `zip` pairs them.
fn lane_slot(lane: &Lane, elements: &HashMap<Local, Slot>) -> Slot {
    match lane {
        Lane::Output(local) | Lane::Input(local) => Slot::Ref(Box::new(elements[local].clone())),
        Lane::Pair(a, b) => Slot::Fields(vec![lane_slot(a, elements), lane_slot(b, elements)]),
    }
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
    /// The numbers of the elements the loop writes, as they are now.
    elements: HashSet<KLocal>,
    stores: bool,
    /// The functions being inlined, innermost last.
    stack: Vec<DefId>,
    /// Every local written, for `each`.
    written: HashSet<KLocal>,
    /// The kernel parameters holding the host's answers to feature tests.
    features: Vec<(Feature, KLocal)>,
    /// The feature tests the code makes, in order.
    wanted: Vec<Feature>,
    /// Positions of pointers that can move (`own_pointer`), which are
    /// never folded to constants: a later move may run before a later use.
    moving: HashSet<KLocal>,
}

/// How deep inlining may go.
const MAX_DEPTH: usize = 24;

/// References and raw pointers, and pattern types over them (`NonNull`'s
/// field), which are pointers too.
fn is_ref(ty: Ty<'_>) -> bool {
    match ty.kind() {
        ty::Ref(..) | ty::RawPtr(..) => true,
        ty::Pat(inner, _) => is_ref(*inner),
        _ => false,
    }
}

/// Whether `local` is assigned at most once in `body` and never borrowed
/// mutably, so its value can't change after it is set.
fn never_reassigned(body: &Body<'_>, local: Local) -> bool {
    let mut assignments = 0;
    for data in body.basic_blocks.iter() {
        for statement in &data.statements {
            let StatementKind::Assign(assign) = &statement.kind else { continue };
            if assign.0.local == local {
                assignments += 1;
            }
            if let Rvalue::Ref(_, rustc_middle::mir::BorrowKind::Mut { .. }, place) | Rvalue::RawPtr(rustc_middle::mir::RawPtrKind::Mut, place) = &assign.1 {
                if place.local == local {
                    return false;
                }
            }
        }
        if let TerminatorKind::Call { destination, .. } = &data.terminator().kind {
            if destination.local == local {
                assignments += 1;
            }
        }
    }
    assignments <= 1
}

/// The largest union held as bytes, at 8 bytes a number.
const MAX_UNION_BYTES: u64 = 1024;

/// Whether a value of `ty` holds a reference, pointer or anything else
/// that isn't plain bytes.
fn holds_pointer<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>, depth: usize) -> bool {
    if depth > 16 {
        return true;
    }
    match ty.kind() {
        ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::Never => false,
        ty::Array(element, _) | ty::Pat(element, _) => holds_pointer(tcx, *element, depth + 1),
        ty::Tuple(fields) => fields.iter().any(|field| holds_pointer(tcx, field, depth + 1)),
        ty::Adt(adt, _) if adt.is_box() => true,
        ty::Adt(adt, args) => adt.all_fields().any(|field| {
            let field = tcx.normalize_erasing_regions(TypingEnv::fully_monomorphized(), field.ty(tcx, args));
            holds_pointer(tcx, field, depth + 1)
        }),
        _ => true,
    }
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
    fn new(tcx: TyCtxt<'tcx>, builder: Builder) -> Self {
        Self {
            tcx,
            b: builder,
            elements: HashSet::new(),
            stores: false,
            stack: Vec::new(),
            written: HashSet::new(),
            features: Vec::new(),
            wanted: Vec::new(),
            moving: HashSet::new(),
        }
    }

    /// Rebuild plain-data parameters in MIR argument order, including parameters with no fields.
    fn bind_scalar_parameters(
        &mut self,
        body: &Body<'tcx>,
        leaves: &[Leaf],
        arguments: &[KLocal],
        slots: &mut HashMap<Local, Slot>,
    ) -> Result<()> {
        let mut by_param: HashMap<Local, Vec<(&Leaf, KLocal)>> = HashMap::new();
        for (leaf, &argument) in leaves.iter().zip(arguments) {
            if let Some(feature) = leaf.feature {
                self.features.push((feature, argument));
            } else {
                by_param.entry(leaf.param).or_default().push((leaf, argument));
            }
        }

        for param in (2..=body.arg_count).map(Local::from_usize) {
            if slots.contains_key(&param) {
                continue;
            }
            let ty = body.local_decls[param].ty;
            let value_type = match ty.kind() {
                ty::Ref(_, inner, _) => *inner,
                _ => ty,
            };
            let value = self.fresh(value_type)?;
            for (leaf, argument) in by_param.remove(&param).unwrap_or_default() {
                let scalar = if leaf.ty == KTy::Bool {
                    KRvalue::Binary(KBinOp::Ne, argument.into(), Const::int(KTy::U32, 0).into())
                } else if leaf.ty != leaf.passed() {
                    KRvalue::Cast(argument.into(), leaf.ty)
                } else {
                    KRvalue::Use(argument.into())
                };
                self.b.set(value.scalar_at(&leaf.fields), scalar);
            }
            let slot = if is_ref(ty) { Slot::Ref(Box::new(value)) } else { value };
            slots.insert(param, slot);
        }
        Ok(())
    }

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
        if self.elements.contains(&local) {
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
                if let Some(live) = constant_target(frame.body, block, self.tcx) {
                    let target = self.target(frame, block, live)?;
                    self.b.goto(target);
                    return Ok(());
                }
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
                if op == IntrinsicOp::Exact {
                    return self.exact(frame, name.as_str(), instance.args, args, destination);
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
        let region = function_region(&body, tcx);
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
