//! Body replacement (M1 question 4), runtime linkage (question 5), and the
//! kernels the rewritten bodies run.
//!
//! Functions are chosen by the recognizer (`analyze`, with
//! `CUPRITE_AUTO=1`), which also translates what they compute, or by hand:
//! `CUPRITE_REPLACE` names functions to rewrite and the multiply-add map
//! each computes, for shapes the recognizer does not handle:
//!
//! ```text
//! CUPRITE_REPLACE=cuprite_probes::transform=3,7,1;cuprite_probes::repeat=3,7,arg2
//! ```
//!
//! Each entry is `<crate>::<path>=<mul>,<add>,<passes>`, where `<passes>` is
//! a constant or `argN`, the function's N-th parameter (1-based). The named
//! function must take `&mut [u32]` first and return `()`.
//!
//! Either way, the driver builds an in-place map kernel
//! (`cuprite_kernel::shapes::in_place_map`), compiles it to PTX through
//! LLVM, and embeds its description in the rewritten body, which first
//! calls `cuprite_runtime::map_slice(values, description, &[scalars...])`.
//! If that returns 0, the GPU did the work and the function returns.
//! Otherwise the original body runs, unchanged, and then passes what
//! `map_slice` returned to `cuprite_runtime::finished`, which lets the
//! runtime time it. The driver loads the runtime crate itself
//! (`--extern force:`), so the rewritten crate never declares it.

use std::sync::OnceLock;

use cuprite_kernel::{
    builder::Builder, description::Description, llvm::Llvm, ptx, shapes, BinOp as KBinOp, Const, Function,
    Kernel, Ty as KTy,
};
use rustc_abi::FieldIdx;
use rustc_hir::def::DefKind;
use rustc_middle::{
    mir::{
        interpret::{Scalar, CTFE_ALLOC_SALT},
        AggregateKind, BasicBlockData, Body, BorrowKind, CallSource, CastKind, CoercionSource, ConstOperand,
        ConstValue, Local, LocalDecl, ProjectionElem, RawPtrKind, UnOp,
        Operand, Place, Rvalue, SourceInfo, Statement, StatementKind, Terminator, TerminatorKind, UnwindAction,
        START_BLOCK,
    },
    ty::{self, adjustment::PointerCoercion, Ty, TyCtxt},
};
use rustc_span::{
    def_id::{DefId, LocalDefId, LOCAL_CRATE},
    Span, Spanned,
};

use crate::{
    analyze::{self, Verdict},
    translate::{self, Leaf},
};

pub(crate) const VAR: &str = "CUPRITE_REPLACE";
pub(crate) const RUNTIME_VAR: &str = "CUPRITE_RUNTIME";
pub(crate) const RUNTIME_DEPS_VAR: &str = "CUPRITE_RUNTIME_DEPS";
pub(crate) const RUNTIME_CRATE: &str = "cuprite_runtime";
const ENTRY: &str = "map_slice";
const FINISHED: &str = "finished";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Passes {
    Const(u32),
    /// A parameter, numbered from 1 like MIR's argument locals.
    Arg(usize),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Replacement {
    pub(crate) path: String,
    pub(crate) mul: u32,
    pub(crate) add: u32,
    pub(crate) passes: Passes,
}

pub(crate) fn parse(spec: &str) -> Result<Vec<Replacement>, String> {
    spec.split(';')
        .filter(|entry| !entry.trim().is_empty())
        .map(|entry| {
            let malformed = || format!("malformed {VAR} entry `{entry}`");
            let (path, values) = entry.trim().split_once('=').ok_or_else(malformed)?;
            let [mul, add, passes] = values.split(',').collect::<Vec<_>>()[..] else {
                return Err(malformed());
            };
            let passes = match passes.strip_prefix("arg") {
                Some(index) => match index.parse() {
                    Ok(index) if index >= 1 => Passes::Arg(index),
                    _ => return Err(malformed()),
                },
                None => Passes::Const(passes.parse().map_err(|_| malformed())?),
            };
            Ok(Replacement {
                path: path.to_owned(),
                mul: mul.parse().map_err(|_| malformed())?,
                add: add.parse().map_err(|_| malformed())?,
                passes,
            })
        })
        .collect()
}

/// What a rewritten function runs on the GPU.
struct Job {
    element: KTy,
    /// Applied to each element in order.
    stages: Vec<Function>,
    /// For a repeated map, how many times.
    passes: Option<Passes>,
    /// Read-only slice parameters zipped with the slice, and their element
    /// types.
    inputs: Vec<(Local, KTy)>,
    /// Read-only slice parameters the code indexes, and their element types.
    tables: Vec<(Local, KTy)>,
    /// Numbers read from the other parameters, passed as the stages'
    /// scalars.
    scalars: Vec<Leaf>,
}

/// `x * mul + add` over `u32`, wrapping: what a `CUPRITE_REPLACE` entry
/// declares.
fn multiply_add(path: &str, mul: u32, add: u32) -> Function {
    let (mut b, params) = Builder::new(path, &[KTy::U32]);
    let product = b.binary(KBinOp::Mul, params[0], Const::int(KTy::U32, mul.into()));
    let sum = b.binary(KBinOp::Add, product, Const::int(KTy::U32, add.into()));
    b.finish_function(Some(sum)).expect("a valid multiply-add")
}

/// Wraps rustc's own `optimized_mir`. A function named in `CUPRITE_REPLACE`
/// gets a rewritten copy; with `CUPRITE_AUTO=1`, so does every function the
/// recognizer accepted. Everything else is returned unchanged.
pub(crate) fn optimized_mir<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> &'tcx Body<'tcx> {
    let body = (rustc_interface::DEFAULT_QUERY_PROVIDERS
        .queries
        .optimized_mir)(tcx, def_id);
    if !matches!(tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn) {
        return body;
    }
    let settings = crate::settings();
    let path = format!(
        "{}::{}",
        tcx.crate_name(LOCAL_CRATE),
        tcx.def_path_str(def_id)
    );
    let manual = settings.replacements.iter().find(|r| r.path == path).map(|replacement| {
        check_signature(tcx, def_id, replacement);
        Job {
            element: KTy::U32,
            stages: vec![multiply_add(&path, replacement.mul, replacement.add)],
            passes: Some(replacement.passes),
            inputs: Vec::new(),
            tables: Vec::new(),
            scalars: Vec::new(),
        }
    });
    let automatic = || {
        if !settings.auto {
            return None;
        }
        Some(match analyze::verdict(tcx, def_id)? {
            Verdict::Candidate(map) => Job {
                element: map.element,
                stages: vec![map.function],
                passes: None,
                inputs: map.inputs,
                tables: map.tables,
                scalars: map.scalars,
            },
            Verdict::Repeated(maps, passes) => Job {
                element: maps[0].element,
                stages: maps.into_iter().map(|map| map.function).collect(),
                passes: Some(passes),
                inputs: Vec::new(),
                tables: Vec::new(),
                scalars: Vec::new(),
            },
            Verdict::Rejected(_) => return None,
        })
    };
    let Some(job) = manual.or_else(automatic) else {
        return body;
    };
    let [entry, finished] = [ENTRY, FINISHED].map(|name| {
        runtime_item(tcx, name).unwrap_or_else(|| {
            tcx.dcx().fatal(format!(
                "cuprite: `{RUNTIME_CRATE}::{name}` is not loaded; set {RUNTIME_VAR}"
            ))
        })
    });
    check_runtime(tcx, entry, finished);
    let description = describe(tcx, &path, &job);
    tcx.arena
        .alloc(rewrite(tcx, body.clone(), [entry, finished], &job, &description))
}

/// The kernel for `job`, compiled, as a description for the runtime.
fn describe(tcx: TyCtxt<'_>, path: &str, job: &Job) -> String {
    let entry = shapes::entry_name(path);
    let stages: Vec<&Function> = job.stages.iter().collect();
    let inputs: Vec<KTy> = job.inputs.iter().map(|&(_, ty)| ty).collect();
    let kernel = shapes::in_place_map(&entry, job.element, &inputs, &stages, job.passes.is_some())
        .unwrap_or_else(|invalid| tcx.dcx().fatal(format!("cuprite: building the kernel for `{path}`: {invalid}")));
    let mut scalars = Vec::new();
    if job.passes.is_some() {
        scalars.push(KTy::U32);
    }
    scalars.extend(job.scalars.iter().map(Leaf::passed));
    Description {
        function: path.to_owned(),
        entry,
        element: job.element,
        inputs,
        tables: job.tables.iter().map(|&(_, ty)| ty).collect(),
        repeated: job.passes.is_some(),
        scalars,
        ptx: compile(tcx, path, &kernel),
    }
    .encode()
}

/// PTX for `kernel`, through the toolchain's LLVM tools, or directly if
/// they aren't installed.
fn compile(tcx: TyCtxt<'_>, path: &str, kernel: &Kernel) -> String {
    static LLVM: OnceLock<Result<Llvm, String>> = OnceLock::new();
    let llvm = LLVM.get_or_init(|| {
        Llvm::in_sysroot(tcx.sess.opts.sysroot.path(), rustc_session::config::host_tuple())
    });
    match llvm {
        Ok(llvm) => llvm.compile(kernel).unwrap_or_else(|error| {
            tcx.dcx()
                .fatal(format!("cuprite: compiling the kernel for `{path}` through LLVM: {error}"))
        }),
        Err(reason) => {
            static WARNED: OnceLock<()> = OnceLock::new();
            WARNED.get_or_init(|| {
                tcx.dcx().warn(format!(
                    "cuprite: compiling kernels directly to PTX, without LLVM's optimizations: {reason}"
                ));
            });
            ptx::emit(kernel)
        }
    }
}

fn check_signature(tcx: TyCtxt<'_>, def_id: LocalDefId, replacement: &Replacement) {
    let fail = |reason: &str| -> ! {
        tcx.dcx().span_fatal(
            tcx.def_span(def_id),
            format!("cuprite: cannot rewrite `{}`: {reason}", replacement.path),
        )
    };
    if tcx.generics_of(def_id).count() != 0 {
        fail("generic functions are not supported");
    }
    let sig = tcx.fn_sig(def_id).instantiate_identity().skip_binder();
    let inputs = sig.inputs();
    let values_ok = inputs.first().is_some_and(|ty| {
        matches!(ty.kind(), ty::Ref(_, inner, ty::Mutability::Mut)
            if matches!(inner.kind(), ty::Slice(element) if *element == tcx.types.u32))
    });
    if !values_ok {
        fail("the first parameter must be `&mut [u32]`");
    }
    if let Passes::Arg(index) = replacement.passes {
        if inputs.get(index - 1) != Some(&tcx.types.u32) {
            fail(&format!("parameter {index} must be `u32`"));
        }
    }
    if !sig.output().is_unit() {
        fail("the function must return `()`");
    }
}

/// The rewrite calls `map_slice::<T>(&mut [T], &'static str, &[u64]) -> u64`
/// and `finished(u64)`; a runtime built for another driver version must not
/// be called with the wrong signature.
fn check_runtime<'tcx>(tcx: TyCtxt<'tcx>, entry: DefId, finished: DefId) {
    let types = &tcx.types;
    let signature = |def_id: DefId| {
        let sig = tcx.fn_sig(def_id).instantiate_identity().skip_binder();
        (sig.inputs().to_vec(), sig.output())
    };
    let (inputs, output) = signature(entry);
    let reference_to = |ty: Ty<'tcx>, mutability: ty::Mutability, check: &dyn Fn(Ty<'tcx>) -> bool| {
        matches!(ty.kind(), ty::Ref(_, inner, m) if *m == mutability && check(*inner))
    };
    let entry_ok = tcx.generics_of(entry).count() == 1
        && output == types.u64
        && matches!(inputs[..], [values, description, args]
            if reference_to(values, ty::Mutability::Mut, &|t| matches!(t.kind(), ty::Slice(_)))
                && reference_to(description, ty::Mutability::Not, &|t| t.is_str())
                && reference_to(args, ty::Mutability::Not, &|t| {
                    matches!(t.kind(), ty::Slice(element) if *element == types.u64)
                }));
    let finished_ok = signature(finished) == (vec![types.u64], types.unit);
    if !(entry_ok && finished_ok) {
        tcx.dcx().fatal(format!(
            "cuprite: `{RUNTIME_CRATE}` does not match this driver; rebuild it"
        ));
    }
}

/// An item of `cuprite_runtime`, found by name among the loaded crates.
fn runtime_item(tcx: TyCtxt<'_>, name: &str) -> Option<DefId> {
    let krate = tcx
        .crates(())
        .iter()
        .copied()
        .find(|&krate| tcx.crate_name(krate).as_str() == RUNTIME_CRATE)?;
    tcx.module_children(krate.as_def_id())
        .iter()
        .find(|child| child.ident.name.as_str() == name)
        .and_then(|child| child.res.opt_def_id())
}

/// A `&'static str` constant.
fn str_constant<'tcx>(tcx: TyCtxt<'tcx>, text: &str, span: Span) -> Operand<'tcx> {
    let alloc_id = tcx.allocate_bytes_dedup(text.as_bytes(), CTFE_ALLOC_SALT);
    let value = ConstValue::Slice {
        alloc_id,
        meta: text.len() as u64,
    };
    Operand::Constant(Box::new(ConstOperand {
        span,
        user_ty: None,
        const_: rustc_middle::mir::Const::Val(value, Ty::new_static_str(tcx)),
    }))
}

/// Statements in the entry block, before the dispatch.
struct Prologue<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a mut Body<'tcx>,
    statements: Vec<Statement<'tcx>>,
    source_info: SourceInfo,
}

impl<'tcx> Prologue<'_, 'tcx> {
    /// `let local: ty = rvalue;` with a new local.
    fn assign(&mut self, ty: Ty<'tcx>, rvalue: Rvalue<'tcx>) -> Local {
        let local = self.body.local_decls.push(LocalDecl::new(ty, self.body.span));
        self.statements.push(Statement::new(
            self.source_info,
            StatementKind::Assign(Box::new((Place::from(local), rvalue))),
        ));
        local
    }

    /// A parameter's bits as a `u64`.
    fn bits(&mut self, param: Local) -> Operand<'tcx> {
        let ty = self.body.local_decls[param].ty;
        self.place_bits(Place::from(param), ty)
    }

    /// The bits of the number a leaf names, through its reference and
    /// fields.
    fn leaf_bits(&mut self, leaf: &Leaf) -> Operand<'tcx> {
        let tcx = self.tcx;
        let mut ty = self.body.local_decls[leaf.param].ty;
        let mut projection = Vec::new();
        if leaf.deref {
            let ty::Ref(_, inner, _) = ty.kind() else { unreachable!("a leaf through a reference") };
            ty = *inner;
            projection.push(ProjectionElem::Deref);
        }
        for &index in &leaf.fields {
            let field = translate::fields_of(tcx, ty).expect("the recognizer walked these fields")[index];
            projection.push(ProjectionElem::Field(FieldIdx::from_usize(index), field));
            ty = field;
        }
        let place = Place::from(leaf.param).project_deeper(&projection, tcx);
        self.place_bits(place, ty)
    }

    /// An input slice's address and length, each as a `u64`.
    fn slice_parts(&mut self, param: Local) -> [Operand<'tcx>; 2] {
        let tcx = self.tcx;
        let ty = self.body.local_decls[param].ty;
        let ty::Ref(_, slice, _) = ty.kind() else { unreachable!("an input is a slice reference") };
        let ty::Slice(element) = slice.kind() else { unreachable!("an input is a slice reference") };
        let (u64_ty, usize_ty) = (tcx.types.u64, tcx.types.usize);
        let place = Place::from(param).project_deeper(&[ProjectionElem::Deref], tcx);
        let raw = self.assign(Ty::new_imm_ptr(tcx, *slice), Rvalue::RawPtr(RawPtrKind::Const, place));
        let element_ptr_ty = Ty::new_imm_ptr(tcx, *element);
        let thin = self.assign(element_ptr_ty, Rvalue::Cast(CastKind::PtrToPtr, Operand::Copy(Place::from(raw)), element_ptr_ty));
        let address = self.assign(
            usize_ty,
            Rvalue::Cast(CastKind::PointerExposeProvenance, Operand::Copy(Place::from(thin)), usize_ty),
        );
        let address = self.assign(u64_ty, Rvalue::Cast(CastKind::IntToInt, Operand::Copy(Place::from(address)), u64_ty));
        let len = self.assign(usize_ty, Rvalue::UnaryOp(UnOp::PtrMetadata, Operand::Copy(Place::from(param))));
        let len = self.assign(u64_ty, Rvalue::Cast(CastKind::IntToInt, Operand::Copy(Place::from(len)), u64_ty));
        [Operand::Copy(Place::from(address)), Operand::Copy(Place::from(len))]
    }

    /// A number's bits as a `u64`: integers and `bool` extended, floats as
    /// their IEEE bits, as `map_slice` expects.
    fn place_bits(&mut self, place: Place<'tcx>, ty: Ty<'tcx>) -> Operand<'tcx> {
        let tcx = self.tcx;
        let value = Operand::Copy(place);
        let u64_ty = tcx.types.u64;
        let local = match ty.kind() {
            ty::Uint(ty::UintTy::U64) => return value,
            ty::Int(_) | ty::Uint(_) | ty::Bool => {
                self.assign(u64_ty, Rvalue::Cast(CastKind::IntToInt, value, u64_ty))
            }
            ty::Float(ty::FloatTy::F32) => {
                let bits = self.assign(tcx.types.u32, Rvalue::Cast(CastKind::Transmute, value, tcx.types.u32));
                self.assign(u64_ty, Rvalue::Cast(CastKind::IntToInt, Operand::Copy(Place::from(bits)), u64_ty))
            }
            ty::Float(ty::FloatTy::F64) => self.assign(u64_ty, Rvalue::Cast(CastKind::Transmute, value, u64_ty)),
            _ => unreachable!("the recognizer accepts only these parameter types"),
        };
        Operand::Copy(Place::from(local))
    }
}

/// Prepends a dispatch to `body`:
///
/// ```text
/// bb0:     args = [scalar bits...]; token = map_slice::<T>(copy _1, description, &args) -> check
/// check:   switchInt(token) -> [0: done, otherwise: original entry]
/// done:    return
/// ```
///
/// The original entry block moves to the end, and jumps back to it are
/// redirected, so the original body runs exactly as before when the runtime
/// declines. Each of its `return`s becomes `finished(copy token) -> done`.
fn rewrite<'tcx>(
    tcx: TyCtxt<'tcx>,
    mut body: Body<'tcx>,
    [entry, finished]: [DefId; 2],
    job: &Job,
    description: &str,
) -> Body<'tcx> {
    let span = body.span;
    let source_info = SourceInfo::outermost(span);
    let element_ty = match body.local_decls[Local::from_usize(1)].ty.kind() {
        ty::Ref(_, inner, _) => match inner.kind() {
            ty::Slice(element) => *element,
            _ => unreachable!("checked: the first parameter is a slice"),
        },
        _ => unreachable!("checked: the first parameter is a slice"),
    };
    let token = body.local_decls.push(LocalDecl::new(tcx.types.u64, span));
    let unit = body.local_decls.push(LocalDecl::new(tcx.types.unit, span));

    // The scalar arguments, as `&[u64]`.
    let mut prologue = Prologue {
        tcx,
        body: &mut body,
        statements: Vec::new(),
        source_info,
    };
    let mut bits = Vec::new();
    match job.passes {
        Some(Passes::Const(passes)) => {
            bits.push(Operand::const_from_scalar(tcx, tcx.types.u64, Scalar::from_u64(passes.into()), span));
        }
        Some(Passes::Arg(index)) => bits.push(prologue.bits(Local::from_usize(index))),
        None => {}
    }
    for leaf in &job.scalars {
        bits.push(prologue.leaf_bits(leaf));
    }
    for &(param, _) in job.inputs.iter().chain(&job.tables) {
        bits.extend(prologue.slice_parts(param));
    }
    let u64_ty = tcx.types.u64;
    let array_ty = Ty::new_array(tcx, u64_ty, bits.len() as u64);
    let array = prologue.assign(
        array_ty,
        Rvalue::Aggregate(Box::new(AggregateKind::Array(u64_ty)), bits.into_iter().collect()),
    );
    let array_ref_ty = Ty::new_imm_ref(tcx, tcx.lifetimes.re_erased, array_ty);
    let array_ref = prologue.assign(
        array_ref_ty,
        Rvalue::Ref(tcx.lifetimes.re_erased, BorrowKind::Shared, Place::from(array)),
    );
    let slice_ty = Ty::new_imm_ref(tcx, tcx.lifetimes.re_erased, Ty::new_slice(tcx, u64_ty));
    let args_slice = prologue.assign(
        slice_ty,
        Rvalue::Cast(
            CastKind::PointerCoercion(PointerCoercion::Unsize, CoercionSource::Implicit),
            Operand::Copy(Place::from(array_ref)),
            slice_ty,
        ),
    );
    let statements = prologue.statements;

    let args = [
        Operand::Copy(Place::from(Local::from_usize(1))),
        str_constant(tcx, description, span),
        Operand::Copy(Place::from(args_slice)),
    ]
    .map(|node| Spanned { node, span });

    let terminator = |kind| Some(Terminator { source_info, kind });
    let call = |func: Operand<'tcx>, args: Box<[Spanned<Operand<'tcx>>]>, destination, target| {
        TerminatorKind::Call {
            func,
            args,
            destination: Place::from(destination),
            target: Some(target),
            unwind: UnwindAction::Continue,
            call_source: CallSource::Normal,
            fn_span: span,
        }
    };
    let blocks = body.basic_blocks_mut();
    let original = blocks.push(blocks[START_BLOCK].clone());
    for block in blocks.iter_mut() {
        block.terminator_mut().successors_mut(|target| {
            if *target == START_BLOCK {
                *target = original;
            }
        });
    }
    let done = blocks.push(BasicBlockData::new(
        terminator(TerminatorKind::Return),
        false,
    ));
    // The original body's returns, before the new blocks: report first.
    let token_arg = || {
        Box::new([Spanned {
            node: Operand::Copy(Place::from(token)),
            span,
        }]) as Box<[_]>
    };
    let finished_fn = || Operand::function_handle(tcx, finished, &[], span);
    for block in blocks.iter_mut().take(done.as_usize()).skip(1) {
        if let TerminatorKind::Return = block.terminator().kind {
            block.terminator_mut().kind = call(finished_fn(), token_arg(), unit, done);
        }
    }
    let check = blocks.push(BasicBlockData::new(
        terminator(TerminatorKind::if_(
            Operand::Copy(Place::from(token)),
            original,
            done,
        )),
        false,
    ));
    let entry_fn = Operand::function_handle(tcx, entry, &[element_ty.into()], span);
    let mut start = BasicBlockData::new(terminator(call(entry_fn, Box::new(args), token, check)), false);
    start.statements = statements;
    blocks[START_BLOCK] = start;
    body
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_constant_and_argument_passes() {
        let spec = "a::transform=3,7,1; a::repeat=4294967295,0,arg2;";
        assert_eq!(
            parse(spec).unwrap(),
            [
                Replacement {
                    path: "a::transform".into(),
                    mul: 3,
                    add: 7,
                    passes: Passes::Const(1),
                },
                Replacement {
                    path: "a::repeat".into(),
                    mul: u32::MAX,
                    add: 0,
                    passes: Passes::Arg(2),
                },
            ]
        );
    }

    #[test]
    fn rejects_malformed_entries() {
        for spec in [
            "a::f",
            "a::f=1,2",
            "a::f=1,2,3,4",
            "a::f=x,2,3",
            "a::f=1,2,arg0",
            "a::f=-1,2,3",
        ] {
            assert!(parse(spec).is_err(), "{spec}");
        }
    }
}
