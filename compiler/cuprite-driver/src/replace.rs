//! Body replacement (M1 question 4) and runtime linkage (question 5).
//!
//! Functions are chosen by the recognizer (`analyze`, with
//! `CUPRITE_AUTO=1`) or by hand: `CUPRITE_REPLACE` names functions to
//! rewrite and the map each computes, for shapes the recognizer does not
//! handle:
//!
//! ```text
//! CUPRITE_REPLACE=cuprite_probes::transform=3,7,1;cuprite_probes::repeat=3,7,arg2
//! ```
//!
//! Each entry is `<crate>::<path>=<mul>,<add>,<passes>`, where `<passes>` is
//! a constant or `argN`, the function's N-th parameter (1-based). The named
//! function must take `&mut [u32]` first and return `()`.
//!
//! The rewritten body first calls
//! `cuprite_runtime::map_u32(values, mul, add, passes)`. If that returns
//! 0, the GPU did the work and the function returns. Otherwise the original
//! body runs, unchanged, and then passes what `map_u32` returned to
//! `cuprite_runtime::map_u32_finished`, which lets the runtime time it. The
//! driver loads the runtime crate itself (`--extern force:`), so the
//! rewritten crate never declares it.

use rustc_hir::def::DefKind;
use rustc_middle::{
    mir::{
        interpret::Scalar, BasicBlockData, Body, CallSource, Local, LocalDecl, Operand, Place,
        SourceInfo, Terminator, TerminatorKind, UnwindAction, START_BLOCK,
    },
    ty::{self, TyCtxt},
};
use rustc_span::{
    def_id::{DefId, LocalDefId, LOCAL_CRATE},
    Spanned,
};

use crate::analyze::{self, Verdict};

pub(crate) const VAR: &str = "CUPRITE_REPLACE";
pub(crate) const RUNTIME_VAR: &str = "CUPRITE_RUNTIME";
pub(crate) const RUNTIME_DEPS_VAR: &str = "CUPRITE_RUNTIME_DEPS";
pub(crate) const RUNTIME_CRATE: &str = "cuprite_runtime";
const ENTRY: &str = "map_u32";
const FINISHED: &str = "map_u32_finished";

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

/// Wraps rustc's own `optimized_mir`. A function named in `CUPRITE_REPLACE`
/// gets a rewritten copy; with `CUPRITE_AUTO=1`, so does every function the
/// recognizer accepted: a map as a single pass, a repeated map with its pass
/// count. Everything else is returned unchanged.
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
    let manual = settings.replacements.iter().find(|r| r.path == path);
    let automatic = || {
        let (map, passes) = match analyze::verdict(tcx, def_id)? {
            Verdict::Candidate(map) => (map, Passes::Const(1)),
            Verdict::Repeated(map, passes) => (map, passes),
            Verdict::Rejected(_) => return None,
        };
        settings.auto.then(|| Replacement {
            path: path.clone(),
            mul: map.mul,
            add: map.add,
            passes,
        })
    };
    let Some(replacement) = manual.cloned().or_else(automatic) else {
        return body;
    };
    let replacement = &replacement;
    check_signature(tcx, def_id, replacement);
    let [entry, finished] = [ENTRY, FINISHED].map(|name| {
        runtime_item(tcx, name).unwrap_or_else(|| {
            tcx.dcx().fatal(format!(
                "cuprite: `{RUNTIME_CRATE}::{name}` is not loaded; set {RUNTIME_VAR}"
            ))
        })
    });
    check_runtime(tcx, entry, finished);
    tcx.arena
        .alloc(rewrite(tcx, body.clone(), [entry, finished], replacement))
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

/// The rewrite calls `map_u32(&mut [u32], u32, u32, u32) -> u64` and
/// `map_u32_finished(u64)`; a runtime built for another driver version must
/// not be called with the wrong signature.
fn check_runtime(tcx: TyCtxt<'_>, entry: DefId, finished: DefId) {
    let types = &tcx.types;
    let signature = |def_id: DefId| {
        let sig = tcx.fn_sig(def_id).instantiate_identity().skip_binder();
        (sig.inputs().to_vec(), sig.output())
    };
    let (inputs, output) = signature(entry);
    let entry_ok = output == types.u64
        && matches!(inputs[..], [values, a, b, c]
            if matches!(values.kind(), ty::Ref(_, inner, ty::Mutability::Mut)
                if matches!(inner.kind(), ty::Slice(element) if *element == types.u32))
                && [a, b, c] == [types.u32; 3]);
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

/// Prepends a dispatch to `body`:
///
/// ```text
/// bb0:     token = map_u32(copy _1, mul, add, passes) -> check
/// check:   switchInt(token) -> [0: done, otherwise: original entry]
/// done:    return
/// ```
///
/// The original entry block moves to the end, and jumps back to it are
/// redirected, so the original body runs exactly as before when the runtime
/// declines. Each of its `return`s becomes
/// `map_u32_finished(copy token) -> done`.
fn rewrite<'tcx>(
    tcx: TyCtxt<'tcx>,
    mut body: Body<'tcx>,
    [entry, finished]: [DefId; 2],
    replacement: &Replacement,
) -> Body<'tcx> {
    let span = body.span;
    let source_info = SourceInfo::outermost(span);
    let token = body.local_decls.push(LocalDecl::new(tcx.types.u64, span));
    let unit = body.local_decls.push(LocalDecl::new(tcx.types.unit, span));
    let constant =
        |value: u32| Operand::const_from_scalar(tcx, tcx.types.u32, Scalar::from_u32(value), span);
    let passes = match replacement.passes {
        Passes::Const(passes) => constant(passes),
        Passes::Arg(index) => Operand::Copy(Place::from(Local::from_usize(index))),
    };
    let args = [
        Operand::Copy(Place::from(Local::from_usize(1))),
        constant(replacement.mul),
        constant(replacement.add),
        passes,
    ]
    .map(|node| Spanned { node, span });

    let terminator = |kind| Some(Terminator { source_info, kind });
    let call = |func: DefId, args: Box<[Spanned<Operand<'tcx>>]>, destination, target| {
        TerminatorKind::Call {
            func: Operand::function_handle(tcx, func, &[], span),
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
    for block in blocks.iter_mut().take(done.as_usize()).skip(1) {
        if let TerminatorKind::Return = block.terminator().kind {
            block.terminator_mut().kind = call(finished, token_arg(), unit, done);
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
    blocks[START_BLOCK] =
        BasicBlockData::new(terminator(call(entry, Box::new(args), token, check)), false);
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
