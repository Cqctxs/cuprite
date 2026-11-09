//! Loop census for surveying real code: `CUPRITE_CENSUS=<dir>` writes
//! `<dir>/<crate>.txt` with one line per loop in every function, closure and
//! generic function, whatever its signature:
//!
//! ```text
//! path: loop over slice::IterMut<u8>; reads u8; writes u8; ops u8{*,+}; branches 1; calls clamp | min
//! ```
//!
//! and one line per function that drives an iterator chain instead, whose
//! loop is inside the standard library:
//!
//! ```text
//! path: chain over slice::Iter<f32>: map, sum
//! ```
//!
//! It records what a loop would need from a GPU rule, not a verdict.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
    sync::Mutex,
};

use rustc_attr_ir::lang_items::LangItem;
use rustc_hir::def::DefKind;
use rustc_middle::{
    mir::{
        AssertKind, BasicBlock, BinOp, Body, Operand, Place, ProjectionElem, Rvalue, StatementKind,
        TerminatorKind, UnOp,
    },
    ty::{Ty, TyCtxt},
};
use rustc_span::def_id::{LocalDefId, LOCAL_CRATE};

pub(crate) const VAR: &str = "CUPRITE_CENSUS";

/// Lines for the crate being compiled. Query providers are plain function
/// pointers, so this is global.
static LINES: Mutex<Vec<String>> = Mutex::new(Vec::new());

pub(crate) fn record<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId, body: &Body<'tcx>) {
    if !matches!(
        tcx.def_kind(def_id),
        DefKind::Fn | DefKind::AssocFn | DefKind::Closure
    ) {
        return;
    }
    let generic = if tcx.generics_of(def_id).count() != 0 {
        " [generic]"
    } else {
        ""
    };
    let path = format!(
        "{}::{}{generic}",
        tcx.crate_name(LOCAL_CRATE),
        tcx.def_path_str(def_id)
    );
    let census = Census { tcx, body };
    let mut lines = LINES.lock().unwrap();
    let loops = census.loops();
    for (header, blocks) in &loops {
        let inner = loops
            .keys()
            .filter(|&&other| other != *header && blocks.contains(&other))
            .count();
        // A loop's own blocks: its inner loops are described separately.
        let own: BTreeSet<BasicBlock> = blocks
            .iter()
            .copied()
            .filter(|block| {
                !loops.iter().any(|(other, inner_blocks)| {
                    other != header && blocks.contains(other) && inner_blocks.contains(block)
                })
            })
            .collect();
        lines.push(format!("{path}: {}", census.describe(&own, inner)));
    }
    if loops.is_empty() {
        if let Some(chain) = census.chain() {
            lines.push(format!("{path}: {chain}"));
        }
    }
}

/// `<dir>/<crate>.txt`, sorted.
pub(crate) fn write(tcx: TyCtxt<'_>, dir: &Path) {
    let mut lines = std::mem::take(&mut *LINES.lock().unwrap());
    lines.sort();
    fs::create_dir_all(dir).expect("cannot create the census directory");
    let text = lines
        .iter()
        .map(|line| format!("{line}\n"))
        .collect::<String>();
    fs::write(
        dir.join(format!("{}.txt", tcx.crate_name(LOCAL_CRATE))),
        text,
    )
    .expect("cannot write the census");
}

/// Shortens a printed type: no lifetimes, and no module paths for the
/// common iterator types.
fn short(ty: impl ToString) -> String {
    let mut text = ty.to_string().replace("'_, ", "").replace("'_ ", "");
    for prefix in ["std::slice::", "std::iter::", "std::ops::", "core::slice::"] {
        text = text.replace(prefix, "");
    }
    text
}

fn op(op: BinOp) -> &'static str {
    use BinOp::*;
    match op {
        Add | AddUnchecked | AddWithOverflow => "+",
        Sub | SubUnchecked | SubWithOverflow => "-",
        Mul | MulUnchecked | MulWithOverflow => "*",
        Div => "/",
        Rem => "%",
        BitXor => "^",
        BitAnd => "&",
        BitOr => "|",
        Shl | ShlUnchecked => "<<",
        Shr | ShrUnchecked => ">>",
        Eq | Lt | Le | Ne | Ge | Gt | Cmp => "cmp",
        Offset => "offset",
    }
}

struct Census<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
}

impl<'a, 'tcx> Census<'a, 'tcx> {
    /// Natural loops, by header: the header and every block that reaches a
    /// back edge into it without passing through it.
    fn loops(&self) -> BTreeMap<BasicBlock, BTreeSet<BasicBlock>> {
        let blocks = &self.body.basic_blocks;
        let dominators = blocks.dominators();
        let predecessors = blocks.predecessors();
        let mut loops: BTreeMap<BasicBlock, BTreeSet<BasicBlock>> = BTreeMap::new();
        for (block, data) in blocks.iter_enumerated() {
            if data.is_cleanup || !dominators.is_reachable(block) {
                continue;
            }
            for header in data.terminator().successors() {
                if !dominators.dominates(header, block) {
                    continue;
                }
                let members = loops.entry(header).or_insert_with(|| [header].into());
                let mut work = vec![block];
                while let Some(member) = work.pop() {
                    if members.insert(member) {
                        work.extend(predecessors[member].iter().copied());
                    }
                }
            }
        }
        loops
    }

    fn ty(&self, place: &Place<'tcx>) -> Ty<'tcx> {
        place.ty(&self.body.local_decls, self.tcx).ty
    }

    /// Whether the place goes through memory: a reference, pointer or index.
    fn in_memory(place: &Place<'tcx>) -> bool {
        place
            .projection
            .iter()
            .any(|elem| matches!(elem, ProjectionElem::Deref | ProjectionElem::Index(_)))
    }

    fn describe(&self, blocks: &BTreeSet<BasicBlock>, inner: usize) -> String {
        let tcx = self.tcx;
        let mut over = None;
        let mut reads = BTreeSet::new();
        let mut writes = BTreeSet::new();
        let mut ops: BTreeMap<String, BTreeSet<&str>> = BTreeMap::new();
        let mut casts = 0;
        let mut indexes = 0;
        let mut bounds_checks = 0;
        let mut other_asserts = 0;
        let mut branches = 0usize;
        let mut calls = BTreeSet::new();

        let read = |operand: &Operand<'tcx>, reads: &mut BTreeSet<String>| {
            if let Some(place) = operand.place() {
                let ty = self.ty(&place);
                if ty.is_numeric() && Self::in_memory(&place) {
                    reads.insert(ty.to_string());
                }
            }
        };
        for &block in blocks {
            let data = &self.body.basic_blocks[block];
            for statement in &data.statements {
                let StatementKind::Assign(assign) = &statement.kind else {
                    continue;
                };
                let (place, rvalue) = &**assign;
                indexes += place
                    .projection
                    .iter()
                    .filter(|elem| matches!(elem, ProjectionElem::Index(_)))
                    .count();
                let ty = self.ty(place);
                if ty.is_numeric() && Self::in_memory(place) {
                    writes.insert(ty.to_string());
                }
                match rvalue {
                    Rvalue::BinaryOp(bin, operands) => {
                        let operand_ty = operands.0.ty(&self.body.local_decls, tcx);
                        ops.entry(operand_ty.to_string())
                            .or_default()
                            .insert(op(*bin));
                        read(&operands.0, &mut reads);
                        read(&operands.1, &mut reads);
                    }
                    Rvalue::UnaryOp(UnOp::Neg, operand) => {
                        let operand_ty = operand.ty(&self.body.local_decls, tcx);
                        ops.entry(operand_ty.to_string()).or_default().insert("neg");
                        read(operand, &mut reads);
                    }
                    Rvalue::Use(operand, _) | Rvalue::UnaryOp(_, operand) => {
                        read(operand, &mut reads)
                    }
                    Rvalue::Cast(_, operand, _) => {
                        casts += 1;
                        read(operand, &mut reads);
                    }
                    _ => {}
                }
            }
            match &data.terminator().kind {
                TerminatorKind::Call { func, args, .. } => {
                    for arg in args {
                        read(&arg.node, &mut reads);
                    }
                    let Some((def_id, fn_args)) = func.const_fn_def() else {
                        calls.insert("a function value".to_string());
                        continue;
                    };
                    if tcx.is_lang_item(def_id, LangItem::IteratorNext) {
                        over = Some(short(fn_args.type_at(0)));
                    } else if !tcx.is_lang_item(def_id, LangItem::IntoIterIntoIter) {
                        calls.insert(tcx.def_path_str(def_id));
                    }
                }
                TerminatorKind::Assert { msg, .. } => match &**msg {
                    AssertKind::BoundsCheck { .. } => bounds_checks += 1,
                    _ => other_asserts += 1,
                },
                TerminatorKind::SwitchInt { .. } => branches += 1,
                _ => {}
            }
        }
        // A `for` loop's own match on `next`'s result isn't a branch in the
        // body.
        if over.is_some() {
            branches = branches.saturating_sub(1);
        }

        let mut parts = vec![format!(
            "loop over {}",
            over.unwrap_or_else(|| "a condition".into())
        )];
        let list = |set: &BTreeSet<String>| set.iter().cloned().collect::<Vec<_>>().join(" ");
        if !reads.is_empty() {
            parts.push(format!("reads {}", list(&reads)));
        }
        if !writes.is_empty() {
            parts.push(format!("writes {}", list(&writes)));
        }
        if !ops.is_empty() {
            let ops = ops
                .iter()
                .map(|(ty, ops)| {
                    format!(
                        "{ty}{{{}}}",
                        ops.iter().copied().collect::<Vec<_>>().join(",")
                    )
                })
                .collect::<Vec<_>>()
                .join(" ");
            parts.push(format!("ops {ops}"));
        }
        for (count, name) in [
            (casts, "casts"),
            (indexes, "indexes"),
            (bounds_checks, "bounds checks"),
            (other_asserts, "other asserts"),
            (branches, "branches"),
            (inner, "inner loops"),
        ] {
            if count > 0 {
                parts.push(format!("{name} {count}"));
            }
        }
        if !calls.is_empty() {
            parts.push(format!(
                "calls {}",
                calls.iter().cloned().collect::<Vec<_>>().join(" | ")
            ));
        }
        parts.join("; ")
    }

    /// The `Iterator` methods a loop-free function calls, in block order,
    /// and the type the first is called on.
    fn chain(&self) -> Option<String> {
        let tcx = self.tcx;
        let iterator = tcx.lang_items().iterator_trait()?;
        let mut methods = Vec::new();
        let mut over = None;
        for data in self.body.basic_blocks.iter() {
            let TerminatorKind::Call { func, .. } = &data.terminator().kind else {
                continue;
            };
            let Some((def_id, args)) = func.const_fn_def() else {
                continue;
            };
            if tcx.trait_of_assoc(def_id) == Some(iterator) {
                over.get_or_insert_with(|| short(args.type_at(0)));
                methods.push(tcx.item_name(def_id).to_string());
            }
        }
        // A lone `next` or `into_iter` call is not a chain.
        let consumes = methods
            .iter()
            .any(|name| !matches!(name.as_str(), "next" | "size_hint"));
        if !consumes {
            return None;
        }
        Some(format!("chain over {}: {}", over?, methods.join(", ")))
    }
}
