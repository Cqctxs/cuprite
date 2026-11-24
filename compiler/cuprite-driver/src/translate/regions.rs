//! Loop regions, reachable blocks, and independence between iterations.

use rustc_middle::mir::{
    BasicBlock, Body, Local, StatementKind, TerminatorKind, RETURN_PLACE, START_BLOCK,
};
use std::collections::HashSet;

use rustc_middle::ty::TyCtxt;

use super::fold::fold_operand;
use super::{Frame, Iteration, Loop, Result};

/// The loop body's blocks: everything reachable from its start without
/// passing the header. Leaving the loop any other way (`break`, `return`)
/// makes later elements depend on earlier ones, so it is refused.
pub(super) fn loop_region(body: &Body<'_>, shape: &Loop<'_, '_>, start: BasicBlock) -> Result<HashSet<BasicBlock>> {
    let mut region = HashSet::new();
    let mut pending = vec![start];
    while let Some(block) = pending.pop() {
        if shape.is_header(block) || !region.insert(block) {
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

/// The blocks before the loop: everything reachable from the start without
/// passing the header, or reaching the `for_each` call.
pub(super) fn setup_region(body: &Body<'_>, shape: &Loop<'_, '_>, loop_blocks: &HashSet<BasicBlock>) -> Result<HashSet<BasicBlock>> {
    let mut region = HashSet::new();
    let mut pending = vec![START_BLOCK];
    while let Some(block) = pending.pop() {
        if shape.is_header(block) || shape.each_block() == Some(block) || !region.insert(block) {
            continue;
        }
        if loop_blocks.contains(&block) {
            return Err("has an unsupported loop shape".into());
        }
        pending.extend(successors(&body.basic_blocks[block].terminator().kind));
    }
    Ok(region)
}

/// Each thread runs the setup and one iteration, so a value one iteration
/// leaves for the next (a counter, a running total) can't be carried:
/// refuse any local the body assigns that is also assigned before the loop
/// or in its header.
pub(super) fn check_carried(
    body: &Body<'_>,
    setup: &HashSet<BasicBlock>,
    loop_blocks: &HashSet<BasicBlock>,
    shape: &Loop<'_, '_>,
) -> Result<()> {
    let assigned_in = |blocks: &dyn Fn(BasicBlock) -> bool| {
        let mut assigned = HashSet::new();
        for (block, data) in body.basic_blocks.iter_enumerated() {
            if !blocks(block) || data.is_cleanup {
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
    };
    let inside = assigned_in(&|block| loop_blocks.contains(&block));
    let outside = assigned_in(&|block| !loop_blocks.contains(&block) && (setup.contains(&block) || shape.is_header(block)));
    let carried = inside.intersection(&outside).any(|&local| {
        let ty = body.local_decls[local].ty;
        local != RETURN_PLACE && !ty.is_unit() && !shape.plumbing.contains(&local)
    });
    if carried {
        return Err("carries a value from one element to the next".into());
    }
    Ok(())
}

/// Every block a function body runs, from its start, ignoring unwinding.
pub(super) fn function_region<'tcx>(body: &Body<'tcx>, tcx: TyCtxt<'tcx>) -> HashSet<BasicBlock> {
    let mut region = HashSet::new();
    let mut pending = vec![START_BLOCK];
    while let Some(block) = pending.pop() {
        if region.insert(block) {
            pending.extend(live_successors(body, block, tcx));
        }
    }
    region
}

/// The one target a switch takes, if its value is a constant: a checked
/// build setting such as `ub_checks()` once inlined, or a constant folded
/// through locals. Its other targets never run, and needn't translate.
pub(super) fn constant_target<'tcx>(body: &Body<'tcx>, block: BasicBlock, tcx: TyCtxt<'tcx>) -> Option<BasicBlock> {
    let TerminatorKind::SwitchInt { discr, targets } = &body.basic_blocks[block].terminator().kind else {
        return None;
    };
    let value = fold_operand(tcx, body, discr)?;
    Some(targets.target_for_value(u128::try_from(value).ok()?))
}

/// `successors`, with only the live target of a constant switch.
fn live_successors<'tcx>(body: &Body<'tcx>, block: BasicBlock, tcx: TyCtxt<'tcx>) -> Vec<BasicBlock> {
    match constant_target(body, block, tcx) {
        Some(target) => vec![target],
        None => successors(&body.basic_blocks[block].terminator().kind),
    }
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
        // Assembly continues: a block with it isn't a panic path, and is
        // refused when it is translated.
        TerminatorKind::InlineAsm { targets, .. } => targets.to_vec(),
        _ => Vec::new(),
    }
}

/// `region` in reverse postorder, so a block comes after the blocks that
/// dominate it, where its values and references are set.
pub(super) fn ordered(body: &Body<'_>, region: &HashSet<BasicBlock>) -> Vec<BasicBlock> {
    body.basic_blocks.reverse_postorder().iter().copied().filter(|block| region.contains(block)).collect()
}

/// Locals assigned outside the body, by statements or calls.
pub(super) fn assigned_outside(body: &Body<'_>, region: &HashSet<BasicBlock>) -> HashSet<Local> {
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

/// The blocks of `region` that can never finish normally: every path from
/// them ends in a call that never returns (a panic), or never ends. A frame
/// finishes at `return`; the loop's frame also at the end of an iteration.
pub(super) fn doomed(frame: &Frame<'_, '_>, region: &[BasicBlock]) -> HashSet<BasicBlock> {
    let body = frame.body;
    let mut finishing: HashSet<BasicBlock> = HashSet::new();
    loop {
        let before = finishing.len();
        for &block in region {
            if finishing.contains(&block) {
                continue;
            }
            let kind = &body.basic_blocks[block].terminator().kind;
            // The `for_each` call is an iteration, which ends there.
            let each = frame.loop_rules.as_ref().is_some_and(|rules| rules.shape.each_block() == Some(block));
            let finishes = matches!(kind, TerminatorKind::Return)
                || each
                || successors(kind).into_iter().any(|next| match frame.loop_rules.as_ref().map(|r| (r, &r.shape.iteration)) {
                    Some((rules, Iteration::For { start, is_header, .. })) if is_header(next) => {
                        // From the body the iteration ends; from the setup
                        // the body starts.
                        rules.body.contains(&block) || finishing.contains(start)
                    }
                    _ => finishing.contains(&next),
                });
            if finishes {
                finishing.insert(block);
            }
        }
        if finishing.len() == before {
            break;
        }
    }
    region.iter().copied().filter(|block| !finishing.contains(block)).collect()
}
