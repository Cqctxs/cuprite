//! Constant folding for assertions, branches, and pointer offsets.

use rustc_middle::mir::{
    BasicBlock, BinOp, Body, CastKind, Local, Operand, RuntimeChecks, Rvalue, StatementKind,
    TerminatorKind, UnOp,
};
use rustc_middle::ty::{self, TyCtxt, TypingEnv};

/// True if the assertion ending `block` always passes: its condition folds
/// to the expected constant.
pub(super) fn assert_holds(body: &Body<'_>, block: BasicBlock) -> bool {
    let TerminatorKind::Assert { cond, expected, .. } = &body.basic_blocks[block].terminator().kind
    else {
        return false;
    };
    Folder { body, tcx: None }.operand(cond, 0) == Some(i128::from(*expected))
}

/// Evaluate an integer or boolean operand using this compilation's constants and check settings.
pub(super) fn fold_operand<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    operand: &Operand<'tcx>,
) -> Option<i128> {
    Folder {
        body,
        tcx: Some(tcx),
    }
    .operand(operand, 0)
}

pub(super) fn runtime_check(tcx: TyCtxt<'_>, check: RuntimeChecks) -> bool {
    match check {
        RuntimeChecks::OverflowChecks => tcx.sess.overflow_checks(),
        RuntimeChecks::UbChecks => tcx.sess.ub_checks(),
        RuntimeChecks::ContractChecks => tcx.sess.contract_checks(),
    }
}

/// Folds integer and `bool` expressions over constants, following locals
/// assigned exactly once; with a compiler context, also which run-time checks the
/// build makes.
struct Folder<'a, 'tcx> {
    body: &'a Body<'tcx>,
    tcx: Option<TyCtxt<'tcx>>,
}

impl<'tcx> Folder<'_, 'tcx> {
    const DEPTH: usize = 16;

    fn operand(&self, operand: &Operand<'tcx>, depth: usize) -> Option<i128> {
        match operand {
            Operand::Constant(c) => {
                let scalar = match (c.const_.try_to_scalar_int(), self.tcx) {
                    (Some(scalar), _) => scalar,
                    // An associated constant, such as `T::IS_ZST` once inlined.
                    (None, Some(tcx)) => c
                        .const_
                        .try_eval_scalar_int(tcx, TypingEnv::fully_monomorphized())?,
                    (None, None) => return None,
                };
                match c.ty().kind() {
                    ty::Bool => Some(i128::from(scalar.try_to_bool().ok()?)),
                    ty::Int(_) => Some(scalar.to_int(scalar.size())),
                    ty::Uint(_) => i128::try_from(scalar.to_uint(scalar.size())).ok(),
                    _ => None,
                }
            }
            Operand::Copy(place) | Operand::Move(place) if depth < Self::DEPTH => {
                let local = place.as_local()?;
                self.definition(local)
                    .and_then(|rvalue| self.rvalue(rvalue, depth + 1))
            }
            Operand::RuntimeChecks(checks) => {
                self.tcx.map(|tcx| i128::from(runtime_check(tcx, *checks)))
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
                let (a, b) = (
                    self.operand(&operands.0, depth),
                    self.operand(&operands.1, depth),
                );
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
