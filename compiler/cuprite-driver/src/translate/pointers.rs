//! Reference operations, pointer casts, and positions within arrays and slices.
//!
//! # Pointers
//!
//! A pointer into an array is a position: a reference to a [`Slot::Choice`]
//! whose index local is the pointer's value. One into a read-only slice is
//! an address ([`Slot::Memory`]). `as_ptr` gives the first position,
//! `Offset` moves it, comparisons and `ptr_offset_from` compare positions,
//! and assigning a pointer moves the target's position (`repoint`); a bound
//! pointer that can move gets a position of its own, never folded to a
//! constant. A position seen as another element type reshapes the array
//! ([`Translator::reshape`]). `slice::from_raw_parts` gives a slice of an
//! array at a known position and length, or of a read-only slice. With
//! `NonNull` (a pattern type over a pointer) moved through transmutes,
//! slice iterators work. In inlined functions, switches on constants,
//! including the build's `ub_checks()`, take only their live target.
//!
//! # Pointer casts
//!
//! A cast from `*const T` to `*const U` is a view of the same bytes: where
//! `U`'s numbers are `T`'s locals at the same offsets and types, the view
//! aliases them. Otherwise it is [`Slot::Bytes`]: reading through it decodes
//! the bytes as the type read, as `transmute` would, and writes update the covered bytes.

use cuprite_kernel::{
    BinOp as KBinOp, Const, Local as KLocal, Operand as KOperand, Rvalue as KRvalue, Ty as KTy,
};
use rustc_middle::mir::{AggregateKind, BinOp, CastKind, Rvalue};
use rustc_middle::ty::{self, adjustment::PointerCoercion, Ty};

use super::fold::fold_operand;
use super::refusal::cast_refusal;
use super::{is_ref, points_to_array, points_to_slice, Frame, Result, Slot, Translator};

impl<'tcx> Translator<'tcx> {
    /// The element at `index` of `options`, values of type `from`, seen as a
    /// pointer to `to`: the array reshaped into elements of `to` (each one
    /// seen as `to`, consecutive ones grouped, or each one split), with the
    /// index scaled, so pointer arithmetic on `to` moves along the array. A
    /// position known now that falls between elements of `to` is bytes at
    /// that offset. One known only at run time that does fails the check,
    /// and the CPU reads it instead.
    fn reshape(&mut self, index: KLocal, options: &[Slot], from: Ty<'tcx>, to: Ty<'tcx>) -> Result<Slot> {
        let (from_size, to_size) = (self.size(from)?, self.size(to)?);
        if from_size == 0 || to_size == 0 {
            return Err(format!("sees `{from}` as `{to}`"));
        }
        // The numbers of consecutive elements, seen as `to` at `at`.
        let seen = |this: &mut Self, slots: &[Slot], at: u64| -> Result<Slot> {
            let mut numbers = Vec::new();
            for (k, slot) in slots.iter().enumerate() {
                this.numbers_of(slot, from, k as u64 * from_size, &mut numbers)?;
            }
            Ok(this.view(to, at, &numbers).unwrap_or(Slot::Bytes { numbers, offset: at }))
        };
        let known = self.b.constant(index, &self.moving).and_then(|i| u64::try_from(i).ok());
        if from_size == to_size {
            let options = options.iter().map(|option| seen(self, std::slice::from_ref(option), 0)).collect::<Result<_>>()?;
            return Ok(Slot::Choice { index, options });
        }
        if to_size % from_size == 0 {
            let n = to_size / from_size;
            if let Some(i) = known.filter(|i| i % n != 0) {
                return seen(self, options, i * from_size);
            }
            let options = options.chunks_exact(n as usize).map(|group| seen(self, group, 0)).collect::<Result<_>>()?;
            if known.is_none() {
                // Between groups: the CPU reads it.
                let n_const = Const::int(KTy::U64, n.into());
                let remainder = self.b.binary(KBinOp::Rem, index, n_const);
                let aligned = self.b.binary(KBinOp::Eq, remainder, Const::int(KTy::U64, 0));
                let (on, between) = (self.b.block(), self.b.block());
                self.b.branch(aligned, on, between);
                self.b.switch_to(between);
                self.b.fail();
                self.b.switch_to(on);
            }
            let index = self.b.binary(KBinOp::Div, index, Const::int(KTy::U64, n.into()));
            return Ok(Slot::Choice { index, options });
        }
        if from_size % to_size == 0 {
            let n = from_size / to_size;
            let mut split = Vec::with_capacity(options.len() * n as usize);
            for option in options {
                for j in 0..n {
                    split.push(seen(self, std::slice::from_ref(option), j * to_size)?);
                }
            }
            let index = self.b.binary(KBinOp::Mul, index, Const::int(KTy::U64, n.into()));
            return Ok(Slot::Choice { index, options: split });
        }
        Err(format!("sees `{from}` as `{to}`"))
    }

    /// Two pointers' positions as numbers that order as the pointers do,
    /// in elements: indices into one array, or addresses in one read-only
    /// slice. `None` for two pointers to the same target at no position.
    pub(super) fn positions(&mut self, a: &Slot, b: &Slot) -> Result<Option<(KOperand, KOperand)>> {
        let different = || "compares pointers to different data".to_string();
        let (Slot::Ref(a), Slot::Ref(b)) = (a, b) else { return Err(different()) };
        match (&**a, &**b) {
            (Slot::Choice { index: i, options: x }, Slot::Choice { index: j, options: y }) if x == y => {
                Ok(Some(((*i).into(), (*j).into())))
            }
            (Slot::Memory { address: p, ty: x }, Slot::Memory { address: q, ty: y }) if x == y => {
                let (p, q) = (self.b.reinterpret(*p, KTy::U64), self.b.reinterpret(*q, KTy::U64));
                let size = Const::int(KTy::U64, x.bytes().into());
                Ok(Some((self.b.binary(KBinOp::Div, p, size).into(), self.b.binary(KBinOp::Div, q, size).into())))
            }
            (Slot::Bytes { numbers: x, offset: p }, Slot::Bytes { numbers: y, offset: q }) if x == y => Ok(Some((
                Const::int(KTy::U64, (*p).into()).into(),
                Const::int(KTy::U64, (*q).into()).into(),
            ))),
            (a, b) if a == b => Ok(None),
            _ => Err(different()),
        }
    }

    /// A pointer into an array or a read-only slice with its own position,
    /// so moving the pointer it came from doesn't move it.
    pub(super) fn own_pointer(&mut self, value: &Slot) -> Slot {
        match value {
            Slot::Ref(target) => match &**target {
                Slot::Choice { index, options } => {
                    let index = self.b.copy(*index);
                    self.moving.insert(index);
                    Slot::Ref(Box::new(Slot::Choice { index, options: options.clone() }))
                }
                Slot::Memory { address, ty } => {
                    let address = self.b.copy(*address);
                    self.moving.insert(address);
                    Slot::Ref(Box::new(Slot::Memory { address, ty: *ty }))
                }
                _ => value.clone(),
            },
            _ => value.clone(),
        }
    }

    /// Points `dest`, a pointer's target, where `src` points, if both are
    /// positions in the same array or read-only slice: the position is the
    /// pointer's value. Returns whether it could.
    pub(super) fn repoint(&mut self, dest: &Slot, src: &Slot) -> bool {
        match (dest, src) {
            (Slot::Choice { index: d, options: a }, Slot::Choice { index: s, options: b }) if a == b => {
                if d != s {
                    self.set(*d, KRvalue::Use((*s).into()));
                }
                true
            }
            (Slot::Memory { address: d, ty: a }, Slot::Memory { address: s, ty: b }) if a == b => {
                if d != s {
                    self.set(*d, KRvalue::Use((*s).into()));
                }
                true
            }
            _ => false,
        }
    }

    /// The slot a reference-valued rvalue points to, as `Slot::Ref`.
    pub(super) fn reference(&mut self, frame: &mut Frame<'_, 'tcx>, rvalue: &Rvalue<'tcx>) -> Result<Slot> {
        match rvalue {
            Rvalue::Ref(_, _, place) | Rvalue::RawPtr(_, place) => Ok(Slot::Ref(Box::new(self.place(frame, place)?))),
            Rvalue::Use(operand, _) => self.operand_slot(frame, operand),
            Rvalue::CopyForDeref(place) => self.place(frame, place),
            // An array as a slice: the same elements, whose length the
            // slot knows.
            Rvalue::Cast(CastKind::PointerCoercion(PointerCoercion::Unsize, _), operand, ty)
                if points_to_array(operand.ty(frame.body, self.tcx)) && points_to_slice(*ty) =>
            {
                self.operand_slot(frame, operand)
            }
            // A value holding a pointer (`NonNull`) as the pointer.
            Rvalue::Cast(CastKind::Transmute, operand, ty) if !is_ref(operand.ty(frame.body, self.tcx)) => {
                self.transmute(frame, operand, *ty)
            }
            // A pointer to the same bytes as another type: a view sharing
            // the numbers they hold. A transmute between references or
            // pointers is the same cast (array `==` makes one).
            Rvalue::Cast(
                kind @ (CastKind::PtrToPtr
                | CastKind::Transmute
                | CastKind::PointerCoercion(PointerCoercion::MutToConstPointer, _)),
                operand,
                ty,
            ) => {
                let from = operand.ty(frame.body, self.tcx).builtin_deref(true);
                let (Some(from), Some(to)) = (from, ty.builtin_deref(true)) else {
                    return Err(cast_refusal(*kind, *ty));
                };
                let Slot::Ref(target) = self.operand_slot(frame, operand)? else {
                    return Err(cast_refusal(*kind, *ty));
                };
                if from == to {
                    return Ok(Slot::Ref(target));
                }
                // An array's or slice's first element (`as_ptr`), which
                // pointer arithmetic moves along it: the element at an index.
                if let ty::Slice(element) | ty::Array(element, _) = from.kind() {
                    if *element == to {
                        let start = Const::int(KTy::U64, 0);
                        match *target {
                            Slot::Fields(elements) => {
                                return Ok(Slot::Ref(Box::new(Slot::Choice { index: self.b.copy(start), options: elements })));
                            }
                            Slot::Unit => return Ok(Slot::Ref(Box::new(Slot::Choice { index: self.b.copy(start), options: Vec::new() }))),
                            Slot::Slice { pointer, element, .. } => {
                                return Ok(Slot::Ref(Box::new(Slot::Memory { address: self.b.copy(pointer), ty: element })));
                            }
                            _ => {}
                        }
                    }
                }
                if !from.is_sized(self.tcx, self.typing_env()) || !to.is_sized(self.tcx, self.typing_env()) {
                    return Err(cast_refusal(*kind, *ty));
                }
                // Bytes stay bytes, whatever type they are seen as.
                if let Slot::Bytes { .. } = *target {
                    return Ok(Slot::Ref(target));
                }
                // A position in an array, as a pointer to another type.
                if let Slot::Choice { index, options } = &*target {
                    let reshaped = self.reshape(*index, options, from, to).map_err(|_| cast_refusal(*kind, *ty))?;
                    return Ok(Slot::Ref(Box::new(reshaped)));
                }
                // A pointer seen as a value holding only that pointer
                // (`NonNull`), or back: the same pointer, whose position the
                // view shares, so moving one moves the other.
                if let (Some(from_path), Some(to_path)) = (self.pointer_path(from, 0), self.pointer_path(to, 0)) {
                    let mut pointer = &*target;
                    for &index in &from_path {
                        let Slot::Fields(fields) = pointer else { return Err(cast_refusal(*kind, *ty)) };
                        pointer = &fields[index];
                    }
                    if matches!(pointer, Slot::Ref(_)) {
                        let view = self.holding_pointer(to, &to_path, pointer.clone())?;
                        return Ok(Slot::Ref(Box::new(view)));
                    }
                }
                let mut numbers = Vec::new();
                self.numbers_of(&target, from, 0, &mut numbers).map_err(|_| cast_refusal(*kind, *ty))?;
                // Numbers that line up alias; others are read as bytes.
                let view = self.view(to, 0, &numbers).unwrap_or(Slot::Bytes { numbers, offset: 0 });
                Ok(Slot::Ref(Box::new(view)))
            }
            Rvalue::Cast(kind, _, ty) => Err(cast_refusal(*kind, *ty)),
            // A pointer from its parts (`slice::from_raw_parts`): a slice of
            // an array at a position and length known now, or of a read-only
            // slice at an address and length known at run time.
            Rvalue::Aggregate(kind, operands) if matches!(**kind, AggregateKind::RawPtr(..)) => {
                let AggregateKind::RawPtr(pointee, _) = **kind else { unreachable!("matched above") };
                let [data, metadata] = &operands.raw[..] else { return Err("builds a pointer from unexpected parts".into()) };
                let Slot::Ref(target) = self.operand_slot(frame, data)? else {
                    return Err("builds a pointer from a pointer it can't follow".into());
                };
                if !matches!(pointee.kind(), ty::Slice(_)) {
                    return Ok(Slot::Ref(target));
                }
                let len = self.scalar_operand(frame, metadata)?;
                let len = self.b.cast(len, KTy::U64);
                match *target {
                    Slot::Choice { index, options } => {
                        let (Some(start), Some(count)) = (self.b.constant(index, &self.moving), self.b.constant(len, &self.moving)) else {
                            return Err("takes a slice of an array at a position known only at run time".into());
                        };
                        let range = usize::try_from(start).ok().zip(usize::try_from(count).ok());
                        let part: Vec<Slot> = match range.and_then(|(start, count)| options.get(start..start.checked_add(count)?)) {
                            Some(part) => part.to_vec(),
                            // Outside the array, which no valid program does:
                            // a path that never runs, such as one for longer
                            // input. If it runs, the CPU takes it. What follows
                            // is unreachable, and gets a placeholder.
                            None => {
                                self.b.fail();
                                let unreachable = self.b.block();
                                self.b.switch_to(unreachable);
                                Vec::new()
                            }
                        };
                        Ok(Slot::Ref(Box::new(if part.is_empty() { Slot::Unit } else { Slot::Fields(part) })))
                    }
                    Slot::Memory { address, ty } => Ok(Slot::Ref(Box::new(Slot::Slice { pointer: address, len, element: ty }))),
                    _ => Err("builds a slice from a pointer that isn't into an array or slice".into()),
                }
            }
            // Pointer arithmetic, in elements of the pointee: along an
            // array, a read-only slice, or bytes seen as another type.
            Rvalue::BinaryOp(BinOp::Offset, operands) => {
                let pointee = operands.0.ty(frame.body, self.tcx).builtin_deref(true).ok_or("moves something other than a pointer")?;
                let Slot::Ref(target) = self.operand_slot(frame, &operands.0)? else {
                    return Err("moves a pointer it can't follow".into());
                };
                let count = self.scalar_operand(frame, &operands.1)?;
                let count = self.b.cast(count, KTy::U64);
                Ok(Slot::Ref(Box::new(match *target {
                    Slot::Choice { index, options } => {
                        Slot::Choice { index: self.b.binary(KBinOp::Add, index, count), options }
                    }
                    Slot::Memory { address, ty } => Slot::Memory { address: self.b.offset(address, count, ty), ty },
                    Slot::Bytes { numbers, offset } => {
                        let steps = fold_operand(self.tcx, frame.body, &operands.1)
                            .ok_or("moves a pointer into bytes by an amount known only at run time")?;
                        let moved = i128::from(offset) + steps * i128::from(self.size(pointee)?);
                        let offset = u64::try_from(moved).map_err(|_| "moves a pointer before the data it's in".to_string())?;
                        Slot::Bytes { numbers, offset }
                    }
                    // Moving by nothing.
                    target if fold_operand(self.tcx, frame.body, &operands.1) == Some(0) => target,
                    _ => return Err("moves a pointer that isn't into an array or slice".into()),
                })))
            }
            _ => Err("uses a reference it can't follow".into()),
        }
    }
}
