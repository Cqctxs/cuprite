//! Storage for MIR values, place projections, reference binding, and copying.
//!
//! # Slots
//!
//! Every MIR value lives in a [`Slot`]: a number in a kernel local, a struct
//! or tuple as one slot per field, or a reference as the slot it points to.
//! Reading through a reference reads the slot it
//! names; writing through it writes that slot. The loop's element is a
//! kernel local like any other, and its reference points to it. Copying a
//! struct copies its fields.
//!
//! # Arrays
//!
//! An array has one slot per element up to [`super::types::MAX_ARRAY`].
//! A constant index names an element. An index known only at run time
//! names a [`Slot::Choice`]: reading it selects among every element,
//! writing it writes the element chosen and keeps the others. MIR checks the
//! index first, and the check runs on the GPU too. An array as a slice
//! keeps its elements, so its length is known. Constant tables (`&[f32; 5]`,
//! `&[u8]`) and read-only statics of plain data become elements the same
//! way.
//!
//! `transmute` (and `to_bits`, `from_le_bytes`...) reinterprets plain data
//! by rustc's layout: each number of the result gathers the bytes it covers
//! from the source's numbers.
//!
//! # Enums
//!
//! An enum is held as its discriminant's value and every variant's fields
//! ([`Slot::Enum`]); only the active variant's fields mean anything.
//! Building a variant sets the discriminant and that variant's fields;
//! `match` reads the discriminant and switches on it, as MIR does. Layout
//! matters only where bytes do: constant enums and `transmute` read and
//! write the tag or niche (`Option<NonZero<u32>>`'s `None` is a zero) as
//! rustc's layout stores it. `Ord::cmp` of numbers gives `Ordering`'s
//! discriminant directly. A `char` is held as its `u32` code, and a pattern
//! type (`NonZero`'s field) as the number it restricts.

use cuprite_kernel::{
    BinOp as KBinOp, Const, Local as KLocal, Operand as KOperand, Rvalue as KRvalue, Ty as KTy,
};
use rustc_abi::VariantIdx;
use rustc_middle::mir::{Local, Operand, Place, ProjectionElem};
use rustc_middle::ty::{self, Ty};

use super::fold::fold_operand;
use super::refusal::projection_refusal;
use super::types::{fields_of, held_ty, kernel_ty};
use super::{
    holds_pointer, is_ref, never_reassigned, unsigned_of, Frame, Result, Translator,
    MAX_UNION_BYTES,
};

/// Where a MIR value lives in the kernel.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Slot {
    Scalar(KLocal),
    /// A struct's or tuple's fields, in declaration order.
    Fields(Vec<Slot>),
    /// A value with no data, such as `()`.
    Unit,
    /// A reference: the slot it points to.
    Ref(Box<Slot>),
    /// A reference not assigned yet.
    Unbound,
    /// The loop's `Option<_>` from `next`, known to be `Some`.
    Some(Box<Slot>),
    /// A value only the iterator may use, such as the slice.
    Opaque,
    /// A read-only slice in GPU memory: its address and length.
    Slice { pointer: KLocal, len: KLocal, element: KTy },
    /// A number in a read-only slice, at an address.
    Memory { address: KLocal, ty: KTy },
    /// An array element chosen at run time: the option `index` names.
    /// Reading it selects among all options; writing it writes the one
    /// chosen. `index` holds the index as it was when the place was
    /// formed, and is below the number of options: MIR checks it first.
    Choice { index: KLocal, options: Vec<Slot> },
    /// An enum: its discriminant's value, and each variant's fields
    /// (`Fields`, or `Unit` for none), whatever variant is active. Only the
    /// active variant's fields mean anything. As a value being stored, a
    /// `Unit` variant is one it doesn't write.
    Enum { tag: KLocal, variants: Vec<Slot> },
    /// The bytes of other data seen as a value of another type, through a
    /// pointer cast whose numbers don't line up (`read_unaligned`,
    /// `bytemuck`): the data's numbers with their offsets, and where this
    /// value starts among them. Reading it decodes the bytes as the type
    /// read. Writes encode the covered bytes back into those locals.
    Bytes { numbers: Vec<(u64, KLocal)>, offset: u64 },
}

impl Slot {
    /// The scalar local named by a path from a validated element or parameter layout.
    pub(super) fn scalar_at(&self, path: &[usize]) -> KLocal {
        let mut slot = self;
        for &index in path {
            let Slot::Fields(fields) = slot else { unreachable!("layout paths follow fields") };
            slot = &fields[index];
        }
        let Slot::Scalar(local) = slot else { unreachable!("layout paths end at scalars") };
        *local
    }

    pub(super) fn scalars(&self, out: &mut Vec<KLocal>) {
        match self {
            Slot::Scalar(local) => out.push(*local),
            Slot::Fields(fields) => fields.iter().for_each(|field| field.scalars(out)),
            Slot::Enum { tag, variants } => {
                out.push(*tag);
                variants.iter().for_each(|variant| variant.scalars(out));
            }
            Slot::Bytes { numbers, .. } => out.extend(numbers.iter().map(|&(_, local)| local)),
            Slot::Choice { index, options } => {
                out.push(*index);
                options.iter().for_each(|option| option.scalars(out));
            }
            Slot::Ref(inner) | Slot::Some(inner) => inner.scalars(out),
            Slot::Slice { pointer, len, .. } => out.extend([*pointer, *len]),
            Slot::Memory { address, .. } => out.push(*address),
            Slot::Unit | Slot::Unbound | Slot::Opaque => {}
        }
    }
}

/// One projection of a place, with any run-time index already read.
enum Step {
    Deref,
    Field(usize),
    Downcast(u32),
    Index(KLocal),
    Constant { offset: u64, from_end: bool },
    Subslice { from: u64, to: u64, from_end: bool },
}

impl<'tcx> Translator<'tcx> {
    /// Like `fresh`, with every reference inside bound to a fresh value of
    /// its own: a parameter of a function translated on its own.
    pub(super) fn fresh_deep(&mut self, ty: Ty<'tcx>, depth: usize) -> Result<Slot> {
        if depth > 16 {
            return Err(format!("nests `{ty}` too deeply"));
        }
        if let Some(pointee) = ty.builtin_deref(true) {
            return Ok(Slot::Ref(Box::new(self.fresh_deep(pointee, depth + 1)?)));
        }
        if held_ty(ty).is_some() || matches!(ty.kind(), ty::Adt(adt, _) if adt.is_union()) {
            return self.fresh(ty);
        }
        if let Some((tag, variants)) = self.enum_shape(ty)? {
            let tag = self.b.local(tag);
            let variants = variants
                .into_iter()
                .map(|fields| self.fresh_fields_deep(fields, depth))
                .collect::<Result<_>>()?;
            return Ok(Slot::Enum { tag, variants });
        }
        let fields = self.fields(ty)?;
        self.fresh_fields_deep(fields, depth)
    }

    fn fresh_fields_deep(&mut self, fields: Vec<Ty<'tcx>>, depth: usize) -> Result<Slot> {
        if fields.is_empty() {
            return Ok(Slot::Unit);
        }
        Ok(Slot::Fields(fields.into_iter().map(|field| self.fresh_deep(field, depth + 1)).collect::<Result<_>>()?))
    }

    /// The kernel type of an enum's discriminant and each variant's field
    /// types, if `ty` is an enum.
    pub(super) fn enum_shape(&self, ty: Ty<'tcx>) -> Result<Option<(KTy, Vec<Vec<Ty<'tcx>>>)>> {
        let ty::Adt(adt, args) = ty.kind() else {
            return Ok(None);
        };
        if !adt.is_enum() {
            return Ok(None);
        }
        let tcx = self.tcx;
        let discriminant = ty.discriminant_ty(tcx);
        let tag = kernel_ty(discriminant)
            .ok_or_else(|| format!("uses the enum `{ty}`, whose discriminant is a `{discriminant}`"))?;
        let variants = adt
            .variants()
            .iter()
            .map(|variant| {
                variant
                    .fields
                    .iter()
                    .map(|field| tcx.normalize_erasing_regions(self.typing_env(), field.ty(tcx, args)))
                    .collect()
            })
            .collect();
        Ok(Some((tag, variants)))
    }

    /// The value of the discriminant of `ty`'s variant `variant`.
    pub(super) fn discriminant(&self, ty: Ty<'tcx>, variant: VariantIdx) -> Result<Const> {
        let tag = kernel_ty(ty.discriminant_ty(self.tcx)).ok_or_else(|| format!("uses the enum `{ty}`"))?;
        let value = ty
            .discriminant_for_variant(self.tcx, variant)
            .ok_or_else(|| format!("uses `{ty}`, which has no variant {}", variant.as_usize()))?;
        Ok(Const::int(tag, value.val as i128))
    }

    /// The slot for a fresh value of type `ty`.
    pub(super) fn fresh(&mut self, ty: Ty<'tcx>) -> Result<Slot> {
        if let Some(kernel) = held_ty(ty) {
            return Ok(Slot::Scalar(self.b.local(kernel)));
        }
        if is_ref(ty) {
            return Ok(Slot::Unbound);
        }
        if matches!(ty.kind(), ty::Adt(adt, _) if adt.is_union()) {
            return self.union_slot(ty);
        }
        if let Some((tag, variants)) = self.enum_shape(ty)? {
            let tag = self.b.local(tag);
            let variants = variants.into_iter().map(|fields| self.fresh_fields(fields)).collect::<Result<_>>()?;
            return Ok(Slot::Enum { tag, variants });
        }
        let fields = self.fields(ty)?;
        self.fresh_fields(fields)
    }

    /// A union, held as its bytes: unsigned numbers covering them, each as
    /// wide as its offset's alignment allows, up to 8 bytes. Reading a field
    /// decodes them as the field's type (`decode`); writing one encodes it
    /// into them (`encode`), keeping the bytes it doesn't cover, as memory
    /// would. References can't be held as bytes.
    fn union_slot(&mut self, ty: Ty<'tcx>) -> Result<Slot> {
        if holds_pointer(self.tcx, ty, 0) {
            return Err(format!("uses the union `{ty}`, which holds a reference or pointer"));
        }
        let size = self.size(ty)?;
        if size > MAX_UNION_BYTES {
            return Err(format!("uses the union `{ty}`, of more than {MAX_UNION_BYTES} bytes"));
        }
        let mut numbers = Vec::new();
        let mut at = 0;
        while at < size {
            let width = [8, 4, 2, 1].into_iter().find(|&width| at % width == 0 && at + width <= size).expect("a byte fits");
            numbers.push((at, self.b.local(unsigned_of(width))));
            at += width;
        }
        Ok(Slot::Bytes { numbers, offset: 0 })
    }

    fn fresh_fields(&mut self, fields: Vec<Ty<'tcx>>) -> Result<Slot> {
        if fields.is_empty() {
            return Ok(Slot::Unit);
        }
        Ok(Slot::Fields(fields.into_iter().map(|field| self.fresh(field)).collect::<Result<_>>()?))
    }

    pub(super) fn fields(&self, ty: Ty<'tcx>) -> Result<Vec<Ty<'tcx>>> {
        fields_of(self.tcx, ty)
    }

    /// The slot of a local, created on first use.
    pub(super) fn local_slot(&mut self, frame: &mut Frame<'_, 'tcx>, local: Local) -> Result<Slot> {
        if let Some(slot) = frame.slots.get(&local) {
            return Ok(slot.clone());
        }
        if let Some(rules) = &frame.loop_rules {
            if (1..=frame.body.arg_count).contains(&local.as_usize()) {
                return Err("uses a parameter a kernel can't take".into());
            }
            if rules.assigned_outside.contains(&local) {
                return Err("uses a value computed outside the loop body".into());
            }
        }
        let slot = self.fresh(frame.body.local_decls[local].ty)?;
        frame.slots.insert(local, slot.clone());
        Ok(slot)
    }

    /// The slot a place names.
    pub(super) fn place(&mut self, frame: &mut Frame<'_, 'tcx>, place: &Place<'tcx>) -> Result<Slot> {
        let mut slot = self.local_slot(frame, place.local)?;
        for (depth, element) in place.projection.iter().enumerate() {
            if let Slot::Bytes { numbers, offset } = &slot {
                let ty = Place::ty_from(place.local, &place.projection[..depth], frame.body, self.tcx).ty;
                let layout = self.layout(ty)?;
                // An index debug builds hold in a local is a constant all the same.
                let constant_index = |index: Local| {
                    fold_operand(self.tcx, frame.body, &Operand::Copy(Place::from(index))).and_then(|i| u64::try_from(i).ok())
                };
                let at = match element {
                    ProjectionElem::Field(field, _) => layout.fields.offset(field.as_usize()).bytes(),
                    ProjectionElem::ConstantIndex { offset: index, from_end: false, .. } => layout.fields.offset(index as usize).bytes(),
                    ProjectionElem::Index(index) if constant_index(index).is_some_and(|i| i < layout.fields.count() as u64) => {
                        layout.fields.offset(constant_index(index).expect("checked") as usize).bytes()
                    }
                    _ => return Err(format!("uses `{ty}` seen through a pointer to other data")),
                };
                slot = Slot::Bytes { numbers: numbers.clone(), offset: offset + at };
                continue;
            }
            if slot == Slot::Opaque {
                return Err("uses a slice parameter other than through the loop's iterator".into());
            }
            let step = match element {
                ProjectionElem::Deref => Step::Deref,
                ProjectionElem::Field(field, _) => Step::Field(field.as_usize()),
                ProjectionElem::Downcast(_, variant) => Step::Downcast(variant.as_u32()),
                // MIR checks the index before it gets here, and the check
                // runs on the GPU too. The index is kept as it is now: a
                // reference to the element must not follow later changes.
                ProjectionElem::Index(index) => {
                    let index = self.scalar_operand(frame, &Operand::Copy(Place::from(index)))?;
                    Step::Index(self.b.copy(index))
                }
                ProjectionElem::ConstantIndex { offset, from_end, .. } => Step::Constant { offset, from_end },
                ProjectionElem::Subslice { from, to, from_end } => Step::Subslice { from, to, from_end },
                element => return Err(projection_refusal(&element)),
            };
            slot = self.step(slot, &step)?;
        }
        Ok(slot)
    }

    /// The value a place holds, as read: bytes seen through a pointer cast
    /// are decoded as the place's type.
    pub(super) fn read_place(&mut self, frame: &mut Frame<'_, 'tcx>, place: &Place<'tcx>) -> Result<Slot> {
        let slot = self.place(frame, place)?;
        let ty = place.ty(frame.body, self.tcx).ty;
        match slot {
            Slot::Bytes { .. } => {
                let value = self.fresh(ty)?;
                self.decode(&value, ty, 0, &slot, ty)?;
                Ok(value)
            }
            // A position among bytes (a pointer reshaped into groups): each
            // group decoded, and the one at the position chosen.
            Slot::Choice { index, options } if options.iter().any(|option| matches!(option, Slot::Bytes { .. })) => {
                let mut decoded = Vec::with_capacity(options.len());
                for option in options {
                    decoded.push(match option {
                        Slot::Bytes { .. } => {
                            let value = self.fresh(ty)?;
                            self.decode(&value, ty, 0, &option, ty)?;
                            value
                        }
                        other => other,
                    });
                }
                Ok(Slot::Choice { index, options: decoded })
            }
            slot => Ok(slot),
        }
    }

    /// The part of `slot` that `step` names.
    fn step(&mut self, slot: Slot, step: &Step) -> Result<Slot> {
        Ok(match (step, slot) {
            // A step into an element chosen at run time is that step into
            // whichever element is chosen.
            (_, Slot::Choice { index, options }) => {
                let options: Vec<Slot> =
                    options.into_iter().map(|option| self.step(option, step)).collect::<Result<_>>()?;
                // Reading one would read every slice at the index, and only
                // the chosen one's bounds were checked.
                if options.iter().any(|option| matches!(option, Slot::Slice { .. } | Slot::Memory { .. })) {
                    return Err("chooses between slices at run time".into());
                }
                Slot::Choice { index, options }
            }
            (Step::Deref, Slot::Ref(inner)) => *inner,
            (Step::Deref, _) => return Err("reads through a reference it can't follow".into()),
            (Step::Field(field), Slot::Fields(fields)) => {
                fields.get(*field).cloned().ok_or("uses a field that doesn't exist")?
            }
            (Step::Downcast(1), Slot::Some(inner)) => Slot::Fields(vec![*inner]),
            (Step::Downcast(variant), Slot::Enum { variants, .. }) => {
                variants.get(*variant as usize).cloned().ok_or("uses a variant that doesn't exist")?
            }
            (Step::Downcast(_), _) => return Err("uses an enum".into()),
            // An array, held as its elements.
            (Step::Index(index), Slot::Fields(elements)) => Slot::Choice { index: *index, options: elements },
            (Step::Index(_), Slot::Unit) => return Err("indexes an empty array".into()),
            (&Step::Constant { offset, from_end }, Slot::Fields(elements)) => {
                let at = if from_end { elements.len().checked_sub(offset as usize) } else { Some(offset as usize) };
                at.and_then(|at| elements.get(at).cloned()).ok_or("indexes past the end of an array")?
            }
            (&Step::Subslice { from, to, from_end }, Slot::Fields(elements)) => {
                // An array's subslice ends at `to`; a slice's `to` elements
                // before its end.
                let end = if from_end { elements.len().checked_sub(to as usize) } else { Some(to as usize) };
                let part = end
                    .and_then(|end| elements.get(from as usize..end))
                    .ok_or("takes a subslice past the end of an array")?;
                if part.is_empty() { Slot::Unit } else { Slot::Fields(part.to_vec()) }
            }
            (&Step::Subslice { from, to, from_end: true }, Slot::Unit) if from == 0 && to == 0 => Slot::Unit,
            // A read-only slice in memory.
            (Step::Index(index), Slot::Slice { pointer, element, .. }) => {
                Slot::Memory { address: self.b.offset(pointer, *index, element), ty: element }
            }
            (&Step::Constant { offset, from_end }, Slot::Slice { pointer, len, element }) => {
                let offset = Const::int(KTy::U64, offset.into());
                let index: KOperand = if from_end { self.b.binary(KBinOp::Sub, len, offset).into() } else { offset.into() };
                Slot::Memory { address: self.b.offset(pointer, index, element), ty: element }
            }
            (&Step::Subslice { from, to, from_end: true }, Slot::Slice { pointer, len, element }) => {
                let start = self.b.offset(pointer, Const::int(KTy::U64, from.into()), element);
                let trimmed = self.b.binary(KBinOp::Sub, len, Const::int(KTy::U64, (from + to).into()));
                Slot::Slice { pointer: start, len: trimmed, element }
            }
            (Step::Field(_), _) => return Err("uses a field of a value the kernel can't hold".into()),
            (Step::Index(_) | Step::Constant { .. }, _) => return Err("indexes something other than an array or slice".into()),
            (Step::Subslice { .. }, _) => return Err("takes a subslice of something other than an array or slice".into()),
        })
    }

    /// Binds a reference local to the slot it points to. A reference
    /// assigned twice must point to the same slot: choosing at run time
    /// between targets would need memory.
    fn bind(&mut self, frame: &mut Frame<'_, 'tcx>, local: Local, value: Slot) -> Result<()> {
        if !matches!(value, Slot::Ref(_)) {
            return Err("uses a reference it can't follow".into());
        }
        if let Some(rules) = &frame.loop_rules {
            if rules.assigned_outside.contains(&local) {
                return Err("uses a value computed outside the loop body".into());
            }
        }
        match frame.slots.get(&local) {
            // A pointer assigned once and never borrowed mutably can't move,
            // so it shares the position it was given, if that can't move
            // either, which keeps it foldable (`Builder::constant`); others
            // get their own.
            None | Some(Slot::Unbound) => {
                let fixed = match &value {
                    Slot::Ref(target) => match &**target {
                        Slot::Choice { index, .. } => !self.moving.contains(index),
                        Slot::Memory { address, .. } => !self.moving.contains(address),
                        _ => true,
                    },
                    _ => true,
                };
                let owned = if fixed && never_reassigned(frame.body, local) { value } else { self.own_pointer(&value) };
                frame.slots.insert(local, owned);
                Ok(())
            }
            Some(existing) if *existing == value => Ok(()),
            Some(Slot::Ref(existing)) => {
                let (existing, Slot::Ref(new)) = (existing.clone(), &value) else { unreachable!("checked above") };
                if self.repoint(&existing, new) {
                    Ok(())
                } else {
                    Err("chooses between references at run time".into())
                }
            }
            Some(_) => Err("chooses between references at run time".into()),
        }
    }

    /// Writes `value` to `place`: copies a value, or binds a reference.
    pub(super) fn store(&mut self, frame: &mut Frame<'_, 'tcx>, place: &Place<'tcx>, value: Slot) -> Result<()> {
        if place.projection.is_empty() && is_ref(frame.body.local_decls[place.local].ty) {
            return self.bind(frame, place.local, value);
        }
        let dest = self.place(frame, place)?;
        // Data held as bytes (a union, or other data seen through a pointer
        // cast) takes the value's bytes.
        if let Slot::Bytes { numbers, offset } = &dest {
            let ty = place.ty(frame.body, self.tcx).ty;
            return self.encode(numbers, *offset, &value, ty);
        }
        // A position among groups of bytes: the chosen group takes them, and
        // the others keep theirs.
        if let Slot::Choice { index, options } = &dest {
            if options.iter().all(|option| matches!(option, Slot::Bytes { .. })) && !options.is_empty() {
                let ty = place.ty(frame.body, self.tcx).ty;
                let value = self.snapshot(&value)?;
                for (at, option) in options.iter().enumerate() {
                    let Slot::Bytes { numbers, offset } = option else { unreachable!("checked above") };
                    let chosen = self.chosen(*index, at);
                    let before: Vec<KLocal> = numbers.iter().map(|&(_, number)| self.b.copy(number)).collect();
                    self.encode(numbers, *offset, &value, ty)?;
                    for (&(_, number), old) in numbers.iter().zip(before) {
                        self.set(number, KRvalue::Select(chosen.into(), number.into(), old.into()));
                    }
                }
                return Ok(());
            }
        }
        let written = self.copy(&dest, &value)?;
        if written == dest {
            return Ok(());
        }
        // The value binds references inside the local, such as a closure's
        // captures. The local's slot records them; a field can't be updated
        // inside a larger slot.
        if !place.projection.is_empty() {
            return Err("stores a reference inside part of a value".into());
        }
        if let Some(rules) = &frame.loop_rules {
            if rules.assigned_outside.contains(&place.local) {
                return Err("uses a value computed outside the loop body".into());
            }
        }
        frame.slots.insert(place.local, written);
        Ok(())
    }

    /// Copies `src` into `dest`, field by field. If they share kernel
    /// locals (`a = (a.1, a.0)`), through temporaries. Returns the slot
    /// `dest` names afterwards, with the references inside it bound.
    pub(super) fn copy(&mut self, dest: &Slot, src: &Slot) -> Result<Slot> {
        let (mut written, mut read) = (Vec::new(), Vec::new());
        dest.scalars(&mut written);
        src.scalars(&mut read);
        if dest != src && written.iter().any(|local| read.contains(local)) && !matches!(dest, Slot::Scalar(_)) {
            let temporary = self.snapshot(src)?;
            return self.copy_fields(dest, &temporary);
        }
        self.copy_fields(dest, src)
    }

    /// Numbers are copied. A reference inside the value is bound, as
    /// reference locals are (`bind`): one not bound yet takes the source's
    /// target, and one already bound must have the same target.
    fn copy_fields(&mut self, dest: &Slot, src: &Slot) -> Result<Slot> {
        match (dest, src) {
            (_, Slot::Choice { .. }) => {
                let src = self.resolve(src.clone())?;
                return self.copy_fields(dest, &src);
            }
            // Writes the option chosen; the others keep their values.
            (Slot::Choice { index, options }, _) => {
                for (at, option) in options.iter().enumerate() {
                    let chosen = self.chosen(*index, at);
                    self.copy_where(chosen, option, src)?;
                }
            }
            (Slot::Scalar(d), Slot::Scalar(s)) => {
                if d != s {
                    self.set(*d, KRvalue::Use((*s).into()));
                }
            }
            (Slot::Fields(d), Slot::Fields(s)) if d.len() == s.len() => {
                let fields = d.iter().zip(s).map(|(d, s)| self.copy_fields(d, s)).collect::<Result<_>>()?;
                return Ok(Slot::Fields(fields));
            }
            // The discriminant, and each variant the value writes.
            (Slot::Enum { tag: d, variants: dv }, Slot::Enum { tag: s, variants: sv }) if dv.len() == sv.len() => {
                if d != s {
                    self.set(*d, KRvalue::Use((*s).into()));
                }
                let variants = dv
                    .iter()
                    .zip(sv)
                    .map(|(d, s)| if *s == Slot::Unit { Ok(d.clone()) } else { self.copy_fields(d, s) })
                    .collect::<Result<_>>()?;
                return Ok(Slot::Enum { tag: *d, variants });
            }
            (Slot::Unit, Slot::Unit) | (Slot::Unbound, Slot::Unbound) => {}
            (Slot::Unbound, Slot::Ref(_)) => return Ok(self.own_pointer(src)),
            // A reference not assigned on any path translated so far, such
            // as an inactive variant's: nothing a program may read.
            (Slot::Ref(_), Slot::Unbound) => {}
            (Slot::Ref(d), Slot::Ref(s)) if d == s => {}
            // A pointer moved along an array or slice.
            (Slot::Ref(d), Slot::Ref(s)) if self.repoint(d, s) => {}
            // A union inside a value: the same type's bytes, number by number.
            (Slot::Bytes { numbers: d, offset: 0 }, Slot::Bytes { numbers: s, offset: 0 }) if self.same_bytes(d, s) => {
                for (&(_, d), &(_, s)) in d.iter().zip(s) {
                    if d != s {
                        self.set(d, KRvalue::Use(s.into()));
                    }
                }
            }
            (Slot::Bytes { .. }, _) => return Err("writes through a pointer to data of another type".into()),
            (Slot::Ref(_), Slot::Ref(_)) => return Err("chooses between references at run time".into()),
            _ => return Err("copies between values of different shapes".into()),
        }
        Ok(dest.clone())
    }

    /// Copies `src` into `dest` where `condition` holds, and leaves `dest`
    /// as it is elsewhere. References can't be written conditionally.
    pub(super) fn copy_where(&mut self, condition: KLocal, dest: &Slot, src: &Slot) -> Result<()> {
        match (dest, src) {
            (Slot::Scalar(d), Slot::Scalar(s)) => {
                if d != s {
                    self.set(*d, KRvalue::Select(condition.into(), (*s).into(), (*d).into()));
                }
            }
            (Slot::Fields(d), Slot::Fields(s)) if d.len() == s.len() => {
                for (d, s) in d.iter().zip(s) {
                    self.copy_where(condition, d, s)?;
                }
            }
            (Slot::Choice { index, options }, _) => {
                for (at, option) in options.iter().enumerate() {
                    let chosen = self.chosen(*index, at);
                    let both = self.b.binary(KBinOp::And, condition, chosen);
                    self.copy_where(both, option, src)?;
                }
            }
            (Slot::Enum { tag: d, variants: dv }, Slot::Enum { tag: s, variants: sv }) if dv.len() == sv.len() => {
                if d != s {
                    self.set(*d, KRvalue::Select(condition.into(), (*s).into(), (*d).into()));
                }
                for (d, s) in dv.iter().zip(sv) {
                    if *s != Slot::Unit {
                        self.copy_where(condition, d, s)?;
                    }
                }
            }
            (Slot::Unit, Slot::Unit) | (Slot::Unbound, Slot::Unbound) | (Slot::Ref(_), Slot::Unbound) => {}
            (Slot::Ref(d), Slot::Ref(s)) if d == s => {}
            // A position in an array or slice, chosen.
            (Slot::Ref(d), Slot::Ref(s)) if matches!((&**d, &**s), (Slot::Choice { options: a, .. }, Slot::Choice { options: b, .. }) if a == b)
                || matches!((&**d, &**s), (Slot::Memory { ty: a, .. }, Slot::Memory { ty: b, .. }) if a == b) =>
            {
                let (Slot::Choice { index: d, .. } | Slot::Memory { address: d, .. }, Slot::Choice { index: s, .. } | Slot::Memory { address: s, .. }) = (&**d, &**s) else {
                    unreachable!("matched above")
                };
                self.set(*d, KRvalue::Select(condition.into(), (*s).into(), (*d).into()));
            }
            (Slot::Ref(_) | Slot::Unbound, Slot::Ref(_)) => return Err("chooses between references at run time".into()),
            (Slot::Bytes { numbers: d, offset: 0 }, Slot::Bytes { numbers: s, offset: 0 }) if self.same_bytes(d, s) => {
                for (&(_, d), &(_, s)) in d.iter().zip(s) {
                    if d != s {
                        self.set(d, KRvalue::Select(condition.into(), s.into(), d.into()));
                    }
                }
            }
            _ => return Err("copies between values of different shapes".into()),
        }
        Ok(())
    }

    /// Whether `index` chooses option `at`.
    pub(super) fn chosen(&mut self, index: KLocal, at: usize) -> KLocal {
        let ty = self.b.ty(index);
        self.b.binary(KBinOp::Eq, index, Const::int(ty, at as i128))
    }

    /// The value a slot holds, with an element chosen at run time read into
    /// new locals.
    pub(super) fn resolve(&mut self, slot: Slot) -> Result<Slot> {
        let Slot::Choice { index, options } = slot else {
            return Ok(slot);
        };
        let options = options.into_iter().map(|option| self.resolve(option)).collect::<Result<Vec<_>>>()?;
        self.select_among(index, &options)
    }

    /// The option `index` chooses, number by number. A reference chosen at
    /// run time points to the chosen target.
    pub(super) fn select_among(&mut self, index: KLocal, options: &[Slot]) -> Result<Slot> {
        let shapes = || "copies between values of different shapes".to_string();
        let Some(first) = options.first() else {
            return Err("indexes an empty array".into());
        };
        if options.iter().all(|option| option == first) {
            return Ok(first.clone());
        }
        Ok(match first {
            Slot::Scalar(_) => {
                let numbers = options
                    .iter()
                    .map(|option| match option {
                        Slot::Scalar(local) => Ok(*local),
                        _ => Err(shapes()),
                    })
                    .collect::<Result<Vec<_>>>()?;
                let value = self.b.copy(numbers[0]);
                for (at, number) in numbers.iter().enumerate().skip(1) {
                    let chosen = self.chosen(index, at);
                    self.b.set(value, KRvalue::Select(chosen.into(), (*number).into(), value.into()));
                }
                Slot::Scalar(value)
            }
            Slot::Fields(fields) => {
                let mut chosen = Vec::with_capacity(fields.len());
                for field in 0..fields.len() {
                    let column = options
                        .iter()
                        .map(|option| match option {
                            Slot::Fields(f) if f.len() == fields.len() => Ok(f[field].clone()),
                            _ => Err(shapes()),
                        })
                        .collect::<Result<Vec<_>>>()?;
                    chosen.push(self.select_among(index, &column)?);
                }
                Slot::Fields(chosen)
            }
            Slot::Enum { variants, .. } => {
                let mut tags = Vec::with_capacity(options.len());
                let mut columns = vec![Vec::with_capacity(options.len()); variants.len()];
                for option in options {
                    let Slot::Enum { tag, variants: v } = option else { return Err(shapes()) };
                    if v.len() != variants.len() {
                        return Err(shapes());
                    }
                    tags.push(Slot::Scalar(*tag));
                    for (column, variant) in columns.iter_mut().zip(v) {
                        column.push(variant.clone());
                    }
                }
                let Slot::Scalar(tag) = self.select_among(index, &tags)? else { return Err(shapes()) };
                let variants = columns.iter().map(|column| self.select_among(index, column)).collect::<Result<_>>()?;
                Slot::Enum { tag, variants }
            }
            Slot::Ref(_) => {
                let targets = options
                    .iter()
                    .map(|option| match option {
                        Slot::Ref(target) if !matches!(**target, Slot::Slice { .. } | Slot::Memory { .. }) => {
                            Ok((**target).clone())
                        }
                        Slot::Ref(_) => Err("chooses between slices at run time".to_string()),
                        _ => Err(shapes()),
                    })
                    .collect::<Result<Vec<_>>>()?;
                Slot::Ref(Box::new(Slot::Choice { index, options: targets }))
            }
            Slot::Unbound => return Err("chooses between references at run time".into()),
            Slot::Slice { .. } | Slot::Memory { .. } => return Err("chooses between slices at run time".into()),
            Slot::Unit | Slot::Some(_) | Slot::Opaque | Slot::Choice { .. } | Slot::Bytes { .. } => return Err(shapes()),
        })
    }

    /// The number a slot holds, read from memory or chosen at run time if
    /// need be.
    pub(super) fn number(&mut self, slot: Slot) -> Result<KOperand> {
        match self.resolve(slot)? {
            Slot::Scalar(local) => Ok(local.into()),
            Slot::Memory { address, ty } => Ok(self.b.load(address, ty).into()),
            _ => Err("uses a value as a number that isn't one".into()),
        }
    }

    /// Writes a number to the slot `place` names.
    pub(super) fn write_number(&mut self, dest: Slot, rvalue: KRvalue) -> Result<()> {
        match dest {
            Slot::Scalar(local) => self.set(local, rvalue),
            Slot::Choice { .. } => {
                // Typed like the options; validation checks the rvalue.
                fn number(slot: &Slot) -> Option<KLocal> {
                    match slot {
                        Slot::Scalar(local) => Some(*local),
                        Slot::Choice { options, .. } => options.first().and_then(number),
                        _ => None,
                    }
                }
                let ty = number(&dest).map(|number| self.b.ty(number)).ok_or("writes a number into something else")?;
                let value = self.b.local(ty);
                self.b.set(value, rvalue);
                self.copy_fields(&dest, &Slot::Scalar(value))?;
            }
            Slot::Bytes { .. } => return Err("writes through a pointer to data of another type".into()),
            Slot::Memory { .. } => return Err("writes to a read-only slice".into()),
            _ => return Err("writes a number into something else".into()),
        }
        Ok(())
    }

    /// A copy of `src`'s numbers in new locals.
    pub(super) fn snapshot(&mut self, src: &Slot) -> Result<Slot> {
        Ok(match src {
            Slot::Scalar(local) => Slot::Scalar(self.b.copy(*local)),
            Slot::Fields(fields) => Slot::Fields(fields.iter().map(|field| self.snapshot(field)).collect::<Result<_>>()?),
            Slot::Choice { .. } => self.resolve(src.clone())?,
            Slot::Enum { tag, variants } => Slot::Enum {
                tag: self.b.copy(*tag),
                variants: variants.iter().map(|variant| self.snapshot(variant)).collect::<Result<_>>()?,
            },
            Slot::Bytes { numbers, offset } => Slot::Bytes {
                numbers: numbers.iter().map(|&(at, number)| (at, self.b.copy(number))).collect(),
                offset: *offset,
            },
            other => other.clone(),
        })
    }
}
