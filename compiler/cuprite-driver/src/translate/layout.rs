//! Constant values, plain-data layouts, and byte operations for unions and transmutes.
//!
//! # Unions
//!
//! A union is held as its bytes ([`Translator::union_slot`]): unsigned
//! numbers covering them, as a [`Slot::Bytes`]. Reading a field decodes the
//! bytes as its type; writing one encodes it and keeps the bytes it doesn't
//! cover ([`Translator::encode`]), which also serves writes through pointer
//! casts to data of another type. `MaybeUninit` and `mem::zeroed` follow.

use cuprite_kernel::{
    BinOp as KBinOp, Const, Local as KLocal, Operand as KOperand, Rvalue as KRvalue, Ty as KTy,
};
use rustc_abi::{TagEncoding, VariantIdx, Variants};
use rustc_middle::mir::{
    interpret::GlobalAlloc, interpret::Scalar, CastKind, ConstOperand, ConstValue, Operand,
};
use rustc_middle::ty::{self, Ty};

use super::refusal::cast_refusal;
use super::types::{constant, held_ty, MAX_ARRAY};
use super::{is_ref, signed_of, unsigned_of, variant_offsets, Frame, Result, Slot, Translator};

impl<'tcx> Translator<'tcx> {
    /// The numbers a value of type `ty` holds, with their offsets in its
    /// layout from `base`.
    pub(super) fn numbers_of(&self, slot: &Slot, ty: Ty<'tcx>, base: u64, out: &mut Vec<(u64, KLocal)>) -> Result<()> {
        match slot {
            Slot::Scalar(local) => out.push((base, *local)),
            Slot::Fields(fields) => {
                let layout = self.layout(ty)?;
                for (index, (field, field_ty)) in fields.iter().zip(self.fields(ty)?).enumerate() {
                    self.numbers_of(field, field_ty, base + layout.fields.offset(index).bytes(), out)?;
                }
            }
            Slot::Unit => {}
            // A union: the numbers holding its bytes.
            Slot::Bytes { numbers, offset: 0 } => out.extend(numbers.iter().map(|&(at, number)| (base + at, number))),
            _ => return Err(format!("views `{ty}` as other data")),
        }
        Ok(())
    }

    /// A value of type `ty` at `base` made of the given numbers: each of its
    /// numbers must be one of them, at its offset and of its type, so reading
    /// and writing through it reads and writes them.
    pub(super) fn view(&self, ty: Ty<'tcx>, base: u64, numbers: &[(u64, KLocal)]) -> Result<Slot> {
        if let Some(kernel) = held_ty(ty) {
            return numbers
                .iter()
                .find(|&&(at, local)| at == base && self.b.ty(local) == kernel)
                .map(|&(_, local)| Slot::Scalar(local))
                .ok_or_else(|| format!("views data as `{ty}`, whose numbers lie elsewhere"));
        }
        if is_ref(ty) || self.enum_shape(ty)?.is_some() {
            return Err(format!("views data as `{ty}`"));
        }
        let layout = self.layout(ty)?;
        let fields = self.fields(ty)?;
        if fields.is_empty() {
            return Ok(Slot::Unit);
        }
        let parts = fields
            .into_iter()
            .enumerate()
            .map(|(index, field)| self.view(field, base + layout.fields.offset(index).bytes(), numbers))
            .collect::<Result<_>>()?;
        Ok(Slot::Fields(parts))
    }

    /// A constant of any supported type, in new locals.
    pub(super) fn constant_slot(&mut self, c: &ConstOperand<'tcx>) -> Result<Slot> {
        let ty = c.ty();
        if let Some(scalar) = constant(self.tcx, c) {
            return Ok(Slot::Scalar(self.b.copy(scalar)));
        }
        if is_ref(ty) {
            return self.constant_ref(c);
        }
        let slot = self.fresh(ty)?;
        if slot == Slot::Unit {
            return Ok(slot);
        }
        let bytes = self.constant_bytes(c)?;
        self.fill(&slot, ty, &bytes, 0)?;
        Ok(slot)
    }

    /// A constant's bytes, laid out as rustc lays out its type.
    pub(super) fn constant_bytes(&self, c: &ConstOperand<'tcx>) -> Result<Vec<u8>> {
        let ty = c.ty();
        let tcx = self.tcx;
        let value = c
            .const_
            .eval(tcx, self.typing_env(), c.span)
            .map_err(|_| format!("uses a constant of type `{ty}` that can't be evaluated"))?;
        let layout = tcx
            .layout_of(self.typing_env().as_query_input(ty))
            .map_err(|_| format!("uses a constant of type `{ty}` without a layout"))?;
        let size = layout.size.bytes_usize();
        let bytes: Vec<u8> = match value {
            ConstValue::Scalar(Scalar::Int(int)) => int.to_bits(int.size()).to_le_bytes()[..size].to_vec(),
            // A value of no size, made of empty fields.
            ConstValue::ZeroSized => Vec::new(),
            ConstValue::Indirect { alloc_id, offset } => match tcx.global_alloc(alloc_id) {
                GlobalAlloc::Memory(alloc) => {
                    let start = offset.bytes_usize();
                    alloc.inner().inspect_with_uninit_and_ptr_outside_interpreter(start..start + size).to_vec()
                }
                _ => return Err(format!("uses a constant of type `{ty}` that isn't plain data")),
            },
            _ => return Err(format!("uses a constant of type `{ty}` that isn't plain data")),
        };
        Ok(bytes)
    }

    /// A reference to constant data, such as a table (`&[f32; 5]`, `&[u8]`)
    /// or a read-only static: a reference to new locals holding its value.
    /// A static that can change (interior mutability, `static mut`) is
    /// refused.
    fn constant_ref(&mut self, c: &ConstOperand<'tcx>) -> Result<Slot> {
        let tcx = self.tcx;
        let ty = c.ty();
        let refusal = || format!("uses a reference to constant data (`{ty}`)");
        let pointee = ty.builtin_deref(true).ok_or_else(refusal)?;
        let value = c.const_.eval(tcx, self.typing_env(), c.span).map_err(|_| refusal())?;
        let (alloc_id, offset, len) = match value {
            ConstValue::Scalar(Scalar::Ptr(pointer, _)) => {
                let (provenance, offset) = pointer.prov_and_relative_offset();
                (provenance.alloc_id(), offset.bytes_usize(), None)
            }
            ConstValue::Slice { alloc_id, meta } => (alloc_id, 0, Some(meta)),
            _ => return Err(refusal()),
        };
        let alloc = match tcx.global_alloc(alloc_id) {
            GlobalAlloc::Memory(alloc) => alloc,
            GlobalAlloc::Static(def_id) => {
                let static_ty = tcx.type_of(def_id).instantiate_identity().skip_normalization();
                if tcx.static_mutability(def_id) != Some(rustc_hir::Mutability::Not)
                    || !static_ty.is_freeze(tcx, self.typing_env())
                {
                    return Err(format!("reads a static that can change (`{}`)", tcx.def_path_str(def_id)));
                }
                tcx.eval_static_initializer(def_id).map_err(|_| refusal())?
            }
            _ => return Err(refusal()),
        };
        let bytes = alloc.inner().inspect_with_uninit_and_ptr_outside_interpreter(0..alloc.inner().size().bytes_usize());
        let target = match (len, pointee.kind()) {
            (None, _) => {
                let slot = self.fresh(pointee)?;
                self.fill(&slot, pointee, bytes, offset)?;
                slot
            }
            // A slice: its elements, one after another.
            (Some(len), ty::Slice(element)) => {
                if len > MAX_ARRAY {
                    return Err(format!("uses a slice of {len} elements, more than the {MAX_ARRAY} a kernel holds in locals"));
                }
                let stride = self.size(*element)? as usize;
                let mut elements = Vec::with_capacity(len as usize);
                for index in 0..len as usize {
                    let slot = self.fresh(*element)?;
                    self.fill(&slot, *element, bytes, offset + index * stride)?;
                    elements.push(slot);
                }
                if elements.is_empty() { Slot::Unit } else { Slot::Fields(elements) }
            }
            (Some(_), _) => return Err(refusal()),
        };
        Ok(Slot::Ref(Box::new(target)))
    }

    /// Sets a slot's locals from a constant's bytes, laid out as rustc lays
    /// out `ty`, little-endian.
    fn fill(&mut self, slot: &Slot, ty: Ty<'tcx>, bytes: &[u8], base: usize) -> Result<()> {
        match slot {
            Slot::Scalar(local) => {
                let kernel = held_ty(ty).ok_or("uses a constant of an unexpected type")?;
                let size = kernel.bytes() as usize;
                let mut buffer = [0u8; 16];
                buffer[..size].copy_from_slice(bytes.get(base..base + size).ok_or("reads past a constant")?);
                let bits = u128::from_le_bytes(buffer);
                self.b.set(*local, KRvalue::Use(Const { ty: kernel, bits }.into()));
                Ok(())
            }
            Slot::Fields(fields) => {
                let layout = self
                    .tcx
                    .layout_of(self.typing_env().as_query_input(ty))
                    .map_err(|_| format!("uses a constant of type `{ty}` without a layout"))?;
                let types = self.fields(ty)?;
                for (index, (field, field_ty)) in fields.iter().zip(types).enumerate() {
                    self.fill(field, field_ty, bytes, base + layout.fields.offset(index).bytes_usize())?;
                }
                Ok(())
            }
            Slot::Enum { tag, variants } => {
                let (variant, offsets) = self.constant_variant(ty, bytes, base)?;
                let discriminant = self.discriminant(ty, variant)?;
                self.b.set(*tag, KRvalue::Use(discriminant.into()));
                let Some((_, shape)) = self.enum_shape(ty)? else {
                    return Err(format!("uses a constant of type `{ty}` that isn't plain data"));
                };
                if let Slot::Fields(fields) = &variants[variant.as_usize()] {
                    for ((field, field_ty), offset) in fields.iter().zip(&shape[variant.as_usize()]).zip(offsets) {
                        self.fill(field, *field_ty, bytes, base + offset)?;
                    }
                }
                Ok(())
            }
            Slot::Unit => Ok(()),
            // A union: its bytes, number by number.
            Slot::Bytes { numbers, offset: 0 } => {
                for &(at, number) in numbers {
                    let kernel = self.b.ty(number);
                    let (start, size) = (base + at as usize, kernel.bytes() as usize);
                    let mut buffer = [0u8; 16];
                    buffer[..size].copy_from_slice(bytes.get(start..start + size).ok_or("reads past a constant")?);
                    self.b.set(number, KRvalue::Use(Const { ty: kernel, bits: u128::from_le_bytes(buffer) }.into()));
                }
                Ok(())
            }
            _ => Err("uses a constant holding a reference".into()),
        }
    }

    /// Which variant a constant enum at `base` in `bytes` holds, and where
    /// its fields are, from rustc's layout: a tag holding the discriminant,
    /// or a niche, a field's invalid values standing for other variants.
    fn constant_variant(&self, ty: Ty<'tcx>, bytes: &[u8], base: usize) -> Result<(VariantIdx, Vec<usize>)> {
        let tcx = self.tcx;
        let layout = tcx
            .layout_of(self.typing_env().as_query_input(ty))
            .map_err(|_| format!("uses a constant of type `{ty}` without a layout"))?;
        let ty::Adt(adt, _) = ty.kind() else {
            return Err(format!("uses a constant of type `{ty}` that isn't plain data"));
        };
        match &layout.variants {
            Variants::Empty => Err(format!("uses a constant of the uninhabited type `{ty}`")),
            Variants::Single { index } => {
                let count = adt.variant(*index).fields.len();
                Ok((*index, (0..count).map(|field| layout.fields.offset(field).bytes_usize()).collect()))
            }
            Variants::Multiple { tag, tag_encoding, tag_field, variants } => {
                let size = tag.size(&tcx).bytes_usize();
                let at = base + layout.fields.offset(tag_field.as_usize()).bytes_usize();
                let mut buffer = [0u8; 16];
                buffer[..size].copy_from_slice(bytes.get(at..at + size).ok_or("reads past a constant")?);
                let raw = u128::from_le_bytes(buffer);
                let mask = if size >= 16 { u128::MAX } else { (1u128 << (8 * size)) - 1 };
                let variant = match tag_encoding {
                    TagEncoding::Direct => adt
                        .discriminants(tcx)
                        .find(|(_, discriminant)| discriminant.val & mask == raw)
                        .map(|(variant, _)| variant)
                        .ok_or_else(|| format!("uses a constant of type `{ty}` with no valid variant"))?,
                    TagEncoding::Niche { untagged_variant, niche_variants, niche_start } => {
                        let relative = raw.wrapping_sub(*niche_start) & mask;
                        let count = u128::from(niche_variants.last.as_u32() - niche_variants.start.as_u32());
                        if relative <= count {
                            VariantIdx::from_u32(niche_variants.start.as_u32() + relative as u32)
                        } else {
                            *untagged_variant
                        }
                    }
                };
                let offsets = variants[variant].field_offsets.iter().map(|offset| offset.bytes_usize()).collect();
                Ok((variant, offsets))
            }
        }
    }

    /// `operand`'s bits as a value of type `to`, as `transmute` keeps them.
    /// Both are plain data of one size, laid out as rustc lays them out,
    /// little-endian; each number of the result gathers the bytes it covers
    /// from the source's numbers. Bytes no number covers are padding, which
    /// a defined transmute never reads into a number. Enums are read and
    /// written through their tags or niches, as their layout stores them.
    pub(super) fn transmute(&mut self, frame: &mut Frame<'_, 'tcx>, operand: &Operand<'tcx>, to: Ty<'tcx>) -> Result<Slot> {
        let from = operand.ty(frame.body, self.tcx);
        // A pointer, or a value holding only a pointer (`NonNull`), as
        // another: the same target.
        if let (Some(from_path), Some(to_path)) = (self.pointer_path(from, 0), self.pointer_path(to, 0)) {
            let mut slot = &self.operand_slot(frame, operand)?;
            let value = slot.clone();
            for &index in &from_path {
                let Slot::Fields(fields) = slot else { return Err(cast_refusal(CastKind::Transmute, to)) };
                slot = &fields[index];
            }
            let pointer = if from_path.is_empty() { value } else { slot.clone() };
            if !matches!(pointer, Slot::Ref(_)) {
                return Err(cast_refusal(CastKind::Transmute, to));
            }
            return self.holding_pointer(to, &to_path, pointer);
        }
        if is_ref(from) || is_ref(to) {
            return Err(cast_refusal(CastKind::Transmute, to));
        }
        let (from_size, to_size) = (self.size(from)?, self.size(to)?);
        if from_size != to_size {
            return Err(format!("reinterprets `{from}` as `{to}`, of another size"));
        }
        let value = self.operand_slot(frame, operand)?;
        let result = self.fresh(to)?;
        self.decode(&result, to, 0, &value, from)?;
        Ok(result)
    }

    /// The field path to the only pointer in a value of type `ty`, if it
    /// holds one pointer and otherwise nothing (`NonNull`, a pointer in
    /// `#[repr(transparent)]` wrappers): empty for a pointer itself.
    pub(super) fn pointer_path(&self, ty: Ty<'tcx>, depth: usize) -> Option<Vec<usize>> {
        if is_ref(ty) {
            return Some(Vec::new());
        }
        let aggregate = match ty.kind() {
            ty::Adt(adt, _) => adt.is_struct(),
            ty::Tuple(_) => true,
            _ => false,
        };
        if !aggregate || depth > 8 {
            return None;
        }
        let mut found = None;
        for (index, field) in self.fields(ty).ok()?.into_iter().enumerate() {
            if let Some(mut path) = self.pointer_path(field, depth + 1) {
                if found.is_some() {
                    return None;
                }
                path.insert(0, index);
                found = Some(path);
            } else if self.size(field).ok()? != 0 {
                return None;
            }
        }
        found
    }

    /// A value of type `ty` holding `pointer` at `path` (`pointer_path`), its
    /// other fields empty.
    pub(super) fn holding_pointer(&mut self, ty: Ty<'tcx>, path: &[usize], pointer: Slot) -> Result<Slot> {
        let Some((&first, rest)) = path.split_first() else {
            return Ok(pointer);
        };
        let fields = self.fields(ty)?;
        let mut slots = Vec::with_capacity(fields.len());
        for (index, field) in fields.into_iter().enumerate() {
            slots.push(if index == first { self.holding_pointer(field, rest, pointer.clone())? } else { self.fresh(field)? });
        }
        Ok(Slot::Fields(slots))
    }

    /// Sets `dest`, of type `ty`, from the bytes at `base` of `src`, a value
    /// of type `src_ty`.
    pub(super) fn decode(&mut self, dest: &Slot, ty: Ty<'tcx>, base: u64, src: &Slot, src_ty: Ty<'tcx>) -> Result<()> {
        match dest {
            Slot::Scalar(local) => {
                let kernel = self.b.ty(*local);
                let width = u64::from(kernel.bytes());
                let rvalue = match src {
                    // The same bytes: one number reinterpreted.
                    Slot::Scalar(number) if base == 0 && self.b.ty(*number).bytes() == kernel.bytes() => {
                        self.same_width((*number).into(), kernel)
                    }
                    Slot::Bytes { numbers, offset } if numbers.iter().any(|&(at, number)| at == offset + base && self.b.ty(number).bytes() == kernel.bytes()) => {
                        let &(_, number) = numbers.iter().find(|&&(at, number)| at == offset + base && self.b.ty(number).bytes() == kernel.bytes()).expect("found above");
                        self.same_width(number.into(), kernel)
                    }
                    // 128 bits: two halves of 64.
                    _ if width == 16 => {
                        let low = self.gather(src, src_ty, 0, base, 8)?;
                        let high = self.gather(src, src_ty, 0, base + 8, 8)?;
                        let (low, high) = (self.b.cast(low, KTy::U128), self.b.cast(high, KTy::U128));
                        let high = self.b.binary(KBinOp::Shl, high, Const::int(KTy::U32, 64));
                        let bits = self.b.binary(KBinOp::Or, low, high);
                        self.same_width(bits.into(), kernel)
                    }
                    _ => {
                        let bits = self.gather(src, src_ty, 0, base, width)?;
                        let narrow = self.b.cast(bits, unsigned_of(width));
                        self.same_width(narrow.into(), kernel)
                    }
                };
                self.set(*local, rvalue);
            }
            Slot::Fields(fields) => {
                let layout = self.layout(ty)?;
                let types = self.fields(ty)?;
                for (index, (field, field_ty)) in fields.iter().zip(types).enumerate() {
                    self.decode(field, field_ty, base + layout.fields.offset(index).bytes(), src, src_ty)?;
                }
            }
            Slot::Unit => {}
            Slot::Enum { tag, variants } => {
                let layout = self.layout(ty)?;
                let Some((tag_ty, shape)) = self.enum_shape(ty)? else {
                    return Err(format!("reinterprets bits as `{ty}`"));
                };
                for (index, variant) in variants.iter().enumerate() {
                    let (Slot::Fields(fields), Some(offsets)) = (variant, variant_offsets(&layout, index)) else {
                        continue;
                    };
                    for ((field, field_ty), offset) in fields.iter().zip(&shape[index]).zip(offsets) {
                        self.decode(field, *field_ty, base + offset, src, src_ty)?;
                    }
                }
                let discriminant: KOperand = match &layout.variants {
                    Variants::Empty => return Err(format!("reinterprets bits as the uninhabited `{ty}`")),
                    Variants::Single { index } => self.discriminant(ty, *index)?.into(),
                    Variants::Multiple { tag: scalar, tag_encoding, tag_field, .. } => {
                        let size = scalar.size(&self.tcx).bytes();
                        if size > 8 {
                            return Err(format!("reinterprets bits as `{ty}`, whose tag is wider than 64 bits"));
                        }
                        let at = base + layout.fields.offset(tag_field.as_usize()).bytes();
                        let raw = self.gather(src, src_ty, 0, at, size)?;
                        match tag_encoding {
                            // The tag is the discriminant, in its width and
                            // signedness.
                            TagEncoding::Direct => {
                                let narrow = self.b.cast(raw, unsigned_of(size));
                                let signed = matches!(scalar.primitive(), rustc_abi::Primitive::Int(_, true));
                                let narrow = if signed { self.b.cast(narrow, signed_of(size)) } else { narrow };
                                self.b.cast(narrow, tag_ty).into()
                            }
                            // Tag values from `niche_start` on stand for the
                            // niche variants in order; others for the
                            // untagged variant.
                            TagEncoding::Niche { untagged_variant, niche_variants, niche_start } => {
                                let mask = if size == 8 { u64::MAX } else { (1u64 << (8 * size)) - 1 };
                                let start = Const::int(KTy::U64, i128::from(*niche_start as u64));
                                let shifted = self.b.binary(KBinOp::Sub, raw, start);
                                let relative = self.b.binary(KBinOp::And, shifted, Const::int(KTy::U64, mask.into()));
                                let mut chosen: KOperand = self.discriminant(ty, *untagged_variant)?.into();
                                let (first, last) = (niche_variants.start.as_u32(), niche_variants.last.as_u32());
                                for variant in first..=last {
                                    let step = Const::int(KTy::U64, (variant - first).into());
                                    let is = self.b.binary(KBinOp::Eq, relative, step);
                                    let value = self.discriminant(ty, VariantIdx::from_u32(variant))?;
                                    chosen = self.b.select(is, value, chosen).into();
                                }
                                chosen
                            }
                        }
                    }
                };
                self.set(*tag, KRvalue::Use(discriminant));
            }
            // A union: each number holding its bytes takes the source's.
            Slot::Bytes { numbers, offset: 0 } => {
                for &(at, number) in numbers {
                    let kernel = self.b.ty(number);
                    let width = u64::from(kernel.bytes());
                    let bits = self.gather(src, src_ty, 0, base + at, width)?;
                    let narrow = self.b.cast(bits, unsigned_of(width));
                    let rvalue = self.same_width(narrow.into(), kernel);
                    self.set(number, rvalue);
                }
            }
            _ => return Err(format!("reinterprets bits as `{ty}`, which holds a reference")),
        }
        Ok(())
    }

    /// Writes `value`, of type `ty`, to the bytes at `start` of data held
    /// as `numbers` (a union, or other data seen through a pointer cast).
    /// Each number it overlaps takes the value's bytes there and keeps its
    /// others, as memory would.
    pub(super) fn encode(&mut self, numbers: &[(u64, KLocal)], start: u64, value: &Slot, ty: Ty<'tcx>) -> Result<()> {
        let end = start + self.size(ty)?;
        let overlap = |at: u64, width: u64| (at.max(start), (at + width).min(end));
        // Bytes no number holds are the data's padding, which its own reads
        // never see, so a later read through this view would lose them.
        let held: u64 = numbers
            .iter()
            .map(|&(at, number)| {
                let (low, high) = overlap(at, u64::from(self.b.ty(number).bytes()));
                high.saturating_sub(low)
            })
            .sum();
        if held != end - start {
            return Err(format!("writes `{ty}` over bytes the data it's seen in doesn't hold"));
        }
        // Every new value first: `value` may read the numbers written.
        let mut updates = Vec::new();
        for &(at, number) in numbers {
            let width = u64::from(self.b.ty(number).bytes());
            let (low, high) = overlap(at, width);
            if low >= high {
                continue;
            }
            if width > 8 {
                return Err(format!("writes `{ty}` over part of a 128-bit number"));
            }
            let bits = self.gather(value, ty, start, low, high - low)?;
            let placed = self.place_bits(bits, low, at);
            let updated = if (low, high) == (at, at + width) {
                placed
            } else {
                let ones = if high - low == 8 { u64::MAX } else { (1u64 << (8 * (high - low))) - 1 };
                let keep = !(ones << (8 * (low - at)));
                let old = self.unsigned_bits(number);
                let kept = self.b.binary(KBinOp::And, old, Const::int(KTy::U64, keep.into()));
                self.b.binary(KBinOp::Or, kept, placed)
            };
            updates.push((number, width, updated));
        }
        for (number, width, bits) in updates {
            let narrow = self.b.cast(bits, unsigned_of(width));
            let rvalue = self.same_width(narrow.into(), self.b.ty(number));
            self.set(number, rvalue);
        }
        Ok(())
    }

    /// Whether two values held as bytes have numbers of the same widths at
    /// the same offsets, as two values of one union type do.
    pub(super) fn same_bytes(&self, a: &[(u64, KLocal)], b: &[(u64, KLocal)]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(&(x, l), &(y, m))| x == y && self.b.ty(l) == self.b.ty(m))
    }

    /// The bits of the bytes `offset..offset + width` (at most 8) of `src`,
    /// a value of type `ty` laid out from `base`, zero-extended to a `u64`.
    pub(super) fn gather(&mut self, src: &Slot, ty: Ty<'tcx>, base: u64, offset: u64, width: u64) -> Result<KOperand> {
        let zero = KOperand::from(Const::int(KTy::U64, 0));
        Ok(match src {
            Slot::Scalar(number) => self.number_bits(*number, i128::from(base), offset, width).unwrap_or(zero),
            Slot::Fields(fields) => {
                let layout = self.layout(ty)?;
                let types = self.fields(ty)?;
                let mut bits = zero;
                for (index, (field, field_ty)) in fields.iter().zip(types).enumerate() {
                    let at = base + layout.fields.offset(index).bytes();
                    if at >= offset + width || offset >= at + self.size(field_ty)? {
                        continue;
                    }
                    let part = self.gather(field, field_ty, at, offset, width)?;
                    bits = self.or(bits, part);
                }
                bits
            }
            // Bytes seen through a pointer: the data's numbers, placed where
            // they fall in the value read.
            Slot::Bytes { numbers, offset: start } => {
                let mut bits = zero;
                for &(at, number) in numbers {
                    let position = i128::from(base) + i128::from(at) - i128::from(*start);
                    if let Some(placed) = self.number_bits(number, position, offset, width) {
                        bits = self.or(bits, placed);
                    }
                }
                bits
            }
            Slot::Unit => zero,
            // Each variant's bytes, with its tag, chosen by the discriminant.
            Slot::Enum { tag, variants } => {
                let layout = self.layout(ty)?;
                let Some((_, shape)) = self.enum_shape(ty)? else {
                    return Err(format!("reinterprets the bits of `{ty}`"));
                };
                let mut chosen = zero;
                for (index, variant) in variants.iter().enumerate() {
                    let Some(offsets) = variant_offsets(&layout, index) else { continue };
                    let mut bits = zero;
                    if let Slot::Fields(fields) = variant {
                        for ((field, field_ty), field_offset) in fields.iter().zip(&shape[index]).zip(offsets) {
                            let at = base + field_offset;
                            if at >= offset + width || offset >= at + self.size(*field_ty)? {
                                continue;
                            }
                            let part = self.gather(field, *field_ty, at, offset, width)?;
                            bits = self.or(bits, part);
                        }
                    }
                    if let Variants::Multiple { tag: scalar, tag_encoding, tag_field, .. } = &layout.variants {
                        let size = scalar.size(&self.tcx).bytes();
                        let at = base + layout.fields.offset(tag_field.as_usize()).bytes();
                        let stored = match tag_encoding {
                            TagEncoding::Direct => Some(self.discriminant(ty, VariantIdx::from_usize(index))?.bits as u64),
                            TagEncoding::Niche { untagged_variant, niche_variants, niche_start } => {
                                let (first, last) = (niche_variants.start.as_usize(), niche_variants.last.as_usize());
                                (index != untagged_variant.as_usize() && (first..=last).contains(&index))
                                    .then(|| (*niche_start as u64).wrapping_add((index - first) as u64))
                            }
                        };
                        if let Some(stored) = stored.filter(|_| at < offset + width && offset < at + size) {
                            let mask = if size >= 8 { u64::MAX } else { (1u64 << (8 * size)) - 1 };
                            let constant = KOperand::from(Const::int(KTy::U64, (stored & mask).into()));
                            let placed = self.place_bits(constant, at, offset);
                            bits = self.or(bits, placed.into());
                        }
                    }
                    let discriminant = self.discriminant(ty, VariantIdx::from_usize(index))?;
                    let is = self.b.binary(KBinOp::Eq, *tag, discriminant);
                    chosen = self.b.select(is, bits, chosen).into();
                }
                chosen
            }
            _ => return Err(format!("reinterprets the bits of `{ty}`, which holds a reference")),
        })
    }

    /// The bits a number at `position` gives the bytes `offset..offset +
    /// width` (at most 8), as a `u64`, if it overlaps them. A 128-bit number
    /// gives its two halves.
    fn number_bits(&mut self, number: KLocal, position: i128, offset: u64, width: u64) -> Option<KOperand> {
        let ty = self.b.ty(number);
        let size = i128::from(ty.bytes());
        let (offset, end) = (i128::from(offset), i128::from(offset + width));
        if position >= end || offset >= position + size {
            return None;
        }
        let halves: Vec<(i128, KLocal)> = if size == 16 {
            let unsigned = self.b.cast(number, KTy::U128);
            let low = self.b.cast(unsigned, KTy::U64);
            let shifted = self.b.binary(KBinOp::Shr, unsigned, Const::int(KTy::U32, 64));
            let high = self.b.cast(shifted, KTy::U64);
            vec![(position, low), (position + 8, high)]
        } else {
            vec![(position, self.unsigned_bits(number))]
        };
        let mut bits: Option<KOperand> = None;
        for (at, half) in halves {
            if at >= end || offset >= at + 8.min(size) {
                continue;
            }
            let placed = if at >= offset {
                self.b.binary(KBinOp::Shl, half, Const::int(KTy::U32, 8 * (at - offset)))
            } else {
                self.b.binary(KBinOp::Shr, half, Const::int(KTy::U32, 8 * (offset - at)))
            };
            bits = Some(match bits {
                Some(previous) => self.b.binary(KBinOp::Or, previous, placed).into(),
                None => placed.into(),
            });
        }
        bits
    }

    /// `bits`, of a value at `at`, moved to where they fall in a range
    /// starting at `offset`.
    fn place_bits(&mut self, bits: KOperand, at: u64, offset: u64) -> KLocal {
        if at >= offset {
            self.b.binary(KBinOp::Shl, bits, Const::int(KTy::U32, (8 * (at - offset)).into()))
        } else {
            self.b.binary(KBinOp::Shr, bits, Const::int(KTy::U32, (8 * (offset - at)).into()))
        }
    }

    fn or(&mut self, a: KOperand, b: KOperand) -> KOperand {
        if a == KOperand::from(Const::int(KTy::U64, 0)) {
            b
        } else {
            self.b.binary(KBinOp::Or, a, b).into()
        }
    }

    pub(super) fn layout(&self, ty: Ty<'tcx>) -> Result<rustc_middle::ty::layout::TyAndLayout<'tcx>> {
        self.tcx.layout_of(self.typing_env().as_query_input(ty)).map_err(|_| format!("uses `{ty}`, which has no layout"))
    }

    /// A number as another type of its width, keeping its bits. A `bool`'s
    /// byte is 0 or 1, and only those bytes are valid `bool`s.
    fn same_width(&mut self, value: KOperand, to: KTy) -> KRvalue {
        let from = self.b.ty(value);
        if from == to {
            KRvalue::Use(value)
        } else if to == KTy::Bool {
            KRvalue::Binary(KBinOp::Ne, value, Const::int(from, 0).into())
        } else if from == KTy::Bool || from.is_int() && to.is_int() {
            KRvalue::Cast(value, to)
        } else {
            KRvalue::Reinterpret(value, to)
        }
    }

    /// A number's bits, zero-extended to a `u64`.
    fn unsigned_bits(&mut self, number: KLocal) -> KLocal {
        let ty = self.b.ty(number);
        let same = self.same_width(number.into(), unsigned_of(u64::from(ty.bytes())));
        let bits = self.b.assign(same);
        self.b.cast(bits, KTy::U64)
    }

    pub(super) fn size(&self, ty: Ty<'tcx>) -> Result<u64> {
        self.tcx
            .layout_of(self.typing_env().as_query_input(ty))
            .map(|layout| layout.size.bytes())
            .map_err(|_| format!("uses `{ty}`, which has no layout"))
    }
}
