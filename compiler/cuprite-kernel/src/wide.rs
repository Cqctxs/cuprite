//! 128-bit integers as pairs of 64-bit halves.
//!
//! Neither backend handles `I128` or `U128` itself: [`lower`] rewrites a
//! kernel so that each 128-bit local is two `U64` locals, its low and high
//! halves, and each operation on them a sequence of 64-bit operations with
//! the kernel's semantics: wrapping arithmetic, shift amounts modulo 128,
//! division truncating toward zero (a 128-step loop, or one 64-bit
//! division when both operands fit in 64 bits), float-to-integer casts that
//! saturate with NaN becoming 0, and integer-to-float casts rounded to
//! nearest, ties to even. Loads and stores move the halves little-endian.

use crate::{
    builder::Builder, rvalue_operands, BinOp, Const, Intrinsic, Kernel, Local, Operand, Rvalue, Statement, Terminator,
    Ty, UnOp,
};

fn is_wide(ty: Ty) -> bool {
    matches!(ty, Ty::I128 | Ty::U128)
}

/// A 128-bit value: its low and high halves, each a `U64`.
type Pair = (Operand, Operand);

#[derive(Clone, Copy)]
enum Mapped {
    One(Local),
    Two(Local, Local),
}

/// `kernel` without 128-bit types, computing the same.
///
/// # Panics
/// If a parameter is 128-bit: a launch can't pass one.
pub fn lower(kernel: &Kernel) -> Kernel {
    let wide_operand = |o: &Operand| is_wide(kernel.ty(*o));
    let uses_wide = kernel.locals.iter().any(|&ty| is_wide(ty))
        || kernel.blocks.iter().flat_map(|block| &block.statements).any(|statement| match statement {
            Statement::Assign(_, rvalue) => rvalue_operands(rvalue).iter().any(wide_operand),
            Statement::Store(pointer, value) => wide_operand(pointer) || wide_operand(value),
        });
    if !uses_wide {
        return kernel.clone();
    }
    let params = &kernel.locals[..kernel.params];
    assert!(params.iter().all(|&ty| !is_wide(ty)), "a 128-bit parameter can't be passed to a kernel");
    let (b, param_locals) = Builder::new(&kernel.name, params);
    let mut lower = Lower { kernel, b, locals: Vec::new() };
    for (index, &ty) in kernel.locals.iter().enumerate() {
        let mapped = if index < kernel.params {
            Mapped::One(param_locals[index])
        } else if is_wide(ty) {
            Mapped::Two(lower.b.local(Ty::U64), lower.b.local(Ty::U64))
        } else {
            Mapped::One(lower.b.local(ty))
        };
        lower.locals.push(mapped);
    }
    let starts: Vec<_> =
        (0..kernel.blocks.len()).map(|index| if index == 0 { crate::BlockId(0) } else { lower.b.block() }).collect();
    for (index, block) in kernel.blocks.iter().enumerate() {
        lower.b.switch_to(starts[index]);
        for statement in &block.statements {
            lower.statement(statement);
        }
        match block.terminator {
            Terminator::Goto(target) => lower.b.goto(starts[target.0 as usize]),
            Terminator::If(condition, then, otherwise) => {
                let condition = lower.one(condition);
                lower.b.branch(condition, starts[then.0 as usize], starts[otherwise.0 as usize]);
            }
            Terminator::Return => lower.b.ret(),
            Terminator::Fail => lower.b.fail(),
        }
    }
    lower.b.into_kernel()
}

struct Lower<'k> {
    kernel: &'k Kernel,
    b: Builder,
    locals: Vec<Mapped>,
}

fn u64c(value: u64) -> Operand {
    Const::int(Ty::U64, value.into()).into()
}

fn u32c(value: u32) -> Operand {
    Const::int(Ty::U32, value.into()).into()
}

impl Lower<'_> {
    /// An operand that isn't 128-bit, renamed.
    fn one(&self, operand: Operand) -> Operand {
        match operand {
            Operand::Local(local) => match self.locals[local.0 as usize] {
                Mapped::One(new) => new.into(),
                Mapped::Two(..) => unreachable!("a 128-bit operand where a narrow one was expected"),
            },
            constant => constant,
        }
    }

    /// A 128-bit operand's halves.
    fn two(&self, operand: Operand) -> Pair {
        match operand {
            Operand::Local(local) => match self.locals[local.0 as usize] {
                Mapped::Two(low, high) => (low.into(), high.into()),
                Mapped::One(_) => unreachable!("a narrow operand where a 128-bit one was expected"),
            },
            Operand::Const(c) => (u64c(c.bits as u64), u64c((c.bits >> 64) as u64)),
        }
    }

    fn ty(&self, operand: Operand) -> Ty {
        self.kernel.ty(operand)
    }

    fn statement(&mut self, statement: &Statement) {
        match statement {
            Statement::Store(pointer, value) => {
                let pointer = self.one(*pointer);
                if is_wide(self.ty(*value)) {
                    let (low, high) = self.two(*value);
                    self.b.store(pointer, low);
                    let upper = self.b.offset(pointer, u64c(1), Ty::U64);
                    self.b.store(upper, high);
                } else {
                    let value = self.one(*value);
                    self.b.store(pointer, value);
                }
            }
            Statement::Assign(dest, rvalue) => {
                let dest_ty = self.kernel.locals[dest.0 as usize];
                let involves = is_wide(dest_ty) || rvalue_operands(rvalue).iter().any(|o| is_wide(self.ty(*o)));
                if !involves {
                    let rvalue = rvalue.map_operands(|o| self.one(o));
                    let Mapped::One(dest) = self.locals[dest.0 as usize] else { unreachable!("checked") };
                    self.b.set(dest, rvalue);
                    return;
                }
                match self.locals[dest.0 as usize] {
                    Mapped::Two(low, high) => {
                        let (l, h) = self.wide_rvalue(rvalue, dest_ty);
                        self.b.set(low, Rvalue::Use(l));
                        self.b.set(high, Rvalue::Use(h));
                    }
                    Mapped::One(dest) => {
                        let value = self.narrow_rvalue(rvalue, dest_ty);
                        self.b.set(dest, value);
                    }
                }
            }
        }
    }

    /// An rvalue with a 128-bit result, as its halves.
    fn wide_rvalue(&mut self, rvalue: &Rvalue, ty: Ty) -> Pair {
        let signed = ty.is_signed();
        match rvalue {
            Rvalue::Use(a) | Rvalue::Reinterpret(a, _) => self.two(*a),
            Rvalue::Binary(op, a, b) => {
                let a = self.two(*a);
                match op {
                    BinOp::Shl | BinOp::Shr => {
                        let amount = self.amount(*b);
                        match (op, signed) {
                            (BinOp::Shl, _) => self.shl(a, amount),
                            (_, false) => self.lshr(a, amount),
                            (_, true) => self.ashr(a, amount),
                        }
                    }
                    _ => {
                        let b = self.two(*b);
                        match op {
                            BinOp::Add => self.add(a, b),
                            BinOp::Sub => self.sub(a, b),
                            BinOp::Mul => self.mul(a, b),
                            BinOp::Div | BinOp::Rem => {
                                let (quotient, remainder) = self.divmod(a, b, signed);
                                if *op == BinOp::Div { quotient } else { remainder }
                            }
                            BinOp::And | BinOp::Or | BinOp::Xor => {
                                (self.b.binary(*op, a.0, b.0).into(), self.b.binary(*op, a.1, b.1).into())
                            }
                            _ => unreachable!("comparisons give `bool`"),
                        }
                    }
                }
            }
            Rvalue::Unary(UnOp::Neg, a) => {
                let a = self.two(*a);
                self.neg(a)
            }
            Rvalue::Unary(UnOp::Not, a) => {
                let (low, high) = self.two(*a);
                (self.b.unary(UnOp::Not, low).into(), self.b.unary(UnOp::Not, high).into())
            }
            Rvalue::Cast(a, _) => {
                let from = self.ty(*a);
                if is_wide(from) {
                    self.two(*a)
                } else if from.is_float() {
                    let a = self.one(*a);
                    self.float_to_wide(a, signed)
                } else {
                    // An integer or `bool`, extended by its signedness.
                    let a = self.one(*a);
                    let low = self.b.cast(a, Ty::U64);
                    let high: Operand = if from.is_signed() {
                        let wide = self.b.cast(a, Ty::I64);
                        let sign = self.b.binary(BinOp::Shr, wide, u32c(63));
                        self.b.cast(sign, Ty::U64).into()
                    } else {
                        u64c(0)
                    };
                    (low.into(), high)
                }
            }
            Rvalue::Call(Intrinsic::RotateLeft | Intrinsic::RotateRight, args) => {
                let a = self.two(args[0]);
                let n = self.one(args[1]);
                let n = self.b.binary(BinOp::And, n, u32c(127));
                let back = self.b.binary(BinOp::Sub, u32c(128), n);
                let back = self.b.binary(BinOp::And, back, u32c(127));
                let (left, right) = if matches!(rvalue, Rvalue::Call(Intrinsic::RotateLeft, _)) {
                    (n, back)
                } else {
                    (back, n)
                };
                let (ll, lh) = self.shl(a, left.into());
                let (rl, rh) = self.lshr(a, right.into());
                (self.b.binary(BinOp::Or, ll, rl).into(), self.b.binary(BinOp::Or, lh, rh).into())
            }
            Rvalue::Load(pointer, _) => {
                let pointer = self.one(*pointer);
                let low = self.b.load(pointer, Ty::U64);
                let upper = self.b.offset(pointer, u64c(1), Ty::U64);
                let high = self.b.load(upper, Ty::U64);
                (low.into(), high.into())
            }
            Rvalue::Select(condition, a, b) => {
                let condition = self.one(*condition);
                let (a, b) = (self.two(*a), self.two(*b));
                (self.b.select(condition, a.0, b.0).into(), self.b.select(condition, a.1, b.1).into())
            }
            other => unreachable!("{other:?} doesn't give a 128-bit value"),
        }
    }

    /// An rvalue with a narrow result and a 128-bit operand.
    fn narrow_rvalue(&mut self, rvalue: &Rvalue, ty: Ty) -> Rvalue {
        match rvalue {
            Rvalue::Binary(op, a, b) if op.is_comparison() => {
                let signed = self.ty(*a).is_signed();
                let (a, b) = (self.two(*a), self.two(*b));
                let holds = match op {
                    BinOp::Eq => self.eq(a, b),
                    BinOp::Ne => {
                        let eq = self.eq(a, b);
                        self.b.unary(UnOp::Not, eq)
                    }
                    BinOp::Lt => self.lt(a, b, signed),
                    BinOp::Gt => self.lt(b, a, signed),
                    BinOp::Le => {
                        let gt = self.lt(b, a, signed);
                        self.b.unary(UnOp::Not, gt)
                    }
                    _ => {
                        let lt = self.lt(a, b, signed);
                        self.b.unary(UnOp::Not, lt)
                    }
                };
                Rvalue::Use(holds.into())
            }
            // A narrow value shifted by a 128-bit amount: the amount modulo
            // the value's width is the low half's.
            Rvalue::Binary(op, a, b) => {
                let a = self.one(*a);
                let (low, _) = self.two(*b);
                Rvalue::Binary(*op, a, low)
            }
            Rvalue::Cast(a, to) => {
                let signed = self.ty(*a).is_signed();
                let a = self.two(*a);
                if to.is_float() {
                    Rvalue::Use(self.wide_to_float(a, signed, *to).into())
                } else {
                    Rvalue::Cast(a.0, *to)
                }
            }
            Rvalue::Call(intrinsic, args) => {
                let (low, high) = self.two(args[0]);
                let count = match intrinsic {
                    Intrinsic::LeadingZeros => {
                        let zero = self.b.binary(BinOp::Eq, high, u64c(0));
                        let (low, high) = (self.b.call(Intrinsic::LeadingZeros, &[low]), self.b.call(Intrinsic::LeadingZeros, &[high]));
                        let below = self.b.binary(BinOp::Add, low, u32c(64));
                        self.b.select(zero, below, high)
                    }
                    Intrinsic::TrailingZeros => {
                        let zero = self.b.binary(BinOp::Eq, low, u64c(0));
                        let (low, high) = (self.b.call(Intrinsic::TrailingZeros, &[low]), self.b.call(Intrinsic::TrailingZeros, &[high]));
                        let above = self.b.binary(BinOp::Add, high, u32c(64));
                        self.b.select(zero, above, low)
                    }
                    Intrinsic::CountOnes => {
                        let (low, high) = (self.b.call(Intrinsic::CountOnes, &[low]), self.b.call(Intrinsic::CountOnes, &[high]));
                        self.b.binary(BinOp::Add, low, high)
                    }
                    other => unreachable!("{other:?} of a 128-bit value"),
                };
                Rvalue::Use(count.into())
            }
            other => unreachable!("{other:?} with a 128-bit operand giving {ty:?}"),
        }
    }

    /// A shift amount, modulo 128, as `U32`.
    fn amount(&mut self, amount: Operand) -> Operand {
        let narrow = if is_wide(self.ty(amount)) { self.two(amount).0 } else { self.one(amount) };
        let narrow = self.b.cast(narrow, Ty::U32);
        self.b.binary(BinOp::And, narrow, u32c(127)).into()
    }

    fn add(&mut self, a: Pair, b: Pair) -> Pair {
        let low = self.b.binary(BinOp::Add, a.0, b.0);
        let carried = self.b.binary(BinOp::Lt, low, a.0);
        let carry = self.b.cast(carried, Ty::U64);
        let high = self.b.binary(BinOp::Add, a.1, b.1);
        let high = self.b.binary(BinOp::Add, high, carry);
        (low.into(), high.into())
    }

    fn sub(&mut self, a: Pair, b: Pair) -> Pair {
        let low = self.b.binary(BinOp::Sub, a.0, b.0);
        let borrowed = self.b.binary(BinOp::Lt, a.0, b.0);
        let borrow = self.b.cast(borrowed, Ty::U64);
        let high = self.b.binary(BinOp::Sub, a.1, b.1);
        let high = self.b.binary(BinOp::Sub, high, borrow);
        (low.into(), high.into())
    }

    fn neg(&mut self, a: Pair) -> Pair {
        self.sub((u64c(0), u64c(0)), a)
    }

    /// The low 128 bits of the product, the same for signed and unsigned.
    fn mul(&mut self, a: Pair, b: Pair) -> Pair {
        let low = self.b.binary(BinOp::Mul, a.0, b.0);
        let carry = self.b.call(Intrinsic::MulHigh, &[a.0, b.0]);
        let cross = self.b.binary(BinOp::Mul, a.0, b.1);
        let other = self.b.binary(BinOp::Mul, a.1, b.0);
        let high = self.b.binary(BinOp::Add, carry, cross);
        let high = self.b.binary(BinOp::Add, high, other);
        (low.into(), high.into())
    }

    fn eq(&mut self, a: Pair, b: Pair) -> Local {
        let (low, high) = (self.b.binary(BinOp::Eq, a.0, b.0), self.b.binary(BinOp::Eq, a.1, b.1));
        self.b.binary(BinOp::And, low, high)
    }

    /// `a < b`: by the high halves (signed for signed values), then the low.
    fn lt(&mut self, a: Pair, b: Pair, signed: bool) -> Local {
        let high_less = if signed {
            let (x, y) = (self.b.cast(a.1, Ty::I64), self.b.cast(b.1, Ty::I64));
            self.b.binary(BinOp::Lt, x, y)
        } else {
            self.b.binary(BinOp::Lt, a.1, b.1)
        };
        let high_equal = self.b.binary(BinOp::Eq, a.1, b.1);
        let low_less = self.b.binary(BinOp::Lt, a.0, b.0);
        let tie = self.b.binary(BinOp::And, high_equal, low_less);
        self.b.binary(BinOp::Or, high_less, tie)
    }

    /// `a << n`, `n` below 128. The kernel's 64-bit shifts take their amount
    /// modulo 64, so the bits crossing halves are shifted in two steps.
    fn shl(&mut self, a: Pair, n: Operand) -> Pair {
        let s = self.b.binary(BinOp::And, n, u32c(63));
        let big = self.b.binary(BinOp::Ge, n, u32c(64));
        let shifted = self.b.binary(BinOp::Shl, a.0, s);
        let rest = self.b.binary(BinOp::Sub, u32c(63), s);
        let half = self.b.binary(BinOp::Shr, a.0, u32c(1));
        let carry = self.b.binary(BinOp::Shr, half, rest);
        let high = self.b.binary(BinOp::Shl, a.1, s);
        let small_high = self.b.binary(BinOp::Or, high, carry);
        (self.b.select(big, u64c(0), shifted).into(), self.b.select(big, shifted, small_high).into())
    }

    /// `a >> n` for unsigned values, `n` below 128.
    fn lshr(&mut self, a: Pair, n: Operand) -> Pair {
        let s = self.b.binary(BinOp::And, n, u32c(63));
        let big = self.b.binary(BinOp::Ge, n, u32c(64));
        let shifted = self.b.binary(BinOp::Shr, a.1, s);
        let small_low = self.low_with_carry(a, s);
        (self.b.select(big, shifted, small_low).into(), self.b.select(big, u64c(0), shifted).into())
    }

    /// `a >> n` for signed values, `n` below 128: the sign fills in.
    fn ashr(&mut self, a: Pair, n: Operand) -> Pair {
        let s = self.b.binary(BinOp::And, n, u32c(63));
        let big = self.b.binary(BinOp::Ge, n, u32c(64));
        let signed_high = self.b.cast(a.1, Ty::I64);
        let shifted = self.b.binary(BinOp::Shr, signed_high, s);
        let shifted = self.b.cast(shifted, Ty::U64);
        let sign = self.b.binary(BinOp::Shr, signed_high, u32c(63));
        let sign = self.b.cast(sign, Ty::U64);
        let small_low = self.low_with_carry(a, s);
        (self.b.select(big, shifted, small_low).into(), self.b.select(big, sign, shifted).into())
    }

    /// The low half shifted right by `s` (below 64), with the high half's
    /// bits shifted in.
    fn low_with_carry(&mut self, a: Pair, s: Local) -> Local {
        let low = self.b.binary(BinOp::Shr, a.0, s);
        let rest = self.b.binary(BinOp::Sub, u32c(63), s);
        let doubled = self.b.binary(BinOp::Shl, a.1, u32c(1));
        let carry = self.b.binary(BinOp::Shl, doubled, rest);
        self.b.binary(BinOp::Or, low, carry)
    }

    /// Quotient and remainder, truncating toward zero. The divisor is
    /// nonzero, and `MIN / -1` doesn't happen: the producer checks first.
    fn divmod(&mut self, a: Pair, b: Pair, signed: bool) -> (Pair, Pair) {
        if !signed {
            return self.udivmod(a, b);
        }
        let a_negative = self.negative(a);
        let b_negative = self.negative(b);
        let a_abs = self.abs(a, a_negative);
        let b_abs = self.abs(b, b_negative);
        let (quotient, remainder) = self.udivmod(a_abs, b_abs);
        let flip = self.b.binary(BinOp::Xor, a_negative, b_negative);
        (self.abs(quotient, flip), self.abs(remainder, a_negative))
    }

    fn negative(&mut self, a: Pair) -> Local {
        let high = self.b.cast(a.1, Ty::I64);
        self.b.binary(BinOp::Lt, high, Const::int(Ty::I64, 0))
    }

    /// `a` negated where `negate` holds.
    fn abs(&mut self, a: Pair, negate: Local) -> Pair {
        let negated = self.neg(a);
        (self.b.select(negate, negated.0, a.0).into(), self.b.select(negate, negated.1, a.1).into())
    }

    /// Unsigned division: one 64-bit division when both fit, else a
    /// restoring division, one quotient bit per step.
    fn udivmod(&mut self, a: Pair, b: Pair) -> (Pair, Pair) {
        let q = (self.b.local(Ty::U64), self.b.local(Ty::U64));
        let r = (self.b.local(Ty::U64), self.b.local(Ty::U64));
        let n = (self.b.local(Ty::U64), self.b.local(Ty::U64));
        let step = self.b.local(Ty::U32);
        let (fast, slow, head, body, join) = (self.b.block(), self.b.block(), self.b.block(), self.b.block(), self.b.block());
        let a_fits = self.b.binary(BinOp::Eq, a.1, u64c(0));
        let b_fits = self.b.binary(BinOp::Eq, b.1, u64c(0));
        let both = self.b.binary(BinOp::And, a_fits, b_fits);
        self.b.branch(both, fast, slow);

        self.b.switch_to(fast);
        self.b.set(q.0, Rvalue::Binary(BinOp::Div, a.0, b.0));
        self.b.set(q.1, Rvalue::Use(u64c(0)));
        self.b.set(r.0, Rvalue::Binary(BinOp::Rem, a.0, b.0));
        self.b.set(r.1, Rvalue::Use(u64c(0)));
        self.b.goto(join);

        self.b.switch_to(slow);
        for local in [q.0, q.1, r.0, r.1] {
            self.b.set(local, Rvalue::Use(u64c(0)));
        }
        self.b.set(n.0, Rvalue::Use(a.0));
        self.b.set(n.1, Rvalue::Use(a.1));
        self.b.set(step, Rvalue::Use(u32c(0)));
        self.b.goto(head);

        self.b.switch_to(head);
        let more = self.b.binary(BinOp::Lt, step, u32c(128));
        self.b.branch(more, body, join);

        // Shift the next dividend bit into the remainder; subtract the
        // divisor where it fits, which sets the next quotient bit.
        self.b.switch_to(body);
        let shift_in = |this: &mut Self, pair: (Local, Local), bit: Operand| {
            let top = this.b.binary(BinOp::Shr, pair.0, u32c(63));
            let high = this.b.binary(BinOp::Shl, pair.1, u32c(1));
            let high = this.b.binary(BinOp::Or, high, top);
            let low = this.b.binary(BinOp::Shl, pair.0, u32c(1));
            let low = this.b.binary(BinOp::Or, low, bit);
            this.b.set(pair.0, Rvalue::Use(low.into()));
            this.b.set(pair.1, Rvalue::Use(high.into()));
        };
        let next = self.b.binary(BinOp::Shr, n.1, u32c(63));
        shift_in(self, r, next.into());
        shift_in(self, n, u64c(0));
        let below = self.lt((r.0.into(), r.1.into()), b, false);
        let fits = self.b.unary(UnOp::Not, below);
        let less = self.sub((r.0.into(), r.1.into()), b);
        let low = self.b.select(fits, less.0, r.0);
        let high = self.b.select(fits, less.1, r.1);
        self.b.set(r.0, Rvalue::Use(low.into()));
        self.b.set(r.1, Rvalue::Use(high.into()));
        let bit = self.b.cast(fits, Ty::U64);
        shift_in(self, q, bit.into());
        self.b.set(step, Rvalue::Binary(BinOp::Add, step.into(), u32c(1)));
        self.b.goto(head);

        self.b.switch_to(join);
        ((q.0.into(), q.1.into()), (r.0.into(), r.1.into()))
    }

    /// A float to a 128-bit integer, as Rust's `as`: truncated, saturating,
    /// NaN to 0. In `f64`, which holds every `f32` exactly.
    fn float_to_wide(&mut self, value: Operand, signed: bool) -> Pair {
        let x: Operand = if self.b.ty(value) == Ty::F32 { self.b.cast(value, Ty::F64).into() } else { value };
        if !signed {
            return self.unsigned_from_f64(x);
        }
        let negative = self.b.binary(BinOp::Lt, x, Const::f64(0.0));
        let magnitude = self.b.call(Intrinsic::Abs, &[x]);
        let too_big = self.b.binary(BinOp::Ge, magnitude, Const::f64(2f64.powi(127)));
        let unsigned = self.unsigned_from_f64(magnitude.into());
        let value = self.abs(unsigned, negative);
        let limit_low = self.b.select(negative, u64c(0), u64c(u64::MAX));
        let limit_high = self.b.select(negative, u64c(1 << 63), u64c(u64::MAX >> 1));
        (self.b.select(too_big, limit_low, value.0).into(), self.b.select(too_big, limit_high, value.1).into())
    }

    fn unsigned_from_f64(&mut self, x: Operand) -> Pair {
        let two64 = Const::f64(2f64.powi(64));
        let at_least_one = self.b.binary(BinOp::Ge, x, Const::f64(1.0));
        let too_big = self.b.binary(BinOp::Ge, x, Const::f64(2f64.powi(128)));
        let fits = self.b.binary(BinOp::Lt, x, two64);
        let direct = self.b.cast(x, Ty::U64);
        // Above 2^64, `x` is a multiple of 2^11: both halves are exact.
        let scaled = self.b.binary(BinOp::Mul, x, Const::f64(2f64.powi(-64)));
        let high_float = self.b.call(Intrinsic::Trunc, &[scaled.into()]);
        let high = self.b.cast(high_float, Ty::U64);
        let high_part = self.b.binary(BinOp::Mul, high_float, two64);
        let low_float = self.b.binary(BinOp::Sub, x, high_part);
        let low = self.b.cast(low_float, Ty::U64);
        let low = self.b.select(fits, direct, low);
        let high = self.b.select(fits, u64c(0), high);
        let low = self.b.select(at_least_one, low, u64c(0));
        let high = self.b.select(at_least_one, high, u64c(0));
        (self.b.select(too_big, u64c(u64::MAX), low).into(), self.b.select(too_big, u64c(u64::MAX), high).into())
    }

    /// A 128-bit integer to a float, rounded to nearest, ties to even.
    fn wide_to_float(&mut self, a: Pair, signed: bool, to: Ty) -> Local {
        if !signed {
            return self.unsigned_to_float(a, to);
        }
        let negative = self.negative(a);
        let magnitude = self.abs(a, negative);
        let converted = self.unsigned_to_float(magnitude, to);
        let negated = self.b.unary(UnOp::Neg, converted);
        self.b.select(negative, negated, converted)
    }

    /// The top 64 bits, with every lower bit folded into the last as a
    /// "sticky" bit, round exactly as the whole value would; the result is
    /// then scaled by a power of two, which is exact.
    fn unsigned_to_float(&mut self, a: Pair, to: Ty) -> Local {
        let small = self.b.binary(BinOp::Eq, a.1, u64c(0));
        let direct = self.b.cast(a.0, to);
        let zeros = self.b.call(Intrinsic::LeadingZeros, &[a.1]);
        let (low, high) = self.shl(a, zeros.into());
        let sticky = self.b.binary(BinOp::Ne, low, u64c(0));
        let sticky = self.b.cast(sticky, Ty::U64);
        let top = self.b.binary(BinOp::Or, high, sticky);
        let rounded = self.b.cast(top, to);
        // 2^(64 - zeros), built from its exponent bits.
        let power = self.b.binary(BinOp::Sub, u32c(64), zeros);
        let scale = if to == Ty::F64 {
            let exponent = self.b.cast(power, Ty::U64);
            let biased = self.b.binary(BinOp::Add, exponent, u64c(1023));
            let bits = self.b.binary(BinOp::Shl, biased, u32c(52));
            self.b.reinterpret(bits, Ty::F64)
        } else {
            let biased = self.b.binary(BinOp::Add, power, u32c(127));
            let bits = self.b.binary(BinOp::Shl, biased, u32c(23));
            self.b.reinterpret(bits, Ty::F32)
        };
        let scaled = self.b.binary(BinOp::Mul, rounded, scale);
        self.b.select(small, direct, scaled)
    }
}
