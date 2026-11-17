//! GPU kernels: a small, typed control-flow representation with Rust's
//! semantics, and two ways to compile it to PTX.
//!
//! - [`ptx::emit`] writes PTX directly.
//! - [`llvm::compile`] writes LLVM IR and runs the toolchain's `opt` and
//!   `llc`, LLVM's NVPTX backend.
//!
//! The compiler translates a loop body's MIR into a [`Kernel`]; both
//! backends must then compute exactly what the CPU computes.
//!
//! # Semantics
//!
//! A kernel is a list of blocks over mutable typed locals, like MIR. The
//! parameters are the first locals. Execution starts at block 0.
//!
//! - **Integers** wrap on `+`, `-`, `*` and negation. `<<` and `>>` use the
//!   shift amount modulo the bit width, as MIR's `Shl` and `Shr` do; `>>`
//!   is arithmetic for signed types. `/` and `%` truncate toward zero; the
//!   producer guarantees the divisor is nonzero and that `MIN / -1` can't
//!   happen.
//! - **Floats** are IEEE 754 with round-to-nearest-even, never fused or
//!   reordered. Comparisons are false when either side is NaN, except
//!   `Ne`, which is true.
//! - **Casts** are Rust's `as`: integers truncate or extend by the source's
//!   signedness; integers round to the nearest float; floats saturate to
//!   integers, with NaN becoming 0; `bool` becomes 0 or 1.
//! - **128-bit integers** have the same semantics as the others;
//!   [`wide::lower`] turns them into pairs of 64-bit halves before either
//!   backend sees them.
//! - **Reinterpreting** keeps the bits and changes the type, between
//!   integers and floats of one width, like `f32::to_bits`; NaN payloads
//!   are kept.
//! - **Pointers** are 64-bit addresses of global memory. A buffer argument
//!   is one; [`Rvalue::Offset`] steps through it.

pub mod builder;
pub mod description;
pub mod llvm;
pub mod ptx;
pub mod shapes;
pub mod wide;

use std::fmt;

/// A scalar type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ty {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    /// 128-bit integers, which `wide::lower` turns into pairs of 64-bit
    /// halves before either backend sees them.
    I128,
    U128,
    F32,
    F64,
    /// A global-memory address.
    Ptr,
}

impl Ty {
    pub fn bits(self) -> u32 {
        match self {
            Ty::Bool => 1,
            Ty::I8 | Ty::U8 => 8,
            Ty::I16 | Ty::U16 => 16,
            Ty::I32 | Ty::U32 | Ty::F32 => 32,
            Ty::I64 | Ty::U64 | Ty::F64 | Ty::Ptr => 64,
            Ty::I128 | Ty::U128 => 128,
        }
    }

    /// Size in memory, in bytes.
    pub fn bytes(self) -> u32 {
        self.bits().div_ceil(8)
    }

    pub fn is_float(self) -> bool {
        matches!(self, Ty::F32 | Ty::F64)
    }

    pub fn is_int(self) -> bool {
        matches!(
            self,
            Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 | Ty::I128 | Ty::U8 | Ty::U16 | Ty::U32 | Ty::U64 | Ty::U128
        )
    }

    pub fn is_signed(self) -> bool {
        matches!(self, Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 | Ty::I128)
    }

    /// What a launch can pass as an argument: `cuprite_memory::Arg`.
    pub fn is_passable(self) -> bool {
        matches!(self, Ty::I32 | Ty::U32 | Ty::I64 | Ty::U64 | Ty::F32 | Ty::F64 | Ty::Ptr)
    }

    /// The name used in kernel descriptions and reports, as in Rust.
    pub fn name(self) -> &'static str {
        match self {
            Ty::Bool => "bool",
            Ty::I8 => "i8",
            Ty::I16 => "i16",
            Ty::I32 => "i32",
            Ty::I64 => "i64",
            Ty::U8 => "u8",
            Ty::U16 => "u16",
            Ty::U32 => "u32",
            Ty::U64 => "u64",
            Ty::I128 => "i128",
            Ty::U128 => "u128",
            Ty::F32 => "f32",
            Ty::F64 => "f64",
            Ty::Ptr => "ptr",
        }
    }

    pub fn from_name(name: &str) -> Option<Ty> {
        [
            Ty::Bool, Ty::I8, Ty::I16, Ty::I32, Ty::I64, Ty::U8, Ty::U16, Ty::U32, Ty::U64, Ty::I128, Ty::U128, Ty::F32,
            Ty::F64, Ty::Ptr,
        ]
        .into_iter()
        .find(|ty| ty.name() == name)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Local(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

/// A constant: its type and its bits, zero-extended to 128. A signed
/// integer holds its two's-complement bits; a float its IEEE bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Const {
    pub ty: Ty,
    pub bits: u128,
}

impl Const {
    pub fn bool(value: bool) -> Self {
        Self {
            ty: Ty::Bool,
            bits: value.into(),
        }
    }

    /// An integer of type `ty`, truncated to its width.
    pub fn int(ty: Ty, value: i128) -> Self {
        assert!(ty.is_int(), "{ty:?} is not an integer type");
        let mask = if ty.bits() == 128 { u128::MAX } else { (1u128 << ty.bits()) - 1 };
        Self {
            ty,
            bits: (value as u128) & mask,
        }
    }

    pub fn f32(value: f32) -> Self {
        Self {
            ty: Ty::F32,
            bits: value.to_bits().into(),
        }
    }

    pub fn f64(value: f64) -> Self {
        Self {
            ty: Ty::F64,
            bits: value.to_bits().into(),
        }
    }

    /// The value as a signed integer, for signed types.
    pub fn signed(self) -> i128 {
        let shift = 128 - self.ty.bits();
        ((self.bits << shift) as i128) >> shift
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand {
    Local(Local),
    Const(Const),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl BinOp {
    pub fn is_comparison(self) -> bool {
        matches!(self, BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge)
    }

    pub fn is_shift(self) -> bool {
        matches!(self, BinOp::Shl | BinOp::Shr)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp {
    /// Arithmetic negation: wrapping for integers, a sign flip for floats.
    Neg,
    /// Bitwise not; logical not for `bool`.
    Not,
}

/// Library operations with exact definitions on both backends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Intrinsic {
    /// Correctly rounded square root.
    Sqrt,
    Floor,
    Ceil,
    /// Toward zero.
    Trunc,
    /// To the nearest integer, ties to even, like `f32::round_ties_even`.
    RoundTiesEven,
    /// Clears the sign bit, like `f32::abs`.
    Abs,
    /// `f32::min`: the other operand when one is NaN.
    Min,
    /// `f32::max`: the other operand when one is NaN.
    Max,
    /// `f32::mul_add`: `a * b + c` rounded once.
    MulAdd,
    RotateLeft,
    RotateRight,
    /// `leading_zeros`, as `U32` whatever the operand's width.
    LeadingZeros,
    /// `trailing_zeros`, as `U32`: the width when the operand is zero.
    TrailingZeros,
    /// `count_ones`, as `U32`.
    CountOnes,
    /// The high half of the double-width product of two unsigned integers
    /// of one type (`u32` or `u64`), such as `((a as u128 * b as u128) >> 64) as u64`.
    MulHigh,
}

impl Intrinsic {
    /// Counts bits of an integer, giving `U32`.
    pub fn counts_bits(self) -> bool {
        matches!(self, Self::LeadingZeros | Self::TrailingZeros | Self::CountOnes)
    }
}

/// The executing thread's position in a launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Special {
    /// `%ctaid.x * %ntid.x + %tid.x`, as `U32`.
    X,
    /// The same for `y`.
    Y,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Rvalue {
    Use(Operand),
    Binary(BinOp, Operand, Operand),
    Unary(UnOp, Operand),
    /// Rust's `as` to the given type.
    Cast(Operand, Ty),
    /// The same bits as the given type, an integer or float of the
    /// operand's width: `f32::to_bits`, `f64::from_bits`.
    Reinterpret(Operand, Ty),
    Call(Intrinsic, Vec<Operand>),
    /// The value of the given type at an address.
    Load(Operand, Ty),
    /// `pointer + index * element_bytes`, with a `U64` index.
    Offset(Operand, Operand, u32),
    Special(Special),
    /// `if condition { a } else { b }`.
    Select(Operand, Operand, Operand),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Statement {
    Assign(Local, Rvalue),
    /// Writes a value to an address.
    Store(Operand, Operand),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Terminator {
    Goto(BlockId),
    If(Operand, BlockId, BlockId),
    Return,
    /// A check the program would make failed here (an index out of
    /// bounds, a zero divisor, an assertion): on the CPU this panics.
    /// Inlining sends it to a block that reports the failure; in a kernel
    /// entry point it stops the whole launch (`trap`).
    Fail,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Block {
    pub statements: Vec<Statement>,
    pub terminator: Terminator,
}

/// A kernel entry point. Locals `0..params` are its parameters, in order.
#[derive(Clone, Debug, PartialEq)]
pub struct Kernel {
    pub name: String,
    pub params: usize,
    pub locals: Vec<Ty>,
    pub blocks: Vec<Block>,
}

/// Why a kernel is malformed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Invalid(pub String);

impl fmt::Display for Invalid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Invalid {}

impl Kernel {
    pub fn ty(&self, operand: Operand) -> Ty {
        match operand {
            Operand::Local(local) => self.locals[local.0 as usize],
            Operand::Const(constant) => constant.ty,
        }
    }

    /// The type an rvalue produces, if its operands' types fit together.
    pub fn rvalue_ty(&self, rvalue: &Rvalue) -> Result<Ty, Invalid> {
        let ty = |o: &Operand| self.ty(*o);
        let fail = |what: String| Err(Invalid(what));
        match rvalue {
            Rvalue::Use(o) => Ok(ty(o)),
            Rvalue::Binary(op, a, b) => {
                let (ta, tb) = (ty(a), ty(b));
                if op.is_shift() {
                    if !ta.is_int() || !tb.is_int() {
                        return fail(format!("{op:?} of {ta:?} by {tb:?}"));
                    }
                    return Ok(ta);
                }
                if ta != tb {
                    return fail(format!("{op:?} of {ta:?} and {tb:?}"));
                }
                let fits = match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => ta.is_int() || ta.is_float(),
                    // Float `%` is `fmod`, which needs an exact algorithm of its own.
                    BinOp::Rem => ta.is_int(),
                    BinOp::And | BinOp::Or | BinOp::Xor => ta.is_int() || ta == Ty::Bool,
                    _ => ta != Ty::Ptr,
                };
                if !fits {
                    return fail(format!("{op:?} of {ta:?}"));
                }
                Ok(if op.is_comparison() { Ty::Bool } else { ta })
            }
            Rvalue::Unary(op, a) => {
                let ta = ty(a);
                let fits = match op {
                    UnOp::Neg => ta.is_int() || ta.is_float(),
                    UnOp::Not => ta.is_int() || ta == Ty::Bool,
                };
                if fits { Ok(ta) } else { fail(format!("{op:?} of {ta:?}")) }
            }
            Rvalue::Cast(a, to) => {
                let from = ty(a);
                let numeric = |t: Ty| t.is_int() || t.is_float();
                // `bool` casts only to integers, as in Rust.
                if numeric(from) && numeric(*to) || from == Ty::Bool && to.is_int() || from == *to {
                    Ok(*to)
                } else {
                    fail(format!("cast from {from:?} to {to:?}"))
                }
            }
            Rvalue::Reinterpret(a, to) => {
                let from = ty(a);
                let plain = |t: Ty| t.is_int() || t.is_float();
                if plain(from) && plain(*to) && from.bits() == to.bits() {
                    Ok(*to)
                } else {
                    fail(format!("reinterpreting {from:?} as {to:?}"))
                }
            }
            Rvalue::Call(intrinsic, args) => {
                let types: Vec<Ty> = args.iter().map(ty).collect();
                let arity = match intrinsic {
                    Intrinsic::Min | Intrinsic::Max | Intrinsic::RotateLeft | Intrinsic::RotateRight | Intrinsic::MulHigh => 2,
                    Intrinsic::MulAdd => 3,
                    _ => 1,
                };
                if types.len() != arity {
                    return fail(format!("{intrinsic:?} takes {arity} operands"));
                }
                let first = types[0];
                let fits = match intrinsic {
                    Intrinsic::RotateLeft | Intrinsic::RotateRight => {
                        first.is_int() && types[1] == Ty::U32
                    }
                    Intrinsic::MulHigh => matches!(first, Ty::U32 | Ty::U64) && types[1] == first,
                    _ if intrinsic.counts_bits() => first.is_int(),
                    _ => first.is_float() && types.iter().all(|t| *t == first),
                };
                if !fits {
                    return fail(format!("{intrinsic:?} of {types:?}"));
                }
                Ok(if intrinsic.counts_bits() { Ty::U32 } else { first })
            }
            Rvalue::Load(pointer, loaded) => {
                if ty(pointer) != Ty::Ptr {
                    return fail("load from a non-pointer".into());
                }
                Ok(*loaded)
            }
            Rvalue::Offset(pointer, index, _) => {
                if ty(pointer) != Ty::Ptr || ty(index) != Ty::U64 {
                    return fail("offset needs a pointer and a U64 index".into());
                }
                Ok(Ty::Ptr)
            }
            Rvalue::Special(_) => Ok(Ty::U32),
            Rvalue::Select(condition, a, b) => {
                if ty(condition) != Ty::Bool || ty(a) != ty(b) {
                    return fail("select needs a bool and two operands of one type".into());
                }
                Ok(ty(a))
            }
        }
    }

    /// Checks every local, block reference and type, and that a launch can
    /// pass every parameter.
    pub fn validate(&self) -> Result<(), Invalid> {
        self.validate_body()?;
        if let Some(ty) = self.locals[..self.params].iter().find(|ty| !ty.is_passable()) {
            return Err(Invalid(format!("a {ty:?} parameter can't be passed to a kernel")));
        }
        Ok(())
    }

    /// Checks every local, block reference and type; parameters may have
    /// any type, as an element function's do.
    pub fn validate_body(&self) -> Result<(), Invalid> {
        let local_ok = |l: Local| (l.0 as usize) < self.locals.len();
        let operand_ok = |o: &Operand| match o {
            Operand::Local(l) => local_ok(*l),
            Operand::Const(_) => true,
        };
        let block_ok = |b: BlockId| (b.0 as usize) < self.blocks.len();
        if self.params > self.locals.len() || self.blocks.is_empty() {
            return Err(Invalid("parameters exceed locals, or no blocks".into()));
        }
        for (i, block) in self.blocks.iter().enumerate() {
            for statement in &block.statements {
                match statement {
                    Statement::Assign(local, rvalue) => {
                        let operands = rvalue_operands(rvalue);
                        if !local_ok(*local) || !operands.iter().all(operand_ok) {
                            return Err(Invalid(format!("block {i}: unknown local")));
                        }
                        let produced = self.rvalue_ty(rvalue)?;
                        let declared = self.locals[local.0 as usize];
                        if produced != declared {
                            return Err(Invalid(format!(
                                "block {i}: {rvalue:?} is {produced:?}, assigned to a {declared:?}"
                            )));
                        }
                    }
                    Statement::Store(pointer, value) => {
                        if !operand_ok(pointer) || !operand_ok(value) {
                            return Err(Invalid(format!("block {i}: unknown local")));
                        }
                        if self.ty(*pointer) != Ty::Ptr {
                            return Err(Invalid(format!("block {i}: store to a non-pointer")));
                        }
                    }
                }
            }
            let targets_ok = match block.terminator {
                Terminator::Goto(b) => block_ok(b),
                Terminator::If(condition, t, f) => {
                    operand_ok(&condition)
                        && self.ty(condition) == Ty::Bool
                        && block_ok(t)
                        && block_ok(f)
                }
                Terminator::Return | Terminator::Fail => true,
            };
            if !targets_ok {
                return Err(Invalid(format!("block {i}: bad terminator")));
            }
        }
        Ok(())
    }
}

/// The operands an rvalue reads.
pub fn rvalue_operands(rvalue: &Rvalue) -> Vec<Operand> {
    match rvalue {
        Rvalue::Use(o) | Rvalue::Unary(_, o) | Rvalue::Cast(o, _) | Rvalue::Reinterpret(o, _) | Rvalue::Load(o, _) => {
            vec![*o]
        }
        Rvalue::Binary(_, a, b) | Rvalue::Offset(a, b, _) => vec![*a, *b],
        Rvalue::Call(_, args) => args.clone(),
        Rvalue::Special(_) => Vec::new(),
        Rvalue::Select(c, a, b) => vec![*c, *a, *b],
    }
}

impl Rvalue {
    /// The same operation on operands replaced by `f`.
    pub fn map_operands(&self, mut f: impl FnMut(Operand) -> Operand) -> Rvalue {
        match self {
            Rvalue::Use(o) => Rvalue::Use(f(*o)),
            Rvalue::Binary(op, a, b) => Rvalue::Binary(*op, f(*a), f(*b)),
            Rvalue::Unary(op, a) => Rvalue::Unary(*op, f(*a)),
            Rvalue::Cast(a, to) => Rvalue::Cast(f(*a), *to),
            Rvalue::Reinterpret(a, to) => Rvalue::Reinterpret(f(*a), *to),
            Rvalue::Call(intrinsic, args) => Rvalue::Call(*intrinsic, args.iter().map(|a| f(*a)).collect()),
            Rvalue::Load(pointer, ty) => Rvalue::Load(f(*pointer), *ty),
            Rvalue::Offset(pointer, index, bytes) => Rvalue::Offset(f(*pointer), f(*index), *bytes),
            Rvalue::Special(special) => Rvalue::Special(*special),
            Rvalue::Select(c, a, b) => Rvalue::Select(f(*c), f(*a), f(*b)),
        }
    }
}

/// A piece of code to inline into kernels, such as what one loop body does
/// to its element. Its kernel's parameters are the function's inputs, of
/// any type; `Return` ends it, with the result, if any, in `result`.
#[derive(Clone, Debug, PartialEq)]
pub struct Function {
    pub kernel: Kernel,
    pub result: Option<Local>,
}

impl Function {
    pub fn params(&self) -> &[Ty] {
        &self.kernel.locals[..self.kernel.params]
    }

    pub fn result_ty(&self) -> Option<Ty> {
        self.result.map(|local| self.kernel.locals[local.0 as usize])
    }

    pub fn validate(&self) -> Result<(), Invalid> {
        self.kernel.validate_body()?;
        match self.result {
            Some(local) if local.0 as usize >= self.kernel.locals.len() => {
                Err(Invalid("the result is not a local".into()))
            }
            _ => Ok(()),
        }
    }

    /// Statements, not counting copies: a rough size for reports.
    pub fn operations(&self) -> usize {
        self.kernel
            .blocks
            .iter()
            .flat_map(|block| &block.statements)
            .filter(|statement| !matches!(statement, Statement::Assign(_, Rvalue::Use(_))))
            .count()
    }
}
