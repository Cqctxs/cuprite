//! Option A: PTX written directly. Each local gets its own register, and
//! each operation one or a few instructions with explicit rounding. The
//! driver's PTX compiler allocates registers and optimizes.

use crate::{BinOp, Intrinsic, Kernel, Operand, Rvalue, Special, Statement, Terminator, Ty, UnOp};

/// Register classes. 8- and 16-bit integers share 16-bit registers; 8-bit
/// values are kept normalized there (zero- or sign-extended).
#[derive(Clone, Copy)]
enum Class {
    Pred,
    B16,
    B32,
    B64,
    F32,
    F64,
}

const CLASSES: [(Class, &str, &str); 6] = [
    (Class::Pred, "%p", ".pred"),
    (Class::B16, "%h", ".b16"),
    (Class::B32, "%r", ".b32"),
    (Class::B64, "%rd", ".b64"),
    (Class::F32, "%f", ".f32"),
    (Class::F64, "%fd", ".f64"),
];

fn class(ty: Ty) -> Class {
    match ty {
        Ty::Bool => Class::Pred,
        Ty::I8 | Ty::U8 | Ty::I16 | Ty::U16 => Class::B16,
        Ty::I32 | Ty::U32 => Class::B32,
        Ty::I64 | Ty::U64 | Ty::Ptr => Class::B64,
        Ty::F32 => Class::F32,
        Ty::F64 => Class::F64,
    }
}

/// The type suffix for arithmetic and comparisons.
fn arith(ty: Ty) -> &'static str {
    match ty {
        Ty::I8 | Ty::I16 => "s16",
        Ty::U8 | Ty::U16 => "u16",
        Ty::I32 => "s32",
        Ty::U32 => "u32",
        Ty::I64 => "s64",
        Ty::U64 | Ty::Ptr => "u64",
        Ty::F32 => "f32",
        Ty::F64 => "f64",
        Ty::Bool => "pred",
    }
}

/// The untyped suffix for moves, bit operations and selects.
fn untyped(ty: Ty) -> &'static str {
    match ty {
        Ty::F32 => "f32",
        Ty::F64 => "f64",
        Ty::Bool => "pred",
        _ => match ty.bits() {
            8 | 16 => "b16",
            32 => "b32",
            _ => "b64",
        },
    }
}

/// The type suffix in memory.
fn memory(ty: Ty) -> &'static str {
    match ty {
        Ty::I8 => "s8",
        Ty::U8 | Ty::Bool => "u8",
        Ty::I16 => "s16",
        Ty::U16 => "u16",
        Ty::I32 => "s32",
        Ty::U32 => "u32",
        Ty::I64 => "s64",
        Ty::U64 | Ty::Ptr => "u64",
        Ty::F32 => "f32",
        Ty::F64 => "f64",
    }
}

/// The bits of a float's sign.
fn sign_bit(ty: Ty) -> &'static str {
    if ty == Ty::F32 {
        "0x80000000"
    } else {
        "0x8000000000000000"
    }
}

struct Emitter<'k> {
    kernel: &'k Kernel,
    body: String,
    counts: [u32; 6],
    names: Vec<String>,
}

impl<'k> Emitter<'k> {
    fn new(kernel: &'k Kernel) -> Self {
        let mut emitter = Self {
            kernel,
            body: String::new(),
            counts: [0; 6],
            names: Vec::new(),
        };
        emitter.names = kernel.locals.iter().map(|ty| emitter_register(&mut emitter.counts, class(*ty))).collect();
        emitter
    }

    fn temp(&mut self, class: Class) -> String {
        emitter_register(&mut self.counts, class)
    }

    fn line(&mut self, text: impl AsRef<str>) {
        self.body.push('\t');
        self.body.push_str(text.as_ref());
        self.body.push_str(";\n");
    }

    fn ty(&self, operand: Operand) -> Ty {
        self.kernel.ty(operand)
    }

    /// A register holding the operand. Constants are moved into one first.
    fn operand(&mut self, operand: Operand) -> String {
        let constant = match operand {
            Operand::Local(local) => return self.names[local.0 as usize].clone(),
            Operand::Const(constant) => constant,
        };
        let register = self.temp(class(constant.ty));
        match constant.ty {
            Ty::Bool => {
                let bits = self.temp(Class::B32);
                self.line(format!("mov.u32 {bits}, {}", constant.bits));
                self.line(format!("setp.ne.u32 {register}, {bits}, 0"));
            }
            Ty::F32 => self.line(format!("mov.f32 {register}, 0f{:08X}", constant.bits)),
            Ty::F64 => self.line(format!("mov.f64 {register}, 0d{:016X}", constant.bits)),
            ty if ty.is_signed() => {
                self.line(format!("mov.{} {register}, {}", untyped(ty), constant.signed()))
            }
            ty => self.line(format!("mov.{} {register}, {}", untyped(ty), constant.bits)),
        }
        register
    }

    /// Restores an 8-bit value in a 16-bit register to its width.
    fn narrow(&mut self, ty: Ty, register: &str) {
        match ty {
            Ty::U8 => self.line(format!("and.b16 {register}, {register}, 255")),
            Ty::I8 => self.line(format!("cvt.s16.s8 {register}, {register}")),
            _ => {}
        }
    }

    /// The operand zero-extended to 32 bits, for 8- and 16-bit operations
    /// done in 32-bit registers.
    fn widen_unsigned(&mut self, ty: Ty, value: &str) -> String {
        let wide = self.temp(Class::B32);
        self.line(format!("cvt.u32.u16 {wide}, {value}"));
        if ty.bits() == 8 {
            self.line(format!("and.b32 {wide}, {wide}, 255"));
        }
        wide
    }

    fn statement(&mut self, statement: &Statement) {
        match statement {
            Statement::Assign(local, rvalue) => {
                let destination = self.names[local.0 as usize].clone();
                let ty = self.kernel.locals[local.0 as usize];
                self.rvalue(&destination, ty, rvalue);
            }
            Statement::Store(pointer, value) => {
                let (pointer, ty) = (self.operand(*pointer), self.ty(*value));
                let value = self.operand(*value);
                if ty == Ty::Bool {
                    let byte = self.temp(Class::B16);
                    self.line(format!("selp.u16 {byte}, 1, 0, {value}"));
                    self.line(format!("st.global.u8 [{pointer}], {byte}"));
                } else {
                    self.line(format!("st.global.{} [{pointer}], {value}", memory(ty)));
                }
            }
        }
    }

    fn rvalue(&mut self, d: &str, ty: Ty, rvalue: &Rvalue) {
        match rvalue {
            Rvalue::Use(a) => {
                let a = self.operand(*a);
                self.line(format!("mov.{} {d}, {a}", untyped(ty)));
            }
            Rvalue::Binary(op, a, b) => self.binary(d, ty, *op, *a, *b),
            Rvalue::Unary(op, a) => self.unary(d, ty, *op, *a),
            Rvalue::Cast(a, to) => self.cast(d, *a, *to),
            Rvalue::Call(intrinsic, args) => self.call(d, ty, *intrinsic, args),
            Rvalue::Load(pointer, loaded) => {
                let pointer = self.operand(*pointer);
                if *loaded == Ty::Bool {
                    let byte = self.temp(Class::B16);
                    self.line(format!("ld.global.u8 {byte}, [{pointer}]"));
                    self.line(format!("setp.ne.u16 {d}, {byte}, 0"));
                } else {
                    self.line(format!("ld.global.{} {d}, [{pointer}]", memory(*loaded)));
                }
            }
            Rvalue::Offset(pointer, index, bytes) => {
                let (pointer, index) = (self.operand(*pointer), self.operand(*index));
                let scaled = self.temp(Class::B64);
                self.line(format!("mul.lo.u64 {scaled}, {index}, {bytes}"));
                self.line(format!("add.u64 {d}, {pointer}, {scaled}"));
            }
            Rvalue::Special(special) => {
                let axis = match special {
                    Special::X => "x",
                    Special::Y => "y",
                };
                let (block, size, thread) = (self.temp(Class::B32), self.temp(Class::B32), self.temp(Class::B32));
                self.line(format!("mov.u32 {block}, %ctaid.{axis}"));
                self.line(format!("mov.u32 {size}, %ntid.{axis}"));
                self.line(format!("mov.u32 {thread}, %tid.{axis}"));
                self.line(format!("mad.lo.u32 {d}, {block}, {size}, {thread}"));
            }
            Rvalue::Select(condition, a, b) => {
                let (condition, a, b) = (self.operand(*condition), self.operand(*a), self.operand(*b));
                if ty == Ty::Bool {
                    let (both, not, other) = (self.temp(Class::Pred), self.temp(Class::Pred), self.temp(Class::Pred));
                    self.line(format!("and.pred {both}, {condition}, {a}"));
                    self.line(format!("not.pred {not}, {condition}"));
                    self.line(format!("and.pred {other}, {not}, {b}"));
                    self.line(format!("or.pred {d}, {both}, {other}"));
                } else {
                    self.line(format!("selp.{} {d}, {a}, {b}, {condition}", untyped(ty)));
                }
            }
        }
    }

    fn binary(&mut self, d: &str, ty: Ty, op: BinOp, a: Operand, b: Operand) {
        let ta = self.ty(a);
        let (tb, x) = (self.ty(b), self.operand(a));
        let y = self.operand(b);
        if op.is_shift() {
            // The amount, as `u32`, modulo the width.
            let amount = self.temp(Class::B32);
            match tb.bits() {
                8 | 16 => self.line(format!("cvt.u32.u16 {amount}, {y}")),
                32 => self.line(format!("mov.b32 {amount}, {y}")),
                _ => self.line(format!("cvt.u32.u64 {amount}, {y}")),
            }
            self.line(format!("and.b32 {amount}, {amount}, {}", ta.bits() - 1));
            if op == BinOp::Shl {
                self.line(format!("shl.{} {d}, {x}, {amount}", untyped(ta)));
            } else {
                self.line(format!("shr.{} {d}, {x}, {amount}", arith(ta)));
            }
            self.narrow(ty, d);
            return;
        }
        if op.is_comparison() {
            if ta == Ty::Bool {
                let t = self.temp(Class::Pred);
                match op {
                    BinOp::Eq => {
                        self.line(format!("xor.pred {t}, {x}, {y}"));
                        self.line(format!("not.pred {d}, {t}"));
                    }
                    BinOp::Ne => self.line(format!("xor.pred {d}, {x}, {y}")),
                    // `false < true`.
                    BinOp::Lt | BinOp::Le => {
                        self.line(format!("not.pred {t}, {x}"));
                        let join = if op == BinOp::Lt { "and" } else { "or" };
                        self.line(format!("{join}.pred {d}, {t}, {y}"));
                    }
                    _ => {
                        self.line(format!("not.pred {t}, {y}"));
                        let join = if op == BinOp::Gt { "and" } else { "or" };
                        self.line(format!("{join}.pred {d}, {x}, {t}"));
                    }
                }
                return;
            }
            let compare = match (op, ta.is_float()) {
                (BinOp::Eq, _) => "eq",
                // Unordered: true when either side is NaN.
                (BinOp::Ne, true) => "neu",
                (BinOp::Ne, false) => "ne",
                (BinOp::Lt, _) => "lt",
                (BinOp::Le, _) => "le",
                (BinOp::Gt, _) => "gt",
                _ => "ge",
            };
            self.line(format!("setp.{compare}.{} {d}, {x}, {y}", arith(ta)));
            return;
        }
        if ta == Ty::Bool {
            let name = match op {
                BinOp::And => "and",
                BinOp::Or => "or",
                _ => "xor",
            };
            self.line(format!("{name}.pred {d}, {x}, {y}"));
            return;
        }
        if ta.is_float() {
            let name = match op {
                BinOp::Add => "add",
                BinOp::Sub => "sub",
                BinOp::Mul => "mul",
                _ => "div",
            };
            self.line(format!("{name}.rn.{} {d}, {x}, {y}", arith(ta)));
            return;
        }
        let instruction = match op {
            BinOp::Add => format!("add.{}", arith(ta)),
            BinOp::Sub => format!("sub.{}", arith(ta)),
            BinOp::Mul => format!("mul.lo.{}", arith(ta)),
            BinOp::Div => format!("div.{}", arith(ta)),
            BinOp::Rem => format!("rem.{}", arith(ta)),
            BinOp::And => format!("and.{}", untyped(ta)),
            BinOp::Or => format!("or.{}", untyped(ta)),
            _ => format!("xor.{}", untyped(ta)),
        };
        self.line(format!("{instruction} {d}, {x}, {y}"));
        self.narrow(ty, d);
    }

    fn unary(&mut self, d: &str, ty: Ty, op: UnOp, a: Operand) {
        let x = self.operand(a);
        match (op, ty) {
            (UnOp::Not, Ty::Bool) => self.line(format!("not.pred {d}, {x}")),
            (UnOp::Not, _) => {
                self.line(format!("not.{} {d}, {x}", untyped(ty)));
                self.narrow(ty, d);
            }
            // A sign flip on the bits, NaNs included, as Rust's `-` is.
            (UnOp::Neg, Ty::F32 | Ty::F64) => {
                let (class, bits) = if ty == Ty::F32 { (Class::B32, "b32") } else { (Class::B64, "b64") };
                let t = self.temp(class);
                self.line(format!("mov.{bits} {t}, {x}"));
                self.line(format!("xor.{bits} {t}, {t}, {}", sign_bit(ty)));
                self.line(format!("mov.{bits} {d}, {t}"));
            }
            (UnOp::Neg, _) => {
                let width = match ty.bits() {
                    8 | 16 => "s16",
                    32 => "s32",
                    _ => "s64",
                };
                self.line(format!("neg.{width} {d}, {x}"));
                self.narrow(ty, d);
            }
        }
    }

    fn cast(&mut self, d: &str, a: Operand, to: Ty) {
        let from = self.ty(a);
        let x = self.operand(a);
        if from == to {
            self.line(format!("mov.{} {d}, {x}", untyped(to)));
        } else if from == Ty::Bool {
            self.line(format!("selp.{} {d}, 1, 0, {x}", arith(to)));
        } else if from.is_int() && to.is_int() {
            // 8-bit values live in 16-bit registers: convert between the
            // register widths by the source's signedness, then narrow.
            if class(from) as u8 == class(to) as u8 {
                self.line(format!("mov.{} {d}, {x}", untyped(to)));
            } else {
                self.line(format!("cvt.{}.{} {d}, {x}", arith(to), arith(from)));
            }
            self.narrow(to, d);
        } else if from.is_int() {
            self.line(format!("cvt.rn.{}.{} {d}, {x}", arith(to), arith(from)));
        } else if to.is_int() {
            // Float to integer saturates and maps NaN to 0. 8-bit targets
            // saturate to 16 bits first, then clamp.
            self.line(format!("cvt.rzi.{}.{} {d}, {x}", arith(to), arith(from)));
            // Measured: some `cvt.rzi` forms (f32 to 64 bits, f64 to 16
            // bits) turn NaN into the integer's minimum, so pick 0 by hand.
            let nan = self.temp(Class::Pred);
            self.line(format!("setp.nan.{} {nan}, {x}, {x}", arith(from)));
            self.line(format!("selp.{} {d}, 0, {d}, {nan}", untyped(to)));
            match to {
                Ty::U8 => self.line(format!("min.u16 {d}, {d}, 255")),
                Ty::I8 => {
                    self.line(format!("max.s16 {d}, {d}, -128"));
                    self.line(format!("min.s16 {d}, {d}, 127"));
                }
                _ => {}
            }
        } else if to == Ty::F64 {
            self.line(format!("cvt.f64.f32 {d}, {x}"));
        } else {
            self.line(format!("cvt.rn.f32.f64 {d}, {x}"));
        }
    }

    fn call(&mut self, d: &str, ty: Ty, intrinsic: Intrinsic, args: &[Operand]) {
        let values: Vec<String> = args.iter().map(|a| self.operand(*a)).collect();
        let f = arith(ty);
        let x = &values[0];
        match intrinsic {
            Intrinsic::Sqrt => self.line(format!("sqrt.rn.{f} {d}, {x}")),
            Intrinsic::Floor => self.line(format!("cvt.rmi.{f}.{f} {d}, {x}")),
            Intrinsic::Ceil => self.line(format!("cvt.rpi.{f}.{f} {d}, {x}")),
            Intrinsic::Trunc => self.line(format!("cvt.rzi.{f}.{f} {d}, {x}")),
            Intrinsic::Abs => {
                let (class, bits, mask) = if ty == Ty::F32 {
                    (Class::B32, "b32", "0x7FFFFFFF")
                } else {
                    (Class::B64, "b64", "0x7FFFFFFFFFFFFFFF")
                };
                let t = self.temp(class);
                self.line(format!("mov.{bits} {t}, {x}"));
                self.line(format!("and.{bits} {t}, {t}, {mask}"));
                self.line(format!("mov.{bits} {d}, {t}"));
            }
            Intrinsic::Min => self.line(format!("min.{f} {d}, {x}, {}", values[1])),
            Intrinsic::Max => self.line(format!("max.{f} {d}, {x}, {}", values[1])),
            Intrinsic::MulAdd => self.line(format!("fma.rn.{f} {d}, {x}, {}, {}", values[1], values[2])),
            Intrinsic::RotateLeft | Intrinsic::RotateRight => {
                let left = intrinsic == Intrinsic::RotateLeft;
                let n = &values[1];
                match ty.bits() {
                    32 => {
                        let direction = if left { "l" } else { "r" };
                        self.line(format!("shf.{direction}.wrap.b32 {d}, {x}, {x}, {n}"));
                    }
                    64 => {
                        let (m, k) = (self.temp(Class::B32), self.temp(Class::B32));
                        let (p, q) = (self.temp(Class::B64), self.temp(Class::B64));
                        let (first, second) = if left { ("shl", "shr") } else { ("shr", "shl") };
                        self.line(format!("and.b32 {m}, {n}, 63"));
                        self.line(format!("sub.u32 {k}, 64, {m}"));
                        // A shift by 64 gives 0, so a rotation by 0 is exact.
                        self.line(format!("{first}.b64 {p}, {x}, {m}"));
                        self.line(format!("{second}.b64 {q}, {x}, {k}"));
                        self.line(format!("or.b64 {d}, {p}, {q}"));
                    }
                    bits => {
                        let wide = self.widen_unsigned(ty, x);
                        let (m, k) = (self.temp(Class::B32), self.temp(Class::B32));
                        let (p, q) = (self.temp(Class::B32), self.temp(Class::B32));
                        let (first, second) = if left { ("shl", "shr") } else { ("shr", "shl") };
                        self.line(format!("and.b32 {m}, {n}, {}", bits - 1));
                        self.line(format!("sub.u32 {k}, {bits}, {m}"));
                        self.line(format!("{first}.b32 {p}, {wide}, {m}"));
                        self.line(format!("{second}.b32 {q}, {wide}, {k}"));
                        self.line(format!("or.b32 {p}, {p}, {q}"));
                        self.line(format!("and.b32 {p}, {p}, {}", (1u64 << bits) - 1));
                        self.line(format!("cvt.u16.u32 {d}, {p}"));
                        self.narrow(ty, d);
                    }
                }
            }
            Intrinsic::LeadingZeros => {
                let operand_ty = self.ty(args[0]);
                match operand_ty.bits() {
                    32 => self.line(format!("clz.b32 {d}, {x}")),
                    64 => self.line(format!("clz.b64 {d}, {x}")),
                    bits => {
                        let wide = self.widen_unsigned(operand_ty, x);
                        let count = self.temp(Class::B32);
                        self.line(format!("clz.b32 {count}, {wide}"));
                        self.line(format!("sub.u32 {d}, {count}, {}", 32 - bits));
                    }
                }
            }
            Intrinsic::CountOnes => {
                let operand_ty = self.ty(args[0]);
                match operand_ty.bits() {
                    32 => self.line(format!("popc.b32 {d}, {x}")),
                    64 => self.line(format!("popc.b64 {d}, {x}")),
                    _ => {
                        let wide = self.widen_unsigned(operand_ty, x);
                        self.line(format!("popc.b32 {d}, {wide}"));
                    }
                }
            }
            // No `ctz` instruction: reverse the bits and count leading zeros.
            Intrinsic::TrailingZeros => {
                let operand_ty = self.ty(args[0]);
                match operand_ty.bits() {
                    32 => {
                        let reversed = self.temp(Class::B32);
                        self.line(format!("brev.b32 {reversed}, {x}"));
                        self.line(format!("clz.b32 {d}, {reversed}"));
                    }
                    64 => {
                        let reversed = self.temp(Class::B64);
                        self.line(format!("brev.b64 {reversed}, {x}"));
                        self.line(format!("clz.b64 {d}, {reversed}"));
                    }
                    bits => {
                        // A bit just above the operand stops zero at its width.
                        let wide = self.widen_unsigned(operand_ty, x);
                        self.line(format!("or.b32 {wide}, {wide}, {}", 1u32 << bits));
                        self.line(format!("brev.b32 {wide}, {wide}"));
                        self.line(format!("clz.b32 {d}, {wide}"));
                    }
                }
            }
        }
    }
}

fn emitter_register(counts: &mut [u32; 6], class: Class) -> String {
    let slot = class as usize;
    let name = format!("{}{}", CLASSES[slot].1, counts[slot]);
    counts[slot] += 1;
    name
}

fn label(block: u32) -> String {
    format!("$L__BB{block}")
}

/// The kernel as a PTX module with one entry point.
///
/// # Panics
/// If the kernel doesn't validate.
pub fn emit(kernel: &Kernel) -> String {
    kernel.validate().unwrap_or_else(|invalid| panic!("{invalid}"));
    let mut emitter = Emitter::new(kernel);
    let name = &kernel.name;

    for (i, ty) in kernel.locals[..kernel.params].iter().enumerate() {
        let register = emitter.names[i].clone();
        emitter.line(format!("ld.param.{} {register}, [{name}_param_{i}]", memory(*ty)));
        if *ty == Ty::Ptr {
            emitter.line(format!("cvta.to.global.u64 {register}, {register}"));
        }
    }
    for (i, block) in kernel.blocks.iter().enumerate() {
        emitter.body.push_str(&format!("{}:\n", label(i as u32)));
        for statement in &block.statements {
            emitter.statement(statement);
        }
        match block.terminator {
            Terminator::Goto(target) => emitter.line(format!("bra.uni {}", label(target.0))),
            Terminator::If(condition, then, otherwise) => {
                let condition = emitter.operand(condition);
                emitter.line(format!("@{condition} bra {}", label(then.0)));
                emitter.line(format!("bra.uni {}", label(otherwise.0)));
            }
            Terminator::Return => emitter.line("ret"),
            Terminator::Fail => emitter.line("trap"),
        }
    }

    let params: Vec<String> = kernel.locals[..kernel.params]
        .iter()
        .enumerate()
        .map(|(i, ty)| format!("\t.param .{} {name}_param_{i}", memory(*ty)))
        .collect();
    let registers: String = CLASSES
        .iter()
        .zip(emitter.counts)
        .filter(|(_, count)| *count > 0)
        .map(|((_, prefix, decl), count)| format!("\t.reg {decl} {prefix}<{count}>;\n"))
        .collect();
    format!(
        ".version 7.0\n.target sm_70\n.address_size 64\n\n.visible .entry {name}(\n{}\n)\n{{\n{registers}{}}}\n",
        params.join(",\n"),
        emitter.body
    )
}
