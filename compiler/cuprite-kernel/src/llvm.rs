//! Option B: LLVM IR, compiled by the toolchain's `opt` and `llc` (LLVM's
//! NVPTX backend). Each local gets a stack slot; `opt` promotes them to
//! registers and optimizes. No fast-math flags are ever set, so LLVM keeps
//! IEEE results and never fuses or reorders float arithmetic.

use std::{
    collections::BTreeSet,
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::{BinOp, Intrinsic, Kernel, Operand, Rvalue, Special, Statement, Terminator, Ty, UnOp};

fn lt(ty: Ty) -> &'static str {
    match ty {
        Ty::I128 | Ty::U128 => "i128",
        Ty::Bool => "i1",
        Ty::I8 | Ty::U8 => "i8",
        Ty::I16 | Ty::U16 => "i16",
        Ty::I32 | Ty::U32 => "i32",
        Ty::I64 | Ty::U64 => "i64",
        Ty::F32 => "float",
        Ty::F64 => "double",
        Ty::Ptr => "ptr addrspace(1)",
    }
}

/// Intrinsic name suffixes: `f32`, `i32`, ...
fn suffix(ty: Ty) -> &'static str {
    match ty {
        Ty::F32 => "f32",
        Ty::F64 => "f64",
        _ => match ty.bits() {
            1 => "i1",
            8 => "i8",
            16 => "i16",
            32 => "i32",
            _ => "i64",
        },
    }
}

struct Emitter<'k> {
    kernel: &'k Kernel,
    body: String,
    temps: u32,
    declarations: BTreeSet<String>,
}

impl Emitter<'_> {
    fn temp(&mut self) -> String {
        self.temps += 1;
        format!("%t{}", self.temps)
    }

    fn line(&mut self, text: impl AsRef<str>) {
        self.body.push_str("  ");
        self.body.push_str(text.as_ref());
        self.body.push('\n');
    }

    /// Computes `text` into a new temporary.
    fn value(&mut self, text: impl AsRef<str>) -> String {
        let t = self.temp();
        self.line(format!("{t} = {}", text.as_ref()));
        t
    }

    fn declare(&mut self, declaration: String) {
        self.declarations.insert(declaration);
    }

    fn ty(&self, operand: Operand) -> Ty {
        self.kernel.ty(operand)
    }

    fn operand(&mut self, operand: Operand) -> String {
        match operand {
            Operand::Local(local) => {
                let ty = self.kernel.locals[local.0 as usize];
                self.value(format!("load {}, ptr %l{}", lt(ty), local.0))
            }
            Operand::Const(constant) => match constant.ty {
                Ty::Bool => if constant.bits != 0 { "true" } else { "false" }.into(),
                // LLVM writes `float` constants as the bits of the equal double.
                Ty::F32 => format!("0x{:016X}", f64::from(f32::from_bits(constant.bits as u32)).to_bits()),
                Ty::F64 => format!("0x{:016X}", constant.bits),
                Ty::Ptr => format!("inttoptr (i64 {} to ptr addrspace(1))", constant.bits),
                ty if ty.is_signed() => constant.signed().to_string(),
                _ => constant.bits.to_string(),
            },
        }
    }

    /// `value` of type `from` as an integer of `to`'s width, by `from`'s
    /// signedness.
    fn resize(&mut self, value: String, from: Ty, to_bits: u32) -> String {
        let to = match to_bits {
            8 => "i8",
            16 => "i16",
            32 => "i32",
            _ => "i64",
        };
        match from.bits().cmp(&to_bits) {
            std::cmp::Ordering::Equal => value,
            std::cmp::Ordering::Greater => self.value(format!("trunc {} {value} to {to}", lt(from))),
            std::cmp::Ordering::Less => {
                let extend = if from.is_signed() { "sext" } else { "zext" };
                self.value(format!("{extend} {} {value} to {to}", lt(from)))
            }
        }
    }

    fn store_local(&mut self, local: u32, ty: Ty, value: &str) {
        self.line(format!("store {} {value}, ptr %l{local}", lt(ty)));
    }

    fn statement(&mut self, statement: &Statement) {
        match statement {
            Statement::Assign(local, rvalue) => {
                let ty = self.kernel.locals[local.0 as usize];
                let value = self.rvalue(ty, rvalue);
                self.store_local(local.0, ty, &value);
            }
            Statement::Store(pointer, value) => {
                let ty = self.ty(*value);
                let (pointer, value) = (self.operand(*pointer), self.operand(*value));
                if ty == Ty::Bool {
                    let byte = self.value(format!("zext i1 {value} to i8"));
                    self.line(format!("store i8 {byte}, ptr addrspace(1) {pointer}"));
                } else {
                    self.line(format!("store {} {value}, ptr addrspace(1) {pointer}", lt(ty)));
                }
            }
        }
    }

    fn rvalue(&mut self, ty: Ty, rvalue: &Rvalue) -> String {
        match rvalue {
            Rvalue::Use(a) => self.operand(*a),
            Rvalue::Binary(op, a, b) => self.binary(*op, *a, *b),
            Rvalue::Unary(op, a) => {
                let x = self.operand(*a);
                match (op, ty.is_float()) {
                    (UnOp::Neg, true) => self.sign_bit(ty, "xor", &x),
                    (UnOp::Neg, false) => self.value(format!("sub {} 0, {x}", lt(ty))),
                    (UnOp::Not, _) => {
                        let ones = if ty == Ty::Bool { "true" } else { "-1" };
                        self.value(format!("xor {} {x}, {ones}", lt(ty)))
                    }
                }
            }
            Rvalue::Cast(a, to) => self.cast(*a, *to),
            Rvalue::Reinterpret(a, to) => {
                let from = self.ty(*a);
                let x = self.operand(*a);
                if lt(from) == lt(*to) {
                    x
                } else {
                    self.value(format!("bitcast {} {x} to {}", lt(from), lt(*to)))
                }
            }
            Rvalue::Call(intrinsic, args) => self.call(ty, *intrinsic, args),
            Rvalue::Load(pointer, loaded) => {
                let pointer = self.operand(*pointer);
                if *loaded == Ty::Bool {
                    let byte = self.value(format!("load i8, ptr addrspace(1) {pointer}"));
                    self.value(format!("icmp ne i8 {byte}, 0"))
                } else {
                    self.value(format!("load {}, ptr addrspace(1) {pointer}", lt(*loaded)))
                }
            }
            Rvalue::Offset(pointer, index, bytes) => {
                let (pointer, index) = (self.operand(*pointer), self.operand(*index));
                let scaled = self.value(format!("mul i64 {index}, {bytes}"));
                self.value(format!("getelementptr i8, ptr addrspace(1) {pointer}, i64 {scaled}"))
            }
            Rvalue::Special(special) => {
                let axis = match special {
                    Special::X => "x",
                    Special::Y => "y",
                };
                let mut read = |register: &str| {
                    let name = format!("llvm.nvvm.read.ptx.sreg.{register}.{axis}");
                    self.declare(format!("declare i32 @{name}()"));
                    self.value(format!("call i32 @{name}()"))
                };
                let (block, size, thread) = (read("ctaid"), read("ntid"), read("tid"));
                let base = self.value(format!("mul i32 {block}, {size}"));
                self.value(format!("add i32 {base}, {thread}"))
            }
            Rvalue::Select(condition, a, b) => {
                let (condition, a, b) = (self.operand(*condition), self.operand(*a), self.operand(*b));
                self.value(format!("select i1 {condition}, {} {a}, {} {b}", lt(ty), lt(ty)))
            }
        }
    }

    fn binary(&mut self, op: BinOp, a: Operand, b: Operand) -> String {
        let (ta, tb) = (self.ty(a), self.ty(b));
        let (x, y) = (self.operand(a), self.operand(b));
        let t = lt(ta);
        if op.is_shift() {
            let amount = self.resize(y, tb, ta.bits());
            let masked = self.value(format!("and {t} {amount}, {}", ta.bits() - 1));
            let name = match (op, ta.is_signed()) {
                (BinOp::Shl, _) => "shl",
                (_, true) => "ashr",
                _ => "lshr",
            };
            return self.value(format!("{name} {t} {x}, {masked}"));
        }
        if op.is_comparison() {
            let predicate = if ta.is_float() {
                match op {
                    BinOp::Eq => "oeq",
                    BinOp::Ne => "une",
                    BinOp::Lt => "olt",
                    BinOp::Le => "ole",
                    BinOp::Gt => "ogt",
                    _ => "oge",
                }
            } else {
                let signed = ta.is_signed();
                match op {
                    BinOp::Eq => "eq",
                    BinOp::Ne => "ne",
                    BinOp::Lt if signed => "slt",
                    BinOp::Lt => "ult",
                    BinOp::Le if signed => "sle",
                    BinOp::Le => "ule",
                    BinOp::Gt if signed => "sgt",
                    BinOp::Gt => "ugt",
                    _ if signed => "sge",
                    _ => "uge",
                }
            };
            let instruction = if ta.is_float() { "fcmp" } else { "icmp" };
            return self.value(format!("{instruction} {predicate} {t} {x}, {y}"));
        }
        let name = match (op, ta.is_float(), ta.is_signed()) {
            (BinOp::Add, true, _) => "fadd",
            (BinOp::Sub, true, _) => "fsub",
            (BinOp::Mul, true, _) => "fmul",
            (BinOp::Div, true, _) => "fdiv",
            (BinOp::Add, ..) => "add",
            (BinOp::Sub, ..) => "sub",
            (BinOp::Mul, ..) => "mul",
            (BinOp::Div, _, true) => "sdiv",
            (BinOp::Div, ..) => "udiv",
            (BinOp::Rem, _, true) => "srem",
            (BinOp::Rem, ..) => "urem",
            (BinOp::And, ..) => "and",
            (BinOp::Or, ..) => "or",
            _ => "xor",
        };
        self.value(format!("{name} {t} {x}, {y}"))
    }

    fn cast(&mut self, a: Operand, to: Ty) -> String {
        let from = self.ty(a);
        let x = self.operand(a);
        if from == to || from.is_int() && to.is_int() && from.bits() == to.bits() {
            return x;
        }
        if from == Ty::Bool {
            return self.value(format!("zext i1 {x} to {}", lt(to)));
        }
        if from.is_int() && to.is_int() {
            return self.resize(x, from, to.bits());
        }
        if from.is_int() {
            let name = if from.is_signed() { "sitofp" } else { "uitofp" };
            return self.value(format!("{name} {} {x} to {}", lt(from), lt(to)));
        }
        if to.is_int() {
            let kind = if to.is_signed() { "fptosi" } else { "fptoui" };
            let name = format!("llvm.{kind}.sat.{}.{}", suffix(to), suffix(from));
            self.declare(format!("declare {} @{name}({})", lt(to), lt(from)));
            return self.value(format!("call {} @{name}({} {x})", lt(to), lt(from)));
        }
        let name = if to == Ty::F64 { "fpext" } else { "fptrunc" };
        self.value(format!("{name} {} {x} to {}", lt(from), lt(to)))
    }

    /// Negation (`xor`) or absolute value (`and`) on the sign bit alone, as
    /// Rust defines them. Measured: the NVPTX backend lowers `fneg` and
    /// `fabs` to `neg` and `abs`, which rewrite NaNs.
    fn sign_bit(&mut self, ty: Ty, op: &str, x: &str) -> String {
        let (int, mask) = match (ty, op) {
            (Ty::F32, "xor") => ("i32", "-2147483648".to_string()),
            (Ty::F32, _) => ("i32", "2147483647".to_string()),
            (_, "xor") => ("i64", "-9223372036854775808".to_string()),
            (_, _) => ("i64", "9223372036854775807".to_string()),
        };
        let bits = self.value(format!("bitcast {} {x} to {int}", lt(ty)));
        let changed = self.value(format!("{op} {int} {bits}, {mask}"));
        self.value(format!("bitcast {int} {changed} to {}", lt(ty)))
    }

    fn call(&mut self, ty: Ty, intrinsic: Intrinsic, args: &[Operand]) -> String {
        let operand_ty = self.ty(args[0]);
        let values: Vec<String> = args.iter().map(|a| self.operand(*a)).collect();
        let t = lt(operand_ty);
        let mut call = |name: &str, args: &[String]| {
            let full = format!("llvm.{name}.{}", suffix(operand_ty));
            let parameters = vec![t; args.len()].join(", ");
            self.declare(format!("declare {t} @{full}({parameters})"));
            let arguments: Vec<String> = args.iter().map(|a| format!("{t} {a}")).collect();
            self.value(format!("call {t} @{full}({})", arguments.join(", ")))
        };
        match intrinsic {
            Intrinsic::Sqrt => call("sqrt", &values),
            Intrinsic::Floor => call("floor", &values),
            Intrinsic::Ceil => call("ceil", &values),
            Intrinsic::Trunc => call("trunc", &values),
            Intrinsic::RoundTiesEven => call("roundeven", &values),
            Intrinsic::MulHigh => {
                let wide = format!("i{}", 2 * operand_ty.bits());
                let (a, b) = (&values[0], &values[1]);
                let wa = self.value(format!("zext {t} {a} to {wide}"));
                let wb = self.value(format!("zext {t} {b} to {wide}"));
                let product = self.value(format!("mul {wide} {wa}, {wb}"));
                let high = self.value(format!("lshr {wide} {product}, {}", operand_ty.bits()));
                self.value(format!("trunc {wide} {high} to {t}"))
            }
            Intrinsic::Abs => self.sign_bit(operand_ty, "and", &values[0]),
            Intrinsic::Min => call("minnum", &values),
            Intrinsic::Max => call("maxnum", &values),
            Intrinsic::MulAdd => call("fma", &values),
            Intrinsic::RotateLeft | Intrinsic::RotateRight => {
                let name = if intrinsic == Intrinsic::RotateLeft { "fshl" } else { "fshr" };
                // Funnel shifts take the amount modulo the width, as Rust does.
                let amount = self.resize(values[1].clone(), Ty::U32, operand_ty.bits());
                let x = values[0].clone();
                let full = format!("llvm.{name}.{}", suffix(operand_ty));
                self.declare(format!("declare {t} @{full}({t}, {t}, {t})"));
                self.value(format!("call {t} @{full}({t} {x}, {t} {x}, {t} {amount})"))
            }
            Intrinsic::LeadingZeros | Intrinsic::TrailingZeros | Intrinsic::CountOnes => {
                let x = &values[0];
                let count = if intrinsic == Intrinsic::CountOnes {
                    let full = format!("llvm.ctpop.{}", suffix(operand_ty));
                    self.declare(format!("declare {t} @{full}({t})"));
                    self.value(format!("call {t} @{full}({t} {x})"))
                } else {
                    // `false`: zero is defined, giving the width.
                    let name = if intrinsic == Intrinsic::LeadingZeros { "ctlz" } else { "cttz" };
                    let full = format!("llvm.{name}.{}", suffix(operand_ty));
                    self.declare(format!("declare {t} @{full}({t}, i1)"));
                    self.value(format!("call {t} @{full}({t} {x}, i1 false)"))
                };
                let unsigned = match operand_ty.bits() {
                    8 => Ty::U8,
                    16 => Ty::U16,
                    32 => Ty::U32,
                    _ => Ty::U64,
                };
                self.resize(count, unsigned, ty.bits())
            }
        }
    }
}

/// The kernel as an LLVM IR module for `nvptx64-nvidia-cuda`.
///
/// # Panics
/// If the kernel doesn't validate.
pub fn emit(kernel: &Kernel) -> String {
    kernel.validate().unwrap_or_else(|invalid| panic!("{invalid}"));
    let lowered = crate::wide::lower(kernel);
    let kernel = &lowered;
    let mut emitter = Emitter {
        kernel,
        body: String::new(),
        temps: 0,
        declarations: BTreeSet::new(),
    };
    for (i, block) in kernel.blocks.iter().enumerate() {
        emitter.body.push_str(&format!("b{i}:\n"));
        for statement in &block.statements {
            emitter.statement(statement);
        }
        match block.terminator {
            Terminator::Goto(target) => emitter.line(format!("br label %b{}", target.0)),
            Terminator::If(condition, then, otherwise) => {
                let condition = emitter.operand(condition);
                emitter.line(format!("br i1 {condition}, label %b{}, label %b{}", then.0, otherwise.0));
            }
            Terminator::Return => emitter.line("ret void"),
            Terminator::Fail => {
                emitter.declare("declare void @llvm.trap()".into());
                emitter.line("call void @llvm.trap()");
                emitter.line("unreachable");
            }
        }
    }

    let params: Vec<String> = kernel.locals[..kernel.params]
        .iter()
        .enumerate()
        .map(|(i, ty)| format!("{} %a{i}", lt(*ty)))
        .collect();
    let mut entry = String::new();
    for (i, ty) in kernel.locals.iter().enumerate() {
        entry.push_str(&format!("  %l{i} = alloca {}\n", lt(*ty)));
    }
    for (i, ty) in kernel.locals[..kernel.params].iter().enumerate() {
        entry.push_str(&format!("  store {} %a{i}, ptr %l{i}\n", lt(*ty)));
    }
    let declarations: String = emitter.declarations.iter().map(|d| format!("{d}\n")).collect();
    format!(
        "target datalayout = \"e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64\"\n\
         target triple = \"nvptx64-nvidia-cuda\"\n\n{declarations}\n\
         define ptx_kernel void @{}({}) {{\nentry:\n{entry}  br label %b0\n{}}}\n",
        kernel.name,
        params.join(", "),
        emitter.body
    )
}

/// The toolchain's `opt` and `llc`, from the `llvm-tools` component.
#[derive(Clone, Debug)]
pub struct Llvm {
    pub opt: PathBuf,
    pub llc: PathBuf,
}

/// Distinguishes the scratch files of concurrent compilations.
static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

impl Llvm {
    /// The tools in the sysroot of the `rustc` on `PATH` (or `$RUSTC`).
    pub fn find() -> Result<Self, String> {
        let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
        let output = |args: &[&str]| -> Result<String, String> {
            let output = Command::new(&rustc)
                .args(args)
                .output()
                .map_err(|e| format!("cannot run rustc: {e}"))?;
            Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
        };
        let sysroot = output(&["--print", "sysroot"])?;
        let version = output(&["-vV"])?;
        let host = version
            .lines()
            .find_map(|line| line.strip_prefix("host: "))
            .ok_or("rustc -vV printed no host")?
            .to_owned();
        Self::in_sysroot(Path::new(&sysroot), &host)
    }

    /// The tools of the toolchain at `sysroot`, built for `host`.
    pub fn in_sysroot(sysroot: &Path, host: &str) -> Result<Self, String> {
        let bin = sysroot.join("lib").join("rustlib").join(host).join("bin");
        let tool = |name: &str| bin.join(format!("{name}{}", env::consts::EXE_SUFFIX));
        let tools = Self {
            opt: tool("opt"),
            llc: tool("llc"),
        };
        if !tools.opt.is_file() || !tools.llc.is_file() {
            return Err(format!(
                "{} has no opt and llc; install the llvm-tools component",
                bin.display()
            ));
        }
        Ok(tools)
    }

    /// The kernel compiled to PTX: `opt -O3`, then `llc -O3` for `sm_70`.
    pub fn compile(&self, kernel: &Kernel) -> Result<String, String> {
        let ir = emit(kernel);
        let stem = env::temp_dir().join(format!(
            "cuprite-kernel-{}-{}",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        let (ll, bc, ptx) = (stem.with_extension("ll"), stem.with_extension("bc"), stem.with_extension("ptx"));
        fs::write(&ll, &ir).map_err(|e| format!("cannot write {}: {e}", ll.display()))?;
        let run = |tool: &Path, args: &[&std::ffi::OsStr]| -> Result<(), String> {
            let output = Command::new(tool)
                .args(args)
                .output()
                .map_err(|e| format!("cannot run {}: {e}", tool.display()))?;
            if output.status.success() {
                Ok(())
            } else {
                Err(format!(
                    "{} failed:\n{}",
                    tool.display(),
                    String::from_utf8_lossy(&output.stderr)
                ))
            }
        };
        let result = run(&self.opt, &["-O3".as_ref(), ll.as_os_str(), "-o".as_ref(), bc.as_os_str()])
            .and_then(|()| {
                run(
                    &self.llc,
                    &["-O3".as_ref(), "-mcpu=sm_70".as_ref(), bc.as_os_str(), "-o".as_ref(), ptx.as_os_str()],
                )
            })
            .and_then(|()| fs::read_to_string(&ptx).map_err(|e| e.to_string()));
        for file in [&ll, &bc, &ptx] {
            let _ = fs::remove_file(file);
        }
        result.map_err(|error| format!("{error}\nIR:\n{ir}"))
    }
}
