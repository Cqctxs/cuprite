#![cfg(feature = "gpu-tests")]

//! Every operation, through both backends, against Rust on the CPU.
//!
//! Each test builds kernels `out[i] = f(a[i], b[i])` over edge values and
//! requires the CPU's result bit for bit. The only freedom is Rust's own:
//! NaN bit patterns (x86 and NVIDIA produce different ones), and the sign
//! of a zero from `min` and `max`.

use std::sync::Mutex;

use cuprite_kernel::{builder::Builder, llvm::Llvm, ptx, BinOp, Const, Intrinsic, Kernel, Local, Special, Ty, UnOp};
use cuprite_memory::{Arg, GpuMemory, Grid, MemoryConfig};

struct Gpu {
    memory: GpuMemory,
    llvm: Llvm,
}

/// One GPU, shared by the tests, which run on several threads.
static GPU: Mutex<Option<Gpu>> = Mutex::new(None);

trait Scalar: Copy + std::fmt::Debug {
    const TY: Ty;
    fn bytes(self) -> Vec<u8>;
}

macro_rules! scalars {
    ($($t:ty => $ty:expr),*) => {
        $(impl Scalar for $t {
            const TY: Ty = $ty;
            fn bytes(self) -> Vec<u8> {
                self.to_le_bytes().to_vec()
            }
        })*
    };
}
scalars!(i8 => Ty::I8, i16 => Ty::I16, i32 => Ty::I32, i64 => Ty::I64, u8 => Ty::U8, u16 => Ty::U16,
         u32 => Ty::U32, u64 => Ty::U64, f32 => Ty::F32, f64 => Ty::F64);

impl Scalar for bool {
    const TY: Ty = Ty::Bool;
    fn bytes(self) -> Vec<u8> {
        vec![self as u8]
    }
}

fn all_bytes<T: Scalar>(values: &[T]) -> Vec<u8> {
    values.iter().flat_map(|v| v.bytes()).collect()
}

/// How results may differ from the CPU's.
#[derive(Clone, Copy, PartialEq)]
enum Freedom {
    None,
    /// Any NaN matches any NaN.
    NanBits,
    /// Also `+0.0` matches `-0.0`.
    NanBitsAndZeroSign,
}

fn same(ty: Ty, freedom: Freedom, a: &[u8], b: &[u8]) -> bool {
    if a == b || freedom == Freedom::None {
        return a == b;
    }
    let (x, y) = match ty {
        Ty::F32 => (
            f64::from(f32::from_le_bytes(a.try_into().unwrap())),
            f64::from(f32::from_le_bytes(b.try_into().unwrap())),
        ),
        Ty::F64 => (f64::from_le_bytes(a.try_into().unwrap()), f64::from_le_bytes(b.try_into().unwrap())),
        _ => return false,
    };
    x.is_nan() && y.is_nan() || freedom == Freedom::NanBitsAndZeroSign && x == 0.0 && y == 0.0
}

/// `out[i] = body(a[i], b[i])` for `i < n`; parameters `(out, a, b, n)`.
fn elementwise(name: &str, ta: Ty, tb: Ty, body: impl FnOnce(&mut Builder, Local, Local) -> Local) -> Kernel {
    let (mut b, p) = Builder::new(name, &[Ty::Ptr, Ty::Ptr, Ty::Ptr, Ty::U32]);
    let (out, a, bb, n) = (p[0], p[1], p[2], p[3]);
    let i = b.special(Special::X);
    let inside = b.binary(BinOp::Lt, i, n);
    let (work, done) = (b.block(), b.block());
    b.branch(inside, work, done);
    b.switch_to(work);
    let x = b.index(a, i, ta);
    let y = b.index(bb, i, tb);
    let result = body(&mut b, x, y);
    let ty = b.ty(result);
    let address = b.offset(out, i, ty);
    b.store(address, result);
    b.goto(done);
    b.switch_to(done);
    b.ret();
    b.finish().unwrap()
}

/// Runs the kernel through both backends and compares with `expected`.
fn check(kernel: &Kernel, a: &[u8], b: &[u8], n: usize, expected: &[u8], freedom: Freedom) {
    let mut lock = GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let gpu = lock.get_or_insert_with(|| Gpu {
        memory: GpuMemory::open(MemoryConfig::default()).unwrap(),
        llvm: Llvm::find().unwrap(),
    });
    let out_ty = expected.len() / n.max(1);
    let ty = kernel
        .blocks
        .iter()
        .flat_map(|block| &block.statements)
        .find_map(|statement| match statement {
            cuprite_kernel::Statement::Store(_, value) => Some(kernel.ty(*value)),
            _ => None,
        })
        .unwrap();
    for (backend, ptx) in [("direct PTX", ptx::emit(kernel)), ("LLVM", gpu.llvm.compile(kernel).unwrap())] {
        let function = gpu
            .memory
            .kernel(&ptx, &kernel.name)
            .unwrap_or_else(|e| panic!("{} via {backend}: {e}\n{ptx}", kernel.name));
        let mut out = vec![0xEEu8; expected.len()];
        let mut job = gpu.memory.job().unwrap();
        let (ba, bb) = (job.input(a).unwrap(), job.input(b).unwrap());
        let output = job.output(&mut out).unwrap();
        let args = [Arg::Buffer(output), Arg::Buffer(ba), Arg::Buffer(bb), Arg::U32(n as u32)];
        job.launch(function, Grid::Linear(n as u64), &args).unwrap();
        job.finish().unwrap();
        for i in 0..n {
            let (got, want) = (&out[i * out_ty..][..out_ty], &expected[i * out_ty..][..out_ty]);
            assert!(
                same(ty, freedom, got, want),
                "{} via {backend}, element {i}: got {got:?}, want {want:?}",
                kernel.name
            );
        }
    }
}

fn binary<T: Scalar, U: Scalar, O: Scalar>(name: &str, pairs: &[(T, U)], op: BinOp, f: impl Fn(T, U) -> O, freedom: Freedom) {
    let kernel = elementwise(name, T::TY, U::TY, |b, x, y| b.binary(op, x, y));
    let (a, b): (Vec<T>, Vec<U>) = pairs.iter().copied().unzip();
    let expected: Vec<O> = pairs.iter().map(|&(x, y)| f(x, y)).collect();
    check(&kernel, &all_bytes(&a), &all_bytes(&b), pairs.len(), &all_bytes(&expected), freedom);
}

fn unary<T: Scalar, O: Scalar>(name: &str, values: &[T], build: impl FnOnce(&mut Builder, Local) -> Local, f: impl Fn(T) -> O, freedom: Freedom) {
    let kernel = elementwise(name, T::TY, T::TY, |b, x, _| build(b, x));
    let expected: Vec<O> = values.iter().map(|&x| f(x)).collect();
    let bytes = all_bytes(values);
    check(&kernel, &bytes, &bytes, values.len(), &all_bytes(&expected), freedom);
}

fn pairs<T: Copy, U: Copy>(xs: &[T], ys: &[U]) -> Vec<(T, U)> {
    xs.iter().flat_map(|&x| ys.iter().map(move |&y| (x, y))).collect()
}

/// Shift and rotation amounts: in range, at the width, and beyond it.
const AMOUNTS: [u32; 13] = [0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 65, 200, u32::MAX];

macro_rules! integer_suite {
    ($name:ident, $t:ty) => {
        #[test]
        fn $name() {
            let n = stringify!($t);
            let zero: $t = 0;
            let values: Vec<$t> = vec![
                0, 1, 2, 3, 7, 0x5A, <$t>::MAX, <$t>::MIN, <$t>::MAX - 1, <$t>::MIN.wrapping_add(1),
                <$t>::MAX / 3, zero.wrapping_sub(1), zero.wrapping_sub(7), 100u8 as $t,
                // Only a high bit: the top of the trailing-zero counts.
                <$t>::MAX ^ (<$t>::MAX >> 1), (<$t>::MAX ^ (<$t>::MAX >> 1)) >> 1,
            ];
            let all = pairs(&values, &values);
            let wrapping: [(BinOp, fn($t, $t) -> $t); 6] = [
                (BinOp::Add, |x, y| x.wrapping_add(y)),
                (BinOp::Sub, |x, y| x.wrapping_sub(y)),
                (BinOp::Mul, |x, y| x.wrapping_mul(y)),
                (BinOp::And, |x, y| x & y),
                (BinOp::Or, |x, y| x | y),
                (BinOp::Xor, |x, y| x ^ y),
            ];
            for (op, f) in wrapping {
                binary(&format!("{n}_{op:?}"), &all, op, f, Freedom::None);
            }
            // The producer rules out zero divisors and `MIN / -1`.
            let divisible: Vec<($t, $t)> = all
                .iter()
                .copied()
                .filter(|&(x, y)| y != 0 && x.checked_div(y).is_some())
                .collect();
            binary(&format!("{n}_div"), &divisible, BinOp::Div, |x: $t, y: $t| x / y, Freedom::None);
            binary(&format!("{n}_rem"), &divisible, BinOp::Rem, |x: $t, y: $t| x % y, Freedom::None);
            let compare: [(BinOp, fn($t, $t) -> bool); 6] = [
                (BinOp::Eq, |x, y| x == y),
                (BinOp::Ne, |x, y| x != y),
                (BinOp::Lt, |x, y| x < y),
                (BinOp::Le, |x, y| x <= y),
                (BinOp::Gt, |x, y| x > y),
                (BinOp::Ge, |x, y| x >= y),
            ];
            for (op, f) in compare {
                binary(&format!("{n}_{op:?}"), &all, op, f, Freedom::None);
            }
            let shifts = pairs(&values, &AMOUNTS);
            binary(&format!("{n}_shl"), &shifts, BinOp::Shl, |x: $t, s: u32| x.wrapping_shl(s), Freedom::None);
            binary(&format!("{n}_shr"), &shifts, BinOp::Shr, |x: $t, s: u32| x.wrapping_shr(s), Freedom::None);
            let rotate = |left: bool| {
                let kernel = elementwise(&format!("{n}_rotate_{left}"), <$t as Scalar>::TY, Ty::U32, |b, x, s| {
                    let which = if left { Intrinsic::RotateLeft } else { Intrinsic::RotateRight };
                    b.call(which, &[x.into(), s.into()])
                });
                let (a, b): (Vec<$t>, Vec<u32>) = shifts.iter().copied().unzip();
                let expected: Vec<$t> = shifts
                    .iter()
                    .map(|&(x, s)| if left { x.rotate_left(s) } else { x.rotate_right(s) })
                    .collect();
                check(&kernel, &all_bytes(&a), &all_bytes(&b), shifts.len(), &all_bytes(&expected), Freedom::None);
            };
            rotate(true);
            rotate(false);
            unary(&format!("{n}_neg"), &values, |b, x| b.unary(UnOp::Neg, x), |x: $t| x.wrapping_neg(), Freedom::None);
            unary(&format!("{n}_not"), &values, |b, x| b.unary(UnOp::Not, x), |x: $t| !x, Freedom::None);
            unary(
                &format!("{n}_clz"),
                &values,
                |b, x| b.call(Intrinsic::LeadingZeros, &[x.into()]),
                |x: $t| x.leading_zeros(),
                Freedom::None,
            );
            unary(
                &format!("{n}_ctz"),
                &values,
                |b, x| b.call(Intrinsic::TrailingZeros, &[x.into()]),
                |x: $t| x.trailing_zeros(),
                Freedom::None,
            );
            unary(
                &format!("{n}_popcount"),
                &values,
                |b, x| b.call(Intrinsic::CountOnes, &[x.into()]),
                |x: $t| x.count_ones(),
                Freedom::None,
            );
        }
    };
}

integer_suite!(i8_operations, i8);
integer_suite!(i16_operations, i16);
integer_suite!(i32_operations, i32);
integer_suite!(i64_operations, i64);
integer_suite!(u8_operations, u8);
integer_suite!(u16_operations, u16);
integer_suite!(u32_operations, u32);
integer_suite!(u64_operations, u64);

macro_rules! float_suite {
    ($name:ident, $t:ident) => {
        #[test]
        fn $name() {
            let n = stringify!($t);
            let values: Vec<$t> = vec![
                0.0, -0.0, 1.0, -1.0, 0.5, 1.5, 2.5, -2.5, 3.0, 0.1, std::$t::consts::PI, 1e10, -1e-10,
                $t::MAX, $t::MIN, $t::MIN_POSITIVE, $t::MIN_POSITIVE / 4.0, -$t::MIN_POSITIVE / 3.0,
                $t::INFINITY, $t::NEG_INFINITY, $t::NAN,
            ];
            let all = pairs(&values, &values);
            let arithmetic: [(BinOp, fn($t, $t) -> $t); 4] = [
                (BinOp::Add, |x, y| x + y),
                (BinOp::Sub, |x, y| x - y),
                (BinOp::Mul, |x, y| x * y),
                (BinOp::Div, |x, y| x / y),
            ];
            for (op, f) in arithmetic {
                binary(&format!("{n}_{op:?}"), &all, op, f, Freedom::NanBits);
            }
            let compare: [(BinOp, fn($t, $t) -> bool); 6] = [
                (BinOp::Eq, |x, y| x == y),
                (BinOp::Ne, |x, y| x != y),
                (BinOp::Lt, |x, y| x < y),
                (BinOp::Le, |x, y| x <= y),
                (BinOp::Gt, |x, y| x > y),
                (BinOp::Ge, |x, y| x >= y),
            ];
            for (op, f) in compare {
                binary(&format!("{n}_{op:?}"), &all, op, f, Freedom::None);
            }
            let exact: [(&str, fn(&mut Builder, Local) -> Local, fn($t) -> $t); 7] = [
                ("neg", |b, x| b.unary(UnOp::Neg, x), |x| -x),
                ("abs", |b, x| b.call(Intrinsic::Abs, &[x.into()]), |x| x.abs()),
                ("sqrt", |b, x| b.call(Intrinsic::Sqrt, &[x.into()]), |x| x.sqrt()),
                ("floor", |b, x| b.call(Intrinsic::Floor, &[x.into()]), |x| x.floor()),
                ("ceil", |b, x| b.call(Intrinsic::Ceil, &[x.into()]), |x| x.ceil()),
                ("trunc", |b, x| b.call(Intrinsic::Trunc, &[x.into()]), |x| x.trunc()),
                ("round_ties_even", |b, x| b.call(Intrinsic::RoundTiesEven, &[x.into()]), |x| x.round_ties_even()),
            ];
            for (name, build, f) in exact {
                // `-` and `abs` change only the sign bit, NaNs included.
                let freedom = if matches!(name, "neg" | "abs") { Freedom::None } else { Freedom::NanBits };
                unary(&format!("{n}_{name}"), &values, build, f, freedom);
            }
            for (name, which, f) in [
                ("min", Intrinsic::Min, $t::min as fn($t, $t) -> $t),
                ("max", Intrinsic::Max, $t::max as fn($t, $t) -> $t),
            ] {
                let kernel = elementwise(&format!("{n}_{name}"), <$t as Scalar>::TY, <$t as Scalar>::TY, |b, x, y| {
                    b.call(which, &[x.into(), y.into()])
                });
                let (a, b): (Vec<$t>, Vec<$t>) = all.iter().copied().unzip();
                let expected: Vec<$t> = all.iter().map(|&(x, y)| f(x, y)).collect();
                check(&kernel, &all_bytes(&a), &all_bytes(&b), all.len(), &all_bytes(&expected), Freedom::NanBitsAndZeroSign);
            }
            // `mul_add(x, y, x)`: rounded once.
            let kernel = elementwise(&format!("{n}_mul_add"), <$t as Scalar>::TY, <$t as Scalar>::TY, |b, x, y| {
                b.call(Intrinsic::MulAdd, &[x.into(), y.into(), x.into()])
            });
            let (a, b): (Vec<$t>, Vec<$t>) = all.iter().copied().unzip();
            let expected: Vec<$t> = all.iter().map(|&(x, y)| x.mul_add(y, x)).collect();
            check(&kernel, &all_bytes(&a), &all_bytes(&b), all.len(), &all_bytes(&expected), Freedom::NanBits);
        }
    };
}

float_suite!(f32_operations, f32);
float_suite!(f64_operations, f64);

/// Every numeric type to every other, with Rust's `as`.
#[test]
fn casts_match_rust_as() {
    macro_rules! from {
        ($from:ty, [$($value:expr),*]) => {{
            let values: Vec<$from> = vec![$($value as $from),*];
            macro_rules! to {
                ($to:ty) => {{
                    let name = format!("cast_{}_{}", stringify!($from), stringify!($to));
                    let freedom = if <$to as Scalar>::TY.is_float() { Freedom::NanBits } else { Freedom::None };
                    unary(&name, &values, |b, x| b.cast(x, <$to as Scalar>::TY), |x: $from| x as $to, freedom);
                }};
            }
            to!(i8); to!(i16); to!(i32); to!(i64); to!(u8); to!(u16); to!(u32); to!(u64); to!(f32); to!(f64);
        }};
    }
    from!(i8, [0, 1, -1, 127, -128, 100, -100]);
    from!(i16, [0, 1, -1, 32767, -32768, 300, -300, 255, 256]);
    from!(i32, [0, 1, -1, i32::MAX, i32::MIN, 16_777_217, -16_777_217, 70_000, 255]);
    from!(i64, [0, 1, -1, i64::MAX, i64::MIN, 9_007_199_254_740_993i64, 4_294_967_296i64, -5_000_000_000i64]);
    from!(u8, [0, 1, 127, 128, 255]);
    from!(u16, [0, 1, 255, 256, 32768, 65535]);
    from!(u32, [0, 1, 255, 65536, 16_777_217, u32::MAX, 2_147_483_648u32]);
    from!(u64, [0, 1, u64::MAX, 18_446_744_073_709_551_557u64, 9_007_199_254_740_993u64, 4_294_967_296u64]);
    let f32s = [0.0f32, -0.0, 0.5, -0.5, 1.5, -1.5, 255.9, 256.0, -129.0, 3e9, -3e9, 1e20, -1e20,
        16_777_217.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, f32::MIN_POSITIVE];
    from!(f32, [f32s[0], f32s[1], f32s[2], f32s[3], f32s[4], f32s[5], f32s[6], f32s[7], f32s[8],
        f32s[9], f32s[10], f32s[11], f32s[12], f32s[13], f32s[14], f32s[15], f32s[16], f32s[17]]);
    from!(f64, [0.0f64, -0.0, 0.5, -1.5, 255.9, 65535.5, -32768.9, 4_294_967_295.9, 1e19, -1e19,
        1e300, 9_007_199_254_740_993.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 3.4028235677973366e38,
        1e-310, 0.1]);
}

/// Floats to integers of their width and back, keeping every bit: NaN
/// payloads, both zeros, subnormals.
#[test]
fn reinterpreting_keeps_bits() {
    let f32_bits = [0u32, 0x8000_0000, 1, 0x8000_0001, 0x007F_FFFF, 0x3F80_0000, 0x7F80_0000, 0xFF80_0000,
        0x7F80_0001, 0x7FC0_0000, 0xFFC0_1234, 0x7FBF_FFFF, u32::MAX];
    let f32s: Vec<f32> = f32_bits.iter().map(|&b| f32::from_bits(b)).collect();
    let i32s: Vec<i32> = f32_bits.iter().map(|&b| b as i32).collect();
    unary("f32_to_bits", &f32s, |b, x| b.reinterpret(x, Ty::U32), f32::to_bits, Freedom::None);
    unary("f32_to_i32_bits", &f32s, |b, x| b.reinterpret(x, Ty::I32), |x: f32| x.to_bits() as i32, Freedom::None);
    unary("f32_from_bits", &f32_bits, |b, x| b.reinterpret(x, Ty::F32), f32::from_bits, Freedom::None);
    unary("f32_from_i32_bits", &i32s, |b, x| b.reinterpret(x, Ty::F32), |x: i32| f32::from_bits(x as u32), Freedom::None);
    unary("i32_as_u32_bits", &i32s, |b, x| b.reinterpret(x, Ty::U32), |x: i32| x as u32, Freedom::None);
    let f64_bits = [0u64, 1 << 63, 1, 0x000F_FFFF_FFFF_FFFF, 0x3FF0_0000_0000_0000, 0x7FF0_0000_0000_0000,
        0xFFF0_0000_0000_0000, 0x7FF0_0000_0000_0001, 0x7FF8_0000_0000_0000, 0xFFF8_0000_DEAD_BEEF, u64::MAX];
    let f64s: Vec<f64> = f64_bits.iter().map(|&b| f64::from_bits(b)).collect();
    let i64s: Vec<i64> = f64_bits.iter().map(|&b| b as i64).collect();
    unary("f64_to_bits", &f64s, |b, x| b.reinterpret(x, Ty::U64), f64::to_bits, Freedom::None);
    unary("f64_to_i64_bits", &f64s, |b, x| b.reinterpret(x, Ty::I64), |x: f64| x.to_bits() as i64, Freedom::None);
    unary("f64_from_bits", &f64_bits, |b, x| b.reinterpret(x, Ty::F64), f64::from_bits, Freedom::None);
    unary("f64_from_i64_bits", &i64s, |b, x| b.reinterpret(x, Ty::F64), |x: i64| f64::from_bits(x as u64), Freedom::None);
    let i8s = [0i8, 1, -1, 127, -128];
    unary("i8_as_u8_bits", &i8s, |b, x| b.reinterpret(x, Ty::U8), |x: i8| x as u8, Freedom::None);
}

/// `bool` operations, casts to integers, stores, and selects.
#[test]
fn booleans_and_selects_match() {
    let bools = [false, true];
    let all = pairs(&bools, &bools);
    type Op = (BinOp, fn(bool, bool) -> bool);
    let ops: [Op; 9] = [
        (BinOp::And, |x, y| x & y),
        (BinOp::Or, |x, y| x | y),
        (BinOp::Xor, |x, y| x ^ y),
        (BinOp::Eq, |x, y| x == y),
        (BinOp::Ne, |x, y| x != y),
        (BinOp::Lt, |x, y| !x & y),
        (BinOp::Le, |x, y| x <= y),
        (BinOp::Gt, |x, y| x & !y),
        (BinOp::Ge, |x, y| x >= y),
    ];
    for (op, f) in ops {
        // Bools travel as `u8`; compare them, then store the result.
        let kernel = elementwise(&format!("bool_{op:?}"), Ty::U8, Ty::U8, |b, x, y| {
            let (x, y) = (b.binary(BinOp::Ne, x, Const::int(Ty::U8, 0)), b.binary(BinOp::Ne, y, Const::int(Ty::U8, 0)));
            b.binary(op, x, y)
        });
        let (a, b): (Vec<u8>, Vec<u8>) = all.iter().map(|&(x, y)| (x as u8, y as u8)).unzip();
        let expected: Vec<bool> = all.iter().map(|&(x, y)| f(x, y)).collect();
        check(&kernel, &a, &b, all.len(), &all_bytes(&expected), Freedom::None);
    }
    let values: Vec<i32> = vec![-5, 0, 3, i32::MAX];
    let kernel = elementwise("select_and_cast", Ty::I32, Ty::I32, |b, x, y| {
        let less = b.binary(BinOp::Lt, x, y);
        let not = b.unary(UnOp::Not, less);
        let flag = b.cast(not, Ty::I32);
        let picked = b.select(less, x, y);
        b.binary(BinOp::Add, picked, flag)
    });
    let all = pairs(&values, &values);
    let (a, b): (Vec<i32>, Vec<i32>) = all.iter().copied().unzip();
    let expected: Vec<i32> = all.iter().map(|&(x, y)| x.min(y).wrapping_add(i32::from(x >= y))).collect();
    check(&kernel, &all_bytes(&a), &all_bytes(&b), all.len(), &all_bytes(&expected), Freedom::None);
}

/// A data-dependent loop with a branch inside: Collatz steps, capped.
#[test]
fn loops_and_branches_match() {
    let kernel = elementwise("collatz", Ty::U32, Ty::U32, |b, x, cap| {
        let (value, steps) = (b.copy(x), b.copy(Const::int(Ty::U32, 0)));
        let (head, body, even, odd, join, exit) = (b.block(), b.block(), b.block(), b.block(), b.block(), b.block());
        b.goto(head);
        b.switch_to(head);
        let not_one = b.binary(BinOp::Gt, value, Const::int(Ty::U32, 1));
        let under = b.binary(BinOp::Lt, steps, cap);
        let more = b.binary(BinOp::And, not_one, under);
        b.branch(more, body, exit);
        b.switch_to(body);
        let low = b.binary(BinOp::And, value, Const::int(Ty::U32, 1));
        let is_even = b.binary(BinOp::Eq, low, Const::int(Ty::U32, 0));
        b.branch(is_even, even, odd);
        b.switch_to(even);
        b.set(value, cuprite_kernel::Rvalue::Binary(BinOp::Shr, value.into(), Const::int(Ty::U32, 1).into()));
        b.goto(join);
        b.switch_to(odd);
        let tripled = b.binary(BinOp::Mul, value, Const::int(Ty::U32, 3));
        b.set(value, cuprite_kernel::Rvalue::Binary(BinOp::Add, tripled.into(), Const::int(Ty::U32, 1).into()));
        b.goto(join);
        b.switch_to(join);
        b.set(steps, cuprite_kernel::Rvalue::Binary(BinOp::Add, steps.into(), Const::int(Ty::U32, 1).into()));
        b.goto(head);
        b.switch_to(exit);
        steps
    });
    let starts: Vec<u32> = (0..2000).chain([27, 97, 871, 77_031, u32::MAX]).collect();
    let caps = vec![500u32; starts.len()];
    let expected: Vec<u32> = starts
        .iter()
        .map(|&start| {
            let (mut value, mut steps) = (start, 0u32);
            while value > 1 && steps < 500 {
                value = if value & 1 == 0 { value >> 1 } else { value.wrapping_mul(3).wrapping_add(1) };
                steps += 1;
            }
            steps
        })
        .collect();
    check(&kernel, &all_bytes(&starts), &all_bytes(&caps), starts.len(), &all_bytes(&expected), Freedom::None);
}
