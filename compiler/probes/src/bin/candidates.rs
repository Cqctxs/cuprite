//! Runs every function the recognizer accepts on many inputs and compares
//! each result with an independent reference that applies the source's
//! operations one by one, pass by pass. Built through the driver with
//! `CUPRITE_AUTO=1`, the calls can run on the GPU.
//!
//! Floats are compared by their bits, so signed zeros must match too; only
//! NaN payloads, which Rust leaves unspecified, may differ.
//!
//! Release builds only: `plain_ops` and `checked_increment` overflow (and
//! so panic) in debug builds, which is why they are rejected there.

use cuprite_probes as probes;

type Case = (&'static str, fn(&mut [u32]), fn(u32) -> u32);

const CASES: [Case; 9] = [
    ("transform", probes::transform, |x| {
        x.wrapping_mul(3).wrapping_add(7)
    }),
    ("transform_iter_mut", probes::transform_iter_mut, |x| {
        x.wrapping_mul(3).wrapping_add(7)
    }),
    ("affine_chain", probes::affine_chain, |x| {
        x.wrapping_mul(3)
            .wrapping_add(7)
            .wrapping_mul(5)
            .wrapping_sub(2)
    }),
    ("plain_ops", probes::plain_ops, |x| {
        x.wrapping_mul(3).wrapping_add(7)
    }),
    ("double", probes::double, |x| x.wrapping_add(x)),
    ("fill", probes::fill, |_| 42),
    ("outside_factor", probes::outside_factor, |x| {
        x.wrapping_mul(3)
    }),
    ("two_stores", probes::two_stores, |x| {
        x.wrapping_add(1).wrapping_mul(2)
    }),
    ("checked_increment", probes::checked_increment, |x| {
        x.wrapping_add(1)
    }),
];

/// A repeated map, its reference for one pass, and the pass counts to try.
/// `repeat_fixed` always makes 4 passes.
type Repeated = (
    &'static str,
    fn(&mut [u32], u32),
    fn(u32) -> u32,
    &'static [u32],
);

const PASSES: &[u32] = &[0, 1, 13];

const REPEATED: [Repeated; 4] = [
    (
        "repeat",
        probes::repeat,
        |x| x.wrapping_mul(3).wrapping_add(7),
        PASSES,
    ),
    (
        "repeat_composed",
        probes::repeat_composed,
        |x| {
            let y = x.wrapping_mul(3).wrapping_add(7);
            y.wrapping_add(y)
        },
        PASSES,
    ),
    (
        "repeat_named",
        probes::repeat_named,
        |x| {
            x.wrapping_mul(3)
                .wrapping_add(7)
                .wrapping_mul(5)
                .wrapping_sub(2)
        },
        PASSES,
    ),
    (
        "repeat_fixed",
        |values, _| probes::repeat_fixed(values),
        |x| x.wrapping_mul(3).wrapping_add(7),
        &[4],
    ),
];

/// Lengths around the GPU's 256-thread blocks, plus a large one.
const LENGTHS: [usize; 10] = [0, 1, 2, 255, 256, 257, 1023, 4097, 65_537, 1 << 20];

/// xorshift64, with a quarter of the values at the edges of the range.
fn next(state: &mut u64) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    let value = (*state >> 32) as u32;
    match *state % 8 {
        0 => 0,
        1 => u32::MAX - (value % 8),
        _ => value,
    }
}

fn check(case: &str, values: &[u32], expected: &[u32]) {
    if let Some(index) = values.iter().zip(expected).position(|(a, b)| a != b) {
        panic!(
            "{case}: index {index} is {} but should be {}",
            values[index], expected[index]
        );
    }
}


/// Element types the general maps use, compared bit for bit, so signed
/// zeros count. Rust leaves the payload of a NaN from arithmetic
/// unspecified (x86 and NVIDIA GPUs differ), so any NaN matches any NaN.
trait Element: Copy + std::fmt::Debug {
    fn random(state: &mut u64) -> Self;
    fn same(self, other: Self) -> bool;
}

macro_rules! integer_elements {
    ($($t:ty),*) => {
        $(impl Element for $t {
            fn random(state: &mut u64) -> Self {
                // Edge values a quarter of the time.
                let bits = u64::from(next(state)) << 32 | u64::from(next(state));
                match bits % 8 {
                    0 => 0,
                    1 => <$t>::MAX,
                    2 => <$t>::MIN,
                    _ => bits as $t,
                }
            }
            fn same(self, other: Self) -> bool {
                self == other
            }
        })*
    };
}
integer_elements!(u8, i16, u32, i64);

impl Element for f32 {
    fn random(state: &mut u64) -> Self {
        // Every kind of float: ordinary values, and from raw bits, NaNs,
        // infinities, subnormals and signed zeros.
        let bits = next(state);
        match bits % 4 {
            0 => f32::from_bits(bits),
            1 => [0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN][(bits / 4 % 5) as usize],
            _ => (bits as i32) as f32 / 65_536.0,
        }
    }
    fn same(self, other: Self) -> bool {
        self.to_bits() == other.to_bits() || self.is_nan() && other.is_nan()
    }
}

impl Element for f64 {
    fn random(state: &mut u64) -> Self {
        let bits = u64::from(next(state)) << 32 | u64::from(next(state));
        match bits % 4 {
            0 => f64::from_bits(bits),
            1 => [0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN][(bits / 4 % 5) as usize],
            _ => (bits as i64) as f64 / 4_294_967_296.0,
        }
    }
    fn same(self, other: Self) -> bool {
        self.to_bits() == other.to_bits() || self.is_nan() && other.is_nan()
    }
}

/// Runs `function` on every length and compares with `reference` applied
/// to each element. Returns the number of cases.
fn general<T: Element>(
    name: &str,
    state: &mut u64,
    function: impl Fn(&mut [T]),
    reference: impl Fn(T) -> T,
) -> usize {
    for len in LENGTHS {
        let input: Vec<T> = (0..len).map(|_| T::random(state)).collect();
        let expected: Vec<T> = input.iter().map(|&x| reference(x)).collect();
        let mut values = input.clone();
        function(&mut values);
        if let Some(index) = values.iter().zip(&expected).position(|(a, b)| !a.same(*b)) {
            panic!(
                "{name}, {len} elements: index {index} is {:?} but should be {:?} (input {:?})",
                values[index], expected[index], input[index]
            );
        }
    }
    LENGTHS.len()
}

/// The general maps, each against an independent reference written with
/// the same operations in the same order.
fn general_cases(state: &mut u64) -> usize {
    let mut checked = 0;
    checked += general(
        "scale_offset",
        state,
        |v: &mut [f32]| probes::scale_offset(v, 1.5, -0.25),
        |x| x * 1.5 + -0.25,
    );
    checked += general("relu", state, probes::relu, |x: f32| if x < 0.0 { 0.0 } else { x });
    checked += general("soft_root", state, probes::soft_root, |x: f64| {
        (x.abs() + 1.0).sqrt().min(1e6).floor()
    });
    checked += general("collatz_steps", state, probes::collatz_steps, |start: u32| {
        let (mut x, mut steps) = (start, 0u32);
        while x > 1 && steps < 500 {
            x = if x.is_multiple_of(2) { x >> 1 } else { x.wrapping_mul(3).wrapping_add(1) };
            steps += 1;
        }
        steps
    });
    checked += general(
        "quantize",
        state,
        |v: &mut [i16]| probes::quantize(v, -7),
        |x| (i32::from(x) / 4).wrapping_mul(-7) as i16,
    );
    checked += general(
        "hash_bytes",
        state,
        |v: &mut [u8]| probes::hash_bytes(v, 0xDEAD_BEEF),
        |x| ((u32::from(x) ^ 0xDEAD_BEEF).wrapping_mul(0x9E37_79B1).rotate_left(5) >> 24) as u8,
    );
    checked += general("levels", state, probes::levels, |x: u8| (x / 64) * 85);
    for (shift, bias) in [(0u32, 0usize), (5, 12_345), (63, usize::MAX), (64, 1), (200, 7)] {
        checked += general(
            "mix_wide",
            state,
            |v: &mut [i64]| probes::mix_wide(v, shift, bias),
            |x| x.wrapping_mul(-3).wrapping_shr(shift) ^ bias as i64,
        );
    }
    for passes in [0u32, 1, 13] {
        checked += general(
            "repeat_damp",
            state,
            |v: &mut [f32]| probes::repeat_damp(v, passes),
            |mut x| {
                for _ in 0..passes {
                    x = x * 0.5 + 0.25;
                    if x < 0.0 {
                        x = 0.0;
                    }
                }
                x
            },
        );
    }
    checked
}

fn main() {
    let mut state = 0x9E37_79B9_7F4A_7C15;
    let mut checked = 0;
    for (name, function, reference) in CASES {
        for len in LENGTHS {
            let input: Vec<u32> = (0..len).map(|_| next(&mut state)).collect();
            let expected: Vec<u32> = input.iter().map(|&x| reference(x)).collect();
            let mut values = input;
            function(&mut values);
            check(&format!("{name}, {len} elements"), &values, &expected);
            checked += 1;
        }
    }
    for (name, function, reference, counts) in REPEATED {
        for len in LENGTHS {
            for &passes in counts {
                let input: Vec<u32> = (0..len).map(|_| next(&mut state)).collect();
                let mut expected = input.clone();
                for _ in 0..passes {
                    for value in &mut expected {
                        *value = reference(*value);
                    }
                }
                let mut values = input;
                function(&mut values, passes);
                let case = format!("{name}, {len} elements, {passes} passes");
                check(&case, &values, &expected);
                checked += 1;
            }
        }
    }
    checked += general_cases(&mut state);
    println!("{checked} cases match");
}
