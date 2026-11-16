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
    general_indexed(name, state, function, |_, _, x| reference(x))
}

/// `general`, with a reference that also sees each element's index and the
/// slice's length.
fn general_indexed<T: Element>(
    name: &str,
    state: &mut u64,
    function: impl Fn(&mut [T]),
    reference: impl Fn(usize, usize, T) -> T,
) -> usize {
    for len in LENGTHS {
        let input: Vec<T> = (0..len).map(|_| T::random(state)).collect();
        let expected: Vec<T> = input.iter().enumerate().map(|(i, &x)| reference(i, len, x)).collect();
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
    // Calls: helpers of this crate, generics, and `glam`, whose references
    // run `glam` itself on the CPU.
    checked += general("with_helper", state, probes::with_helper, |x: u32| x.wrapping_mul(3));
    checked += general("swap_halves", state, probes::swap_halves, |x: u32| x.rotate_left(16));
    checked += general("bump_twice", state, probes::bump_twice, |x: f32| (x + 1.0) + 0.5);
    checked += general("generic_twice", state, probes::generic_twice, |x: f32| (x + x) - 1.0);
    checked += general("scaled", state, probes::scaled, |x: f64| x * 2.5 + -1.0);
    checked += general("vector_lengths", state, probes::vector_lengths, |x: f32| {
        glam::Vec3::new(x, 1.0, -2.0).length()
    });
    checked += general("normalized_y", state, probes::normalized_y, |x: f32| {
        glam::Vec3::new(x, x * 0.5, 1.0).normalize().y
    });
    checked += general("cross_dot", state, probes::cross_dot, |x: f32| {
        let a = glam::Vec3::new(x, 2.0, -1.0);
        a.cross(glam::Vec3::Y).dot(a + glam::Vec3::Y)
    });
    // Reading data: values computed before the loop, plain-data parameters
    // and zipped input slices. `b` is 3 shorter than the output, so the
    // last elements stay as they are, as `zip` leaves them.
    let largest = LENGTHS[LENGTHS.len() - 1];
    let a: Vec<u32> = (0..largest).map(|_| u32::random(state)).collect();
    let b: Vec<u32> = (0..largest).map(|_| u32::random(state)).collect();
    let xs: Vec<f32> = (0..largest).map(|_| f32::random(state)).collect();
    let weights: Vec<u8> = (0..largest).map(|_| u8::random(state)).collect();
    let short = |len: usize| len.saturating_sub(3);
    checked += general("normalize_range", state, |v: &mut [f32]| probes::normalize_range(v, -2.5, 7.0), |x| {
        (x - (-2.5f32 + 7.0) * 0.5) / (7.0 - -2.5)
    });
    for double in [false, true] {
        checked += general(
            "conditional_scale",
            state,
            |v: &mut [f32]| probes::conditional_scale(v, double),
            |x| x * if double { 2.0 } else { 3.0 },
        );
    }
    for clamp in [false, true] {
        let settings = probes::Adjust { gain: 1.75, bias: -0.5, clamp, limit: 200 };
        checked += general("adjust", state, |v: &mut [f32]| probes::adjust(v, &settings), |x| {
            let y = x * settings.gain + settings.bias;
            if settings.clamp && y > f32::from(settings.limit) { f32::from(settings.limit) } else { y }
        });
    }
    let settings = probes::Adjust { gain: -3.0, bias: 0.0, clamp: false, limit: 0 };
    checked += general(
        "adjust_by_value",
        state,
        |v: &mut [f32]| probes::adjust_by_value(v, settings, (0.125, -300)),
        |x| (f64::from(x * -3.0) * 0.125) as f32 + -300.0,
    );
    checked += general_indexed(
        "add_into",
        state,
        |v: &mut [u32]| {
            let len = v.len();
            probes::add_into(v, &a[..len], &b[..short(len)])
        },
        |i, len, x| if i < short(len) { a[i].wrapping_add(b[i]) } else { x },
    );
    checked += general_indexed(
        "add_pairs",
        state,
        |v: &mut [u32]| {
            let len = v.len();
            probes::add_pairs(v, &a[..len], &b[..short(len)])
        },
        |i, len, x| if i < short(len) { x.wrapping_add(a[i].wrapping_mul(b[i])) } else { x },
    );
    checked += general_indexed(
        "saxpy",
        state,
        |v: &mut [f32]| probes::saxpy(v, &xs[..v.len()], 0.75),
        |i, _, y| 0.75 * xs[i] + y,
    );
    checked += general_indexed(
        "weighted",
        state,
        |v: &mut [f64]| probes::weighted(v, &weights[..v.len()]),
        |i, _, x| x * (f64::from(weights[i]) / 255.0),
    );
    checked += general("scene_distances", state, probes::scene_distances, |x: f32| {
        use sdfu::SDF;
        let scene = sdfu::Sphere::new(1.0)
            .union_smooth(sdfu::Sphere::new(0.5).translate(glam::Vec3A::new(1.0, 0.0, 0.0)), 0.3)
            .subtract(sdfu::Box::new(glam::Vec3A::new(0.25, 2.0, 0.25)));
        scene.dist(glam::Vec3A::new(x, 0.25, -0.5))
    });
    checked += general("rounded_box", state, |v: &mut [f32]| probes::rounded_box(v, 0.8, 0.1), |x: f32| {
        use sdfu::SDF;
        sdfu::Box::new(glam::Vec3A::splat(0.8)).round(0.1).dist(glam::Vec3A::new(x, x * 0.5, 0.1))
    });
    // Held out: `ultraviolet`, validated after the translator was built.
    checked += general("uv_lengths", state, probes::uv_lengths, |x: f32| {
        ultraviolet::Vec3::new(x, 1.0, -2.0).mag()
    });
    checked += general("uv_normalized_y", state, probes::uv_normalized_y, |x: f32| {
        ultraviolet::Vec3::new(x, x * 0.5, 1.0).normalized().y
    });
    checked += general("uv_cross_dot", state, probes::uv_cross_dot, |x: f32| {
        let a = ultraviolet::Vec3::new(x, 2.0, -1.0);
        let b = ultraviolet::Vec3::new(0.0, 1.0, 0.0);
        a.cross(b).dot(a + b)
    });
    for passes in [0u32, 1, 13] {
        checked += general(
            "repeat_helper",
            state,
            |v: &mut [u32]| probes::repeat_helper(v, passes),
            |x| (0..passes).fold(x, |x, _| x.wrapping_mul(3)),
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
