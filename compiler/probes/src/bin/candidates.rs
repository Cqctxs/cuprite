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
integer_elements!(u8, u16, i16, i32, u32, i64, u64);

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

/// x86's float to `i32` conversion: rounded or truncated, and `i32::MIN`
/// for NaN and out of range.
fn x86_to_i32(x: f32, round: bool) -> i32 {
    let r = if round { x.round_ties_even() } else { x.trunc() };
    if (-2_147_483_648.0..2_147_483_648.0).contains(&r) { r as i32 } else { i32::MIN }
}

/// `sse_integers`, on 128-bit values held as `u128`, lane by lane.
fn sse_integers_reference(v: u32) -> u32 {
    let lanes16 = |x: u128| -> [i16; 8] { std::array::from_fn(|i| (x >> (16 * i)) as u16 as i16) };
    let from16 = |l: [i16; 8]| -> u128 { l.iter().enumerate().map(|(i, &x)| u128::from(x as u16) << (16 * i)).sum() };
    let lanes32 = |x: u128| -> [i32; 4] { std::array::from_fn(|i| (x >> (32 * i)) as u32 as i32) };
    let from32 = |l: [i32; 4]| -> u128 { l.iter().enumerate().map(|(i, &x)| u128::from(x as u32) << (32 * i)).sum() };
    let from8 = |l: [u8; 16]| -> u128 { l.iter().enumerate().map(|(i, &x)| u128::from(x) << (8 * i)).sum() };
    let x = lanes16(from32([v as i32; 4]));
    let y: [i16; 8] = [-12345, 12345, -300, 300, 7, -1, 32767, -32768];
    let high: [i16; 8] = std::array::from_fn(|i| ((i32::from(x[i]) * i32::from(y[i])) >> 16) as i16);
    let high_unsigned: [i16; 8] =
        std::array::from_fn(|i| ((u32::from(x[i] as u16) * u32::from(y[i] as u16)) >> 16) as u16 as i16);
    let summed: [i32; 4] = std::array::from_fn(|j| {
        (i32::from(x[2 * j]) * i32::from(y[2 * j])).wrapping_add(i32::from(x[2 * j + 1]) * i32::from(y[2 * j + 1]))
    });
    let summed16 = lanes16(from32(summed));
    let signed_byte = |h: i16| h.clamp(-128, 127) as i8 as u8;
    let unsigned_byte = |h: i16| h.clamp(0, 255) as u8;
    let packed: [u8; 16] =
        std::array::from_fn(|k| if k < 8 { signed_byte(summed16[k]) } else { signed_byte(high[k - 8]) });
    let packed_unsigned: [u8; 16] =
        std::array::from_fn(|k| if k < 8 { unsigned_byte(high_unsigned[k]) } else { unsigned_byte(summed16[k - 8]) });
    let group = |g: usize| -> u128 {
        (8 * g..8 * g + 8).map(|k| u128::from(packed[k].abs_diff(packed_unsigned[k]))).sum()
    };
    let sad = group(0) | group(1) << 64;
    let count = (v % 20).min(15);
    let shifted = from16(high.map(|h| h >> count));
    let (p, q) = (lanes32(from8(packed)), lanes32(from8(packed_unsigned)));
    let added = from32(std::array::from_fn(|j| p[j].wrapping_add(q[j])));
    let all = sad ^ shifted ^ added;
    let folded = all ^ (all >> 64);
    let folded = folded ^ (folded >> 32);
    folded as u32
}

/// `sse_mix`, on 128-bit values held as `u128`, instruction by instruction.
fn sse_mix_reference(v: u32) -> u32 {
    let bytes = |x: u128| -> [u8; 16] { x.to_le_bytes() };
    let from_bytes = u128::from_le_bytes;
    let lanes16 = |x: u128| -> [i16; 8] { std::array::from_fn(|i| (x >> (16 * i)) as u16 as i16) };
    let from16 = |l: [i16; 8]| -> u128 { l.iter().enumerate().map(|(i, &x)| u128::from(x as u16) << (16 * i)).sum() };
    let lanes32 = |x: u128| -> [u32; 4] { std::array::from_fn(|i| (x >> (32 * i)) as u32) };
    let from32 = |l: [u32; 4]| -> u128 { l.iter().enumerate().map(|(i, &x)| u128::from(x) << (32 * i)).sum() };
    let lanes64 = |x: u128| -> [u64; 2] { [x as u64, (x >> 64) as u64] };
    let from64 = |l: [u64; 2]| -> u128 { u128::from(l[0]) | u128::from(l[1]) << 64 };
    let mask32 = |holds: bool| if holds { u32::MAX } else { 0 };
    let to_i32 = |x: f64, round: bool| {
        let r = if round { x.round_ties_even() } else { x.trunc() };
        if (-2_147_483_648.0..2_147_483_648.0).contains(&r) { r as i32 } else { i32::MIN }
    };
    let x = from32([v; 4]);
    let y = from32([i32::MIN as u32, 0x0102_0304, u32::MAX, 0x7F80_FF01]);
    let (xb, yb) = (bytes(x), bytes(y));
    let saturated: [u8; 16] = std::array::from_fn(|i| xb[i].saturating_add(yb[i]));
    let (x16, y16) = (lanes16(x), lanes16(y));
    let signed = from16(std::array::from_fn(|i| x16[i].saturating_sub(y16[i])));
    let average: [u8; 16] = std::array::from_fn(|i| ((u16::from(xb[i]) + u16::from(yb[i]) + 1) >> 1) as u8);
    let unpacked = from_bytes(std::array::from_fn(|i| if i % 2 == 0 { saturated[i / 2] } else { average[i / 2] }));
    let mask = (0..16).filter(|&i| xb[i] & yb[i] == yb[i]).fold(0u32, |m, i| m | 1 << i);
    let mixed = lanes32(unpacked ^ signed);
    let shuffled = [mixed[3], mixed[2], mixed[1], mixed[0]];
    let count = v % 40;
    let shifted = from32(shuffled.map(|l| if count > 31 { 0 } else { l << count }));
    let count = v % 70;
    let right = from64(lanes64(shifted).map(|l| if count > 63 { 0 } else { l >> count }));
    let floats: [f32; 4] = lanes32(right).map(|l| l as i32 as f32);
    let doubles = [f64::from(v) * -0.75, f64::from(f32::from_bits(v))];
    let narrowed = [to_i32(doubles[0], true) as u32, to_i32(doubles[1], true) as u32, 0, 0];
    let c = f32::from_bits(v);
    let compared = from32(floats.map(|f| mask32(f != c)));
    let ordered = from64(doubles.map(|d| if d.is_nan() { 0 } else { u64::MAX }));
    let scalar = [mask32(0.0 <= floats[0]), floats[1].to_bits(), floats[2].to_bits(), floats[3].to_bits()];
    let added = from32(std::array::from_fn(|i| narrowed[i].wrapping_add(scalar[i])));
    let all = compared ^ ordered ^ added;
    let folded = all ^ (all >> 64);
    let folded = folded ^ (folded >> 32);
    let truncated = doubles[0].trunc();
    let truncated = if (-9.223_372_036_854_776e18..9.223_372_036_854_776e18).contains(&truncated) {
        truncated as i64
    } else {
        i64::MIN
    };
    folded as u32 ^ mask ^ (truncated as u32).rotate_left(13)
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
    // Closures.
    checked += general("closure_affine", state, |v: &mut [f32]| probes::closure_affine(v, 1.5, -0.25), |x| {
        x * 1.5 + -0.25
    });
    checked += general("closure_in_body", state, |v: &mut [u32]| probes::closure_in_body(v, 5), |x| {
        5u32.wrapping_add(x) ^ 1u32.wrapping_add(x)
    });
    checked += general("closure_pair", state, |v: &mut [u32]| probes::closure_pair(v, 0x1234_5678), |x| {
        x.wrapping_mul(3) ^ 0x1234_5678u32.rotate_left(5)
    });
    checked += general("closure_counter", state, probes::closure_counter, |x: u32| {
        let total = x.rotate_left(3);
        total.wrapping_add(7).rotate_left(3)
    });
    checked += general("closure_move", state, |v: &mut [i32]| probes::closure_move(v, 9), |x| x.wrapping_sub(9) >> 1);
    checked += general("closure_generic", state, |v: &mut [f32]| probes::closure_generic(v, 0.75), |x| {
        (x + 0.75) + 0.75
    });
    checked += general("closure_once", state, |v: &mut [u32]| probes::closure_once(v, 0xF0F0_1234), |x| {
        (x & 0xF0F0_1234).wrapping_mul(3)
    });
    checked += general("function_item", state, probes::function_item, |x: f32| (x * 0.5) * 0.5);
    let table: Vec<u32> = (0..37).map(|_| next(state)).collect();
    checked += general("closure_table", state, |v: &mut [u32]| probes::closure_table(v, &table), |x| {
        table[x as usize % 37].wrapping_add(table[(x >> 8) as usize % 37])
    });
    // Iterator chains: `for_each`, sequential and Rayon's.
    checked += general("each_affine", state, probes::each_affine, |x: u32| x.wrapping_mul(5).wrapping_add(3));
    checked += general_indexed(
        "each_saxpy",
        state,
        |v: &mut [f32]| probes::each_saxpy(v, &xs[..v.len()], 0.75),
        |i, _, y| y + 0.75 * xs[i],
    );
    checked += general("each_function", state, probes::each_function, |x: f32| x.clamp(0.0, 1.0));
    checked += general("each_setup", state, |v: &mut [f64]| probes::each_setup(v, 0.3), |x| {
        x * (0.3 * 2.0 - 1.0) + 0.5
    });
    checked += general("transform_rayon", state, probes::transform_rayon, |x: u32| {
        x.wrapping_mul(3).wrapping_add(7)
    });
    // Rayon's `zip` also stops at the shorter side.
    checked += general_indexed(
        "par_saxpy",
        state,
        |v: &mut [f32]| {
            let len = v.len();
            probes::par_saxpy(v, &xs[..short(len)], -1.5)
        },
        |i, len, y| if i < short(len) { y + -1.5 * xs[i] } else { y },
    );
    // Refused, so they run on the CPU; here so that accepting one by
    // mistake shows up as a wrong result.
    checked += general_indexed("each_counter", state, probes::each_counter, |i, _, x: u32| x.wrapping_add(i as u32));
    checked += general_indexed("each_then_more", state, probes::each_then_more, |i, _, x: u32| {
        if i == 0 { 0 } else { x.wrapping_add(1) }
    });
    // Held out before, now development code: `num-integer`'s square root,
    // a closure passed to a generic fixed-point loop.
    checked += general("ni_sqrt", state, probes::ni_sqrt, |x: u64| x.isqrt());
    // Arrays and bit reinterpretation.
    let coeffs = [0.5f32, -1.25, 2.0, 0.75];
    checked += general("array_poly", state, |v: &mut [f32]| probes::array_poly(v, coeffs), |x| {
        (((0.0 * x + 0.75) * x + 2.0) * x + -1.25) * x + 0.5
    });
    checked += general("array_bytes", state, probes::array_bytes, u32::swap_bytes);
    checked += general("array_histogram", state, probes::array_histogram, |x: u32| {
        let groups: Vec<u32> = (0..10).map(|k| (x >> (3 * k)) & 7).collect();
        (0..8u32).fold(0u32, |mix, g| mix.wrapping_mul(31).wrapping_add(groups.iter().filter(|&&n| n == g).count() as u32))
    });
    checked += general("array_const_table", state, probes::array_const_table, |x: u32| (x & 0xFF).count_ones() | (x & !0xFF));
    checked += general("array_matrix", state, |v: &mut [u32]| probes::array_matrix(v, [[3, 5], [7, 11]]), |x| {
        let picked = match (x & 1, (x >> 1) & 1) {
            (0, 0) => 3u32,
            (0, _) => 5,
            (_, 0) => 7,
            _ => 11,
        };
        picked.wrapping_mul(5).wrapping_add(x ^ 7)
    });
    checked += general("array_gradients", state, probes::array_gradients, |x: f32| {
        let (gx, gy) = match x.to_bits() & 3 {
            0 => (1.0, 0.0),
            1 => (0.0, 1.0),
            2 => (-1.0, 0.0),
            _ => (0.0, -1.0),
        };
        gx * x + gy * 0.5
    });
    checked += general("array_patterns", state, probes::array_patterns, |x: u32| {
        (x & 0xFF) << 24 | (((x >> 8) ^ (x >> 16)) & 0xFF) << 8 | 3 << 16 | x >> 24
    });
    checked += general("float_bits", state, probes::float_bits, |x: u32| (x & 0x7FFF_FFFF) ^ (x >> 31));
    checked += general("float_exponent", state, probes::float_exponent, |x: f64| {
        if x.is_normal() && (x.to_bits() >> 52) & 0x7FF < 0x7FE { x * 2.0 } else { x }
    });
    // Held out before, now development code: `half`'s bfloat16, rounded to
    // nearest even on the bits.
    checked += general("bf16_round_trip", state, probes::bf16_round_trip, |x: f32| {
        let bits = x.to_bits();
        if x.is_nan() { f32::NAN } else { f32::from_bits(bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000) }
    });
    // Enums.
    checked += general("enum_option", state, probes::enum_option, |x: u32| {
        if x % 2 == 0 { x / 2 } else { x.wrapping_mul(3).wrapping_add(1) }
    });
    checked += general("enum_match", state, probes::enum_match, |x: f32| match x.to_bits() & 3 {
        0 => 3.0 * x * x,
        1 => x * x,
        2 => x * 0.5,
        _ => 0.0,
    });
    checked += general("enum_ordering", state, |v: &mut [i32]| probes::enum_ordering(v, 0), |x| {
        let rank = if x < 0 { 0 } else if x == 0 { 1 } else { 2 };
        rank + x.clamp(-5, 100)
    });
    checked += general("enum_partial_cmp", state, probes::enum_partial_cmp, |x: f32| {
        if x.is_nan() { 2.0 } else if x < 0.5 { -1.0 } else if x == 0.5 { 0.0 } else { 1.0 }
    });
    checked += general("enum_inner_for", state, probes::enum_inner_for, |x: u32| {
        let mut acc = 0u32;
        for k in 0..8 {
            acc = acc.wrapping_mul(31).wrapping_add((x >> k) & 1);
        }
        acc ^ (x >> 24) ^ (x >> 16) ^ (x >> 8)
    });
    checked += general("enum_iter_sum", state, probes::enum_iter_sum, |x: u32| {
        x.to_le_bytes().iter().map(|&b| u32::from(b)).filter(|b| b % 3 != 0).sum()
    });
    checked += general("enum_char", state, probes::enum_char, |x: u32| {
        let c = x & 0x7F;
        match c {
            0x30..=0x39 => c - 0x30,
            0x41..=0x46 => c - 0x41 + 10,
            0x61..=0x66 => c - 0x61 + 10,
            0x67..=0x7A => c - 32,
            _ => c + 1000,
        }
    });
    checked += general("enum_nonzero", state, probes::enum_nonzero, |x: u32| {
        if x == 0 { 32 } else { x.trailing_zeros() + x.leading_zeros() * 64 }
    });
    checked += general("enum_table", state, probes::enum_table, |x: u32| {
        let applied = match x & 3 {
            0 => x.wrapping_add(3),
            1 => x.wrapping_mul(5),
            2 => x ^ 0xFF,
            _ => x,
        };
        applied.rotate_left([1, 4, 2][((x >> 2) % 3) as usize])
    });
    // SIMD.
    checked += general("simd_roots", state, probes::simd_roots, f32::sqrt);
    checked += general("sse_dot", state, probes::sse_dot, |x: f32| {
        let s = |k: f32| x * k + 0.5;
        (s(1.0) + s(3.0)) + (s(2.0) + s(4.0))
    });
    checked += general("sse_min_mask", state, probes::sse_min_mask, |v: u32| {
        let x = f32::from_bits(v);
        let limits = [-1.0f32, 1.0, -0.0, f32::NAN];
        let low = |i: usize| if x < limits[i] { x } else { limits[i] };
        let mut bits = 0u32;
        for (i, &limit) in limits.iter().enumerate() {
            bits |= u32::from(x < limit) << i | u32::from(x.is_nan() || limit.is_nan()) << (i + 4);
        }
        let picked = if x < limits[0] { low(0).to_bits() } else { 0 };
        picked ^ bits ^ low(2).to_bits()
    });
    checked += general("sse_convert", state, probes::sse_convert, |v: u32| {
        let x = f32::from_bits(v);
        let lanes = [x, x * 2.0, -x, x + 0.5];
        let mixed = lanes.map(|l| x86_to_i32(l, true) ^ x86_to_i32(l, false).wrapping_shl(1));
        let sum = mixed.iter().fold(0i32, |a, &m| a.wrapping_add(m));
        sum as u32 ^ (x86_to_i32(x, false) as u32).rotate_left(7)
    });
    checked += general("sse_integers", state, probes::sse_integers, sse_integers_reference);
    checked += general("sse_mix", state, probes::sse_mix, sse_mix_reference);
    checked += general("pointer_views", state, probes::pointer_views, |v: u64| {
        v.rotate_left(8) ^ (((v >> 32) & 0xFFFF) ^ 0xFF)
    });
    // Refused: a write through a pointer to data of another type.
    checked += general("pointer_write", state, probes::pointer_write, |v: u64| (v >> 16) & 0xFFFF);
    // 128-bit integers, computed on the GPU as pairs of halves; the
    // references use the CPU's own 128-bit arithmetic, except the product,
    // done by hand in 32-bit pieces.
    checked += general("wide_mul_hash", state, probes::wide_mul_hash, |v: u64| {
        let k = 0x9E37_79B9_7F4A_7C15u64;
        let (a, b) = ([v & 0xFFFF_FFFF, v >> 32], [k & 0xFFFF_FFFF, k >> 32]);
        let (ll, lh, hl, hh) = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]);
        let middle = (ll >> 32) + (lh & 0xFFFF_FFFF) + (hl & 0xFFFF_FFFF);
        let low = (ll & 0xFFFF_FFFF) | middle << 32;
        let high = hh + (lh >> 32) + (hl >> 32) + (middle >> 32);
        high ^ low
    });
    checked += general("wide_signed", state, probes::wide_signed, |v: i64| {
        let wide = i128::from(v) * -3 + i128::MAX / 7;
        let mixed = (wide >> 70) ^ (wide << 5) ^ (wide / -1_000_003);
        mixed as i64 ^ (mixed >> 64) as i64
    });
    checked += general("wide_float", state, probes::wide_float, |v: u64| {
        let wide = u128::from(v) << 40 | u128::from(v >> 7);
        let float = wide as f64 * 0.75;
        let back = float as u128;
        let signed = -(float * 1e10) as i128;
        back as u64 ^ (back >> 64) as u64 ^ signed as u64 ^ ((signed >> 100) as u64).rotate_left(3)
    });
    checked += general("wide_bits", state, probes::wide_bits, |v: u64| {
        let wide = u128::from(v) << 17 ^ u128::from(v).rotate_right(29);
        let counts = wide.leading_zeros() + wide.trailing_zeros() * 7 + wide.count_ones() * 131;
        let swapped = wide.swap_bytes() ^ (wide >> 3).swap_bytes().rotate_left(1);
        u64::from(counts) ^ swapped as u64 ^ (swapped >> 64) as u64
    });
    // Runtime CPU feature detection: the GPU follows the host's answer.
    let ssse3 = is_x86_feature_detected!("ssse3");
    let f16c = is_x86_feature_detected!("f16c");
    checked += general("detect_shuffle", state, probes::detect_shuffle, |v: u32| {
        if ssse3 { v.swap_bytes() } else { v.swap_bytes() ^ 1 }
    });
    checked += general("detect_f16c", state, probes::detect_f16c, |v: u32| {
        let bits = u32::from(half::f16::from_f32(f32::from_bits(v)).to_bits());
        if f16c { bits } else { bits | 0x1_0000 }
    });
    // Held out before, now development code: `half`'s encoding, through
    // F16C where the host has it.
    checked += general_indexed(
        "half_encode",
        state,
        |v: &mut [u16]| probes::half_encode(v, &xs[..v.len()]),
        |i, _, _| half::f16::from_f32(xs[i]).to_bits(),
    );
    checked += general("transmute_halves", state, probes::transmute_halves, |x: u32| x.rotate_left(16));
    checked += general("transmute_struct", state, probes::transmute_struct, |x: u64| {
        let c = u64::from(((x >> 16) as u16).rotate_left(4));
        let d = (x >> 32) & 0x7FFF_FFFF;
        (x >> 8) & 0xFF | (x & 0xFF) << 8 | c << 16 | d << 32
    });
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
