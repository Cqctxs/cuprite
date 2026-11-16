//! Ordinary Rust, written without annotations, that the driver inspects.
//! Candidates are shapes a first GPU version could plausibly accept;
//! rejections are shapes it must refuse. Nothing here knows about Cuprite.

use std::sync::atomic::{AtomicU32, Ordering};

use rayon::prelude::*;

// Candidates.

/// The design's first target: an independent in-place map.
pub fn transform(values: &mut [u32]) {
    for value in values {
        *value = (*value).wrapping_mul(3).wrapping_add(7);
    }
}

/// Same map through an explicit `iter_mut()`.
pub fn transform_iter_mut(values: &mut [u32]) {
    for value in values.iter_mut() {
        *value = value.wrapping_mul(3).wrapping_add(7);
    }
}

/// Several steps that compose into one map: `x * 15 + 33`.
pub fn affine_chain(values: &mut [u32]) {
    for value in values {
        *value = value
            .wrapping_mul(3)
            .wrapping_add(7)
            .wrapping_mul(5)
            .wrapping_sub(2);
    }
}

/// Plain operators: wrapping in release builds, panicking in debug ones.
pub fn plain_ops(values: &mut [u32]) {
    for value in values {
        *value = *value * 3 + 7;
    }
}

/// The element added to itself is still a map: `x * 2`.
pub fn double(values: &mut [u32]) {
    for value in values {
        *value = value.wrapping_add(*value);
    }
}

/// Overwrites every element: `x * 0 + 42`.
pub fn fill(values: &mut [u32]) {
    for value in values {
        *value = 42;
    }
}

/// The design's residency target: repeated calls to a visible helper.
pub fn repeat(values: &mut [u32], passes: u32) {
    for _ in 0..passes {
        transform(values);
    }
}

/// Two inputs and a separate output, through iterator adapters.
pub fn add_into(output: &mut [u32], a: &[u32], b: &[u32]) {
    for ((out, x), y) in output.iter_mut().zip(a).zip(b) {
        *out = x.wrapping_add(*y);
    }
}

/// Indexed loop: same map as `transform`, but with bounds checks in MIR.
#[allow(clippy::needless_range_loop)]
pub fn transform_indexed(values: &mut [u32]) {
    for i in 0..values.len() {
        values[i] = values[i].wrapping_mul(3).wrapping_add(7);
    }
}

/// Library-provided parallelism: the body is a closure Rayon calls.
pub fn transform_rayon(values: &mut [u32]) {
    values
        .par_iter_mut()
        .for_each(|value| *value = value.wrapping_mul(3).wrapping_add(7));
}

/// Reduction: excluded from the first version, kept to see its shape.
pub fn wrapping_sum(values: &[u32]) -> u32 {
    values.iter().fold(0, |sum, &value| sum.wrapping_add(value))
}

/// Reduction with an empty-input case.
pub fn minimum(values: &[u32]) -> Option<u32> {
    values.iter().copied().min()
}

/// Neighbor reads into a separate output: reads overlap, writes do not.
pub fn smooth(input: &[u32], output: &mut [u32]) {
    for (out, window) in output.iter_mut().zip(input.windows(3)) {
        *out = window[0] / 4 + window[1] / 2 + window[2] / 4;
    }
}

// Rejections.

/// Each element depends on the previous one.
pub fn prefix_sum(values: &mut [u32]) {
    for i in 1..values.len() {
        values[i] = values[i].wrapping_add(values[i - 1]);
    }
}

/// `+` panics on overflow in debug builds.
pub fn checked_increment(values: &mut [u32]) {
    for value in values {
        *value += 1;
    }
}

/// Division panics when `divisor` is zero.
pub fn divide(values: &mut [u32], divisor: u32) {
    for value in values {
        *value /= divisor;
    }
}

static CALLS: AtomicU32 = AtomicU32::new(0);

/// Hidden effect on shared state.
pub fn counted(values: &mut [u32]) {
    for value in values {
        CALLS.fetch_add(1, Ordering::Relaxed);
        *value = value.wrapping_add(1);
    }
}

/// The body cannot be seen at compile time.
pub fn apply(values: &mut [u32], f: fn(u32) -> u32) {
    for value in values {
        *value = f(*value);
    }
}

/// The element times itself, which the first, multiply-add rule refused.
pub fn square(values: &mut [u32]) {
    for value in values {
        *value = value.wrapping_mul(*value);
    }
}

/// A branch inside the loop, which the first rule refused.
pub fn clamp(values: &mut [u32]) {
    for value in values {
        if *value > 9 {
            *value = 9;
        }
    }
}

/// Division by the element panics when it is zero.
pub fn reciprocal(values: &mut [u32]) {
    for value in values {
        *value = 100 / *value;
    }
}

#[inline(never)]
fn triple(value: u32) -> u32 {
    value.wrapping_mul(3)
}

/// A call to a function the recognizer has no summary for.
pub fn with_helper(values: &mut [u32]) {
    for value in values {
        *value = triple(*value);
    }
}

/// A factor computed before the loop. A real map, but the first recognizer
/// only follows values computed inside the loop body.
pub fn outside_factor(values: &mut [u32]) {
    let factor = 3;
    for value in values {
        *value = value.wrapping_mul(factor);
    }
}

/// Two stores in one iteration still make one map: `(x + 1) * 2`.
pub fn two_stores(values: &mut [u32]) {
    for value in values {
        *value = value.wrapping_add(1);
        *value = value.wrapping_mul(2);
    }
}

// Shapes that look close to a map but are not one, to check that the
// recognizer refuses them.

/// Stops at the first zero, leaving the rest unchanged.
pub fn until_zero(values: &mut [u32]) {
    for value in values {
        if *value == 0 {
            break;
        }
        *value = value.wrapping_mul(3);
    }
}

/// Same map, through an iterator adapter the recognizer has no summary for.
pub fn reversed(values: &mut [u32]) {
    for value in values.iter_mut().rev() {
        *value = value.wrapping_mul(3);
    }
}

/// Two loops, which together are the map `x * 0 + 1`; the first recognizer
/// handles only one.
pub fn two_loops(values: &mut [u32]) {
    for value in values.iter_mut() {
        *value = value.wrapping_mul(0);
    }
    for value in values.iter_mut() {
        *value = value.wrapping_add(1);
    }
}

/// Each element depends on how many came before it.
pub fn numbered(values: &mut [u32]) {
    let mut count = 0u32;
    for value in values {
        *value = value.wrapping_add(count);
        count = count.wrapping_add(1);
    }
}

/// Looks like a map over its parameter, but iterates over an empty slice:
/// the caller's slice must stay unchanged.
#[allow(unused_assignments)]
pub fn reassigned(mut values: &mut [u32]) {
    values = &mut [];
    for value in values {
        *value = value.wrapping_add(1);
    }
}

// Repeated maps (M3): counted loops around calls to maps.

/// A fixed number of passes.
pub fn repeat_fixed(values: &mut [u32]) {
    for _ in 0..4 {
        transform(values);
    }
}

/// Two maps per pass, which compose into `x * 6 + 14`.
pub fn repeat_composed(values: &mut [u32], passes: u32) {
    for _ in 0..passes {
        transform(values);
        double(values);
    }
}

/// The pass number is bound but unused.
pub fn repeat_named(values: &mut [u32], passes: u32) {
    for _pass in 0..passes {
        affine_chain(values);
    }
}

/// Repeats a function that is not a map.
pub fn repeat_helper(values: &mut [u32], passes: u32) {
    for _ in 0..passes {
        with_helper(values);
    }
}

/// Runs one pass fewer than `passes`.
pub fn repeat_from_one(values: &mut [u32], passes: u32) {
    for _ in 1..passes {
        transform(values);
    }
}

/// The work depends on the pass number.
pub fn repeat_after_first(values: &mut [u32], passes: u32) {
    for pass in 0..passes {
        if pass > 0 {
            transform(values);
        }
    }
}

/// Changes the count before the loop.
#[allow(unused_assignments)]
pub fn repeat_reassigned(values: &mut [u32], mut passes: u32) {
    passes = 2;
    for _ in 0..passes {
        transform(values);
    }
}

/// Repeats a repeated map.
pub fn repeat_nested(values: &mut [u32], passes: u32) {
    for _ in 0..passes {
        repeat(values, 2);
    }
}

// General element code (code generation): other element types, scalar
// parameters, floats, branches, inner loops and library methods.

/// Floats and scalar parameters: a linear transform.
pub fn scale_offset(values: &mut [f32], scale: f32, offset: f32) {
    for value in values {
        *value = *value * scale + offset;
    }
}

/// A branch on a float comparison.
pub fn relu(values: &mut [f32]) {
    for value in values.iter_mut() {
        if *value < 0.0 {
            *value = 0.0;
        }
    }
}

/// `f64` library methods.
pub fn soft_root(values: &mut [f64]) {
    for value in values {
        *value = (value.abs() + 1.0).sqrt().min(1e6).floor();
    }
}

/// An inner `while` loop with a branch: steps to reach 1, capped.
/// Dividing by the constant 2 can't panic.
pub fn collatz_steps(values: &mut [u32]) {
    for value in values {
        let mut x = *value;
        let mut steps = 0u32;
        while x > 1 && steps < 500 {
            x = if x % 2 == 0 { x / 2 } else { x.wrapping_mul(3).wrapping_add(1) };
            steps = steps.wrapping_add(1);
        }
        *value = steps;
    }
}

/// Narrow signed elements, a signed division by a constant (which can't
/// overflow), and an `i32` parameter.
pub fn quantize(values: &mut [i16], step: i32) {
    for value in values {
        *value = ((*value as i32) / 4).wrapping_mul(step) as i16;
    }
}

/// Bytes, integer hashing and rotation.
pub fn hash_bytes(values: &mut [u8], seed: u32) {
    for value in values {
        let h = (*value as u32 ^ seed).wrapping_mul(0x9E37_79B1).rotate_left(5);
        *value = (h >> 24) as u8;
    }
}

/// A `match` on ranges.
pub fn levels(values: &mut [u8]) {
    for value in values {
        *value = match *value {
            0..=63 => 0,
            64..=127 => 85,
            128..=191 => 170,
            _ => 255,
        };
    }
}

/// 64-bit integers, a shift by a parameter, and a `usize` parameter.
pub fn mix_wide(values: &mut [i64], shift: u32, bias: usize) {
    for value in values {
        *value = value.wrapping_mul(-3).wrapping_shr(shift) ^ bias as i64;
    }
}

/// A map over floats that `repeat_damp` repeats.
pub fn damp(values: &mut [f32]) {
    for value in values {
        *value = *value * 0.5 + 0.25;
    }
}

/// A repeated map over floats.
pub fn repeat_damp(values: &mut [f32], passes: u32) {
    for _ in 0..passes {
        damp(values);
        relu(values);
    }
}

/// Float remainder has no exact GPU version yet.
pub fn float_rem(values: &mut [f32]) {
    for value in values {
        *value %= 2.5;
    }
}

/// Booleans can't be elements.
pub fn invert(values: &mut [bool]) {
    for value in values {
        *value = !*value;
    }
}

// Calls (code generation): helpers of this crate, generic functions, and
// vector math from `glam`, all inlined into the kernel.

/// Splits a word into its high and low halves: a helper returning a tuple.
fn halves(x: u32) -> (u16, u16) {
    ((x >> 16) as u16, x as u16)
}

/// Swaps the halves of each word, through a tuple-returning helper.
pub fn swap_halves(values: &mut [u32]) {
    for value in values {
        let (high, low) = halves(*value);
        *value = (u32::from(low) << 16) | u32::from(high);
    }
}

/// Adds to a float through a mutable reference.
fn bump(x: &mut f32, by: f32) {
    *x += by;
}

/// Passes the element's reference on to a helper, twice.
pub fn bump_twice(values: &mut [f32]) {
    for value in values {
        bump(value, 1.0);
        bump(value, 0.5);
    }
}

/// A generic helper, instantiated with `f32` and with `u64`.
fn twice<T: std::ops::Add<Output = T> + Copy>(x: T) -> T {
    x + x
}

pub fn generic_twice(values: &mut [f32]) {
    for value in values {
        *value = twice(*value) - 1.0;
    }
}

/// A method with `&self` on a struct built inside the loop.
struct Scale {
    factor: f64,
    offset: f64,
}

impl Scale {
    fn apply(&self, x: f64) -> f64 {
        x * self.factor + self.offset
    }
}

pub fn scaled(values: &mut [f64]) {
    for value in values {
        *value = Scale { factor: 2.5, offset: -1.0 }.apply(*value);
    }
}

/// Recursion can't be inlined.
fn halvings(x: u32) -> u32 {
    if x < 2 { 0 } else { halvings(x / 2).wrapping_add(1) }
}

pub fn count_halvings(values: &mut [u32]) {
    for value in values {
        *value = halvings(*value);
    }
}

/// `glam`: building a vector and taking its length.
pub fn vector_lengths(values: &mut [f32]) {
    for value in values {
        *value = glam::Vec3::new(*value, 1.0, -2.0).length();
    }
}

/// `glam`: normalizing, through operator traits and a method chain.
pub fn normalized_y(values: &mut [f32]) {
    for value in values {
        let v = glam::Vec3::new(*value, *value * 0.5, 1.0);
        *value = v.normalize().y;
    }
}

/// `glam`: a cross product, an associated constant, `+` and a dot product.
pub fn cross_dot(values: &mut [f32]) {
    for value in values {
        let a = glam::Vec3::new(*value, 2.0, -1.0);
        let b = glam::Vec3::Y;
        *value = a.cross(b).dot(a + b);
    }
}

// Held-out validation for calls: the same three computations with
// `ultraviolet`, written after the translator was finished, with no changes
// to it.

pub fn uv_lengths(values: &mut [f32]) {
    for value in values {
        *value = ultraviolet::Vec3::new(*value, 1.0, -2.0).mag();
    }
}

pub fn uv_normalized_y(values: &mut [f32]) {
    for value in values {
        let v = ultraviolet::Vec3::new(*value, *value * 0.5, 1.0);
        *value = v.normalized().y;
    }
}

pub fn uv_cross_dot(values: &mut [f32]) {
    for value in values {
        let a = ultraviolet::Vec3::new(*value, 2.0, -1.0);
        let b = ultraviolet::Vec3::new(0.0, 1.0, 0.0);
        *value = a.cross(b).dot(a + b);
    }
}

// Reading data (code generation): values computed before the loop,
// parameters of plain data, and read-only slices zipped with the slice.

/// Values computed before the loop from parameters.
pub fn normalize_range(values: &mut [f32], low: f32, high: f32) {
    let middle = (low + high) * 0.5;
    let span = high - low;
    for value in values {
        *value = (*value - middle) / span;
    }
}

/// A branch before the loop, on a `bool` parameter.
pub fn conditional_scale(values: &mut [f32], double: bool) {
    let factor = if double { 2.0 } else { 3.0 };
    for value in values {
        *value *= factor;
    }
}

/// Plain data, passed by reference and by value.
#[derive(Clone, Copy)]
pub struct Adjust {
    pub gain: f32,
    pub bias: f32,
    pub clamp: bool,
    pub limit: u8,
}

pub fn adjust(values: &mut [f32], settings: &Adjust) {
    for value in values {
        let x = *value * settings.gain + settings.bias;
        *value = if settings.clamp && x > f32::from(settings.limit) { f32::from(settings.limit) } else { x };
    }
}

pub fn adjust_by_value(values: &mut [f32], settings: Adjust, extra: (f64, i16)) {
    for value in values {
        *value = (f64::from(*value * settings.gain) * extra.0) as f32 + f32::from(extra.1);
    }
}

/// Two input slices zipped with the output.
pub fn add_pairs(output: &mut [u32], a: &[u32], b: &[u32]) {
    for ((out, x), y) in output.iter_mut().zip(a).zip(b) {
        *out = out.wrapping_add(x.wrapping_mul(*y));
    }
}

/// `y = a * x + y`, with `.iter()` on the input.
pub fn saxpy(y: &mut [f32], x: &[f32], a: f32) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += a * xi;
    }
}

/// The input first, and of another type than the output.
pub fn weighted(values: &mut [f64], weights: &[u8]) {
    for (weight, value) in weights.iter().zip(values.iter_mut()) {
        *value *= f64::from(*weight) / 255.0;
    }
}

/// An `sdfu` scene built before the loop, sampled along a line.
pub fn scene_distances(values: &mut [f32]) {
    use sdfu::SDF;
    let scene = sdfu::Sphere::new(1.0)
        .union_smooth(sdfu::Sphere::new(0.5).translate(glam::Vec3A::new(1.0, 0.0, 0.0)), 0.3)
        .subtract(sdfu::Box::new(glam::Vec3A::new(0.25, 2.0, 0.25)));
    for value in values {
        *value = scene.dist(glam::Vec3A::new(*value, 0.25, -0.5));
    }
}

/// A rounded box from parameters, through `sdfu`'s modifiers.
pub fn rounded_box(values: &mut [f32], size: f32, rounding: f32) {
    use sdfu::SDF;
    let scene = sdfu::Box::new(glam::Vec3A::splat(size)).round(rounding);
    for value in values {
        *value = scene.dist(glam::Vec3A::new(*value, *value * 0.5, 0.1));
    }
}

/// Indexing a slice needs a bounds check whose result depends on lengths
/// known only at run time.
pub fn lookup(values: &mut [u8], table: &[u8]) {
    for value in values {
        *value = table[*value as usize];
    }
}

/// Explicit SIMD can't be expressed in the kernel.
#[cfg(target_arch = "x86_64")]
pub fn simd_roots(values: &mut [f32]) {
    use std::arch::x86_64::{_mm_cvtss_f32, _mm_set_ss, _mm_sqrt_ss};
    for value in values {
        // SAFETY: SSE is part of the x86_64 baseline.
        *value = unsafe { _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(*value))) };
    }
}

// Held-out validation for reading data: `palette`, written after the
// translator was finished, with no changes to it.

/// Mixes towards a colour built before the loop from parameters.
pub fn palette_mix(values: &mut [f32], red: f32, green: f32, blue: f32) {
    use palette::Mix;
    let base = palette::LinSrgb::new(0.2f32, 0.4, 0.6);
    let target = palette::LinSrgb::new(red, green, blue);
    for value in values {
        *value = base.mix(target, *value).green;
    }
}

/// Decodes sRGB to linear light.
pub fn palette_linear(values: &mut [f32]) {
    for value in values {
        *value = palette::Srgb::new(*value, 0.5, 0.25).into_linear().red;
    }
}

/// Converts to HSV and takes the saturation.
pub fn palette_saturation(values: &mut [f32]) {
    use palette::FromColor;
    for value in values {
        *value = palette::Hsv::from_color(palette::Srgb::new(*value, 0.3, 0.7)).saturation;
    }
}

/// Blends each element with a second slice's colours.
pub fn palette_blend(values: &mut [f32], others: &[f32]) {
    use palette::Mix;
    let tint = palette::LinSrgb::new(1.0f32, 0.5, 0.0);
    for (value, other) in values.iter_mut().zip(others) {
        *value = palette::LinSrgb::new(*value, *other, 0.5).mix(tint, 0.25).red;
    }
}

// Checks (error flag): code whose checks can fail at run time. The GPU
// makes the same checks; when one fails, the CPU runs the code and panics
// where it always would.

/// An index into a table that wraps around its length: a remainder by a
/// length that may be zero, then a bounds check.
pub fn lookup_wrapped(values: &mut [f32], table: &[f32]) {
    for value in values {
        let index = (value.abs() * 16.0) as usize % table.len();
        *value *= table[index];
    }
}

/// Division and remainder by a parameter: zero and `MIN / -1` panic.
pub fn div_mod(values: &mut [i32], divisor: i32) {
    for value in values {
        *value = *value / divisor + *value % divisor;
    }
}

/// `f32::clamp` asserts its bounds are in order.
pub fn clamp_range(values: &mut [f32], low: f32, high: f32) {
    for value in values {
        *value = value.clamp(low, high);
    }
}

/// `+=` on bytes overflows, and so panics, in debug builds.
pub fn add_bytes(values: &mut [u8], amount: u8) {
    for value in values {
        *value += amount;
    }
}

/// `*` on 64-bit integers: the widest overflow check.
pub fn square_wide(values: &mut [i64]) {
    for value in values {
        *value *= *value;
    }
}

/// An explicit assertion.
pub fn asserted_halves(values: &mut [u32]) {
    for value in values {
        assert!(*value % 2 == 0, "odd value {value}");
        *value /= 2;
    }
}

// Held-out validation for checks: `strength_reduce` and `num-integer`,
// written after the error flag was finished, with no changes to it.

/// A divisor reduced before the loop (which panics for zero), then used.
pub fn sr_divide(values: &mut [u32], divisor: u32) {
    let reduced = strength_reduce::StrengthReducedU32::new(divisor);
    for value in values {
        *value = *value / reduced;
    }
}

/// The same with 64-bit division and remainder.
pub fn sr_div_rem(values: &mut [u64], divisor: u64) {
    let reduced = strength_reduce::StrengthReducedU64::new(divisor);
    for value in values {
        *value = *value / reduced + *value % reduced;
    }
}

/// Floored division, which panics for a zero divisor.
pub fn ni_div_floor(values: &mut [i32], divisor: i32) {
    for value in values {
        *value = num_integer::div_floor(*value, divisor);
    }
}

/// Greatest common divisors, with a loop inside.
pub fn ni_gcd(values: &mut [u32], other: u32) {
    for value in values {
        *value = num_integer::gcd(*value, other);
    }
}

/// Integer square roots.
pub fn ni_sqrt(values: &mut [u64]) {
    for value in values {
        *value = num_integer::Roots::sqrt(value);
    }
}

/// Number of calls `counted` has made, for tests.
pub fn counted_calls() -> u32 {
    CALLS.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> Vec<u32> {
        vec![0, 1, 2, u32::MAX - 1, u32::MAX]
    }

    fn map(value: u32) -> u32 {
        value.wrapping_mul(3).wrapping_add(7)
    }

    #[test]
    fn maps_agree() {
        let expected: Vec<u32> = input().into_iter().map(map).collect();
        for f in [
            transform,
            transform_iter_mut,
            transform_indexed,
            transform_rayon,
        ] {
            let mut values = input();
            f(&mut values);
            assert_eq!(values, expected);
        }
        let mut values = input();
        apply(&mut values, map);
        assert_eq!(values, expected);
    }

    fn mapped(f: fn(&mut [u32]), mut values: Vec<u32>) -> Vec<u32> {
        f(&mut values);
        values
    }

    #[test]
    fn composed_maps() {
        let chain = |v: u32| v.wrapping_mul(15).wrapping_add(33);
        let expected: Vec<u32> = input().into_iter().map(chain).collect();
        assert_eq!(mapped(affine_chain, input()), expected);

        let expected: Vec<u32> = input().into_iter().map(|v| v.wrapping_mul(2)).collect();
        assert_eq!(mapped(double, input()), expected);
        assert_eq!(mapped(fill, input()), [42; 5]);
        // Small inputs: `plain_ops` panics on overflow in debug builds.
        assert_eq!(mapped(plain_ops, vec![0, 1, 2]), [7, 10, 13]);
    }

    #[test]
    fn single_slice_rejections_behave_like_ordinary_rust() {
        assert_eq!(mapped(square, vec![3, 1 << 16]), [9, 0]);
        assert_eq!(mapped(clamp, vec![3, 10, 9]), [3, 9, 9]);
        assert_eq!(mapped(reciprocal, vec![1, 3, 200]), [100, 33, 0]);
        assert_eq!(mapped(with_helper, vec![1, u32::MAX]), [3, u32::MAX - 2]);
        assert_eq!(mapped(outside_factor, vec![1, 2]), [3, 6]);
        assert_eq!(mapped(two_stores, vec![0, u32::MAX]), [2, 0]);
        assert_eq!(mapped(until_zero, vec![1, 0, 2]), [3, 0, 2]);
        assert_eq!(mapped(reversed, vec![1, 2]), [3, 6]);
        assert_eq!(mapped(two_loops, vec![5, 6]), [1, 1]);
        assert_eq!(mapped(numbered, vec![10, 10, 10]), [10, 11, 12]);
        assert_eq!(mapped(reassigned, vec![1, 2]), [1, 2]);
    }

    fn passes(f: fn(u32) -> u32, passes: u32, values: Vec<u32>) -> Vec<u32> {
        values
            .into_iter()
            .map(|mut v| {
                for _ in 0..passes {
                    v = f(v);
                }
                v
            })
            .collect()
    }

    #[test]
    fn repeated_maps_behave_like_ordinary_rust() {
        let mut values = input();
        repeat_fixed(&mut values);
        assert_eq!(values, passes(map, 4, input()));

        let composed = |v| map(v).wrapping_mul(2);
        for count in [0, 1, 3] {
            let mut values = input();
            repeat_composed(&mut values, count);
            assert_eq!(values, passes(composed, count, input()));
        }

        let chain = |v: u32| v.wrapping_mul(15).wrapping_add(33);
        let mut values = input();
        repeat_named(&mut values, 3);
        assert_eq!(values, passes(chain, 3, input()));

        let mut values = input();
        repeat_helper(&mut values, 2);
        assert_eq!(values, passes(|v| v.wrapping_mul(3), 2, input()));

        let mut values = input();
        repeat_from_one(&mut values, 3);
        assert_eq!(values, passes(map, 2, input()));
        let mut values = input();
        repeat_from_one(&mut values, 0);
        assert_eq!(values, input());

        let mut values = input();
        repeat_after_first(&mut values, 3);
        assert_eq!(values, passes(map, 2, input()));

        let mut values = input();
        repeat_reassigned(&mut values, 5);
        assert_eq!(values, passes(map, 2, input()));

        let mut values = input();
        repeat_nested(&mut values, 3);
        assert_eq!(values, passes(map, 6, input()));
    }

    #[test]
    fn repeat_applies_transform_each_pass() {
        let mut values = input();
        repeat(&mut values, 3);
        let expected: Vec<u32> = input().into_iter().map(|v| map(map(map(v)))).collect();
        assert_eq!(values, expected);
    }

    #[test]
    fn add_into_stops_at_the_shortest_slice() {
        let mut output = [9; 4];
        add_into(&mut output, &[1, u32::MAX, 3], &[1, 2, 3, 4]);
        assert_eq!(output, [2, 1, 6, 9]);
    }

    #[test]
    fn reductions() {
        assert_eq!(wrapping_sum(&[u32::MAX, 2]), 1);
        assert_eq!(wrapping_sum(&[]), 0);
        assert_eq!(minimum(&[5, 3, 9]), Some(3));
        assert_eq!(minimum(&[]), None);
    }

    #[test]
    fn smooth_weights_neighbors() {
        let mut output = [0; 2];
        smooth(&[4, 8, 12, 16], &mut output);
        assert_eq!(output, [8, 12]);
    }

    #[test]
    fn rejections_behave_like_ordinary_rust() {
        let mut values = vec![1, 2, 3];
        prefix_sum(&mut values);
        assert_eq!(values, [1, 3, 6]);

        let mut values = vec![1, 2];
        checked_increment(&mut values);
        assert_eq!(values, [2, 3]);

        let mut values = vec![9, 10];
        divide(&mut values, 3);
        assert_eq!(values, [3, 3]);

        let before = counted_calls();
        let mut values = vec![1, 2];
        counted(&mut values);
        assert_eq!(values, [2, 3]);
        assert_eq!(counted_calls() - before, 2);
    }

    #[test]
    #[should_panic]
    fn divide_by_zero_panics() {
        divide(&mut [1], 0);
    }
}
