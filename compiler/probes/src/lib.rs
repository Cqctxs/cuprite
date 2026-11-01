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

/// Not a multiply-add map: the element is multiplied by itself.
pub fn square(values: &mut [u32]) {
    for value in values {
        *value = value.wrapping_mul(*value);
    }
}

/// A branch inside the loop.
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
