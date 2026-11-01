//! Runs every function the recognizer accepts on many inputs and compares
//! each result with an independent reference that applies the source's
//! operations one by one, pass by pass. Built through the driver with
//! `CUPRITE_AUTO=1`, the calls can run on the GPU.
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
    println!("{checked} cases match");
}
