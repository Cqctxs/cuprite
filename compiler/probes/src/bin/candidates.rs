//! Runs every function the recognizer accepts on many inputs and compares
//! each result with an independent reference that applies the source's
//! operations one by one. Built through the driver with `CUPRITE_AUTO=1`,
//! the calls can run on the GPU.
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

fn main() {
    let mut state = 0x9E37_79B9_7F4A_7C15;
    let mut checked = 0;
    for (name, function, reference) in CASES {
        for len in LENGTHS {
            let input: Vec<u32> = (0..len).map(|_| next(&mut state)).collect();
            let expected: Vec<u32> = input.iter().map(|&x| reference(x)).collect();
            let mut values = input;
            function(&mut values);
            if let Some(index) = values.iter().zip(&expected).position(|(a, b)| a != b) {
                panic!(
                    "{name}, {len} elements: index {index} is {} but should be {}",
                    values[index], expected[index]
                );
            }
            checked += 1;
        }
    }
    println!("{checked} cases match");
}
