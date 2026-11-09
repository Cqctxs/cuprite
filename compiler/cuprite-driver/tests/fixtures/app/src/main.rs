//! `cuprite-fixture-app <len> <passes>`: runs the library's simulation and
//! its own map over `0..len`, then prints a checksum and a value from the
//! build script.

use rayon::prelude::*;

fn mix(values: &mut [u32]) {
    for value in values.iter_mut() {
        *value = value
            .wrapping_mul(7)
            .wrapping_add(fixture_macros::answer!());
    }
}

fn main() {
    let args: Vec<u64> = std::env::args()
        .skip(1)
        .map(|arg| arg.parse().expect("arguments are numbers"))
        .collect();
    let [len, passes] = args[..] else {
        panic!("usage: cuprite-fixture-app <len> <passes>");
    };
    let mut values: Vec<u32> = (0..len as u32).collect();
    fixture_lib::simulate(&mut values, passes as u32);
    mix(&mut values);
    let checksum: u64 = values.par_iter().map(|&value| u64::from(value)).sum();
    println!("{checksum} {}", env!("FIXTURE_BUILT"));
}
