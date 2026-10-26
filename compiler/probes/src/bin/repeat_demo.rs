//! Times ordinary calls to `cuprite_probes::repeat` and checks each result.
//! Built normally, it runs the CPU code; built through the Cuprite driver
//! with `repeat` rewritten, the same source can run on the GPU.
//!
//! Usage: `repeat_demo <elements> <passes> <samples>`. Prints the first
//! (cold) call and the median of the following (warm) calls.

use std::{env, hint::black_box, time::Instant};

fn main() {
    let args: Vec<usize> = env::args()
        .skip(1)
        .map(|arg| arg.parse().expect("arguments are numbers"))
        .collect();
    let [len, passes, samples] = args[..] else {
        panic!("usage: repeat_demo <elements> <passes> <samples>");
    };
    let passes = u32::try_from(passes).expect("passes fit in u32");

    let input: Vec<u32> = (0..len)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761))
        .collect();
    // Reference computed here, in a crate the driver does not rewrite.
    let mut expected = input.clone();
    for _ in 0..passes {
        for value in &mut expected {
            *value = value.wrapping_mul(3).wrapping_add(7);
        }
    }

    let mut times = Vec::with_capacity(samples + 1);
    for _ in 0..=samples {
        let mut values = input.clone();
        let start = Instant::now();
        cuprite_probes::repeat(black_box(&mut values), passes);
        times.push(start.elapsed());
        assert!(values == expected, "repeat produced a wrong result");
    }
    let cold = times[0];
    let mut warm = times[1..].to_vec();
    warm.sort();
    let warm = warm.get(warm.len() / 2).copied().unwrap_or(cold);
    println!(
        "elements={len} passes={passes} cold_us={:.1} warm_us={:.1}",
        cold.as_secs_f64() * 1e6,
        warm.as_secs_f64() * 1e6
    );
}
