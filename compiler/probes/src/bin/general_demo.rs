//! Times three general maps on large inputs, for comparing a build through
//! the driver with the GPU off and on:
//!
//! ```text
//! CUPRITE_GPU=0 general_demo    # the CPU bodies
//! CUPRITE_GPU=1 general_demo    # the generated kernels
//! ```
//!
//! Each map runs once to warm up (the GPU's first call also starts CUDA
//! and compiles the kernel), then the best of 5 calls is printed, including
//! transfers. A checksum shows both builds compute the same.

use std::time::Instant;

use cuprite_probes as probes;

fn best(mut call: impl FnMut()) -> f64 {
    call();
    (0..5)
        .map(|_| {
            let start = Instant::now();
            call();
            start.elapsed().as_secs_f64() * 1e3
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let len = 1 << 24;
    let starts: Vec<u32> = (0..len as u32).map(|i| i.wrapping_mul(2_654_435_761) >> 8).collect();
    let mut steps = starts.clone();
    let collatz = best(|| {
        steps.copy_from_slice(&starts);
        probes::collatz_steps(&mut steps);
    });
    let sum: u64 = steps.iter().map(|&s| u64::from(s)).sum();
    println!("collatz_steps  {len} u32: {collatz:8.2} ms  (steps {sum})");

    let mut floats: Vec<f32> = (0..len).map(|i| i as f32 * 0.001).collect();
    let scale = best(|| probes::scale_offset(&mut floats, 0.999, 0.001));
    let sum: f64 = floats.iter().map(|&x| f64::from(x)).sum();
    println!("scale_offset   {len} f32: {scale:8.2} ms  (sum {sum:.6e})");

    let mut doubles: Vec<f64> = (0..len).map(|i| i as f64 * 0.37).collect();
    let root = best(|| probes::soft_root(&mut doubles));
    let sum: f64 = doubles.iter().sum();
    println!("soft_root      {len} f64: {root:8.2} ms  (sum {sum:.6e})");
}
