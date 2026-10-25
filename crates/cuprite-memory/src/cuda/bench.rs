//! Residency measurement. Run on a GPU host with:
//!
//! ```text
//! cargo test --release -p cuprite-memory --features cuda residency -- --ignored --nocapture
//! ```
//!
//! Every strategy applies a kernel `reps` times to the same input and is
//! checked against the CPU result before it is timed. Times are wall-clock,
//! from CPU input to completed CPU output, median of several samples after
//! one warm-up run (which absorbs PTX compilation and fills the pool).

use std::{
    hint::black_box,
    sync::Arc,
    time::{Duration, Instant},
};

use cudarc::driver::{CudaFunction, CudaStream};
use rayon::prelude::*;

use super::{
    fixture::{add_seven, input, launch_in_session, launch_on, lcg64, lcg_step, LCG_ROUNDS},
    CudaBackend,
};
use crate::{backend::MemoryBackend, manager::Manager, MemoryConfig};

struct Workload {
    name: &'static str,
    function: CudaFunction,
    /// One CPU pass over a slice, equivalent to one kernel launch.
    cpu_pass: fn(&mut [u32]),
    /// Relative CPU cost per element, used only to pick sample counts.
    weight: usize,
    lengths: &'static [usize],
    repetitions: &'static [usize],
}

fn add_seven_pass(values: &mut [u32]) {
    for value in values {
        *value = value.wrapping_add(7);
    }
}

fn lcg_pass(values: &mut [u32]) {
    // Rounds inside small chunks: the chunk stays in registers and the
    // innermost loop vectorizes, which is the strongest simple CPU form.
    for chunk in values.chunks_mut(16) {
        for _ in 0..LCG_ROUNDS {
            for value in chunk.iter_mut() {
                *value = lcg_step(*value);
            }
        }
    }
}

fn samples(cost: usize) -> usize {
    if cost > 50_000_000 {
        3
    } else {
        9
    }
}

fn median(mut times: Vec<Duration>) -> Duration {
    times.sort();
    times[times.len() / 2]
}

/// Check `run` against `expected`, then time it on a working copy.
fn measure(
    base: &[u32],
    expected: &[u32],
    samples: usize,
    mut run: impl FnMut(&mut [u32]),
) -> Duration {
    let mut check = base.to_vec();
    run(&mut check);
    assert_eq!(check, expected, "strategy disagrees with the CPU result");

    let mut values = base.to_vec();
    run(&mut values); // Warm-up.
    let times = (0..samples)
        .map(|_| {
            let start = Instant::now();
            run(&mut values);
            start.elapsed()
        })
        .collect();
    median(times)
}

fn show(duration: Duration) -> String {
    let seconds = duration.as_secs_f64();
    if seconds < 1e-3 {
        format!("{:.1} us", seconds * 1e6)
    } else if seconds < 1.0 {
        format!("{:.2} ms", seconds * 1e3)
    } else {
        format!("{seconds:.2} s")
    }
}

fn ratio(numerator: Duration, denominator: Duration) -> String {
    format!(
        "{:.2}x",
        numerator.as_secs_f64() / denominator.as_secs_f64()
    )
}

fn gigabytes_per_second(bytes: usize, duration: Duration) -> String {
    format!("{:.1}", bytes as f64 / duration.as_secs_f64() / 1e9)
}

fn region_table(manager: &mut Manager<CudaBackend>, stream: &Arc<CudaStream>, workload: &Workload) {
    println!(
        "\n{}: complete region, CPU input -> completed CPU output (median)\n",
        workload.name
    );
    println!(
        "{:>10} {:>5} | {:>10} {:>10} {:>10} {:>10} {:>12} {:>10} | {:>11} {:>11}",
        "elements",
        "reps",
        "cpu 1t",
        "cpu rayon",
        "raw gpu",
        "resident",
        "+alloc/free",
        "per-op",
        "per-op/res",
        "rayon/res"
    );
    let function = &workload.function;
    let pass = workload.cpu_pass;
    for &len in workload.lengths {
        let base = input(len);
        let mut raw_buffer = stream.alloc_zeros::<u32>(len).unwrap();
        for &reps in workload.repetitions {
            let mut expected = base.clone();
            for _ in 0..reps {
                pass(&mut expected);
            }
            let samples = samples(len * reps * workload.weight);

            let cpu = measure(&base, &expected, samples, |values| {
                for _ in 0..reps {
                    pass(values);
                    // Keep the compiler from merging passes into one.
                    black_box(&mut *values);
                }
            });

            let parallel = measure(&base, &expected, samples, |values| {
                for _ in 0..reps {
                    values.par_chunks_mut(1 << 14).for_each(pass);
                    black_box(&mut *values);
                }
            });

            // Plain cudarc, no staging copies, preallocated buffer.
            let raw = measure(&base, &expected, samples, |values| {
                stream.memcpy_htod(&*values, &mut raw_buffer).unwrap();
                for _ in 0..reps {
                    launch_on(stream, function, &mut raw_buffer);
                }
                stream.memcpy_dtoh(&raw_buffer, values).unwrap();
                stream.synchronize().unwrap();
            });

            let resident = measure(&base, &expected, samples, |values| {
                let mut session = manager.begin_mut(values).unwrap();
                for _ in 0..reps {
                    launch_in_session(&mut session, function);
                }
                session.finish().unwrap();
            });

            // Same, but with no reuse: `trim` frees the pooled device buffer
            // and host staging, so every session allocates both afresh.
            let allocating = measure(&base, &expected, samples, |values| {
                manager.trim();
                let mut session = manager.begin_mut(values).unwrap();
                for _ in 0..reps {
                    launch_in_session(&mut session, function);
                }
                session.finish().unwrap();
            });

            // No residency: upload and download around every operation.
            let per_op = measure(&base, &expected, samples, |values| {
                for _ in 0..reps {
                    let mut session = manager.begin_mut(values).unwrap();
                    launch_in_session(&mut session, function);
                    session.finish().unwrap();
                }
            });

            println!(
                "{:>10} {:>5} | {:>10} {:>10} {:>10} {:>10} {:>12} {:>10} | {:>11} {:>11}",
                len,
                reps,
                show(cpu),
                show(parallel),
                show(raw),
                show(resident),
                show(allocating),
                show(per_op),
                ratio(per_op, resident),
                ratio(parallel, resident),
            );
        }
    }
}

fn phase_table(
    manager: &mut Manager<CudaBackend>,
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
) {
    println!("\nPhases of one resident add_seven session with 8 kernels (median)\n");
    println!(
        "{:>10} | {:>10} {:>10} {:>10} | {:>10} {:>10} {:>10} | {:>8} {:>8}",
        "elements",
        "host copy",
        "raw up",
        "begin_mut",
        "8 kernels",
        "raw down",
        "finish",
        "up GB/s",
        "down GB/s"
    );
    for len in [1, 4096, 65_536, 1_048_576, 16_777_216] {
        let base = input(len);
        let bytes = len * size_of::<u32>();
        let mut raw_buffer = stream.alloc_zeros::<u32>(len).unwrap();
        // Reused like a session's staging, so copies hit touched pages.
        let mut staging: Vec<u32> = Vec::with_capacity(len);
        let mut phases: [Vec<Duration>; 6] = Default::default();
        for sample in 0..=9 {
            let mut values = base.clone();

            let start = Instant::now();
            staging.clear();
            staging.extend_from_slice(&values);
            black_box(&mut staging);
            let host_copy = start.elapsed();

            let start = Instant::now();
            stream.memcpy_htod(&values, &mut raw_buffer).unwrap();
            stream.synchronize().unwrap();
            let raw_up = start.elapsed();

            let start = Instant::now();
            stream.memcpy_dtoh(&raw_buffer, &mut staging).unwrap();
            stream.synchronize().unwrap();
            let raw_down = start.elapsed();

            let start = Instant::now();
            let mut session = manager.begin_mut(&mut values).unwrap();
            let begin = start.elapsed();

            let start = Instant::now();
            for _ in 0..8 {
                launch_in_session(&mut session, function);
            }
            session.write_parts().unwrap().0.synchronize().unwrap();
            let kernels = start.elapsed();

            let start = Instant::now();
            session.finish().unwrap();
            let finish = start.elapsed();

            if sample > 0 {
                // Sample 0 is the warm-up.
                for (phase, time) in phases
                    .iter_mut()
                    .zip([host_copy, raw_up, begin, kernels, raw_down, finish])
                {
                    phase.push(time);
                }
            }
        }
        let [host_copy, raw_up, begin, kernels, raw_down, finish] = phases.map(median);
        println!(
            "{:>10} | {:>10} {:>10} {:>10} | {:>10} {:>10} {:>10} | {:>8} {:>8}",
            len,
            show(host_copy),
            show(raw_up),
            show(begin),
            show(kernels),
            show(raw_down),
            show(finish),
            gigabytes_per_second(bytes, raw_up),
            gigabytes_per_second(bytes, raw_down),
        );
    }
}

#[test]
#[ignore = "benchmark: run in release with --ignored --nocapture"]
fn residency() {
    let config = MemoryConfig::default();
    let mut manager = Manager::new(CudaBackend::open(config).unwrap(), config);
    let stream = manager.backend.stream.clone();
    println!("\nrayon threads: {}", rayon::current_num_threads());

    let workloads = [
        Workload {
            name: "add_seven (memory-bound)",
            function: add_seven(&stream),
            cpu_pass: add_seven_pass,
            weight: 1,
            lengths: &[1, 257, 4096, 65_536, 1_048_576, 16_777_216],
            repetitions: &[1, 8, 32, 128],
        },
        Workload {
            name: "lcg64 (compute-bound, 64 multiply-adds per element)",
            function: lcg64(&stream),
            cpu_pass: lcg_pass,
            weight: 16,
            lengths: &[4096, 65_536, 1_048_576, 16_777_216],
            repetitions: &[1, 8],
        },
    ];
    for workload in &workloads {
        region_table(&mut manager, &stream, workload);
    }
    phase_table(&mut manager, &stream, &workloads[0].function);

    let stats = manager.stats();
    println!(
        "\nManager totals: {} allocations, {} pool hits, {} uploads, {} downloads",
        stats.device_allocations, stats.pool_hits, stats.uploads, stats.downloads
    );
}
