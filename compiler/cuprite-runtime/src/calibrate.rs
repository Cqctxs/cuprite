//! Measures complete calls on both sides over a grid of sizes and pass
//! counts, plus the GPU's cold start, and fits a [`Profile`] to them.

use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use cuprite_memory::{GpuMemory, MapStep, MemoryConfig};

use crate::{
    cpu_map,
    policy::{fit, HostKey, Profile, Sample},
    run_region,
};

/// 1K to 16M elements, every factor of 4, so that interpolation between
/// them stays within one cache level. 16M `u32` (64 MiB) is half the default
/// budget.
pub const LENGTHS: [usize; 8] = [
    1 << 10,
    1 << 12,
    1 << 14,
    1 << 16,
    1 << 18,
    1 << 20,
    1 << 22,
    1 << 24,
];
pub const PASSES: [u32; 4] = [1, 4, 16, 64];

pub struct Calibration {
    pub profile: Profile,
    pub gpu: Vec<Sample>,
    pub cpu: Vec<Sample>,
}

/// Runs the measurements; `log` receives progress lines. Takes a few
/// seconds. Must run in a fresh process, since the first GPU call in the
/// process is what measures cold start.
pub fn calibrate(mut log: impl FnMut(String)) -> Result<Calibration, String> {
    let host = HostKey::current();
    let step = MapStep { mul: 3, add: 7 };

    let mut small = input(LENGTHS[0]);
    let start = Instant::now();
    let mut memory = GpuMemory::open(MemoryConfig::default()).map_err(|e| e.to_string())?;
    run_region(&mut memory, &mut small, step, 1)?;
    let first = start.elapsed();
    let warm = median(5, || {
        let start = Instant::now();
        run_region(&mut memory, &mut small, step, 1).map(|()| start.elapsed())
    })?;
    let cold_us = micros(first.saturating_sub(warm));
    log(format!(
        "{}: cold start {:.1} ms",
        memory.device().name,
        cold_us / 1e3
    ));

    let mut gpu = Vec::new();
    let mut cpu = Vec::new();
    for len in LENGTHS {
        let input = input(len);
        for passes in PASSES {
            let runs = if len as u64 * u64::from(passes) > 1 << 26 {
                3
            } else {
                7
            };
            let mut values = input.clone();
            let gpu_time = median(runs, || {
                values.copy_from_slice(&input);
                let start = Instant::now();
                run_region(&mut memory, &mut values, step, passes).map(|()| start.elapsed())
            })?;
            // Small CPU calls are near the timer's resolution, so each run
            // repeats the call on the same values (which costs the same) and
            // takes the mean.
            let repeats = (1 << 22) / (len as u64 * u64::from(passes)).min(1 << 22);
            let cpu_time = median(runs, || {
                values.copy_from_slice(&input);
                let start = Instant::now();
                for _ in 0..repeats {
                    cpu_map(black_box(&mut values), step, passes);
                }
                Ok(start.elapsed() / repeats as u32)
            })?;
            for (samples, time) in [(&mut gpu, gpu_time), (&mut cpu, cpu_time)] {
                samples.push(Sample {
                    len,
                    passes,
                    micros: micros(time),
                });
            }
            log(format!(
                "{len:>9} elements {passes:>3} passes: gpu {:>10.1} us, cpu {:>10.1} us",
                micros(gpu_time),
                micros(cpu_time)
            ));
        }
    }

    let fitted = |samples: &[Sample], side: &str| {
        fit(samples).ok_or_else(|| format!("cannot fit a {side} model to the measurements"))
    };
    let profile = Profile {
        host,
        device: memory.device().clone(),
        cold_us,
        gpu: fitted(&gpu, "GPU")?,
        cpu: fitted(&cpu, "CPU")?,
    };
    Ok(Calibration { profile, gpu, cpu })
}

fn median(
    runs: usize,
    mut run: impl FnMut() -> Result<Duration, String>,
) -> Result<Duration, String> {
    let mut times = (0..runs).map(|_| run()).collect::<Result<Vec<_>, _>>()?;
    times.sort();
    Ok(times[times.len() / 2])
}

fn micros(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e6
}

fn input(len: usize) -> Vec<u32> {
    (0..len as u32)
        .map(|i| i.wrapping_mul(2_654_435_761))
        .collect()
}
