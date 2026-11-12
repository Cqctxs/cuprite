//! Grid workloads from real crates, measured on the CPU and the GPU.
//!
//! Each [`Case`] runs three ways:
//! - **cpu**: the crate itself, on one thread;
//! - **cpu_parallel**: the crate's own per-point code on every core, the
//!   strongest CPU version without rewriting the algorithm;
//! - **gpu**: a port of the same code (`kernels/grid.rs`), end to end:
//!   uploads, one launch, and the download.
//!
//! Results are compared bit for bit, so a speedup only counts where the GPU
//! computes exactly what the crate computes.

use std::{
    sync::OnceLock,
    time::{Duration, Instant},
};

use cuprite_memory::{GpuMemory, Kernel, MemoryConfig};

pub mod hashes;
pub mod images;
pub mod noises;
pub mod sdfs;

include!("../kernels/shared.rs");

/// The kernels of `kernels/grid.rs`, compiled to PTX by `build.rs`.
pub const GRID_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/grid.ptx"));

/// The GPU, with a budget large enough for 4096 × 4096 images and maps.
pub struct Gpu {
    pub memory: GpuMemory,
}

impl Gpu {
    pub fn open() -> Self {
        let config = MemoryConfig {
            max_device_bytes: 4 << 30,
            ..MemoryConfig::default()
        };
        Self {
            memory: GpuMemory::open(config).expect("a CUDA GPU"),
        }
    }

    /// A kernel of [`GRID_PTX`], loaded on first use.
    pub fn kernel(&mut self, name: &str) -> Kernel {
        self.memory
            .kernel(GRID_PTX, name)
            .expect("the kernel loads")
    }
}

/// What a case computes.
#[derive(Clone, Debug, PartialEq)]
pub enum Output {
    F64(Vec<f64>),
    F32(Vec<f32>),
    Bytes(Vec<u8>),
}

/// How two outputs differ.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Comparison {
    pub values: usize,
    pub differing: usize,
    /// For floats, the largest difference in units in the last place (0
    /// when only the sign of a zero differs); for bytes, the largest
    /// absolute difference.
    pub largest: u64,
}

impl Comparison {
    pub fn exact(&self) -> bool {
        self.differing == 0
    }
}

/// `f32` bits ordered like the values, so their distance counts units in
/// the last place.
fn ordered32(x: f32) -> i64 {
    let bits = x.to_bits() as i32;
    i64::from(if bits < 0 { i32::MIN - bits } else { bits })
}

fn ulps(a: f64, b: f64) -> u64 {
    let ordered = |x: f64| {
        let bits = x.to_bits() as i64;
        if bits < 0 {
            i64::MIN - bits
        } else {
            bits
        }
    };
    ordered(a).abs_diff(ordered(b))
}

impl Output {
    pub fn compare(&self, other: &Output) -> Comparison {
        match (self, other) {
            (Output::F64(a), Output::F64(b)) => {
                assert_eq!(a.len(), b.len());
                let differences = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits());
                let (differing, largest) = differences.fold((0, 0), |(n, largest), (x, y)| {
                    (n + 1, largest.max(ulps(*x, *y)))
                });
                Comparison {
                    values: a.len(),
                    differing,
                    largest,
                }
            }
            (Output::F32(a), Output::F32(b)) => {
                assert_eq!(a.len(), b.len());
                let differences = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits());
                let (differing, largest) = differences.fold((0, 0), |(n, largest), (x, y)| {
                    (n + 1, largest.max(ordered32(*x).abs_diff(ordered32(*y))))
                });
                Comparison {
                    values: a.len(),
                    differing,
                    largest,
                }
            }
            (Output::Bytes(a), Output::Bytes(b)) => {
                assert_eq!(a.len(), b.len());
                let differences = a.iter().zip(b).filter(|(x, y)| x != y);
                let (differing, largest) = differences.fold((0, 0), |(n, largest), (x, y)| {
                    (n + 1, largest.max(u64::from(x.abs_diff(*y))))
                });
                Comparison {
                    values: a.len(),
                    differing,
                    largest,
                }
            }
            _ => panic!("outputs of different kinds"),
        }
    }
}

/// One grid workload.
pub trait Case {
    /// What it computes, such as "fbm<perlin> 3D, 6 octaves".
    fn name(&self) -> String;
    /// Output width and height.
    fn size(&self) -> (usize, usize);
    /// The size for reports.
    fn shape(&self) -> String {
        let (width, height) = self.size();
        format!("{width}x{height}")
    }
    /// The crate itself, on one thread.
    fn cpu(&self) -> Output;
    /// The crate's own per-point code on every core.
    fn cpu_parallel(&self) -> Output;
    /// The GPU port, end to end.
    fn gpu(&self, gpu: &mut Gpu) -> Output;
    /// The kernel the GPU version launches, to time its first load.
    fn kernel_name(&self) -> &'static str;
}

/// Runs `f` on a single-thread rayon pool, so crates that parallelize with
/// rayon run their code on one thread.
pub fn one_thread<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
    })
    .install(f)
}

/// The median of timed runs after one warm-up: at least 3 samples, and
/// more while they take under a second in total, up to 15. A warm-up over
/// 1.5 s is followed by a single sample.
pub fn time(mut run: impl FnMut()) -> Duration {
    let start = Instant::now();
    run();
    if start.elapsed() > Duration::from_millis(1500) {
        let start = Instant::now();
        run();
        return start.elapsed();
    }
    let mut samples = Vec::new();
    let mut total = Duration::ZERO;
    while samples.len() < 3 || (total < Duration::from_secs(1) && samples.len() < 15) {
        let start = Instant::now();
        run();
        let elapsed = start.elapsed();
        total += elapsed;
        samples.push(elapsed);
    }
    samples.sort();
    samples[samples.len() / 2]
}

/// A field of a value's `Debug` output, which prints floats in their
/// shortest form that parses back to the same value. It reads settings the
/// crates keep private, such as `scale_factor`, exactly as they hold them.
pub fn debug_field<'a>(debug: &'a str, field: &str) -> &'a str {
    let start = debug
        .find(&format!("{field}: "))
        .unwrap_or_else(|| panic!("no `{field}` in {debug}"))
        + field.len()
        + 2;
    let rest = &debug[start..];
    let end = if rest.starts_with('[') {
        rest.find(']').unwrap() + 1
    } else {
        rest.find([',', ' ', '}']).unwrap_or(rest.len())
    };
    &rest[..end]
}
