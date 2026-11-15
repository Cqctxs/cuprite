//! Gravitational accelerations with `particular`: brute force, every
//! particle against every massive particle.

use std::time::Duration;

use cuprite_memory::{Arg, Grid};
use particular::prelude::*;

use crate::{time, Case, Gpu, Output};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Precision {
    F32,
    F64,
}

#[derive(Clone, Debug)]
pub struct NbodyCase {
    pub precision: Precision,
    pub dims: usize,
    pub n: usize,
    /// Particles with mass; the rest are massless, so they're pulled but
    /// pull nothing, like test particles around a few stars.
    pub massive: usize,
    /// `BruteForceSoftenedScalar` when nonzero, else `BruteForceScalar`.
    pub softening: f64,
    /// Every this many particles sits exactly on the one before it, which
    /// exercises `force_scalar`'s zero-distance branch. 0 for never.
    pub coincident_every: usize,
}

impl NbodyCase {
    pub fn new(precision: Precision, dims: usize, n: usize) -> Self {
        Self {
            precision,
            dims,
            n,
            massive: n,
            softening: 0.0,
            coincident_every: 0,
        }
    }

    /// Deterministic positions in `[-1, 1]` and gravitational parameters
    /// around `1 / n`; the first `massive` particles have mass.
    fn bodies(&self) -> Vec<(Vec<f64>, f64)> {
        let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ self.n as u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut bodies: Vec<(Vec<f64>, f64)> = (0..self.n)
            .map(|i| {
                let position = (0..self.dims).map(|_| next() * 2.0 - 1.0).collect();
                let mu = if i < self.massive {
                    (0.5 + next()) / self.n as f64
                } else {
                    0.0
                };
                (position, mu)
            })
            .collect();
        if self.coincident_every > 0 {
            for i in (self.coincident_every..self.n).step_by(self.coincident_every) {
                bodies[i].0 = bodies[i - 1].0.clone();
            }
        }
        bodies
    }

    fn kernel(&self) -> &'static str {
        match (self.precision, self.dims) {
            (Precision::F32, 3) => "nbody_f32_3d",
            (Precision::F64, 3) => "nbody_f64_3d",
            (Precision::F32, 2) => "nbody_f32_2d",
            other => panic!("no n-body kernel for {other:?}"),
        }
    }

    /// The crate's other CPU methods, which compute approximately the same
    /// accelerations: SIMD brute force with an approximate reciprocal
    /// square root, and Barnes-Hut. Timed, not compared.
    pub fn alternatives(&self) -> Vec<(&'static str, Duration)> {
        if (self.precision, self.dims, self.softening) != (Precision::F32, 3, 0.0) {
            return Vec::new();
        }
        let bodies = f32_bodies::<3>(&self.bodies());
        vec![
            (
                "parallel BruteForceSIMD<8>",
                time(|| {
                    drop(
                        bodies
                            .iter()
                            .accelerations(&mut parallel::BruteForceSIMD::<8>)
                            .collect::<Vec<_>>(),
                    )
                }),
            ),
            (
                "parallel BarnesHut theta 0.5",
                time(|| {
                    let mut method = parallel::BarnesHut { theta: 0.5f32 };
                    drop(bodies.iter().accelerations(&mut method).collect::<Vec<_>>())
                }),
            ),
        ]
    }

    /// Runs `particular` sequentially or with its parallel module.
    fn run(&self, in_parallel: bool) -> Output {
        let bodies = self.bodies();
        let softened = self.softening != 0.0;
        macro_rules! accelerations {
            ($bodies:expr, $s:ty) => {{
                let bodies = $bodies;
                let softening = self.softening as $s;
                let accelerations: Vec<_> = match (in_parallel, softened) {
                    (false, false) => bodies
                        .iter()
                        .accelerations(&mut sequential::BruteForceScalar)
                        .collect(),
                    (false, true) => bodies
                        .iter()
                        .accelerations(&mut sequential::BruteForceSoftenedScalar { softening })
                        .collect(),
                    (true, false) => bodies
                        .iter()
                        .accelerations(&mut parallel::BruteForceScalar)
                        .collect(),
                    (true, true) => bodies
                        .iter()
                        .accelerations(&mut parallel::BruteForceSoftenedScalar { softening })
                        .collect(),
                };
                accelerations.into_iter().flatten().collect::<Vec<$s>>()
            }};
        }
        match (self.precision, self.dims) {
            (Precision::F32, 3) => Output::F32(accelerations!(f32_bodies::<3>(&bodies), f32)),
            (Precision::F32, 2) => Output::F32(accelerations!(f32_bodies::<2>(&bodies), f32)),
            (Precision::F64, 3) => Output::F64(accelerations!(f64_bodies::<3>(&bodies), f64)),
            other => panic!("no n-body case for {other:?}"),
        }
    }
}

fn f32_bodies<const D: usize>(bodies: &[(Vec<f64>, f64)]) -> Vec<([f32; D], f32)> {
    bodies
        .iter()
        .map(|(p, mu)| (std::array::from_fn(|k| p[k] as f32), *mu as f32))
        .collect()
}

fn f64_bodies<const D: usize>(bodies: &[(Vec<f64>, f64)]) -> Vec<([f64; D], f64)> {
    bodies
        .iter()
        .map(|(p, mu)| (std::array::from_fn(|k| p[k]), *mu))
        .collect()
}

impl Case for NbodyCase {
    fn name(&self) -> String {
        let precision = match self.precision {
            Precision::F32 => "f32",
            Precision::F64 => "f64",
        };
        let mut name = format!("n-body {}D {precision}", self.dims);
        if self.massive < self.n {
            name += &format!(", {} massive", self.massive);
        }
        if self.softening != 0.0 {
            name += ", softened";
        }
        name
    }

    fn size(&self) -> (usize, usize) {
        (self.n, 1)
    }

    fn shape(&self) -> String {
        format!("{} bodies", self.n)
    }

    fn kernel_name(&self) -> &'static str {
        self.kernel()
    }

    fn cpu(&self) -> Output {
        self.run(false)
    }

    /// particular's `parallel` module: the same per-particle sums, on rayon.
    fn cpu_parallel(&self) -> Output {
        self.run(true)
    }

    fn gpu(&self, gpu: &mut Gpu) -> Output {
        let bodies = self.bodies();
        let kernel = gpu.kernel(self.kernel());
        let n = self.n as u32;
        macro_rules! launch {
            ($s:ty, $arg:path, $output:path) => {{
                // `particular` keeps massive particles in their original
                // order, after converting each to the case's precision.
                let affected: Vec<$s> = bodies
                    .iter()
                    .flat_map(|b| b.0.iter().map(|&c| c as $s))
                    .collect();
                let heavy = bodies.iter().filter(|b| b.1 as $s != 0.0);
                let positions: Vec<$s> = heavy
                    .clone()
                    .flat_map(|b| b.0.iter().map(|&c| c as $s))
                    .collect();
                let masses: Vec<$s> = heavy.map(|b| b.1 as $s).collect();
                let massive_n = masses.len() as u32;
                let mut out = vec![0 as $s; affected.len()];
                let mut job = gpu.memory.job().unwrap();
                let affected = job.input(&affected).unwrap();
                let positions = job.input(&positions).unwrap();
                let masses = job.input(&masses).unwrap();
                let output = job.output(&mut out).unwrap();
                let args = [
                    Arg::Buffer(output),
                    Arg::Buffer(affected),
                    Arg::U32(n),
                    Arg::Buffer(positions),
                    Arg::Buffer(masses),
                    Arg::U32(massive_n),
                    $arg(self.softening as $s),
                ];
                job.launch(kernel, Grid::Linear(u64::from(n)), &args)
                    .unwrap();
                job.finish().unwrap();
                $output(out)
            }};
        }
        match self.precision {
            Precision::F32 => launch!(f32, Arg::F32, Output::F32),
            Precision::F64 => launch!(f64, Arg::F64, Output::F64),
        }
    }
}
