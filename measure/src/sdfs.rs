//! Signed distance fields from `sdfu`, over glam's `Vec3A`, sampled on an
//! `n³` lattice as an application does before meshing them (for example
//! with `fast-surface-nets`).

use cuprite_memory::{Arg, Grid};
use glam::Vec3A;
use rayon::prelude::*;
use sdfu::{Axis, SDF};

use crate::{
    Case, Gpu, Output, BALLS, BALL_BLEND, BALL_RADIUS, PART_BLEND, PART_BODY, PART_BORE,
    PART_BOSS_CENTER, PART_BOSS_RADIUS, PART_ROUNDING, PART_SLOT_OFFSET, PART_SLOT_RADIUS,
    WIRE_EDGES, WIRE_THICKNESS, WIRE_TORUS,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scene {
    /// A single sphere: the least work per point.
    Sphere,
    /// A machined part: rounded box, blended boss, bore and slot.
    Part,
    /// Eight smoothly unioned spheres.
    Metaballs,
    /// Twelve capsules and a torus.
    Wireframe,
}

impl Scene {
    pub const ALL: [Scene; 4] = [
        Scene::Sphere,
        Scene::Part,
        Scene::Metaballs,
        Scene::Wireframe,
    ];

    fn kernel(self) -> &'static str {
        match self {
            Scene::Sphere => "sdf_sphere",
            Scene::Part => "sdf_part",
            Scene::Metaballs => "sdf_metaballs",
            Scene::Wireframe => "sdf_wireframe",
        }
    }
}

fn v(a: [f32; 3]) -> Vec3A {
    Vec3A::from(a)
}

/// Samples `sdf` at every lattice point, x fastest.
fn sample(
    sdf: impl SDF<f32, Vec3A>,
    n: usize,
    min: f32,
    step: f32,
    z_range: std::ops::Range<usize>,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * n * z_range.len());
    for z in z_range {
        for y in 0..n {
            for x in 0..n {
                let p = Vec3A::new(x as f32, y as f32, z as f32) * step + Vec3A::splat(min);
                out.push(sdf.dist(p));
            }
        }
    }
    out
}

/// The scenes, built with sdfu's combinators.
fn sample_scene(
    scene: Scene,
    n: usize,
    min: f32,
    step: f32,
    z_range: std::ops::Range<usize>,
) -> Vec<f32> {
    match scene {
        Scene::Sphere => sample(sdfu::Sphere::new(1.0), n, min, step, z_range),
        Scene::Part => {
            let body = sdfu::Box::<Vec3A, sdfu::Dim3D>::new(v(PART_BODY)).round(PART_ROUNDING);
            let boss = sdfu::Sphere::new(PART_BOSS_RADIUS).translate(v(PART_BOSS_CENTER));
            let bore = sdfu::CappedCylinder::new(PART_BORE[0], PART_BORE[1], Axis::Z);
            let slot =
                sdfu::Cylinder::new(PART_SLOT_RADIUS, Axis::X).translate(v(PART_SLOT_OFFSET));
            let part = body
                .union_smooth(boss, PART_BLEND)
                .subtract(bore)
                .subtract(slot);
            sample(part, n, min, step, z_range)
        }
        Scene::Metaballs => {
            let ball = |i: usize| sdfu::Sphere::new(BALL_RADIUS).translate(v(BALLS[i]));
            let balls = ball(0)
                .union_smooth(ball(1), BALL_BLEND)
                .union_smooth(ball(2), BALL_BLEND)
                .union_smooth(ball(3), BALL_BLEND)
                .union_smooth(ball(4), BALL_BLEND)
                .union_smooth(ball(5), BALL_BLEND)
                .union_smooth(ball(6), BALL_BLEND)
                .union_smooth(ball(7), BALL_BLEND);
            sample(balls, n, min, step, z_range)
        }
        Scene::Wireframe => {
            let edge = |i: usize| {
                sdfu::Line::new(v(WIRE_EDGES[i][0]), v(WIRE_EDGES[i][1]), WIRE_THICKNESS)
            };
            let frame = edge(0)
                .union(edge(1))
                .union(edge(2))
                .union(edge(3))
                .union(edge(4))
                .union(edge(5))
                .union(edge(6))
                .union(edge(7))
                .union(edge(8))
                .union(edge(9))
                .union(edge(10))
                .union(edge(11))
                .union(sdfu::Torus::new(WIRE_TORUS[0], WIRE_TORUS[1]));
            sample(frame, n, min, step, z_range)
        }
    }
}

/// A scene sampled on an `n³` lattice spanning `[min, -min]` on each axis.
#[derive(Clone, Debug)]
pub struct SdfCase {
    pub scene: Scene,
    pub n: usize,
    pub min: f32,
}

impl SdfCase {
    pub fn new(scene: Scene, n: usize) -> Self {
        Self {
            scene,
            n,
            min: -1.5,
        }
    }

    fn step(&self) -> f32 {
        -2.0 * self.min / self.n as f32
    }
}

impl Case for SdfCase {
    fn name(&self) -> String {
        let scene = match self.scene {
            Scene::Sphere => "sphere",
            Scene::Part => "machined part (CSG)",
            Scene::Metaballs => "8 metaballs (smooth union)",
            Scene::Wireframe => "wireframe: 12 capsules + torus",
        };
        format!("sdf {scene}")
    }

    fn size(&self) -> (usize, usize) {
        (self.n * self.n * self.n, 1)
    }

    fn shape(&self) -> String {
        format!("{}^3", self.n)
    }

    fn kernel_name(&self) -> &'static str {
        self.scene.kernel()
    }

    fn cpu(&self) -> Output {
        Output::F32(sample_scene(
            self.scene,
            self.n,
            self.min,
            self.step(),
            0..self.n,
        ))
    }

    /// Slices of the lattice on every core.
    fn cpu_parallel(&self) -> Output {
        let n = self.n;
        let mut out = vec![0f32; n * n * n];
        out.par_chunks_mut(n * n)
            .enumerate()
            .for_each(|(z, slice)| {
                slice.copy_from_slice(&sample_scene(
                    self.scene,
                    n,
                    self.min,
                    self.step(),
                    z..z + 1,
                ));
            });
        Output::F32(out)
    }

    fn gpu(&self, gpu: &mut Gpu) -> Output {
        let kernel = gpu.kernel(self.scene.kernel());
        let points = self.n * self.n * self.n;
        let mut out = vec![0f32; points];
        let mut job = gpu.memory.job().unwrap();
        let output = job.output(&mut out).unwrap();
        let args = [
            Arg::Buffer(output),
            Arg::U32(self.n as u32),
            Arg::F32(self.min),
            Arg::F32(self.step()),
        ];
        job.launch(kernel, Grid::Linear(points as u64), &args)
            .unwrap();
        job.finish().unwrap();
        Output::F32(out)
    }
}
