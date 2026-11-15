//! Brute-force k-nearest neighbours with `linfa-nn`'s `LinearSearch` and
//! squared Euclidean distance: every query against every data point.

use cuprite_memory::{Arg, Grid};
use linfa_nn::{distance::L2Dist, LinearSearch, NearestNeighbour};
use ndarray::Array2;
use rayon::prelude::*;

use crate::{Case, Gpu, Output};

#[derive(Clone, Debug)]
pub struct KnnCase {
    pub queries: usize,
    pub points: usize,
    pub dims: usize,
    pub k: usize,
}

impl KnnCase {
    /// Uniform points in the unit cube, queries likewise. Continuous random
    /// values make tied distances among the nearest `k` vanishingly rare;
    /// linfa's heap breaks ties in its own order.
    fn data(&self) -> (Array2<f32>, Array2<f32>) {
        let mut state = 0xd1b5_4a32_d192_ed03u64 ^ (self.points * 7 + self.dims) as u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        };
        let points = Array2::from_shape_fn((self.points, self.dims), |_| next());
        let queries = Array2::from_shape_fn((self.queries, self.dims), |_| next());
        (points, queries)
    }

    fn run(&self, in_parallel: bool) -> Output {
        let (points, queries) = self.data();
        let index = LinearSearch::new().from_batch(&points, L2Dist).unwrap();
        let nearest = |q: usize| -> Vec<u32> {
            index
                .k_nearest(queries.row(q), self.k)
                .unwrap()
                .into_iter()
                .map(|(_, i)| i as u32)
                .collect()
        };
        let all: Vec<Vec<u32>> = if in_parallel {
            (0..self.queries).into_par_iter().map(nearest).collect()
        } else {
            (0..self.queries).map(nearest).collect()
        };
        Output::Bytes(
            all.into_iter()
                .flatten()
                .flat_map(u32::to_le_bytes)
                .collect(),
        )
    }
}

impl KnnCase {
    /// The GPU's selection on every CPU core: the same distances, keeping a
    /// sorted list of the best `k` instead of a heap of every point. It
    /// separates what the GPU gains from what the selection method gains.
    pub fn insertion_parallel(&self) -> Output {
        let (points, queries) = self.data();
        let k = self.k.min(self.points);
        let nearest: Vec<Vec<u32>> = (0..self.queries)
            .into_par_iter()
            .map(|q| {
                let query = queries.row(q);
                let mut best: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
                for (i, point) in points.rows().into_iter().enumerate() {
                    let mut distance = 0.0f32;
                    for (a, b) in query.iter().zip(point) {
                        let diff = a - b;
                        distance += diff * diff;
                    }
                    if best.len() < k || distance < best[k - 1].0 {
                        let at = best.partition_point(|(d, _)| *d <= distance);
                        best.insert(at, (distance, i as u32));
                        best.truncate(k);
                    }
                }
                best.into_iter().map(|(_, i)| i).collect()
            })
            .collect();
        Output::Bytes(
            nearest
                .into_iter()
                .flatten()
                .flat_map(u32::to_le_bytes)
                .collect(),
        )
    }
}

impl Case for KnnCase {
    fn name(&self) -> String {
        format!("linfa-nn {}-nearest, {} dims", self.k, self.dims)
    }

    fn size(&self) -> (usize, usize) {
        (self.queries, self.points)
    }

    fn shape(&self) -> String {
        format!("{} queries x {} points", self.queries, self.points)
    }

    fn kernel_name(&self) -> &'static str {
        "knn_f32"
    }

    fn cpu(&self) -> Output {
        self.run(false)
    }

    /// Queries on every core, sharing one index.
    fn cpu_parallel(&self) -> Output {
        self.run(true)
    }

    fn gpu(&self, gpu: &mut Gpu) -> Output {
        let (points, queries) = self.data();
        let kernel = gpu.kernel("knn_f32");
        let k = self.k.min(self.points);
        let mut out = vec![0u32; self.queries * k];
        let mut job = gpu.memory.job().unwrap();
        let queries_buffer = job.input(queries.as_slice().unwrap()).unwrap();
        let points_buffer = job.input(points.as_slice().unwrap()).unwrap();
        let output = job.output(&mut out).unwrap();
        let args = [
            Arg::Buffer(output),
            Arg::Buffer(queries_buffer),
            Arg::U32(self.queries as u32),
            Arg::Buffer(points_buffer),
            Arg::U32(self.points as u32),
            Arg::U32(self.dims as u32),
            Arg::U32(self.k as u32),
        ];
        job.launch(kernel, Grid::Linear(self.queries as u64), &args)
            .unwrap();
        job.finish().unwrap();
        Output::Bytes(out.into_iter().flat_map(u32::to_le_bytes).collect())
    }
}
