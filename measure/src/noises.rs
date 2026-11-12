//! Noise maps from the `noise` crate, as `PlaneMapBuilder` builds them.

use cuprite_memory::{Arg, Grid};
use noise::{
    permutationtable::{NoiseHasher, PermutationTable},
    utils::{NoiseMap, NoiseMapBuilder, PlaneMapBuilder},
    Fbm, MultiFractal, NoiseFn, Perlin, RidgedMulti, Simplex, Worley,
};
use rayon::prelude::*;

use crate::{debug_field, Case, Gpu, Output};

/// `noise::math::interpolate::linear`, which the crate keeps private.
fn linear(a: f64, b: f64, alpha: f64) -> f64 {
    b * alpha + a * (1.0 - alpha)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Noise {
    /// `Perlin`, 2D.
    Perlin,
    /// `Simplex`, 2D.
    Simplex,
    /// `Worley` with its default distance and return type, 2D.
    Worley,
    /// `Fbm<Simplex>`, 2D.
    FbmSimplex,
    /// `Fbm<Perlin>`, 3D at `z = 0`, the crate's usual map.
    FbmPerlin,
    /// `RidgedMulti<Perlin>`, 3D at `z = 0`.
    RidgedPerlin,
}

impl Noise {
    pub const ALL: [Noise; 6] = [
        Noise::Perlin,
        Noise::Simplex,
        Noise::Worley,
        Noise::FbmSimplex,
        Noise::FbmPerlin,
        Noise::RidgedPerlin,
    ];

    fn fractal(self) -> bool {
        matches!(
            self,
            Noise::FbmSimplex | Noise::FbmPerlin | Noise::RidgedPerlin
        )
    }

    fn kernel(self) -> &'static str {
        match self {
            Noise::Perlin => "perlin_2d_map",
            Noise::Simplex => "simplex_2d_map",
            Noise::Worley => "worley_2d_map",
            Noise::FbmSimplex => "fbm_simplex_2d_map",
            Noise::FbmPerlin => "fbm_perlin_3d_map",
            Noise::RidgedPerlin => "ridged_perlin_3d_map",
        }
    }
}

/// One noise map: the noise, its settings, and the builder's bounds.
#[derive(Clone, Debug)]
pub struct NoiseCase {
    pub noise: Noise,
    pub seed: u32,
    pub width: usize,
    pub height: usize,
    pub x_bounds: (f64, f64),
    pub y_bounds: (f64, f64),
    pub seamless: bool,
    /// Fractal settings; `frequency` is also Worley's.
    pub octaves: usize,
    pub frequency: f64,
    pub lacunarity: f64,
    pub persistence: f64,
    /// `RidgedMulti` only.
    pub attenuation: f64,
}

impl NoiseCase {
    /// The crate's defaults over its default bounds.
    pub fn new(noise: Noise, width: usize, height: usize) -> Self {
        let (frequency, lacunarity) = (1.0, core::f64::consts::PI * 2.0 / 3.0);
        let persistence = if noise == Noise::RidgedPerlin {
            1.0
        } else {
            0.5
        };
        Self {
            noise,
            seed: 0,
            width,
            height,
            x_bounds: (-1.0, 1.0),
            y_bounds: (-1.0, 1.0),
            seamless: false,
            octaves: 6,
            frequency,
            lacunarity,
            persistence,
            attenuation: 2.0,
        }
    }

    fn fbm<T: Default + noise::Seedable>(&self) -> Fbm<T> {
        Fbm::<T>::new(self.seed)
            .set_octaves(self.octaves)
            .set_frequency(self.frequency)
            .set_lacunarity(self.lacunarity)
            .set_persistence(self.persistence)
    }

    fn ridged(&self) -> RidgedMulti<Perlin> {
        RidgedMulti::<Perlin>::new(self.seed)
            .set_octaves(self.octaves)
            .set_frequency(self.frequency)
            .set_lacunarity(self.lacunarity)
            .set_persistence(self.persistence)
            .set_attenuation(self.attenuation)
    }

    fn worley(&self) -> Worley {
        Worley::new(self.seed).set_frequency(self.frequency)
    }

    /// The noise at a point of the plane, as the builder samples it.
    fn source(&self) -> Box<dyn Fn(f64, f64) -> f64> {
        match self.noise {
            Noise::Perlin => {
                let noise = Perlin::new(self.seed);
                Box::new(move |x, y| noise.get([x, y]))
            }
            Noise::Simplex => {
                let noise = Simplex::new(self.seed);
                Box::new(move |x, y| noise.get([x, y]))
            }
            Noise::Worley => {
                let noise = self.worley();
                Box::new(move |x, y| noise.get([x, y]))
            }
            Noise::FbmSimplex => {
                let noise = self.fbm::<Simplex>();
                Box::new(move |x, y| noise.get([x, y]))
            }
            Noise::FbmPerlin => {
                let noise = self.fbm::<Perlin>();
                Box::new(move |x, y| noise.get([x, y, 0.0]))
            }
            Noise::RidgedPerlin => {
                let noise = self.ridged();
                Box::new(move |x, y| noise.get([x, y, 0.0]))
            }
        }
    }

    /// The crate's private `scale_factor`, read from its `Debug` output.
    fn scale_factor(&self) -> f64 {
        let debug = match self.noise {
            Noise::FbmSimplex => format!("{:?}", self.fbm::<Simplex>()),
            Noise::FbmPerlin => format!("{:?}", self.fbm::<Perlin>()),
            Noise::RidgedPerlin => format!("{:?}", self.ridged()),
            _ => return 0.0,
        };
        debug_field(&debug, "scale_factor").parse().unwrap()
    }

    /// The permutation tables of the noise's sources, 256 bytes each:
    /// fractal source `i` is seeded `seed + i`, as `build_sources` does.
    fn tables(&self) -> Vec<u8> {
        let sources = if self.noise.fractal() {
            self.octaves
        } else {
            1
        };
        (0..sources)
            .flat_map(|i| {
                let table = PermutationTable::new(self.seed.wrapping_add(i as u32));
                (0..256).map(move |byte| table.hash(&[byte]) as u8)
            })
            .collect()
    }

    fn builder<S: NoiseFn<f64, D>, const D: usize>(
        &self,
        builder: PlaneMapBuilder<S, D>,
    ) -> PlaneMapBuilder<S, D> {
        builder
            .set_x_bounds(self.x_bounds.0, self.x_bounds.1)
            .set_y_bounds(self.y_bounds.0, self.y_bounds.1)
            .set_is_seamless(self.seamless)
    }

    /// `build` for a 2D source given as a closure.
    fn build_2d(&self, noise: impl Fn([f64; 2]) -> f64) -> NoiseMap {
        self.builder(PlaneMapBuilder::new_fn(noise))
            .set_size(self.width, self.height)
            .build()
    }

    fn extents(&self) -> (f64, f64, f64, f64) {
        let x_extent = self.x_bounds.1 - self.x_bounds.0;
        let y_extent = self.y_bounds.1 - self.y_bounds.0;
        (
            x_extent,
            y_extent,
            x_extent / self.width as f64,
            y_extent / self.height as f64,
        )
    }
}

impl Case for NoiseCase {
    fn name(&self) -> String {
        let name = match self.noise {
            Noise::Perlin => "perlin 2D".to_string(),
            Noise::Simplex => "simplex 2D".to_string(),
            Noise::Worley => "worley 2D".to_string(),
            Noise::FbmSimplex => format!("fbm<simplex> 2D, {} octaves", self.octaves),
            Noise::FbmPerlin => format!("fbm<perlin> 3D, {} octaves", self.octaves),
            Noise::RidgedPerlin => format!("ridged<perlin> 3D, {} octaves", self.octaves),
        };
        if self.seamless {
            format!("{name}, seamless")
        } else {
            name
        }
    }

    fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    fn kernel_name(&self) -> &'static str {
        self.noise.kernel()
    }

    /// `PlaneMapBuilder::build`, exactly as an application calls it.
    fn cpu(&self) -> Output {
        let map = match self.noise {
            Noise::Perlin => {
                let noise = Perlin::new(self.seed);
                self.build_2d(|p| noise.get(p))
            }
            Noise::Simplex => {
                let noise = Simplex::new(self.seed);
                self.build_2d(|p| noise.get(p))
            }
            Noise::Worley => {
                let noise = self.worley();
                self.build_2d(|p| noise.get(p))
            }
            Noise::FbmSimplex => {
                let noise = self.fbm::<Simplex>();
                self.build_2d(|p| noise.get(p))
            }
            Noise::FbmPerlin => self
                .builder(PlaneMapBuilder::new(self.fbm::<Perlin>()))
                .set_size(self.width, self.height)
                .build(),
            Noise::RidgedPerlin => self
                .builder(PlaneMapBuilder::new(self.ridged()))
                .set_size(self.width, self.height)
                .build(),
        };
        Output::F64(map.into_iter().collect())
    }

    /// The builder's loop, rows on every core, each thread with its own
    /// copy of the noise (`Worley` isn't `Sync`).
    fn cpu_parallel(&self) -> Output {
        let (x_extent, y_extent, x_step, y_step) = self.extents();
        let (x0, y0) = (self.x_bounds.0, self.y_bounds.0);
        let mut out = vec![0.0; self.width * self.height];
        out.par_chunks_mut(self.width).enumerate().for_each_init(
            || self.source(),
            |source, (y, row)| {
                let current_y = y0 + y_step * y as f64;
                for (x, value) in row.iter_mut().enumerate() {
                    let current_x = x0 + x_step * x as f64;
                    *value = if self.seamless {
                        let sw = source(current_x, current_y);
                        let se = source(current_x + x_extent, current_y);
                        let nw = source(current_x, current_y + y_extent);
                        let ne = source(current_x + x_extent, current_y + y_extent);
                        let x_blend = 1.0 - ((current_x - x0) / x_extent);
                        let y_blend = 1.0 - ((current_y - y0) / y_extent);
                        let y0 = linear(sw, se, x_blend);
                        let y1 = linear(nw, ne, x_blend);
                        linear(y0, y1, y_blend)
                    } else {
                        source(current_x, current_y)
                    };
                }
            },
        );
        Output::F64(out)
    }

    fn gpu(&self, gpu: &mut Gpu) -> Output {
        let (x_extent, y_extent, x_step, y_step) = self.extents();
        let tables = self.tables();
        let kernel = gpu.kernel(self.noise.kernel());
        let mut out = vec![0.0f64; self.width * self.height];
        let (width, height) = (self.width as u32, self.height as u32);
        let mut job = gpu.memory.job().unwrap();
        let tables = job.input(&tables).unwrap();
        let output = job.output(&mut out).unwrap();
        let args = [
            Arg::Buffer(output),
            Arg::Buffer(tables),
            Arg::U32(width),
            Arg::U32(height),
            Arg::F64(self.x_bounds.0),
            Arg::F64(x_extent),
            Arg::F64(x_step),
            Arg::F64(self.y_bounds.0),
            Arg::F64(y_extent),
            Arg::F64(y_step),
            Arg::U32(u32::from(self.seamless)),
            Arg::U32(self.octaves as u32),
            Arg::F64(self.frequency),
            Arg::F64(self.lacunarity),
            Arg::F64(self.persistence),
            Arg::F64(self.scale_factor()),
            Arg::F64(self.attenuation),
        ];
        job.launch(kernel, Grid::Plane { width, height }, &args)
            .unwrap();
        job.finish().unwrap();
        Output::F64(out)
    }
}
