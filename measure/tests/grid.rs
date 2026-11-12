//! Every GPU port must compute exactly what its crate computes, and the
//! parallel CPU versions must too. Sizes are small; the settings vary.
//! Needs a CUDA GPU.

use std::{f32::consts::FRAC_PI_2, sync::Mutex};

use cuprite_measure::{
    hashes::{Algorithm, HashCase},
    images::{test_image, HueCase, Transform, WarpCase},
    noises::{Noise, NoiseCase},
    sdfs::{Scene, SdfCase},
    Case, Gpu,
};
use image::Rgba;

/// One GPU, shared by the tests, which run on several threads.
static GPU: Mutex<Option<Gpu>> = Mutex::new(None);

/// Runs the case three ways and requires identical results.
fn check(case: &dyn Case) {
    let label = format!("{} at {}", case.name(), case.shape());
    let cpu = case.cpu();
    let parallel = case.cpu_parallel();
    let comparison = cpu.compare(&parallel);
    assert!(
        comparison.exact(),
        "{label}: the parallel CPU version differs: {comparison:?}"
    );
    let mut lock = GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let gpu = case.gpu(lock.get_or_insert_with(Gpu::open));
    drop(lock);
    let comparison = cpu.compare(&gpu);
    assert!(
        comparison.exact(),
        "{label}: the GPU differs: {comparison:?}"
    );
}

const SIZES: [(usize, usize); 4] = [(1, 1), (37, 23), (64, 64), (200, 3)];

/// Bounds over which the builder's points include exact integers and zero
/// (`floor_to_isize` treats those specially), large negative coordinates,
/// and the default.
const BOUNDS: [((f64, f64), (f64, f64)); 4] = [
    ((-1.0, 1.0), (-1.0, 1.0)),
    ((0.0, 16.0), (-8.0, 8.0)),
    ((-1000.5, 1000.5), (-3.75, 2.0)),
    ((12.25, 12.75), (-0.001, 0.001)),
];

fn variations(noise: Noise) -> Vec<NoiseCase> {
    let mut cases = Vec::new();
    for (i, &(width, height)) in SIZES.iter().enumerate() {
        for (j, &(x_bounds, y_bounds)) in BOUNDS.iter().enumerate() {
            let mut case = NoiseCase::new(noise, width, height);
            case.x_bounds = x_bounds;
            case.y_bounds = y_bounds;
            case.seed = [0, 1, 42, 0xDEAD_BEEF][(i + j) % 4];
            case.seamless = (i + j) % 2 == 1;
            cases.push(case);
        }
    }
    // Fractal and frequency settings, at one size.
    for (octaves, frequency, lacunarity, persistence, attenuation) in [
        (1, 1.0, 2.0, 0.5, 2.0),
        (12, 3.7, 2.0, 0.8, 1.3),
        (4, 0.01, 1.7, 0.3, 4.0),
    ] {
        let mut case = NoiseCase::new(noise, 50, 40);
        case.octaves = octaves;
        case.frequency = frequency;
        case.lacunarity = lacunarity;
        case.persistence = persistence;
        case.attenuation = attenuation;
        case.x_bounds = (-20.0, 20.0);
        case.y_bounds = (-20.0, 20.0);
        cases.push(case);
    }
    cases
}

#[test]
fn perlin_matches() {
    variations(Noise::Perlin)
        .iter()
        .for_each(|case| check(case));
}

#[test]
fn simplex_matches() {
    variations(Noise::Simplex)
        .iter()
        .for_each(|case| check(case));
}

#[test]
fn worley_matches() {
    variations(Noise::Worley)
        .iter()
        .for_each(|case| check(case));
}

#[test]
fn fbm_simplex_matches() {
    variations(Noise::FbmSimplex)
        .iter()
        .for_each(|case| check(case));
}

#[test]
fn fbm_perlin_matches() {
    variations(Noise::FbmPerlin)
        .iter()
        .for_each(|case| check(case));
}

#[test]
fn ridged_perlin_matches() {
    variations(Noise::RidgedPerlin)
        .iter()
        .for_each(|case| check(case));
}

const IMAGE_SIZES: [(u32, u32); 4] = [(1, 1), (17, 9), (100, 77), (256, 256)];

fn transforms(width: u32, height: u32) -> Vec<Transform> {
    let (w, h) = (width as f32, height as f32);
    vec![
        Transform::Rotate { theta: 0.0 },
        Transform::Rotate { theta: 0.3 },
        Transform::Rotate { theta: FRAC_PI_2 },
        Transform::Rotate { theta: -2.0 },
        // A keystone: the top edge pulled in.
        Transform::Perspective {
            from: [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)],
            to: [(w * 0.2, 0.0), (w * 0.8, 0.0), (w, h), (0.0, h)],
        },
        // Tilted past the edges, so some pixels come from outside.
        Transform::Perspective {
            from: [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)],
            to: [
                (w * 0.3, h * 0.1),
                (w * 1.1, h * 0.4),
                (w * 0.7, h * 1.2),
                (-w * 0.1, h * 0.9),
            ],
        },
        // Shrunk into the middle, so most pixels come from outside.
        Transform::Perspective {
            from: [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)],
            to: [
                (w * 0.4, h * 0.3),
                (w * 0.6, h * 0.35),
                (w * 0.65, h * 0.7),
                (w * 0.3, h * 0.6),
            ],
        },
        Transform::Barrel { k: 0.0 },
        Transform::Barrel {
            k: 4.0 / (w * w + h * h),
        },
        Transform::Barrel {
            k: -2.0 / (w * w + h * h),
        },
    ]
}

#[test]
fn warps_match() {
    for (width, height) in IMAGE_SIZES {
        let image = test_image(width, height);
        for transform in transforms(width, height) {
            for bilinear in [false, true] {
                let default = Rgba([7, 0, 200, 99]);
                check(&WarpCase {
                    image: image.clone(),
                    transform,
                    bilinear,
                    default,
                });
            }
        }
    }
}

#[test]
fn huerotate_matches() {
    for (width, height) in IMAGE_SIZES {
        let image = test_image(width, height);
        for degrees in [0, 45, 90, 180, -30, 361, 720] {
            check(&HueCase {
                image: image.clone(),
                degrees,
            });
        }
    }
}

/// Lattices whose points land exactly on surfaces, faces and the origin
/// (min -1.0 with even n), off them (-1.5), and asymmetric spacing (-2.3).
#[test]
fn sdfs_match() {
    for scene in Scene::ALL {
        for n in [1, 2, 17, 32, 64] {
            for min in [-1.5, -1.0, -2.3] {
                check(&SdfCase { scene, n, min });
            }
        }
    }
}

/// Message lengths on both sides of each padding boundary, nonce counts
/// that aren't whole blocks of threads, a nonce range that wraps, and whole
/// digests as well as scores.
#[test]
fn hashes_match() {
    let sha256_lengths = [0, 32, 47, 48, 112];
    let sha512_lengths = [0, 32, 103, 104, 200];
    for (algorithm, lengths) in [
        (Algorithm::Sha256, sha256_lengths),
        (Algorithm::Sha512, sha512_lengths),
    ] {
        for prefix_len in lengths {
            for count in [1, 1000, 4099] {
                for digests in [false, true] {
                    let mut case = HashCase::new(algorithm, prefix_len, count);
                    case.digests = digests;
                    check(&case);
                }
            }
            let mut wrapping = HashCase::new(algorithm, prefix_len, 300);
            wrapping.first = u64::MAX - 100;
            wrapping.digests = true;
            check(&wrapping);
        }
    }
}
