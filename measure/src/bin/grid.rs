//! Times grid workloads on one CPU thread, on every core, and on the GPU,
//! and checks that the GPU's results are bit-identical at full size.
//!
//! ```text
//! cargo run --release --bin grid            # everything
//! cargo run --release --bin grid -- noise   # cases whose name contains "noise"
//! ```

use std::{
    collections::HashSet,
    f32::consts::FRAC_PI_6,
    time::{Duration, Instant},
};

use cuprite_measure::{
    hashes::{Algorithm, HashCase},
    images::{test_image, HueCase, Transform, WarpCase},
    noises::{Noise, NoiseCase},
    sdfs::{Scene, SdfCase},
    time, Case, Gpu,
};
use image::Rgba;

fn show(duration: Duration) -> String {
    let seconds = duration.as_secs_f64();
    if seconds < 1e-3 {
        format!("{:.0} us", seconds * 1e6)
    } else if seconds < 1.0 {
        format!("{:.1} ms", seconds * 1e3)
    } else {
        format!("{seconds:.2} s")
    }
}

fn ratio(cpu: Duration, gpu: Duration) -> String {
    format!("{:.1}x", cpu.as_secs_f64() / gpu.as_secs_f64())
}

fn noise_cases() -> Vec<Box<dyn Case>> {
    let mut cases: Vec<Box<dyn Case>> = Vec::new();
    for size in [512, 2048, 4096] {
        for noise in Noise::ALL {
            let mut case = NoiseCase::new(noise, size, size);
            // A terrain-sized region rather than the default 2 × 2.
            case.x_bounds = (0.0, 16.0);
            case.y_bounds = (0.0, 16.0);
            if noise.eq(&Noise::Worley) {
                case.frequency = 2.0;
            }
            cases.push(Box::new(case));
        }
    }
    // A heavier map: 12 octaves, tiling seamlessly (four samples a point).
    for size in [512, 2048] {
        let mut case = NoiseCase::new(Noise::FbmPerlin, size, size);
        case.octaves = 12;
        case.seamless = true;
        case.x_bounds = (0.0, 16.0);
        case.y_bounds = (0.0, 16.0);
        cases.push(Box::new(case));
    }
    cases
}

fn image_cases() -> Vec<Box<dyn Case>> {
    let mut cases: Vec<Box<dyn Case>> = Vec::new();
    for size in [1024u32, 4096] {
        let image = test_image(size, size);
        let s = size as f32;
        let default = Rgba([0, 0, 0, 0]);
        let warp = |transform, bilinear| {
            Box::new(WarpCase {
                image: image.clone(),
                transform,
                bilinear,
                default,
            }) as Box<dyn Case>
        };
        cases.push(warp(Transform::Rotate { theta: FRAC_PI_6 }, false));
        cases.push(warp(Transform::Rotate { theta: FRAC_PI_6 }, true));
        cases.push(warp(
            Transform::Perspective {
                from: [(0.0, 0.0), (s, 0.0), (s, s), (0.0, s)],
                to: [(s * 0.2, 0.0), (s * 0.8, 0.0), (s, s), (0.0, s)],
            },
            true,
        ));
        cases.push(warp(Transform::Barrel { k: 1.0 / (s * s) }, true));
        cases.push(Box::new(HueCase {
            image: image.clone(),
            degrees: 45,
        }));
    }
    cases
}

fn sdf_cases() -> Vec<Box<dyn Case>> {
    let mut cases: Vec<Box<dyn Case>> = Vec::new();
    for n in [64, 128, 256] {
        for scene in Scene::ALL {
            cases.push(Box::new(SdfCase::new(scene, n)));
        }
    }
    cases
}

fn hash_cases() -> Vec<Box<dyn Case>> {
    let mut cases: Vec<Box<dyn Case>> = Vec::new();
    for count in [1 << 20, 1 << 24] {
        // 40-byte messages: one block for both; 120 bytes: three for SHA-256.
        cases.push(Box::new(HashCase::new(Algorithm::Sha256, 32, count)));
        cases.push(Box::new(HashCase::new(Algorithm::Sha256, 112, count)));
        cases.push(Box::new(HashCase::new(Algorithm::Sha512, 32, count)));
    }
    // Every digest kept: 32 bytes back per hash instead of one.
    for count in [1 << 20, 1 << 22] {
        let mut case = HashCase::new(Algorithm::Sha256, 32, count);
        case.digests = true;
        cases.push(Box::new(case));
    }
    cases
}

fn main() {
    let filter = std::env::args().nth(1).unwrap_or_default();
    let mut gpu = Gpu::open();
    let mut loaded = HashSet::new();
    println!(
        "rayon threads: {}; GPU: {}; CPU SHA instructions: {}\n",
        rayon::current_num_threads(),
        gpu.memory.device().name,
        std::arch::is_x86_feature_detected!("sha"),
    );
    println!(
        "| case | size | cpu 1 thread | cpu all cores | gpu | vs 1 thread | vs all cores | bit-identical | first kernel load |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");

    let cases = noise_cases()
        .into_iter()
        .chain(image_cases())
        .chain(sdf_cases())
        .chain(hash_cases());
    for case in cases {
        let name = case.name();
        let kernel = case.kernel_name();
        let kind = if kernel.ends_with("_map") {
            "noise"
        } else if kernel.starts_with("sdf_") {
            "sdf"
        } else if kernel.starts_with("sha") {
            "hash"
        } else {
            "image"
        };
        if !format!("{kind} {name}").contains(&filter) {
            continue;
        }
        // The first load compiles the module's PTX for this GPU.
        let load = if loaded.insert(case.kernel_name()) {
            let start = Instant::now();
            gpu.kernel(case.kernel_name());
            show(start.elapsed())
        } else {
            String::new()
        };

        let expected = case.cpu();
        let comparison = expected.compare(&case.gpu(&mut gpu));
        let exact = if comparison.exact() {
            "yes".to_string()
        } else {
            format!("no: {} of {}", comparison.differing, comparison.values)
        };
        assert!(expected.compare(&case.cpu_parallel()).exact());

        let cpu = time(|| drop(case.cpu()));
        let parallel = time(|| drop(case.cpu_parallel()));
        let on_gpu = time(|| drop(case.gpu(&mut gpu)));
        let shape = case.shape();
        println!(
            "| {name} | {shape} | {} | {} | {} | {} | {} | {exact} | {load} |",
            show(cpu),
            show(parallel),
            show(on_gpu),
            ratio(cpu, on_gpu),
            ratio(parallel, on_gpu),
        );
    }
}
