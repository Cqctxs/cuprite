//! Per-pixel image transforms: imageproc's warps (rotation, perspective,
//! and lens distortion through `warp_with`) and image's `huerotate`.

use std::sync::Arc;

use cuprite_memory::{Arg, Grid};
use image::{imageops, GenericImageView, Rgba, RgbaImage};
use imageproc::geometric_transformations::{
    rotate_about_center, warp, warp_with, Interpolation, Projection,
};
use rayon::prelude::*;

use crate::{barrel, debug_field, one_thread, Case, Gpu, Output};

/// A deterministic test image: smooth gradients with hashed detail, and
/// alpha that varies, so blending exercises every channel's rounding.
pub fn test_image(width: u32, height: u32) -> Arc<RgbaImage> {
    Arc::new(RgbaImage::from_fn(width, height, |x, y| {
        let hash =
            (x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663)).wrapping_mul(2_654_435_761);
        Rgba([
            (x * 255 / width.max(1)) as u8 ^ (hash >> 28) as u8,
            (y * 255 / height.max(1)) as u8,
            (hash >> 8) as u8,
            if (x / 7 + y / 5) % 3 == 0 {
                255
            } else {
                (hash >> 16) as u8
            },
        ])
    }))
}

/// How an output pixel finds its source.
#[derive(Clone, Copy, Debug)]
pub enum Transform {
    /// `rotate_about_center` by `theta` radians.
    Rotate { theta: f32 },
    /// `warp` with `Projection::from_control_points(from, to)`.
    Perspective {
        from: [(f32, f32); 4],
        to: [(f32, f32); 4],
    },
    /// `warp_with` and the `barrel` mapping about the image's center.
    Barrel { k: f32 },
}

#[derive(Clone)]
pub struct WarpCase {
    pub image: Arc<RgbaImage>,
    pub transform: Transform,
    pub bilinear: bool,
    pub default: Rgba<u8>,
}

impl WarpCase {
    fn interpolation(&self) -> Interpolation {
        if self.bilinear {
            Interpolation::Bilinear
        } else {
            Interpolation::Nearest
        }
    }

    fn center(&self) -> (f32, f32) {
        let (w, h) = self.image.dimensions();
        (w as f32 / 2.0, h as f32 / 2.0)
    }

    /// The projection `warp` is given: `rotate` builds it this way.
    fn projection(&self) -> Option<Projection> {
        match self.transform {
            Transform::Rotate { theta } => {
                let (cx, cy) = self.center();
                Some(
                    Projection::translate(cx, cy)
                        * Projection::rotate(theta)
                        * Projection::translate(-cx, -cy),
                )
            }
            Transform::Perspective { from, to } => Projection::from_control_points(from, to),
            Transform::Barrel { .. } => None,
        }
    }

    /// The image call an application makes.
    fn run(&self) -> RgbaImage {
        let (image, interpolation, default) = (&*self.image, self.interpolation(), self.default);
        match self.transform {
            Transform::Rotate { theta } => {
                rotate_about_center(image, theta, interpolation, default)
            }
            Transform::Perspective { .. } => {
                warp(image, &self.projection().unwrap(), interpolation, default)
            }
            Transform::Barrel { k } => {
                let (cx, cy) = self.center();
                warp_with(
                    image,
                    move |x, y| barrel(x, y, cx, cy, k),
                    interpolation,
                    default,
                )
            }
        }
    }

    /// The kernel's mapping and its nine coefficients: for projections, the
    /// inverse matrix `warp_into` maps with and its class, read from
    /// `Projection`'s `Debug` output (the fields are private).
    fn mapping(&self) -> (u32, [f32; 9]) {
        if let Transform::Barrel { k } = self.transform {
            let (cx, cy) = self.center();
            return (3, [cx, cy, k, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }
        let inverse = format!("{:?}", self.projection().unwrap().invert());
        let matrix = debug_field(&inverse, "transform");
        let values: Vec<f32> = matrix
            .trim_matches(['[', ']'])
            .split(", ")
            .map(|value| value.parse().unwrap())
            .collect();
        let kind = match debug_field(&inverse, "class") {
            "Translation" => 0,
            "Affine" => 1,
            "Projection" => 2,
            class => panic!("unknown transformation class {class}"),
        };
        (kind, values.try_into().unwrap())
    }
}

impl Case for WarpCase {
    fn name(&self) -> String {
        let sampling = if self.bilinear { "bilinear" } else { "nearest" };
        match self.transform {
            Transform::Rotate { theta } => format!("rotate {theta} rad, {sampling}"),
            Transform::Perspective { .. } => format!("perspective warp, {sampling}"),
            Transform::Barrel { k } => format!("lens distortion k={k:e}, {sampling}"),
        }
    }

    fn size(&self) -> (usize, usize) {
        let (w, h) = self.image.dimensions();
        (w as usize, h as usize)
    }

    fn kernel_name(&self) -> &'static str {
        "warp_rgba8"
    }

    /// imageproc parallelizes warps with rayon; on one thread here.
    fn cpu(&self) -> Output {
        Output::Bytes(one_thread(|| self.run()).into_raw())
    }

    fn cpu_parallel(&self) -> Output {
        Output::Bytes(self.run().into_raw())
    }

    fn gpu(&self, gpu: &mut Gpu) -> Output {
        let (kind, t) = self.mapping();
        let (width, height) = self.image.dimensions();
        let kernel = gpu.kernel("warp_rgba8");
        let mut out = vec![0u8; self.image.as_raw().len()];
        let mut job = gpu.memory.job().unwrap();
        let source = job.input(self.image.as_raw()).unwrap();
        let output = job.output(&mut out).unwrap();
        let mut args = vec![
            Arg::Buffer(output),
            Arg::Buffer(source),
            Arg::U32(width),
            Arg::U32(height),
            Arg::U32(kind),
        ];
        args.extend(t.map(Arg::F32));
        args.push(Arg::U32(u32::from(self.bilinear)));
        args.push(Arg::U32(u32::from_le_bytes(self.default.0)));
        job.launch(kernel, Grid::Plane { width, height }, &args)
            .unwrap();
        job.finish().unwrap();
        Output::Bytes(out)
    }
}

#[derive(Clone)]
pub struct HueCase {
    pub image: Arc<RgbaImage>,
    pub degrees: i32,
}

impl HueCase {
    /// `huerotate`'s matrix, computed exactly as it does.
    fn matrix(&self) -> [f64; 9] {
        let angle = self.degrees as f64;
        let cosv = angle.to_radians().cos();
        let sinv = angle.to_radians().sin();
        [
            0.213 + cosv * 0.787 - sinv * 0.213,
            0.715 - cosv * 0.715 - sinv * 0.715,
            0.072 - cosv * 0.072 + sinv * 0.928,
            0.213 - cosv * 0.213 + sinv * 0.143,
            0.715 + cosv * 0.285 + sinv * 0.140,
            0.072 - cosv * 0.072 - sinv * 0.283,
            0.213 - cosv * 0.213 - sinv * 0.787,
            0.715 - cosv * 0.715 + sinv * 0.715,
            0.072 + cosv * 0.928 + sinv * 0.072,
        ]
    }
}

/// Rows per band when `huerotate` runs on bands in parallel.
const BAND: u32 = 16;

impl Case for HueCase {
    fn name(&self) -> String {
        format!("huerotate {} degrees", self.degrees)
    }

    fn size(&self) -> (usize, usize) {
        let (w, h) = self.image.dimensions();
        (w as usize, h as usize)
    }

    fn kernel_name(&self) -> &'static str {
        "huerotate_rgba8"
    }

    fn cpu(&self) -> Output {
        Output::Bytes(imageops::huerotate(&*self.image, self.degrees).into_raw())
    }

    /// `huerotate` itself on bands of rows, each on its own core.
    fn cpu_parallel(&self) -> Output {
        let (width, height) = self.image.dimensions();
        let row_bytes = width as usize * 4;
        let mut out = vec![0u8; self.image.as_raw().len()];
        out.par_chunks_mut(row_bytes * BAND as usize)
            .enumerate()
            .for_each(|(band, chunk)| {
                let top = band as u32 * BAND;
                let rows = (height - top).min(BAND);
                let view = self.image.view(0, top, width, rows);
                chunk.copy_from_slice(imageops::huerotate(&*view, self.degrees).as_raw());
            });
        Output::Bytes(out)
    }

    fn gpu(&self, gpu: &mut Gpu) -> Output {
        let pixels = u64::from(self.image.width()) * u64::from(self.image.height());
        let kernel = gpu.kernel("huerotate_rgba8");
        let mut out = vec![0u8; self.image.as_raw().len()];
        let mut job = gpu.memory.job().unwrap();
        let source = job.input(self.image.as_raw()).unwrap();
        let output = job.output(&mut out).unwrap();
        let mut args = vec![Arg::Buffer(output), Arg::Buffer(source), Arg::U64(pixels)];
        args.extend(self.matrix().map(Arg::F64));
        job.launch(kernel, Grid::Linear(pixels), &args).unwrap();
        job.finish().unwrap();
        Output::Bytes(out)
    }
}
