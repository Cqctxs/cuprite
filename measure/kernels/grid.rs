//! GPU ports of grid computations from real crates: every output point is
//! a pure function of its coordinates and read-only inputs. Compiled to PTX
//! by rustc's nvptx backend (`build.rs`).
//!
//! Each port follows its crate's source operation for operation (the
//! comments name the function), so results can match the CPU bit for bit:
//! rustc never fuses multiply-adds, and LLVM emits float arithmetic with
//! explicit rounding modes, which the driver keeps.
//!
//! Launch layouts:
//! - `*_map` kernels: a `Grid::Plane` over the `width × height` output, in
//!   `NoiseMap` order (`x + y * width`).
//! - `warp_rgba8`: a `Grid::Plane` over the output image.
//! - `huerotate_rgba8`: a `Grid::Linear` over the pixels.

#![no_std]
#![feature(abi_ptx, core_intrinsics, stdarch_nvptx)]
#![allow(internal_features, clippy::too_many_arguments)]

use core::{
    arch::nvptx::{
        _block_dim_x, _block_dim_y, _block_idx_x, _block_idx_y, _thread_idx_x, _thread_idx_y, trap,
    },
    f64::consts::{FRAC_1_SQRT_2, SQRT_2},
};

include!("shared.rs");

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    // SAFETY: trapping ends the kernel; the job's next wait reports it.
    unsafe { trap() }
}

fn floor32(x: f32) -> f32 {
    core::intrinsics::floorf32(x)
}

fn sqrt64(x: f64) -> f64 {
    core::intrinsics::sqrtf64(x)
}

/// This thread's `(x, y)` in a `Grid::Plane` launch.
fn plane_index() -> (u32, u32) {
    // SAFETY: reading the thread's position has no preconditions.
    unsafe {
        let x = _block_idx_x()
            .wrapping_mul(_block_dim_x())
            .wrapping_add(_thread_idx_x());
        let y = _block_idx_y()
            .wrapping_mul(_block_dim_y())
            .wrapping_add(_thread_idx_y());
        (x as u32, y as u32)
    }
}

/// This thread's index in a `Grid::Linear` launch.
fn linear_index() -> u64 {
    // SAFETY: as above.
    unsafe { _block_idx_x() as u64 * _block_dim_x() as u64 + _thread_idx_x() as u64 }
}

// ---------------------------------------------------------------------------
// noise 0.9.0

/// One source's `PermutationTable`: 256 bytes.
#[derive(Clone, Copy)]
struct Table(*const u8);

impl Table {
    fn at(self, index: usize) -> usize {
        // SAFETY: every index below is masked or xor-ed below 256.
        unsafe { *self.0.add(index) as usize }
    }

    /// `NoiseHasher::hash`: `reduce(|a, b| values[a] ^ b)` over the masked
    /// coordinates, then one more lookup.
    fn hash2(self, a: isize, b: isize) -> usize {
        let index = self.at((a & 0xff) as usize) ^ (b & 0xff) as usize;
        self.at(index)
    }

    fn hash3(self, a: isize, b: isize, c: isize) -> usize {
        let index = self.at((a & 0xff) as usize) ^ (b & 0xff) as usize;
        let index = self.at(index) ^ (c & 0xff) as usize;
        self.at(index)
    }
}

/// `Vector::floor_to_isize`, quirk included: zero and below map one lower,
/// so whole negative numbers do too.
fn floor_to_isize(x: f64) -> isize {
    if x <= 0.0 {
        x as isize - 1
    } else {
        x as isize
    }
}

/// `math::interpolate::linear`.
fn linear(a: f64, b: f64, alpha: f64) -> f64 {
    b * alpha + a * (1.0 - alpha)
}

/// `Quintic::map_quintic` for `f64`.
fn quintic(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * x * (x * (x * 6.0 - 15.0) + 10.0)
}

/// `core::perlin::perlin_2d`.
fn perlin_2d(px: f64, py: f64, table: Table) -> f64 {
    const SCALE_FACTOR: f64 = 2.0 / SQRT_2;
    let (cx, cy) = (floor_to_isize(px), floor_to_isize(py));
    let (dx, dy) = (px - cx as f64, py - cy as f64);
    let gradient = |ox: isize, oy: isize| {
        let (x, y) = (dx - ox as f64, dy - oy as f64);
        match table.hash2(cx + ox, cy + oy) & 0b11 {
            0 => x + y,
            1 => -x + y,
            2 => x - y,
            _ => -x - y,
        }
    };
    let (g00, g10, g01, g11) = (
        gradient(0, 0),
        gradient(1, 0),
        gradient(0, 1),
        gradient(1, 1),
    );
    let (ux, uy) = (quintic(dx), quintic(dy));
    let result = linear(linear(g00, g01, uy), linear(g10, g11, uy), ux) * SCALE_FACTOR;
    result.clamp(-1.0, 1.0)
}

/// `core::perlin::perlin_3d`.
fn perlin_3d(p: [f64; 3], table: Table) -> f64 {
    const SCALE_FACTOR: f64 = 1.154_700_538_379_251_5;
    let c = [
        floor_to_isize(p[0]),
        floor_to_isize(p[1]),
        floor_to_isize(p[2]),
    ];
    let d = [p[0] - c[0] as f64, p[1] - c[1] as f64, p[2] - c[2] as f64];
    let gradient = |ox: isize, oy: isize, oz: isize| {
        let (x, y, z) = (d[0] - ox as f64, d[1] - oy as f64, d[2] - oz as f64);
        match table.hash3(c[0] + ox, c[1] + oy, c[2] + oz) & 0b1111 {
            0 | 12 => x + y,
            1 | 13 => -x + y,
            2 => x - y,
            3 => -x - y,
            4 => x + z,
            5 => -x + z,
            6 => x - z,
            7 => -x - z,
            8 => y + z,
            9 | 14 => -y + z,
            10 => y - z,
            _ => -y - z,
        }
    };
    let g000 = gradient(0, 0, 0);
    let g100 = gradient(1, 0, 0);
    let g010 = gradient(0, 1, 0);
    let g110 = gradient(1, 1, 0);
    let g001 = gradient(0, 0, 1);
    let g101 = gradient(1, 0, 1);
    let g011 = gradient(0, 1, 1);
    let g111 = gradient(1, 1, 1);
    let u = [quintic(d[0]), quintic(d[1]), quintic(d[2])];
    let result = linear(
        linear(linear(g000, g001, u[2]), linear(g010, g011, u[2]), u[1]),
        linear(linear(g100, g101, u[2]), linear(g110, g111, u[2]), u[1]),
        u[0],
    ) * SCALE_FACTOR;
    result.clamp(-1.0, 1.0)
}

/// `gradient::grad2`.
fn grad2(index: usize) -> (f64, f64) {
    const DIAG: f64 = FRAC_1_SQRT_2;
    match index % 8 {
        0 => (1.0, 0.0),
        1 => (-1.0, 0.0),
        2 => (0.0, 1.0),
        3 => (0.0, -1.0),
        4 => (DIAG, DIAG),
        5 => (-DIAG, DIAG),
        6 => (DIAG, -DIAG),
        _ => (-DIAG, -DIAG),
    }
}

/// The `surflet` value inside `simplex_2d`. Vector sums and dot products
/// start from zero, as the crate's `Vector::dot` does.
fn surflet_2d(index: usize, x: f64, y: f64) -> f64 {
    let t = 1.0 - (0.0 + x * x + y * y) * 2.0;
    if t > 0.0 {
        let (gx, gy) = grad2(index);
        let t2 = t * t;
        let t4 = t2 * t2;
        (2.0 * t2 + t4) * (0.0 + x * gx + y * gy)
    } else {
        0.0
    }
}

/// `core::simplex::simplex_2d`, value only (`Simplex::get` drops the
/// derivative).
fn simplex_2d(px: f64, py: f64, table: Table) -> f64 {
    let n = 2.0f64;
    let skew_factor = (sqrt64(n + 1.0) - 1.0) / n;
    let unskew_factor = (1.0 - (1.0 / sqrt64(n + 1.0))) / n;

    let skew = (0.0 + px + py) * skew_factor;
    let (cx, cy) = (floor_to_isize(px + skew), floor_to_isize(py + skew));
    let (fx, fy) = (cx as f64, cy as f64);
    let unskew = (0.0 + fx + fy) * unskew_factor;
    let (x1, y1) = (px - (fx - unskew), py - (fy - unskew));
    let (ox, oy) = if x1 > y1 { (1.0, 0.0) } else { (0.0, 1.0) };
    let (x2, y2) = (x1 - ox + unskew_factor, y1 - oy + unskew_factor);
    let corner = 2.0 * unskew_factor;
    let (x3, y3) = (x1 - 1.0 + corner, y1 - 1.0 + corner);

    let gi0 = table.hash2(cx, cy);
    let gi1 = table.hash2(cx + ox as isize, cy + oy as isize);
    let gi2 = table.hash2(cx + 1, cy + 1);
    surflet_2d(gi0, x1, y1) + surflet_2d(gi1, x2, y2) + surflet_2d(gi2, x3, y3)
}

/// `core::worley::get_vec2`.
fn worley_vec2(index: usize) -> (f64, f64) {
    let length = ((index & 0xF8) >> 3) as f64 * 0.5 / 31.0;
    let diag = length * FRAC_1_SQRT_2;
    match index & 0x07 {
        0 => (diag, diag),
        1 => (diag, -diag),
        2 => (-diag, diag),
        3 => (-diag, -diag),
        4 => (length, 0.0),
        5 => (-length, 0.0),
        6 => (0.0, length),
        _ => (0.0, -length),
    }
}

/// `worley_2d`'s `get_point`: the cell's feature point.
fn worley_point(table: Table, cell: (isize, isize)) -> (f64, f64) {
    let (vx, vy) = worley_vec2(table.hash2(cell.0, cell.1));
    (vx + cell.0 as f64, vy + cell.1 as f64)
}

/// `worley::distance_functions::euclidean`, folded from zero.
fn euclidean(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (a.0 - b.0, a.1 - b.1);
    sqrt64(0.0 + dx * dx + dy * dy)
}

/// `core::worley::worley_2d` with the default euclidean distance and
/// `ReturnType::Value`.
fn worley_2d(px: f64, py: f64, table: Table) -> f64 {
    let point = (px, py);
    let (cx, cy) = (floor_to_isize(px), floor_to_isize(py));
    let (fx, fy) = (px - cx as f64, py - cy as f64);
    let (hx, hy) = (fx > 0.5, fy > 0.5);
    let near = (hx as isize + cx, hy as isize + cy);
    let far = ((!hx) as isize + cx, (!hy) as isize + cy);

    let mut seed_cell = near;
    let mut distance = euclidean(point, worley_point(table, near));
    // `(0.5 - x).powf(2.0)`, which is exactly a square.
    let range = ((0.5 - fx) * (0.5 - fx), (0.5 - fy) * (0.5 - fy));

    let mut test = |cell: (isize, isize), distance: &mut f64| {
        let candidate = euclidean(point, worley_point(table, cell));
        if candidate < *distance {
            *distance = candidate;
            seed_cell = cell;
        }
    };
    if range.0 < distance {
        test((far.0, near.1), &mut distance);
    }
    if range.1 < distance {
        test((near.0, far.1), &mut distance);
    }
    if range.0 < distance && range.1 < distance {
        test((far.0, far.1), &mut distance);
    }

    let value = table.hash2(seed_cell.0, seed_cell.1) as f64 / 255.0;
    value * 2.0 - 1.0
}

/// The fractal settings, as `Fbm` and `RidgedMulti` hold them. `scale` is
/// their private `scale_factor`; source `i` has table `i`.
struct Fractal {
    tables: *const u8,
    octaves: u32,
    frequency: f64,
    lacunarity: f64,
    persistence: f64,
    scale: f64,
    attenuation: f64,
}

impl Fractal {
    fn table(&self, source: u32) -> Table {
        // SAFETY: the host uploads one table for each source.
        Table(unsafe { self.tables.add(source as usize * 256) })
    }

    /// `Fbm::get` for any dimension.
    fn fbm<const N: usize>(
        &self,
        mut point: [f64; N],
        source: impl Fn([f64; N], Table) -> f64,
    ) -> f64 {
        let mut result = 0.0;
        let mut attenuation = self.persistence;
        for c in &mut point {
            *c *= self.frequency;
        }
        for octave in 0..self.octaves {
            let mut signal = source(point, self.table(octave));
            signal *= attenuation;
            attenuation *= self.persistence;
            result += signal;
            for c in &mut point {
                *c *= self.lacunarity;
            }
        }
        result * self.scale
    }

    /// `RidgedMulti::get` for any dimension.
    fn ridged<const N: usize>(
        &self,
        mut point: [f64; N],
        source: impl Fn([f64; N], Table) -> f64,
    ) -> f64 {
        let mut result = 0.0;
        let mut weight = 1.0;
        let mut attenuation = 1.0;
        for c in &mut point {
            *c *= self.frequency;
        }
        for octave in 0..self.octaves {
            let mut signal = source(point, self.table(octave));
            signal = signal.abs();
            signal = 1.0 - signal;
            signal *= signal;
            signal *= weight;
            weight = signal / self.attenuation;
            weight = weight.clamp(0.0, 1.0);
            signal *= attenuation;
            attenuation *= self.persistence;
            result += signal;
            for c in &mut point {
                *c *= self.lacunarity;
            }
        }
        result *= self.scale;
        result - 1.0
    }
}

/// `PlaneMapBuilder`'s bounds, as `build` derives them.
struct Plane {
    x0: f64,
    x_extent: f64,
    x_step: f64,
    y0: f64,
    y_extent: f64,
    y_step: f64,
    seamless: bool,
}

impl Plane {
    /// One point of `PlaneMapBuilder::build`: the source at the point, or,
    /// when seamless, the blend of it with its three wrapped copies.
    fn value(&self, x: u32, y: u32, source: impl Fn(f64, f64) -> f64) -> f64 {
        let current_y = self.y0 + self.y_step * y as f64;
        let current_x = self.x0 + self.x_step * x as f64;
        if !self.seamless {
            return source(current_x, current_y);
        }
        let sw = source(current_x, current_y);
        let se = source(current_x + self.x_extent, current_y);
        let nw = source(current_x, current_y + self.y_extent);
        let ne = source(current_x + self.x_extent, current_y + self.y_extent);
        let x_blend = 1.0 - ((current_x - self.x0) / self.x_extent);
        let y_blend = 1.0 - ((current_y - self.y0) / self.y_extent);
        let y0 = linear(sw, se, x_blend);
        let y1 = linear(nw, ne, x_blend);
        linear(y0, y1, y_blend)
    }
}

/// A noise map kernel. They all take the same parameters, so the host
/// launches them the same way; each uses the ones its noise needs.
macro_rules! map_kernel {
    ($name:ident, |$plane:ident, $fractal:ident, $x:ident, $y:ident| $value:expr) => {
        #[no_mangle]
        pub unsafe extern "ptx-kernel" fn $name(
            out: *mut f64,
            tables: *const u8,
            width: u32,
            height: u32,
            x0: f64,
            x_extent: f64,
            x_step: f64,
            y0: f64,
            y_extent: f64,
            y_step: f64,
            seamless: u32,
            octaves: u32,
            frequency: f64,
            lacunarity: f64,
            persistence: f64,
            scale: f64,
            attenuation: f64,
        ) {
            let ($x, $y) = plane_index();
            if $x >= width || $y >= height {
                return;
            }
            let $plane = Plane {
                x0,
                x_extent,
                x_step,
                y0,
                y_extent,
                y_step,
                seamless: seamless != 0,
            };
            let $fractal = Fractal {
                tables,
                octaves,
                frequency,
                lacunarity,
                persistence,
                scale,
                attenuation,
            };
            let value = $value;
            // SAFETY: the point is inside the `width × height` output.
            unsafe { *out.add($y as usize * width as usize + $x as usize) = value };
        }
    };
}

// `PlaneMapBuilder::new_fn(|p| noise.get(p))`, 2D.
map_kernel!(perlin_2d_map, |plane, fractal, x, y| plane.value(
    x,
    y,
    |px, py| perlin_2d(px, py, fractal.table(0))
));
map_kernel!(simplex_2d_map, |plane, fractal, x, y| plane.value(
    x,
    y,
    |px, py| simplex_2d(px, py, fractal.table(0))
));
// `Worley::get` scales the point by its frequency first.
map_kernel!(worley_2d_map, |plane, fractal, x, y| plane.value(
    x,
    y,
    |px, py| {
        worley_2d(
            px * fractal.frequency,
            py * fractal.frequency,
            fractal.table(0),
        )
    }
));
map_kernel!(fbm_simplex_2d_map, |plane, fractal, x, y| plane.value(
    x,
    y,
    |px, py| { fractal.fbm([px, py], |p, table| simplex_2d(p[0], p[1], table)) }
));

// `PlaneMapBuilder::new(source)`, 3D at `z = 0`.
map_kernel!(fbm_perlin_3d_map, |plane, fractal, x, y| plane.value(
    x,
    y,
    |px, py| fractal.fbm([px, py, 0.0], perlin_3d)
));
map_kernel!(ridged_perlin_3d_map, |plane, fractal, x, y| plane.value(
    x,
    y,
    |px, py| fractal.ridged([px, py, 0.0], perlin_3d)
));

// ---------------------------------------------------------------------------
// imageproc 0.25.1 and image 0.25.10, `Rgba<u8>` images

#[derive(Clone, Copy)]
struct Rgba8 {
    pixels: *const u8,
    width: u32,
    height: u32,
}

impl Rgba8 {
    /// `unsafe_get_pixel`. Callers pass coordinates inside the image.
    fn get(self, x: u32, y: u32) -> [u8; 4] {
        let index = (y as usize * self.width as usize + x as usize) * 4;
        // SAFETY: the pixel is inside the uploaded image.
        unsafe { *self.pixels.add(index).cast::<[u8; 4]>() }
    }
}

/// imageproc's `Clamp<f32> for u8`.
fn clamp_u8(x: f32) -> u8 {
    if x < 255.0 {
        if x > 0.0 {
            x as u8
        } else {
            0
        }
    } else {
        255
    }
}

/// `blend_bilinear`'s per-channel mix.
fn mix(a: [u8; 4], b: [u8; 4], weight: f32) -> [u8; 4] {
    core::array::from_fn(|c| clamp_u8((1f32 - weight) * a[c] as f32 + weight * b[c] as f32))
}

/// `interpolate_bilinear`.
fn bilinear(image: Rgba8, x: f32, y: f32, default: [u8; 4]) -> [u8; 4] {
    let left = floor32(x);
    let right = left + 1f32;
    let top = floor32(y);
    let bottom = top + 1f32;
    let right_weight = x - left;
    let bottom_weight = y - top;
    if left >= 0f32 && right < image.width as f32 && top >= 0f32 && bottom < image.height as f32 {
        let top_row = mix(
            image.get(left as u32, top as u32),
            image.get(right as u32, top as u32),
            right_weight,
        );
        let bottom_row = mix(
            image.get(left as u32, bottom as u32),
            image.get(right as u32, bottom as u32),
            right_weight,
        );
        mix(top_row, bottom_row, bottom_weight)
    } else {
        default
    }
}

/// `interpolate_nearest`.
fn nearest(image: Rgba8, x: f32, y: f32, default: [u8; 4]) -> [u8; 4] {
    if x < -0.5 || y < -0.5 {
        return default;
    }
    let rx = (x + 0.5) as u32;
    let ry = (y + 0.5) as u32;
    if rx >= image.width || ry >= image.height {
        default
    } else {
        image.get(rx, ry)
    }
}

/// How `warp_rgba8` maps output to input coordinates.
const TRANSLATION: u32 = 0;
const AFFINE: u32 = 1;
const PROJECTIVE: u32 = 2;
const BARREL: u32 = 3;

/// `warp_inner` for one output pixel, with the mapping `warp_into` or
/// `warp_into_with` chose: `Projection::map_translation`, `map_affine` or
/// `map_projective` with the inverse matrix `t`, or `barrel` (`shared.rs`)
/// with `t[0..3]` as its center and strength.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn warp_rgba8(
    out: *mut u8,
    source: *const u8,
    width: u32,
    height: u32,
    mapping: u32,
    t0: f32,
    t1: f32,
    t2: f32,
    t3: f32,
    t4: f32,
    t5: f32,
    t6: f32,
    t7: f32,
    t8: f32,
    bilinear_sampling: u32,
    default: u32,
) {
    let (x, y) = plane_index();
    if x >= width || y >= height {
        return;
    }
    let (fx, fy) = (x as f32, y as f32);
    let (px, py) = match mapping {
        TRANSLATION => (fx + t2, fy + t5),
        AFFINE => (t0 * fx + t1 * fy + t2, t3 * fx + t4 * fy + t5),
        PROJECTIVE => {
            let d = t6 * fx + t7 * fy + t8;
            ((t0 * fx + t1 * fy + t2) / d, (t3 * fx + t4 * fy + t5) / d)
        }
        BARREL => barrel(fx, fy, t0, t1, t2),
        _ => return,
    };
    let image = Rgba8 {
        pixels: source,
        width,
        height,
    };
    let default = default.to_le_bytes();
    let pixel = if bilinear_sampling != 0 {
        bilinear(image, px, py, default)
    } else {
        nearest(image, px, py, default)
    };
    let index = (y as usize * width as usize + x as usize) * 4;
    // SAFETY: the pixel is inside the output image.
    unsafe { *out.add(index).cast::<[u8; 4]>() = pixel };
}

/// image's `utils::clamp`.
fn clamp_channel(a: f64) -> f64 {
    if a < 0.0 {
        0.0
    } else if a > 255.0 {
        255.0
    } else {
        a
    }
}

/// `imageops::huerotate` for one pixel, with the host's matrix `m`.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn huerotate_rgba8(
    out: *mut u8,
    source: *const u8,
    pixels: u64,
    m0: f64,
    m1: f64,
    m2: f64,
    m3: f64,
    m4: f64,
    m5: f64,
    m6: f64,
    m7: f64,
    m8: f64,
) {
    let i = linear_index();
    if i >= pixels {
        return;
    }
    // SAFETY: the pixel is inside the uploaded image.
    let p = unsafe { *source.add(i as usize * 4).cast::<[u8; 4]>() };
    let (r, g, b, a) = (p[0] as f64, p[1] as f64, p[2] as f64, p[3] as f64);
    let new_r = m0 * r + m1 * g + m2 * b;
    let new_g = m3 * r + m4 * g + m5 * b;
    let new_b = m6 * r + m7 * g + m8 * b;
    let pixel = [
        clamp_channel(new_r) as u8,
        clamp_channel(new_g) as u8,
        clamp_channel(new_b) as u8,
        clamp_channel(a) as u8,
    ];
    // SAFETY: as above, for the output.
    unsafe { *out.add(i as usize * 4).cast::<[u8; 4]>() = pixel };
}

// ---------------------------------------------------------------------------
// sdfu 0.3.1-alpha.1 over glam 0.34.3: `Vec3A` (SSE) and `Vec2` (scalar)

fn sqrt32(x: f32) -> f32 {
    core::intrinsics::sqrtf32(x)
}

/// `_mm_max_ps` and glam's `Vec2::max`: the second operand on ties and NaNs.
fn lane_max(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}

/// sdfu's `Clamp` for `f32`: `self.max(low).min(high)` with `f32::max` and
/// `f32::min`.
fn clamp_scalar(t: f32, low: f32, high: f32) -> f32 {
    t.max(low).min(high)
}

#[derive(Clone, Copy)]
struct V3 {
    x: f32,
    y: f32,
    z: f32,
}

impl V3 {
    const ZERO: V3 = V3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    fn of(a: [f32; 3]) -> V3 {
        V3 {
            x: a[0],
            y: a[1],
            z: a[2],
        }
    }

    fn sub(self, o: V3) -> V3 {
        V3 {
            x: self.x - o.x,
            y: self.y - o.y,
            z: self.z - o.z,
        }
    }

    fn scale(self, s: f32) -> V3 {
        V3 {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    /// `dot3_in_x`: the products, then `(x + y) + z`.
    fn dot(self, o: V3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    fn length(self) -> f32 {
        sqrt32(self.dot(self))
    }

    fn abs(self) -> V3 {
        V3 {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        }
    }

    fn max(self, o: V3) -> V3 {
        V3 {
            x: lane_max(self.x, o.x),
            y: lane_max(self.y, o.y),
            z: lane_max(self.z, o.z),
        }
    }
}

/// glam's scalar `Vec2::length`.
fn length2(x: f32, y: f32) -> f32 {
    sqrt32(x * x + y * y)
}

/// `Sphere`.
fn sphere(p: V3, radius: f32) -> f32 {
    p.length() - radius
}

/// `Box<_, Dim3D>`.
fn cuboid(p: V3, dims: V3) -> f32 {
    let d = p.abs().sub(dims);
    d.max(V3::ZERO).length() + d.y.max(d.z).max(d.x).min(0.0)
}

/// `CappedCylinder` along z: `Vec2::new(|(x, y)|, z).abs() - (radius,
/// height)`.
fn capped_cylinder_z(p: V3, radius: f32, height: f32) -> f32 {
    let r = length2(p.x, p.y);
    let (dx, dy) = (r.abs() - radius, p.z.abs() - height);
    dx.max(dy).min(0.0) + length2(lane_max(dx, 0.0), lane_max(dy, 0.0))
}

/// `Cylinder` along x.
fn cylinder_x(p: V3, radius: f32) -> f32 {
    length2(p.y, p.z) - radius
}

/// `Torus`.
fn torus(p: V3, radius: f32, thickness: f32) -> f32 {
    length2(length2(p.x, p.z) - thickness, p.y) - radius
}

/// `Line`: a capsule from `a` to `b`.
fn capsule(p: V3, a: V3, b: V3, thickness: f32) -> f32 {
    let pa = p.sub(a);
    let ba = b.sub(a);
    let t = pa.dot(ba) / ba.dot(ba);
    let h = clamp_scalar(t, 0.0, 1.0);
    pa.sub(ba.scale(h)).length() - thickness
}

/// `PolySmoothMin::min`, with `Lerp::lerp`.
fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let t = 0.5 + 0.5 * (b - a) / k;
    let h = clamp_scalar(t, 0.0, 1.0);
    (b * (1.0 - h) + a * h) - k * h * (1.0 - h)
}

/// `Box::new(PART_BODY).round(..).union_smooth(boss, ..).subtract(bore)
/// .subtract(slot)`; `subtract` is `(-other).max(self)`.
fn part(p: V3) -> f32 {
    let body = cuboid(p, V3::of(PART_BODY)) - PART_ROUNDING;
    let boss = sphere(p.sub(V3::of(PART_BOSS_CENTER)), PART_BOSS_RADIUS);
    let shape = smooth_min(body, boss, PART_BLEND);
    let bore = capped_cylinder_z(p, PART_BORE[0], PART_BORE[1]);
    let shape = (-bore).max(shape);
    let slot = cylinder_x(p.sub(V3::of(PART_SLOT_OFFSET)), PART_SLOT_RADIUS);
    (-slot).max(shape)
}

/// Each ball `union_smooth`ed onto the ones before it.
fn metaballs(p: V3) -> f32 {
    let mut d = sphere(p.sub(V3::of(BALLS[0])), BALL_RADIUS);
    for ball in &BALLS[1..] {
        d = smooth_min(d, sphere(p.sub(V3::of(*ball)), BALL_RADIUS), BALL_BLEND);
    }
    d
}

/// Each edge capsule `union`ed onto the ones before it, then the torus.
fn wireframe(p: V3) -> f32 {
    let edge = |e: &[[f32; 3]; 2]| capsule(p, V3::of(e[0]), V3::of(e[1]), WIRE_THICKNESS);
    let mut d = edge(&WIRE_EDGES[0]);
    for e in &WIRE_EDGES[1..] {
        d = d.min(edge(e));
    }
    d.min(torus(p, WIRE_TORUS[0], WIRE_TORUS[1]))
}

/// Samples a scene at every point of an `n³` lattice, x fastest, at
/// `Vec3A::new(x, y, z) * step + Vec3A::splat(min)`.
macro_rules! sdf_kernel {
    ($name:ident, $scene:expr) => {
        #[no_mangle]
        pub unsafe extern "ptx-kernel" fn $name(out: *mut f32, n: u32, min: f32, step: f32) {
            let i = linear_index();
            let n = n as u64;
            // Ruling out `n == 0` also removes the remainder's panic path,
            // which would call into `core`, absent from the PTX.
            if n == 0 || i >= n * n * n {
                return;
            }
            let (x, y, z) = (i % n, i / n % n, i / (n * n));
            let p = V3 {
                x: x as f32 * step + min,
                y: y as f32 * step + min,
                z: z as f32 * step + min,
            };
            // SAFETY: `i` is inside the `n³` output.
            unsafe { *out.add(i as usize) = $scene(p) };
        }
    };
}

sdf_kernel!(sdf_sphere, |p: V3| sphere(p, 1.0));
sdf_kernel!(sdf_part, part);
sdf_kernel!(sdf_metaballs, metaballs);
sdf_kernel!(sdf_wireframe, wireframe);

// ---------------------------------------------------------------------------
// SHA-2 over `prefix ‖ nonce`, the nonce as 8 little-endian bytes

/// Byte `index` of the padded message: the host's template, with this
/// thread's nonce at `nonce_at`.
fn message_byte(template: *const u8, nonce_at: u32, nonce: u64, index: u32) -> u8 {
    if index >= nonce_at && index < nonce_at + 8 {
        (nonce >> (8 * (index - nonce_at))) as u8
    } else {
        // SAFETY: the template holds every block of the padded message.
        unsafe { *template.add(index as usize) }
    }
}

fn sha256(template: *const u8, blocks: u32, nonce_at: u32, nonce: u64) -> [u8; 32] {
    let mut state = H256_256;
    for block in 0..blocks {
        let mut w = [0u32; 64];
        for (t, word) in w.iter_mut().enumerate().take(16) {
            let at = block * 64 + t as u32 * 4;
            *word = u32::from_be_bytes(core::array::from_fn(|k| {
                message_byte(template, nonce_at, nonce, at + k as u32)
            }));
        }
        for t in 16..64 {
            let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
            let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16]
                .wrapping_add(s0)
                .wrapping_add(w[t - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for t in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K32[t])
                .wrapping_add(w[t]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (s, v) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *s = s.wrapping_add(v);
        }
    }
    let mut digest = [0u8; 32];
    for (bytes, word) in digest.chunks_exact_mut(4).zip(state) {
        bytes.copy_from_slice(&word.to_be_bytes());
    }
    digest
}

fn sha512(template: *const u8, blocks: u32, nonce_at: u32, nonce: u64) -> [u8; 64] {
    let mut state = H512_512;
    for block in 0..blocks {
        let mut w = [0u64; 80];
        for (t, word) in w.iter_mut().enumerate().take(16) {
            let at = block * 128 + t as u32 * 8;
            *word = u64::from_be_bytes(core::array::from_fn(|k| {
                message_byte(template, nonce_at, nonce, at + k as u32)
            }));
        }
        for t in 16..80 {
            let s0 = w[t - 15].rotate_right(1) ^ w[t - 15].rotate_right(8) ^ (w[t - 15] >> 7);
            let s1 = w[t - 2].rotate_right(19) ^ w[t - 2].rotate_right(61) ^ (w[t - 2] >> 6);
            w[t] = w[t - 16]
                .wrapping_add(s0)
                .wrapping_add(w[t - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for t in 0..80 {
            let s1 = e.rotate_right(14) ^ e.rotate_right(18) ^ e.rotate_right(41);
            let ch = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K64[t])
                .wrapping_add(w[t]);
            let s0 = a.rotate_right(28) ^ a.rotate_right(34) ^ a.rotate_right(39);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (s, v) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *s = s.wrapping_add(v);
        }
    }
    let mut digest = [0u8; 64];
    for (bytes, word) in digest.chunks_exact_mut(8).zip(state) {
        bytes.copy_from_slice(&word.to_be_bytes());
    }
    digest
}

/// One candidate per thread: nonce `first + i`. With `digests` set, writes
/// each whole digest; otherwise one score byte (`leading_zero_bits`).
macro_rules! hash_kernel {
    ($name:ident, $hash:ident) => {
        #[no_mangle]
        pub unsafe extern "ptx-kernel" fn $name(
            out: *mut u8,
            template: *const u8,
            blocks: u32,
            nonce_at: u32,
            first: u64,
            count: u64,
            digests: u32,
        ) {
            let i = linear_index();
            if i >= count {
                return;
            }
            let digest = $hash(template, blocks, nonce_at, first.wrapping_add(i));
            // SAFETY: the output holds `count` scores or digests.
            unsafe {
                if digests != 0 {
                    let at = out.add(i as usize * digest.len());
                    for (k, byte) in digest.iter().enumerate() {
                        *at.add(k) = *byte;
                    }
                } else {
                    *out.add(i as usize) = leading_zero_bits(&digest);
                }
            }
        }
    };
}

hash_kernel!(sha256_candidates, sha256);
hash_kernel!(sha512_candidates, sha512);
