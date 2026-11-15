//! The benchmark kernels as Rust source, for rustc's own nvptx backend.
//! Operation for operation the same as the builder versions in `main.rs`,
//! so all three compilations must agree bit for bit.

#![no_std]
#![feature(abi_ptx, core_intrinsics, stdarch_nvptx)]
#![allow(internal_features)]

use core::arch::nvptx::{_block_dim_x, _block_dim_y, _block_idx_x, _block_idx_y, _thread_idx_x, _thread_idx_y, trap};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    // SAFETY: trapping ends the kernel.
    unsafe { trap() }
}

fn x_index() -> u32 {
    // SAFETY: reading the thread's position has no preconditions.
    unsafe { (_block_idx_x() as u32).wrapping_mul(_block_dim_x() as u32).wrapping_add(_thread_idx_x() as u32) }
}

fn y_index() -> u32 {
    // SAFETY: as above.
    unsafe { (_block_idx_y() as u32).wrapping_mul(_block_dim_y() as u32).wrapping_add(_thread_idx_y() as u32) }
}

/// particular's softened brute force for one affected body, f32 in 3D.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn nbody(
    out: *mut f32,
    affected: *const f32,
    n_affected: u32,
    massive: *const f32,
    masses: *const f32,
    n_massive: u32,
    softening: f32,
) {
    let i = x_index();
    if i >= n_affected {
        return;
    }
    let base = i as u64 * 3;
    // SAFETY: the host uploads every position and mass.
    unsafe {
        let (x1, y1, z1) = (*affected.add(base as usize), *affected.add(base as usize + 1), *affected.add(base as usize + 2));
        let (mut ax, mut ay, mut az) = (0.0f32, 0.0f32, 0.0f32);
        let mut j = 0u32;
        while j < n_massive {
            let b = j as u64 * 3;
            let dx = *massive.add(b as usize) - x1;
            let dy = *massive.add(b as usize + 1) - y1;
            let dz = *massive.add(b as usize + 2) - z1;
            let mass = *masses.add(j as usize);
            let mut norm = dx * dx;
            norm = norm + dy * dy;
            norm = norm + dz * dz;
            let norm_s = norm + softening * softening;
            let (fx, fy, fz) = if norm == 0.0 {
                (dx, dy, dz)
            } else {
                let f = mass / (norm_s * core::intrinsics::sqrtf32(norm_s));
                (dx * f, dy * f, dz * f)
            };
            ax = ax + fx;
            ay = ay + fy;
            az = az + fz;
            j += 1;
        }
        *out.add(base as usize) = ax;
        *out.add(base as usize + 1) = ay;
        *out.add(base as usize + 2) = az;
    }
}

/// psdm's Hamming distance for each pair `i <= j`, upper triangle by rows.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn snp(out: *mut u64, sequences: *const u8, n: u32, len: u32) {
    let (j, i) = (x_index(), y_index());
    if i >= n || j >= n || j < i {
        return;
    }
    let (i, j, n, len) = (i as u64, j as u64, n as u64, len as u64);
    let mut distance = 0u64;
    if i != j {
        let mut k = 0u64;
        while k < len {
            // SAFETY: both sequences hold `len` bytes.
            let (a, b) = unsafe { (*sequences.add((i * len + k) as usize), *sequences.add((j * len + k) as usize)) };
            distance += (a != b && a != b'.' && b != b'.') as u64;
            k += 1;
        }
    }
    let row = i * n - i * i.saturating_sub(1) / 2;
    // SAFETY: the output holds every pair `i <= j`.
    unsafe { *out.add((row + (j - i)) as usize) = distance };
}

fn hash(x: i32, y: i32, seed: u32) -> u32 {
    let mut h = seed ^ (x as u32).wrapping_mul(0x27d4_eb2d) ^ (y as u32).wrapping_mul(0x1656_67b1);
    h = (h ^ (h >> 15)).wrapping_mul(0x85eb_ca6b);
    h = (h ^ (h >> 13)).wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

fn lattice(x: i32, y: i32, seed: u32) -> f32 {
    (hash(x, y, seed) >> 8) as f32 * (1.0 / 16_777_216.0)
}

fn value(px: f32, py: f32, seed: u32) -> f32 {
    let (fx, fy) = (core::intrinsics::floorf32(px), core::intrinsics::floorf32(py));
    let (x0, y0) = (fx as i32, fy as i32);
    let (x1, y1) = (x0.wrapping_add(1), y0.wrapping_add(1));
    let (tx, ty) = (px - fx, py - fy);
    let sx = tx * tx * (3.0 - 2.0 * tx);
    let sy = ty * ty * (3.0 - 2.0 * ty);
    let (a, b) = (lattice(x0, y0, seed), lattice(x1, y0, seed));
    let (c, d) = (lattice(x0, y1, seed), lattice(x1, y1, seed));
    let top = a + (b - a) * sx;
    let bottom = c + (d - c) * sx;
    top + (bottom - top) * sy
}

/// Fractal value noise over a `width × height` plane.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fbm(out: *mut f32, width: u32, height: u32, scale: f32, octaves: u32, seed: u32) {
    let (x, y) = (x_index(), y_index());
    if x >= width || y >= height {
        return;
    }
    let (px, py) = (x as f32 * scale, y as f32 * scale);
    let (mut sum, mut amplitude, mut frequency) = (0.0f32, 1.0f32, 1.0f32);
    let mut o = 0u32;
    while o < octaves {
        sum = sum + value(px * frequency, py * frequency, seed.wrapping_add(o)) * amplitude;
        amplitude = amplitude * 0.5;
        frequency = frequency * 2.0;
        o += 1;
    }
    // SAFETY: the output holds the whole plane.
    unsafe { *out.add((y as u64 * width as u64 + x as u64) as usize) = sum };
}
