//! Direct PTX against LLVM (`opt -O3`, `llc -O3`) on the same kernels, with
//! rustc's nvptx backend compiling the same code from source as a reference.
//!
//! Every result is checked bit for bit against the other compilations and
//! against the CPU on samples. Kernel time per launch is the difference
//! between jobs of 1 and many launches, so transfers cancel out; the GPU is
//! warmed up first and the backends take turns over several rounds.
//!
//! `cargo run -p cuprite-kernel --release --example backends`

use std::{
    path::PathBuf,
    process::Command,
    time::{Duration, Instant},
};

use cuprite_kernel::{builder::Builder, llvm::Llvm, ptx, BinOp, Const, Intrinsic, Kernel, Local, Rvalue, Special, Ty};
use cuprite_memory::{Arg, GpuMemory, Grid, MemoryConfig};

fn u32c(value: u32) -> Const {
    Const::int(Ty::U32, value.into())
}

fn u64c(value: u64) -> Const {
    Const::int(Ty::U64, value.into())
}

/// `local = local + 1`, of its own type.
fn increment(b: &mut Builder, local: Local) {
    let one = Const::int(b.ty(local), 1);
    b.set(local, Rvalue::Binary(BinOp::Add, local.into(), one.into()));
}

fn add_to(b: &mut Builder, local: Local, value: Local) {
    b.set(local, Rvalue::Binary(BinOp::Add, local.into(), value.into()));
}

// ---------------------------------------------------------------------------
// The kernels, as a MIR translator would hand them over: one statement per
// operation, helpers inlined, nothing hoisted or shared by hand.

fn nbody_kernel() -> Kernel {
    let (mut b, p) = Builder::new("nbody", &[Ty::Ptr, Ty::Ptr, Ty::U32, Ty::Ptr, Ty::Ptr, Ty::U32, Ty::F32]);
    let (out, affected, n_affected, massive, masses, n_massive, softening) = (p[0], p[1], p[2], p[3], p[4], p[5], p[6]);
    let i = b.special(Special::X);
    let outside = b.binary(BinOp::Ge, i, n_affected);
    let (start, head, body, zero, nonzero, join, exit, done) =
        (b.block(), b.block(), b.block(), b.block(), b.block(), b.block(), b.block(), b.block());
    b.branch(outside, done, start);

    b.switch_to(start);
    let wide = b.cast(i, Ty::U64);
    let base = b.binary(BinOp::Mul, wide, u64c(3));
    let mut own = Vec::new();
    for k in 0..3 {
        let at = b.binary(BinOp::Add, base, u64c(k));
        own.push(b.index(affected, at, Ty::F32));
    }
    let acceleration: Vec<Local> = (0..3).map(|_| b.copy(Const::f32(0.0))).collect();
    let force: Vec<Local> = (0..3).map(|_| b.local(Ty::F32)).collect();
    let j = b.copy(u32c(0));
    b.goto(head);

    b.switch_to(head);
    let more = b.binary(BinOp::Lt, j, n_massive);
    b.branch(more, body, exit);

    b.switch_to(body);
    let wide = b.cast(j, Ty::U64);
    let other = b.binary(BinOp::Mul, wide, u64c(3));
    let mut dir = Vec::new();
    for k in 0..3 {
        let at = b.binary(BinOp::Add, other, u64c(k));
        let position = b.index(massive, at, Ty::F32);
        dir.push(b.binary(BinOp::Sub, position, own[k as usize]));
    }
    let mass = b.index(masses, j, Ty::F32);
    let norm = b.binary(BinOp::Mul, dir[0], dir[0]);
    for &d in &dir[1..] {
        let square = b.binary(BinOp::Mul, d, d);
        add_to(&mut b, norm, square);
    }
    let soft_square = b.binary(BinOp::Mul, softening, softening);
    let norm_s = b.binary(BinOp::Add, norm, soft_square);
    let is_zero = b.binary(BinOp::Eq, norm, Const::f32(0.0));
    b.branch(is_zero, zero, nonzero);

    b.switch_to(zero);
    for k in 0..3 {
        b.set(force[k], Rvalue::Use(dir[k].into()));
    }
    b.goto(join);

    b.switch_to(nonzero);
    let root = b.call(Intrinsic::Sqrt, &[norm_s.into()]);
    let denominator = b.binary(BinOp::Mul, norm_s, root);
    let f = b.binary(BinOp::Div, mass, denominator);
    for k in 0..3 {
        b.set(force[k], Rvalue::Binary(BinOp::Mul, dir[k].into(), f.into()));
    }
    b.goto(join);

    b.switch_to(join);
    for k in 0..3 {
        add_to(&mut b, acceleration[k], force[k]);
    }
    increment(&mut b, j);
    b.goto(head);

    b.switch_to(exit);
    for (k, &value) in acceleration.iter().enumerate() {
        let at = b.binary(BinOp::Add, base, u64c(k as u64));
        let address = b.offset(out, at, Ty::F32);
        b.store(address, value);
    }
    b.goto(done);
    b.switch_to(done);
    b.ret();
    b.finish().unwrap()
}

fn snp_kernel() -> Kernel {
    let (mut b, p) = Builder::new("snp", &[Ty::Ptr, Ty::Ptr, Ty::U32, Ty::U32]);
    let (out, sequences, n, len) = (p[0], p[1], p[2], p[3]);
    let j = b.special(Special::X);
    let i = b.special(Special::Y);
    let (start, compare, head, body, exit, done) = (b.block(), b.block(), b.block(), b.block(), b.block(), b.block());
    let i_out = b.binary(BinOp::Ge, i, n);
    let j_out = b.binary(BinOp::Ge, j, n);
    let below = b.binary(BinOp::Lt, j, i);
    let either = b.binary(BinOp::Or, i_out, j_out);
    let skip = b.binary(BinOp::Or, either, below);
    b.branch(skip, done, start);

    b.switch_to(start);
    let (i, j, n, len) = (b.cast(i, Ty::U64), b.cast(j, Ty::U64), b.cast(n, Ty::U64), b.cast(len, Ty::U64));
    let distance = b.copy(u64c(0));
    let same = b.binary(BinOp::Eq, i, j);
    b.branch(same, exit, compare);

    b.switch_to(compare);
    let k = b.copy(u64c(0));
    b.goto(head);

    b.switch_to(head);
    let more = b.binary(BinOp::Lt, k, len);
    b.branch(more, body, exit);

    b.switch_to(body);
    let row_a = b.binary(BinOp::Mul, i, len);
    let at_a = b.binary(BinOp::Add, row_a, k);
    let x = b.index(sequences, at_a, Ty::U8);
    let row_b = b.binary(BinOp::Mul, j, len);
    let at_b = b.binary(BinOp::Add, row_b, k);
    let y = b.index(sequences, at_b, Ty::U8);
    let differ = b.binary(BinOp::Ne, x, y);
    let x_ok = b.binary(BinOp::Ne, x, Const::int(Ty::U8, b'.'.into()));
    let y_ok = b.binary(BinOp::Ne, y, Const::int(Ty::U8, b'.'.into()));
    let both = b.binary(BinOp::And, differ, x_ok);
    let counted = b.binary(BinOp::And, both, y_ok);
    let one = b.cast(counted, Ty::U64);
    add_to(&mut b, distance, one);
    increment(&mut b, k);
    b.goto(head);

    b.switch_to(exit);
    let positive = b.binary(BinOp::Gt, i, u64c(0));
    let minus = b.binary(BinOp::Sub, i, u64c(1));
    let saturated = b.select(positive, minus, u64c(0));
    let product = b.binary(BinOp::Mul, i, saturated);
    let half = b.binary(BinOp::Div, product, u64c(2));
    let full = b.binary(BinOp::Mul, i, n);
    let row = b.binary(BinOp::Sub, full, half);
    let column = b.binary(BinOp::Sub, j, i);
    let at = b.binary(BinOp::Add, row, column);
    let address = b.offset(out, at, Ty::U64);
    b.store(address, distance);
    b.goto(done);
    b.switch_to(done);
    b.ret();
    b.finish().unwrap()
}

fn hash(b: &mut Builder, x: Local, y: Local, seed: Local) -> Local {
    let (x, y) = (b.cast(x, Ty::U32), b.cast(y, Ty::U32));
    let hx = b.binary(BinOp::Mul, x, u32c(0x27d4_eb2d));
    let hy = b.binary(BinOp::Mul, y, u32c(0x1656_67b1));
    let h = b.binary(BinOp::Xor, seed, hx);
    let h = b.binary(BinOp::Xor, h, hy);
    let mut h = h;
    for (shift, multiplier) in [(15, 0x85eb_ca6b), (13, 0xc2b2_ae35)] {
        let shifted = b.binary(BinOp::Shr, h, u32c(shift));
        let mixed = b.binary(BinOp::Xor, h, shifted);
        h = b.binary(BinOp::Mul, mixed, u32c(multiplier));
    }
    let shifted = b.binary(BinOp::Shr, h, u32c(16));
    b.binary(BinOp::Xor, h, shifted)
}

fn lattice(b: &mut Builder, x: Local, y: Local, seed: Local) -> Local {
    let h = hash(b, x, y, seed);
    let top = b.binary(BinOp::Shr, h, u32c(8));
    let float = b.cast(top, Ty::F32);
    b.binary(BinOp::Mul, float, Const::f32(1.0 / 16_777_216.0))
}

fn smooth(b: &mut Builder, t: Local) -> Local {
    let square = b.binary(BinOp::Mul, t, t);
    let twice = b.binary(BinOp::Mul, Const::f32(2.0), t);
    let rest = b.binary(BinOp::Sub, Const::f32(3.0), twice);
    b.binary(BinOp::Mul, square, rest)
}

fn lerp(b: &mut Builder, from: Local, to: Local, t: Local) -> Local {
    let span = b.binary(BinOp::Sub, to, from);
    let part = b.binary(BinOp::Mul, span, t);
    b.binary(BinOp::Add, from, part)
}

fn value(b: &mut Builder, px: Local, py: Local, seed: Local) -> Local {
    let fx = b.call(Intrinsic::Floor, &[px.into()]);
    let fy = b.call(Intrinsic::Floor, &[py.into()]);
    let (x0, y0) = (b.cast(fx, Ty::I32), b.cast(fy, Ty::I32));
    let x1 = b.binary(BinOp::Add, x0, Const::int(Ty::I32, 1));
    let y1 = b.binary(BinOp::Add, y0, Const::int(Ty::I32, 1));
    let tx = b.binary(BinOp::Sub, px, fx);
    let ty = b.binary(BinOp::Sub, py, fy);
    let (sx, sy) = (smooth(b, tx), smooth(b, ty));
    let a = lattice(b, x0, y0, seed);
    let bb = lattice(b, x1, y0, seed);
    let c = lattice(b, x0, y1, seed);
    let d = lattice(b, x1, y1, seed);
    let top = lerp(b, a, bb, sx);
    let bottom = lerp(b, c, d, sx);
    lerp(b, top, bottom, sy)
}

fn fbm_kernel() -> Kernel {
    let (mut b, p) = Builder::new("fbm", &[Ty::Ptr, Ty::U32, Ty::U32, Ty::F32, Ty::U32, Ty::U32]);
    let (out, width, height, scale, octaves, seed) = (p[0], p[1], p[2], p[3], p[4], p[5]);
    let x = b.special(Special::X);
    let y = b.special(Special::Y);
    let (start, head, body, exit, done) = (b.block(), b.block(), b.block(), b.block(), b.block());
    let x_out = b.binary(BinOp::Ge, x, width);
    let y_out = b.binary(BinOp::Ge, y, height);
    let skip = b.binary(BinOp::Or, x_out, y_out);
    b.branch(skip, done, start);

    b.switch_to(start);
    let (fx, fy) = (b.cast(x, Ty::F32), b.cast(y, Ty::F32));
    let px = b.binary(BinOp::Mul, fx, scale);
    let py = b.binary(BinOp::Mul, fy, scale);
    let sum = b.copy(Const::f32(0.0));
    let amplitude = b.copy(Const::f32(1.0));
    let frequency = b.copy(Const::f32(1.0));
    let o = b.copy(u32c(0));
    b.goto(head);

    b.switch_to(head);
    let more = b.binary(BinOp::Lt, o, octaves);
    b.branch(more, body, exit);

    b.switch_to(body);
    let qx = b.binary(BinOp::Mul, px, frequency);
    let qy = b.binary(BinOp::Mul, py, frequency);
    let octave_seed = b.binary(BinOp::Add, seed, o);
    let v = value(&mut b, qx, qy, octave_seed);
    let weighted = b.binary(BinOp::Mul, v, amplitude);
    add_to(&mut b, sum, weighted);
    b.set(amplitude, Rvalue::Binary(BinOp::Mul, amplitude.into(), Const::f32(0.5).into()));
    b.set(frequency, Rvalue::Binary(BinOp::Mul, frequency.into(), Const::f32(2.0).into()));
    increment(&mut b, o);
    b.goto(head);

    b.switch_to(exit);
    let (wide_y, wide_x, wide_w) = (b.cast(y, Ty::U64), b.cast(x, Ty::U64), b.cast(width, Ty::U64));
    let row = b.binary(BinOp::Mul, wide_y, wide_w);
    let at = b.binary(BinOp::Add, row, wide_x);
    let address = b.offset(out, at, Ty::F32);
    b.store(address, sum);
    b.goto(done);
    b.switch_to(done);
    b.ret();
    b.finish().unwrap()
}

// ---------------------------------------------------------------------------
// CPU references, for samples of the outputs.

fn nbody_cpu(positions: &[f32], masses: &[f32], softening: f32, i: usize) -> [f32; 3] {
    let p1 = &positions[i * 3..][..3];
    let mut a = [0.0f32; 3];
    for (j, &mass) in masses.iter().enumerate() {
        let d: Vec<f32> = (0..3).map(|k| positions[j * 3 + k] - p1[k]).collect();
        let mut norm = d[0] * d[0];
        norm += d[1] * d[1];
        norm += d[2] * d[2];
        let norm_s = norm + softening * softening;
        let f = mass / (norm_s * norm_s.sqrt());
        for (sum, &dk) in a.iter_mut().zip(&d) {
            *sum += if norm == 0.0 { dk } else { dk * f };
        }
    }
    a
}

fn snp_cpu(sequences: &[u8], len: usize, i: usize, j: usize) -> u64 {
    let (a, b) = (&sequences[i * len..][..len], &sequences[j * len..][..len]);
    a.iter().zip(b).map(|(&x, &y)| u64::from(x != y && x != b'.' && y != b'.')).sum()
}

fn hash_cpu(x: i32, y: i32, seed: u32) -> u32 {
    let mut h = seed ^ (x as u32).wrapping_mul(0x27d4_eb2d) ^ (y as u32).wrapping_mul(0x1656_67b1);
    h = (h ^ (h >> 15)).wrapping_mul(0x85eb_ca6b);
    h = (h ^ (h >> 13)).wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

fn value_cpu(px: f32, py: f32, seed: u32) -> f32 {
    let lattice = |x, y| (hash_cpu(x, y, seed) >> 8) as f32 * (1.0 / 16_777_216.0);
    let (fx, fy) = (px.floor(), py.floor());
    let (x0, y0) = (fx as i32, fy as i32);
    let (x1, y1) = (x0.wrapping_add(1), y0.wrapping_add(1));
    let (tx, ty) = (px - fx, py - fy);
    let sx = tx * tx * (3.0 - 2.0 * tx);
    let sy = ty * ty * (3.0 - 2.0 * ty);
    let (a, b, c, d) = (lattice(x0, y0), lattice(x1, y0), lattice(x0, y1), lattice(x1, y1));
    let top = a + (b - a) * sx;
    let bottom = c + (d - c) * sx;
    top + (bottom - top) * sy
}

fn fbm_cpu(x: u32, y: u32, scale: f32, octaves: u32, seed: u32) -> f32 {
    let (px, py) = (x as f32 * scale, y as f32 * scale);
    let (mut sum, mut amplitude, mut frequency) = (0.0f32, 1.0f32, 1.0f32);
    for o in 0..octaves {
        sum += value_cpu(px * frequency, py * frequency, seed.wrapping_add(o)) * amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    sum
}

// ---------------------------------------------------------------------------

/// A deterministic stream of pseudo-random numbers.
struct Stream(u64);

impl Stream {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 7) as f32 / (1u64 << 24) as f32
    }
}

/// Runs one compiled kernel `launches` times in one job; returns the
/// output bytes.
type Run = dyn Fn(&mut GpuMemory, cuprite_memory::Kernel, usize) -> Vec<u8>;

/// `Err` with a description if the output is wrong.
type Check = dyn Fn(&[u8]) -> Result<(), String>;

struct Benchmark {
    name: &'static str,
    detail: String,
    kernel: Kernel,
    run: Box<Run>,
    check: Box<Check>,
}

fn benchmarks() -> Vec<Benchmark> {
    let mut stream = Stream(7);
    let bodies = 32_768usize;
    let positions: Vec<f32> = (0..bodies * 3).map(|_| stream.unit() * 200.0 - 100.0).collect();
    let mut masses: Vec<f32> = (0..bodies).map(|_| stream.unit() * 10.0 + 0.1).collect();
    masses[5] = 0.0;
    let softening = 0.01f32;
    let nbody = {
        let (gpu_positions, gpu_masses) = (positions.clone(), masses.clone());
        Benchmark {
            name: "n-body f32 3D",
            detail: format!("{bodies} bodies, all pairs"),
            kernel: nbody_kernel(),
            run: Box::new(move |memory, kernel, launches| {
                let mut out = vec![0f32; bodies * 3];
                let mut job = memory.job().unwrap();
                let p = job.input(&gpu_positions).unwrap();
                let m = job.input(&gpu_masses).unwrap();
                let o = job.output(&mut out).unwrap();
                let args = [Arg::Buffer(o), Arg::Buffer(p), Arg::U32(bodies as u32), Arg::Buffer(p), Arg::Buffer(m), Arg::U32(bodies as u32), Arg::F32(softening)];
                for _ in 0..launches {
                    job.launch(kernel, Grid::Linear(bodies as u64), &args).unwrap();
                }
                job.finish().unwrap();
                out.iter().flat_map(|v| v.to_le_bytes()).collect()
            }),
            check: Box::new(move |bytes| {
                for i in (0..64).chain(bodies - 64..bodies) {
                    let want = nbody_cpu(&positions, &masses, softening, i);
                    for k in 0..3 {
                        let got = f32::from_le_bytes(bytes[(i * 3 + k) * 4..][..4].try_into().unwrap());
                        if got.to_bits() != want[k].to_bits() {
                            return Err(format!("body {i}[{k}]: {got} against {}", want[k]));
                        }
                    }
                }
                Ok(())
            }),
        }
    };

    let (n, len) = (2048usize, 4096usize);
    let sequences: Vec<u8> = (0..n * len)
        .map(|_| match stream.next() % 20 {
            0 => b'.',
            1..=4 => b"ACGT"[(stream.next() % 4) as usize],
            _ => b'A',
        })
        .collect();
    let pairs = n * (n + 1) / 2;
    let snp = {
        let gpu_sequences = sequences.clone();
        Benchmark {
            name: "SNP distances",
            detail: format!("{n} sequences of {len} bytes"),
            kernel: snp_kernel(),
            run: Box::new(move |memory, kernel, launches| {
                let mut out = vec![0u64; pairs];
                let mut job = memory.job().unwrap();
                let s = job.input(&gpu_sequences).unwrap();
                let o = job.output(&mut out).unwrap();
                let args = [Arg::Buffer(o), Arg::Buffer(s), Arg::U32(n as u32), Arg::U32(len as u32)];
                for _ in 0..launches {
                    job.launch(kernel, Grid::Plane { width: n as u32, height: n as u32 }, &args).unwrap();
                }
                job.finish().unwrap();
                out.iter().flat_map(|v| v.to_le_bytes()).collect()
            }),
            check: Box::new(move |bytes| {
                for i in [0, 1, 2, n / 2, n - 2, n - 1] {
                    let row = i * n - i * i.saturating_sub(1) / 2;
                    for j in i..n {
                        let got = u64::from_le_bytes(bytes[(row + j - i) * 8..][..8].try_into().unwrap());
                        let want = snp_cpu(&sequences, len, i, j);
                        if got != want {
                            return Err(format!("pair ({i}, {j}): {got} against {want}"));
                        }
                    }
                }
                Ok(())
            }),
        }
    };

    let (width, height, scale, octaves, seed) = (2048u32, 2048u32, 0.013f32, 8u32, 1234u32);
    let fbm = Benchmark {
        name: "fbm value noise",
        detail: format!("{width}×{height}, {octaves} octaves"),
        kernel: fbm_kernel(),
        run: Box::new(move |memory, kernel, launches| {
            let mut out = vec![0f32; (width * height) as usize];
            let mut job = memory.job().unwrap();
            let o = job.output(&mut out).unwrap();
            let args = [Arg::Buffer(o), Arg::U32(width), Arg::U32(height), Arg::F32(scale), Arg::U32(octaves), Arg::U32(seed)];
            for _ in 0..launches {
                job.launch(kernel, Grid::Plane { width, height }, &args).unwrap();
            }
            job.finish().unwrap();
            out.iter().flat_map(|v| v.to_le_bytes()).collect()
        }),
        check: Box::new(move |bytes| {
            for at in (0..(width * height) as usize).step_by(61) {
                let (x, y) = (at as u32 % width, at as u32 / width);
                let got = f32::from_le_bytes(bytes[at * 4..][..4].try_into().unwrap());
                let want = fbm_cpu(x, y, scale, octaves, seed);
                if got.to_bits() != want.to_bits() {
                    return Err(format!("({x}, {y}): {got} against {want}"));
                }
            }
            Ok(())
        }),
    };
    vec![nbody, snp, fbm]
}

/// Compiles `reference.rs` with rustc's nvptx backend.
fn rustc_reference(dir: &std::path::Path) -> (String, Duration) {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = dir.join("reference.ptx");
    let start = Instant::now();
    let status = Command::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()))
        .current_dir(&manifest)
        .args(["--edition", "2021", "--crate-type", "cdylib", "--target", "nvptx64-nvidia-cuda"])
        .args(["-C", "opt-level=3", "-C", "target-cpu=sm_70", "--emit", "asm", "-o"])
        .arg(&out)
        .arg(manifest.join("examples/backends/reference.rs"))
        .status()
        .expect("rustc runs");
    assert!(status.success(), "rustc failed on reference.rs");
    (std::fs::read_to_string(out).unwrap(), start.elapsed())
}

/// Wall time of one job running `launches` launches.
fn job_time(memory: &mut GpuMemory, kernel: cuprite_memory::Kernel, run: &Run, launches: usize) -> f64 {
    let start = Instant::now();
    run(memory, kernel, launches);
    start.elapsed().as_secs_f64()
}

/// Kernel time per launch: a job of `1 + repeats` launches minus a job of
/// one, so transfers and job setup cancel out.
fn per_launch(memory: &mut GpuMemory, kernel: cuprite_memory::Kernel, run: &Run, repeats: usize) -> f64 {
    let one = job_time(memory, kernel, run, 1);
    let many = job_time(memory, kernel, run, 1 + repeats);
    (many - one) / repeats as f64
}

/// Instructions in a PTX entry: lines ending in `;` that aren't
/// declarations.
fn instructions(ptx: &str) -> usize {
    ptx.lines()
        .map(str::trim)
        .filter(|line| line.ends_with(';') && !line.starts_with('.') && !line.starts_with("//"))
        .count()
}

/// Rounds of interleaved measurements per benchmark.
const ROUNDS: usize = 7;
/// Kernel work per measurement.
const TARGET_SECONDS: f64 = 0.3;

fn main() {
    let dir = std::env::temp_dir().join("cuprite-backends");
    std::fs::create_dir_all(&dir).unwrap();
    let llvm = Llvm::find().expect("llvm-tools in the pinned toolchain");
    let mut memory = GpuMemory::open(MemoryConfig::default()).expect("a CUDA GPU");
    let (reference, reference_compile) = rustc_reference(&dir);
    println!("rustc reference: all three kernels compiled in {:.0} ms", reference_compile.as_secs_f64() * 1e3);
    println!("PTX written to {}", dir.display());
    println!("kernel time: median of {ROUNDS} interleaved rounds, each ≈{TARGET_SECONDS} s of launches\n");

    for benchmark in benchmarks() {
        println!("{} ({})", benchmark.name, benchmark.detail);
        let start = Instant::now();
        let direct = ptx::emit(&benchmark.kernel);
        let direct_compile = start.elapsed();
        let start = Instant::now();
        let through_llvm = llvm.compile(&benchmark.kernel).unwrap();
        let llvm_compile = start.elapsed();
        let name = benchmark.kernel.name.as_str();
        let compilations = [
            ("rustc", reference.clone(), None),
            ("direct PTX", direct, Some(direct_compile)),
            ("LLVM", through_llvm, Some(llvm_compile)),
        ];

        // Load and check each compilation.
        let mut loaded = Vec::new();
        let mut first: Option<Vec<u8>> = None;
        for (backend, text, compile) in compilations {
            let file = format!("{name}_{}.ptx", backend.replace(' ', "_").to_lowercase());
            std::fs::write(dir.join(file), &text).unwrap();
            let start = Instant::now();
            let kernel = memory.kernel(&text, name).unwrap_or_else(|e| panic!("{backend}: {e}"));
            let jit = start.elapsed();
            let output = (benchmark.run)(&mut memory, kernel, 1);
            if let Err(wrong) = (benchmark.check)(&output) {
                panic!("{backend} disagrees with the CPU: {wrong}");
            }
            match &first {
                Some(expected) => assert!(expected == &output, "{backend} disagrees with rustc"),
                None => first = Some(output),
            }
            loaded.push((backend, kernel, jit, compile, text));
        }

        // Warm the GPU's clocks up, then size the measurements.
        let warm = Instant::now();
        while warm.elapsed() < Duration::from_secs(1) {
            job_time(&mut memory, loaded[0].1, &benchmark.run, 10);
        }
        let estimate = per_launch(&mut memory, loaded[0].1, &benchmark.run, 10).max(1e-6);
        let repeats = ((TARGET_SECONDS / estimate) as usize).clamp(10, 20_000);

        let mut times = vec![Vec::new(); loaded.len()];
        for round in 0..ROUNDS {
            for offset in 0..loaded.len() {
                let which = (round + offset) % loaded.len();
                times[which].push(per_launch(&mut memory, loaded[which].1, &benchmark.run, repeats));
            }
        }
        let medians: Vec<f64> = times
            .iter_mut()
            .map(|t| {
                t.sort_by(f64::total_cmp);
                t[t.len() / 2]
            })
            .collect();

        println!("  {repeats} launches per measurement");
        println!(
            "  {:<12} {:>12} {:>10} {:>7} {:>10} {:>17} {:>8}",
            "backend", "compile", "driver JIT", "instrs", "kernel", "spread", "vs rustc"
        );
        for (i, (backend, _, jit, compile, text)) in loaded.iter().enumerate() {
            let compile = compile.map_or("(whole file)".into(), |c| format!("{:.2} ms", c.as_secs_f64() * 1e3));
            // The reference's PTX holds all three kernels.
            let count = if i == 0 { "-".to_string() } else { instructions(text).to_string() };
            let (low, high) = (times[i][0], times[i][times[i].len() - 1]);
            println!(
                "  {backend:<12} {compile:>12} {:>7.1} ms {count:>7} {:>7.3} ms {:>7.3}–{:<6.3} ms {:>7.2}×",
                jit.as_secs_f64() * 1e3,
                medians[i] * 1e3,
                low * 1e3,
                high * 1e3,
                medians[0] / medians[i]
            );
        }
        println!("  outputs: identical across all three, and match the CPU on samples\n");
    }
}
