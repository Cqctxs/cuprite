//! CUDA backend built on cudarc: one context, one stream, byte buffers
//! (`CudaSlice<u8>`), and PTX kernels the driver compiles.
//!
//! cudarc's async copies and launches are queued on the stream; the
//! manager's explicit `synchronize` calls are what establish completion.
//! Device buffers free themselves in stream order when dropped, after work
//! already queued.

use std::sync::Arc;

use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
    },
    nvrtc::Ptx,
};

use crate::{
    backend::{MemoryBackend, Value},
    device::fnv1a,
    step::MAP_PTX,
    DeviceInfo, Grid, MemoryConfig, MemoryError,
};

pub(crate) struct CudaBackend {
    pub(crate) stream: Arc<CudaStream>,
}

impl CudaBackend {
    pub(crate) fn open(config: MemoryConfig) -> Result<(Self, DeviceInfo), MemoryError> {
        // cudarc panics on first use if the driver library cannot be loaded,
        // so probe for it to report a missing driver as an ordinary error.
        // SAFETY: this loads the CUDA driver library, whose initialization
        // code runs.
        //
        // KNOWN ISSUE (DLL hijacking, deferred): cudarc loads by bare name,
        // trying `cuda.dll`, `cuda64.dll`, ... before `nvcuda.dll`, with the
        // default Windows search order. None of the `cuda*.dll` names exist
        // legitimately, so a planted one in the working directory or any PATH
        // directory is loaded even when the real driver is installed. A fix
        // is on the local branch `cudarc-dll-fix`. Linux `dlopen` does not
        // search the working directory.
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return Err(MemoryError::Unavailable(
                "CUDA driver library not found".into(),
            ));
        }
        let unavailable = |error: DriverError| MemoryError::Unavailable(error.to_string());
        let context = CudaContext::new(config.device_ordinal).map_err(unavailable)?;
        let stream = context.new_stream().map_err(unavailable)?;
        let mut driver_version = 0;
        // SAFETY: the driver library is loaded (checked above), and the call
        // writes one `c_int` through a valid pointer.
        let status = unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut driver_version) };
        status.result().map_err(unavailable)?;
        let device = DeviceInfo {
            name: context.name().map_err(unavailable)?,
            compute_capability: context.compute_capability().map_err(unavailable)?,
            driver_version,
            kernels: fnv1a(MAP_PTX.as_bytes()),
        };
        Ok((Self { stream }, device))
    }
}

fn check_lengths(source: usize, destination: usize) -> Result<(), MemoryError> {
    // cudarc asserts (panics) on a short destination; return an error instead.
    if source != destination {
        return Err(MemoryError::Transfer(format!(
            "source has {source} bytes but destination has {destination}"
        )));
    }
    Ok(())
}

fn blocks(threads: u64, per_block: u32) -> Result<u32, MemoryError> {
    u32::try_from(threads.div_ceil(u64::from(per_block)))
        .map_err(|_| MemoryError::Launch(format!("{threads} threads need too many blocks")))
}

fn config(grid: Grid) -> Result<LaunchConfig, MemoryError> {
    let (grid_dim, block_dim) = match grid {
        Grid::Linear(threads) => (
            (blocks(threads, Grid::LINEAR_BLOCK)?, 1, 1),
            (Grid::LINEAR_BLOCK, 1, 1),
        ),
        Grid::Plane { width, height } => {
            let side = Grid::PLANE_BLOCK;
            let rows = blocks(height.into(), side)?;
            // CUDA allows at most 65535 blocks in y.
            if rows > 65_535 {
                return Err(MemoryError::Launch(format!(
                    "{height} rows need too many blocks"
                )));
            }
            ((blocks(width.into(), side)?, rows, 1), (side, side, 1))
        }
    };
    Ok(LaunchConfig {
        grid_dim,
        block_dim,
        shared_mem_bytes: 0,
    })
}

impl MemoryBackend for CudaBackend {
    type Buffer = CudaSlice<u8>;
    type Kernel = CudaFunction;

    fn allocate(&mut self, bytes: usize) -> Result<Self::Buffer, MemoryError> {
        // Zeroed storage avoids handing out uninitialized device memory.
        self.stream
            .alloc_zeros::<u8>(bytes)
            .map_err(|error| MemoryError::DeviceAllocation(error.to_string()))
    }

    fn upload(&mut self, buffer: &mut Self::Buffer, input: &[u8]) -> Result<(), MemoryError> {
        check_lengths(input.len(), buffer.len())?;
        self.stream
            .memcpy_htod(input, buffer)
            .map_err(|error| MemoryError::Transfer(error.to_string()))
    }

    fn download(&mut self, buffer: &Self::Buffer, output: &mut [u8]) -> Result<(), MemoryError> {
        check_lengths(buffer.len(), output.len())?;
        self.stream
            .memcpy_dtoh(buffer, output)
            .map_err(|error| MemoryError::Transfer(error.to_string()))
    }

    fn zero(&mut self, buffer: &mut Self::Buffer) -> Result<(), MemoryError> {
        self.stream
            .memset_zeros(buffer)
            .map_err(|error| MemoryError::Transfer(error.to_string()))
    }

    fn load(&mut self, ptx: &str, name: &str) -> Result<Self::Kernel, MemoryError> {
        self.stream
            .context()
            .load_module(Ptx::from_src(ptx))
            .and_then(|module| module.load_function(name))
            .map_err(|error| MemoryError::Kernel(format!("`{name}`: {error}")))
    }

    fn launch(
        &mut self,
        kernel: &Self::Kernel,
        grid: Grid,
        args: &[Value<'_, Self::Buffer>],
    ) -> Result<(), MemoryError> {
        let config = config(grid)?;
        // An empty buffer has no storage; kernels see a null pointer.
        let null = 0u64;
        let mut launch = self.stream.launch_builder(kernel);
        for arg in args {
            match arg {
                Value::Buffer(Some(buffer)) => launch.arg(*buffer),
                Value::Buffer(None) => launch.arg(&null),
                Value::U32(value) => launch.arg(value),
                Value::I32(value) => launch.arg(value),
                Value::U64(value) => launch.arg(value),
                Value::I64(value) => launch.arg(value),
                Value::F32(value) => launch.arg(value),
                Value::F64(value) => launch.arg(value),
            };
        }
        // SAFETY: the arguments live until the launch is queued. A kernel
        // whose parameters don't match them, or that misbehaves, can only
        // write device memory or fault the device: buffers are plain bytes
        // that any values may fill, no CPU memory is mapped to the device,
        // and a fault is reported by the next synchronize.
        unsafe { launch.launch(config) }
            .map(|_| ())
            .map_err(|error| MemoryError::Launch(error.to_string()))
    }

    fn synchronize(&mut self) -> Result<(), MemoryError> {
        // Also reports errors cudarc recorded from drops, such as a failed
        // free, and faults in kernels.
        self.stream
            .synchronize()
            .map_err(|error| MemoryError::Completion(error.to_string()))
    }
}

#[cfg(test)]
mod bench;
#[cfg(test)]
mod differential;

/// Test kernels and helpers shared by the hardware tests and benchmark.
#[cfg(test)]
pub(crate) mod fixture {
    use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};

    use super::*;
    use crate::{manager::Manager, Arg, Kernel};

    /// Handwritten PTX, which the driver compiles for the installed GPU, so
    /// no CUDA Toolkit is needed. Both kernels take `(values, len)`:
    /// - `add_seven` adds 7 to each element with wrapping (memory-bound).
    /// - `lcg64` applies [`LCG_ROUNDS`] wrapping steps of [`lcg_step`] to each
    ///   element (compute-bound: 64 multiply-adds per 8 bytes moved).
    pub(crate) const KERNELS_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

.visible .entry add_seven(
    .param .u64 values,
    .param .u64 len
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [values];
    ld.param.u64 %rd2, [len];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    cvt.u64.u32 %rd3, %r3;
    mad.wide.u32 %rd4, %r1, %r2, %rd3;
    setp.ge.u64 %p1, %rd4, %rd2;
    @%p1 bra DONE;
    cvta.to.global.u64 %rd5, %rd1;
    shl.b64 %rd6, %rd4, 2;
    add.s64 %rd7, %rd5, %rd6;
    ld.global.u32 %r4, [%rd7];
    add.u32 %r5, %r4, 7;
    st.global.u32 [%rd7], %r5;
DONE:
    ret;
}

.visible .entry lcg64(
    .param .u64 values,
    .param .u64 len
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [values];
    ld.param.u64 %rd2, [len];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    cvt.u64.u32 %rd3, %r3;
    mad.wide.u32 %rd4, %r1, %r2, %rd3;
    setp.ge.u64 %p1, %rd4, %rd2;
    @%p1 bra DONE;
    cvta.to.global.u64 %rd5, %rd1;
    shl.b64 %rd6, %rd4, 2;
    add.s64 %rd7, %rd5, %rd6;
    ld.global.u32 %r4, [%rd7];
    mov.u32 %r5, 0;
LOOP:
    mad.lo.u32 %r4, %r4, 1664525, 1013904223;
    add.u32 %r5, %r5, 1;
    setp.lt.u32 %p2, %r5, 64;
    @%p2 bra LOOP;
    st.global.u32 [%rd7], %r4;
DONE:
    ret;
}
"#;

    pub(crate) const LCG_ROUNDS: usize = 64;

    /// CPU reference for one `lcg64` round.
    pub(crate) fn lcg_step(value: u32) -> u32 {
        value.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
    }

    pub(crate) fn add_seven(manager: &mut Manager<CudaBackend>) -> Kernel {
        manager.kernel(KERNELS_PTX, "add_seven").unwrap()
    }

    pub(crate) fn lcg64(manager: &mut Manager<CudaBackend>) -> Kernel {
        manager.kernel(KERNELS_PTX, "lcg64").unwrap()
    }

    /// Applies a `(values, len)` kernel `reps` times to `values` in one job,
    /// committing it if `commit` and discarding it otherwise.
    pub(crate) fn run_job(
        manager: &mut Manager<CudaBackend>,
        values: &mut [u32],
        kernel: Kernel,
        reps: usize,
        commit: bool,
    ) {
        let len = values.len() as u64;
        let mut job = manager.job().unwrap();
        let buffer = job.in_out(values).unwrap();
        for _ in 0..reps {
            job.launch(
                kernel,
                Grid::Linear(len),
                &[Arg::Buffer(buffer), Arg::U64(len)],
            )
            .unwrap();
        }
        if commit {
            job.finish().unwrap();
        } else {
            drop(job);
        }
    }

    /// A fixture kernel loaded directly through cudarc, for the benchmark's
    /// baseline without the manager.
    pub(crate) fn raw_function(stream: &Arc<CudaStream>, name: &str) -> CudaFunction {
        stream
            .context()
            .load_module(Ptx::from_src(KERNELS_PTX))
            .unwrap()
            .load_function(name)
            .unwrap()
    }

    /// Queue one fixture kernel over `buffer` on `stream`, which must own it.
    pub(crate) fn launch_on(
        stream: &Arc<CudaStream>,
        function: &CudaFunction,
        buffer: &mut CudaSlice<u32>,
    ) {
        let len = buffer.len() as u64;
        let blocks = u32::try_from(len.div_ceil(256)).unwrap();
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(function);
        launch.arg(buffer).arg(&len);
        // SAFETY: both fixture kernels take a u32 pointer and a u64 length, matching
        // the arguments pushed above. It writes only indices below `len`,
        // which is the buffer's element count, and runs on the buffer's stream.
        unsafe { launch.launch(config) }.unwrap();
    }

    pub(crate) fn input(len: usize) -> Vec<u32> {
        // Cover zero, small values, and values that wrap when 7 is added.
        let specials = [0, 1, u32::MAX, u32::MAX - 6, u32::MAX - 7];
        (0..len)
            .map(|i| {
                specials
                    .get(i)
                    .copied()
                    .unwrap_or_else(|| (i as u32).wrapping_mul(2_654_435_761))
            })
            .collect()
    }
}

/// Hardware tests: they need an NVIDIA GPU and fail without one, since
/// enabling the `cuda` feature declares this a GPU test host.
#[cfg(test)]
mod tests {
    use super::fixture::{input, lcg_step, KERNELS_PTX, LCG_ROUNDS};
    use crate::{Arg, GpuMemory, Grid, MapStep, MemoryConfig, MemoryError};

    const LENGTHS: [usize; 6] = [0, 1, 255, 256, 257, 1025];

    /// Kernels exercising the general job API: several buffers, `f32`,
    /// `f64` and `u8` elements, scalar arguments, and a 2D grid. Float
    /// arithmetic uses explicit rounding (`.rn`), which the driver never
    /// fuses into a multiply-add, so results match the CPU bit for bit.
    const GENERAL_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

// out[i] = a * x[i] + y[i]
.visible .entry axpy_f32(
    .param .u64 x,
    .param .u64 y,
    .param .u64 out,
    .param .f32 a,
    .param .u64 len
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .f32 %f<5>;
    .reg .b64 %rd<12>;

    ld.param.u64 %rd1, [x];
    ld.param.u64 %rd2, [y];
    ld.param.u64 %rd3, [out];
    ld.param.f32 %f1, [a];
    ld.param.u64 %rd4, [len];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    cvt.u64.u32 %rd5, %r3;
    mad.wide.u32 %rd6, %r1, %r2, %rd5;
    setp.ge.u64 %p1, %rd6, %rd4;
    @%p1 bra DONE;
    shl.b64 %rd7, %rd6, 2;
    cvta.to.global.u64 %rd8, %rd1;
    add.s64 %rd8, %rd8, %rd7;
    ld.global.f32 %f2, [%rd8];
    cvta.to.global.u64 %rd9, %rd2;
    add.s64 %rd9, %rd9, %rd7;
    ld.global.f32 %f3, [%rd9];
    mul.rn.f32 %f4, %f1, %f2;
    add.rn.f32 %f4, %f4, %f3;
    cvta.to.global.u64 %rd10, %rd3;
    add.s64 %rd10, %rd10, %rd7;
    st.global.f32 [%rd10], %f4;
DONE:
    ret;
}

// out[y * width + x] = x as f64 * sx + y as f64 * sy, over a 2D grid
.visible .entry plane_f64(
    .param .u64 out,
    .param .u32 width,
    .param .u32 height,
    .param .f64 sx,
    .param .f64 sy
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<8>;
    .reg .f64 %fd<6>;
    .reg .b64 %rd<5>;

    ld.param.u64 %rd1, [out];
    ld.param.u32 %r1, [width];
    ld.param.u32 %r2, [height];
    ld.param.f64 %fd1, [sx];
    ld.param.f64 %fd2, [sy];
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.u32 %r6, %r3, %r4, %r5;
    mov.u32 %r3, %ctaid.y;
    mov.u32 %r4, %ntid.y;
    mov.u32 %r5, %tid.y;
    mad.lo.u32 %r7, %r3, %r4, %r5;
    setp.ge.u32 %p1, %r6, %r1;
    setp.ge.u32 %p2, %r7, %r2;
    or.pred %p1, %p1, %p2;
    @%p1 bra DONE;
    cvt.rn.f64.u32 %fd3, %r6;
    cvt.rn.f64.u32 %fd4, %r7;
    mul.rn.f64 %fd3, %fd3, %fd1;
    mul.rn.f64 %fd4, %fd4, %fd2;
    add.rn.f64 %fd5, %fd3, %fd4;
    mul.wide.u32 %rd2, %r7, %r1;
    cvt.u64.u32 %rd3, %r6;
    add.s64 %rd2, %rd2, %rd3;
    shl.b64 %rd2, %rd2, 3;
    cvta.to.global.u64 %rd4, %rd1;
    add.s64 %rd4, %rd4, %rd2;
    st.global.f64 [%rd4], %fd5;
DONE:
    ret;
}

// dst[i] = src[i] + 1, wrapping u8
.visible .entry bump_u8(
    .param .u64 src,
    .param .u64 dst,
    .param .u64 len
)
{
    .reg .pred %p<2>;
    .reg .b16 %h<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<7>;

    ld.param.u64 %rd1, [src];
    ld.param.u64 %rd2, [dst];
    ld.param.u64 %rd3, [len];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    cvt.u64.u32 %rd4, %r3;
    mad.wide.u32 %rd4, %r1, %r2, %rd4;
    setp.ge.u64 %p1, %rd4, %rd3;
    @%p1 bra DONE;
    cvta.to.global.u64 %rd5, %rd1;
    add.s64 %rd5, %rd5, %rd4;
    ld.global.u8 %h1, [%rd5];
    add.u16 %h1, %h1, 1;
    cvta.to.global.u64 %rd6, %rd2;
    add.s64 %rd6, %rd6, %rd4;
    st.global.u8 [%rd6], %h1;
DONE:
    ret;
}
"#;

    fn open() -> GpuMemory {
        GpuMemory::open(MemoryConfig::default()).unwrap()
    }

    /// Floats covering signs, zeros, rounding, subnormals and infinities.
    fn floats(len: usize) -> Vec<f32> {
        let specials = [
            0.0,
            -0.0,
            1.0,
            -1.5,
            0.1,
            1e-40,
            -1e-45,
            3.4e38,
            f32::INFINITY,
        ];
        (0..len)
            .map(|i| {
                specials
                    .get(i)
                    .copied()
                    .unwrap_or_else(|| f32::from_bits((i as u32).wrapping_mul(2_654_435_761) >> 2))
            })
            .collect()
    }

    #[test]
    fn repeated_kernels_share_one_buffer_with_one_upload_and_download() {
        let mut memory = open();
        let add_seven = memory.kernel(KERNELS_PTX, "add_seven").unwrap();

        for len in LENGTHS {
            let mut values = input(len);
            let expected: Vec<u32> = values.iter().map(|v| v.wrapping_add(56)).collect();
            let before = memory.stats();

            let mut job = memory.job().unwrap();
            let buffer = job.in_out(&mut values).unwrap();
            let args = [Arg::Buffer(buffer), Arg::U64(len as u64)];
            for _ in 0..8 {
                job.launch(add_seven, Grid::Linear(len as u64), &args)
                    .unwrap();
            }
            job.finish().unwrap();

            assert_eq!(values, expected, "length {len}");
            let after = memory.stats();
            let nonempty = u64::from(len > 0);
            assert_eq!(after.uploads - before.uploads, nonempty);
            assert_eq!(after.downloads - before.downloads, nonempty);
            assert_eq!(
                after.device_allocations - before.device_allocations,
                nonempty
            );
        }
    }

    #[test]
    fn the_device_reports_its_identity() {
        let memory = open();
        let device = memory.device();
        assert!(!device.name.is_empty());
        assert!(device.compute_capability.0 >= 5, "{device:?}");
        assert!(device.driver_version >= 12000, "{device:?}");
        assert_eq!(
            device.kernels,
            crate::device::fnv1a(crate::step::MAP_PTX.as_bytes())
        );
    }

    #[test]
    fn map_steps_match_the_cpu_reference() {
        let mut memory = open();
        let steps = [
            MapStep { mul: 3, add: 7 },
            MapStep {
                mul: u32::MAX,
                add: u32::MAX,
            },
            MapStep { mul: 0, add: 1 },
            MapStep { mul: 1, add: 0 },
        ];

        for len in LENGTHS {
            let mut values = input(len);
            let expected: Vec<u32> = values
                .iter()
                .map(|&v| steps.iter().fold(v, |x, step| step.apply(x)))
                .collect();

            let mut session = memory.begin_mut(&mut values).unwrap();
            for step in steps {
                session.map(step).unwrap();
            }
            session.finish().unwrap();

            assert_eq!(values, expected, "length {len}");
        }
    }

    #[test]
    fn dropping_a_job_discards_gpu_writes() {
        let mut memory = open();
        let add_seven = memory.kernel(KERNELS_PTX, "add_seven").unwrap();
        let mut values = input(257);
        let original = values.clone();

        let mut job = memory.job().unwrap();
        let buffer = job.in_out(&mut values).unwrap();
        job.launch(
            add_seven,
            Grid::Linear(257),
            &[Arg::Buffer(buffer), Arg::U64(257)],
        )
        .unwrap();
        drop(job);

        assert_eq!(values, original);
        assert_eq!(memory.stats().downloads, 0);

        // The manager stays usable after a discarded job.
        memory.begin_mut(&mut values).unwrap().finish().unwrap();
    }

    #[test]
    fn the_compute_kernel_matches_the_cpu_reference() {
        let mut memory = open();
        let lcg = memory.kernel(KERNELS_PTX, "lcg64").unwrap();
        let mut values = input(1025);
        let expected: Vec<u32> = values
            .iter()
            .map(|&v| (0..LCG_ROUNDS).fold(v, |x, _| lcg_step(x)))
            .collect();

        let mut job = memory.job().unwrap();
        let buffer = job.in_out(&mut values).unwrap();
        job.launch(
            lcg,
            Grid::Linear(1025),
            &[Arg::Buffer(buffer), Arg::U64(1025)],
        )
        .unwrap();
        job.finish().unwrap();

        assert_eq!(values, expected);
    }

    #[test]
    fn a_later_job_sees_cpu_changes_made_in_between() {
        let mut memory = open();
        let add_seven = memory.kernel(KERNELS_PTX, "add_seven").unwrap();
        let mut values = input(1025);
        let run = |memory: &mut GpuMemory, values: &mut [u32]| {
            let mut job = memory.job().unwrap();
            let buffer = job.in_out(values).unwrap();
            job.launch(
                add_seven,
                Grid::Linear(1025),
                &[Arg::Buffer(buffer), Arg::U64(1025)],
            )
            .unwrap();
            job.finish().unwrap();
        };

        run(&mut memory, &mut values);
        // Change the same allocation in place; the next job must upload it.
        for value in &mut values {
            *value ^= 0xA5A5_A5A5;
        }
        let expected: Vec<u32> = values.iter().map(|v| v.wrapping_add(7)).collect();
        run(&mut memory, &mut values);

        assert_eq!(values, expected);
        let stats = memory.stats();
        assert_eq!((stats.uploads, stats.downloads), (2, 2));
        assert_eq!((stats.device_allocations, stats.pool_hits), (1, 1));
    }

    /// Two inputs, an output and a float scalar: `a * x + y` in `f32`
    /// matches the CPU bit for bit, including signed zeros, subnormals,
    /// infinities and NaNs.
    #[test]
    fn several_float_buffers_match_the_cpu_bit_for_bit() {
        let mut memory = open();
        let axpy = memory.kernel(GENERAL_PTX, "axpy_f32").unwrap();

        for len in LENGTHS {
            let x = floats(len);
            let y: Vec<f32> = floats(len).into_iter().rev().collect();
            let a = -0.3f32;
            let mut out = vec![f32::NAN; len];

            let mut job = memory.job().unwrap();
            let bx = job.input(&x).unwrap();
            let by = job.input(&y).unwrap();
            let bout = job.output(&mut out).unwrap();
            let args = [
                Arg::Buffer(bx),
                Arg::Buffer(by),
                Arg::Buffer(bout),
                Arg::F32(a),
                Arg::U64(len as u64),
            ];
            job.launch(axpy, Grid::Linear(len as u64), &args).unwrap();
            job.finish().unwrap();

            let expected: Vec<u32> = x
                .iter()
                .zip(&y)
                .map(|(x, y)| (a * x + y).to_bits())
                .collect();
            let actual: Vec<u32> = out.iter().map(|v| v.to_bits()).collect();
            if len > 0 {
                assert_eq!(actual, expected, "length {len}");
            } else {
                assert!(out.is_empty());
            }
        }
    }

    /// A 2D grid over a plane that isn't a multiple of the block size, with
    /// `f64` scalars: every point is written once, exactly as on the CPU.
    #[test]
    fn a_plane_grid_covers_every_point_once() {
        let mut memory = open();
        let plane = memory.kernel(GENERAL_PTX, "plane_f64").unwrap();
        let (sx, sy) = (0.1f64, -3.7f64);

        for (width, height) in [(1u32, 1u32), (17, 5), (16, 16), (300, 31)] {
            let mut out = vec![f64::NAN; (width * height) as usize];
            let mut job = memory.job().unwrap();
            let buffer = job.output(&mut out).unwrap();
            let args = [
                Arg::Buffer(buffer),
                Arg::U32(width),
                Arg::U32(height),
                Arg::F64(sx),
                Arg::F64(sy),
            ];
            job.launch(plane, Grid::Plane { width, height }, &args)
                .unwrap();
            job.finish().unwrap();

            for y in 0..height {
                for x in 0..width {
                    let expected = f64::from(x) * sx + f64::from(y) * sy;
                    let actual = out[(y * width + x) as usize];
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "({x}, {y}) of {width}x{height}"
                    );
                }
            }
        }
    }

    /// `u8` elements at byte granularity, a scratch buffer that keeps its
    /// contents across launches, and one buffer passed as both source and
    /// destination.
    #[test]
    fn scratch_persists_across_launches_and_bytes_wrap() {
        let mut memory = open();
        let bump = memory.kernel(GENERAL_PTX, "bump_u8").unwrap();
        let input: Vec<u8> = (0..=255).chain(250..=255).collect();
        let len = input.len();
        let mut out = vec![0u8; len];

        let mut job = memory.job().unwrap();
        let source = job.input(&input).unwrap();
        let scratch = job.scratch::<u8>(len).unwrap();
        let output = job.output(&mut out).unwrap();
        let grid = Grid::Linear(len as u64);
        let n = Arg::U64(len as u64);
        job.launch(bump, grid, &[Arg::Buffer(source), Arg::Buffer(scratch), n])
            .unwrap();
        job.launch(bump, grid, &[Arg::Buffer(scratch), Arg::Buffer(scratch), n])
            .unwrap();
        job.launch(bump, grid, &[Arg::Buffer(scratch), Arg::Buffer(output), n])
            .unwrap();
        job.finish().unwrap();

        let expected: Vec<u8> = input.iter().map(|v| v.wrapping_add(3)).collect();
        assert_eq!(out, expected);
        // Only the output comes back; the scratch buffer never leaves the GPU.
        assert_eq!(memory.stats().downloaded_bytes, len as u64);
    }

    #[test]
    fn ptx_the_driver_rejects_is_a_kernel_error() {
        let mut memory = open();
        let error = memory
            .kernel(".version 7.0\nnot ptx", "missing")
            .unwrap_err();
        assert!(matches!(error, MemoryError::Kernel(_)), "{error:?}");
        let error = memory.kernel(GENERAL_PTX, "no_such_entry").unwrap_err();
        assert!(matches!(error, MemoryError::Kernel(_)), "{error:?}");

        // Neither failure affects later work.
        let mut values = [1u32];
        memory.begin_mut(&mut values).unwrap().finish().unwrap();
    }
}
