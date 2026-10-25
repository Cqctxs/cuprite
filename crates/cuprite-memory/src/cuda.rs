//! CUDA backend built on cudarc: one context, one stream, `CudaSlice<u32>` buffers.
//!
//! cudarc's async copies are queued on the stream; the manager's explicit
//! `synchronize` calls are what establish completion. Device buffers free
//! themselves in stream order when dropped, after work already queued.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DriverError};

use crate::{backend::MemoryBackend, MemoryConfig, MemoryError};

pub(crate) struct CudaBackend {
    pub(crate) stream: Arc<CudaStream>,
}

impl CudaBackend {
    pub(crate) fn open(config: MemoryConfig) -> Result<Self, MemoryError> {
        // cudarc panics on first use if the driver library cannot be loaded,
        // so probe for it to report a missing driver as an ordinary error.
        // SAFETY: this loads the system CUDA driver library (`nvcuda.dll` on
        // Windows, `libcuda.so` on Linux), whose initialization code runs.
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return Err(MemoryError::Unavailable(
                "CUDA driver library not found".into(),
            ));
        }
        let unavailable = |error: DriverError| MemoryError::Unavailable(error.to_string());
        let context = CudaContext::new(config.device_ordinal).map_err(unavailable)?;
        let stream = context.new_stream().map_err(unavailable)?;
        Ok(Self { stream })
    }
}

fn check_lengths(source: usize, destination: usize) -> Result<(), MemoryError> {
    // cudarc asserts (panics) on a short destination; return an error instead.
    if source != destination {
        return Err(MemoryError::Transfer(format!(
            "source has {source} elements but destination has {destination}"
        )));
    }
    Ok(())
}

impl MemoryBackend for CudaBackend {
    type Buffer = CudaSlice<u32>;

    fn allocate(&mut self, len: usize) -> Result<Self::Buffer, MemoryError> {
        // Zeroed storage avoids handing out uninitialized device memory.
        self.stream
            .alloc_zeros::<u32>(len)
            .map_err(|error| MemoryError::DeviceAllocation(error.to_string()))
    }

    fn upload(&mut self, buffer: &mut Self::Buffer, input: &[u32]) -> Result<(), MemoryError> {
        check_lengths(input.len(), buffer.len())?;
        self.stream
            .memcpy_htod(input, buffer)
            .map_err(|error| MemoryError::Transfer(error.to_string()))
    }

    fn download(&mut self, buffer: &Self::Buffer, output: &mut [u32]) -> Result<(), MemoryError> {
        check_lengths(buffer.len(), output.len())?;
        self.stream
            .memcpy_dtoh(buffer, output)
            .map_err(|error| MemoryError::Transfer(error.to_string()))
    }

    fn synchronize(&mut self) -> Result<(), MemoryError> {
        // Also reports errors cudarc recorded from drops, such as a failed free.
        self.stream
            .synchronize()
            .map_err(|error| MemoryError::Completion(error.to_string()))
    }
}

/// Hardware tests: they need an NVIDIA GPU and fail without one, since
/// enabling the `cuda` feature declares this a GPU test host.
#[cfg(test)]
mod tests {
    use cudarc::{
        driver::{CudaFunction, LaunchConfig, PushKernelArg},
        nvrtc::Ptx,
    };

    use super::*;
    use crate::{session::Session, GpuMemory};

    /// Adds 7 to each element with wrapping. Handwritten PTX, which the
    /// driver compiles for the installed GPU, so no CUDA Toolkit is needed.
    const ADD_SEVEN_PTX: &str = r#"
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
"#;

    const LENGTHS: [usize; 6] = [0, 1, 255, 256, 257, 1025];

    fn add_seven(stream: &Arc<CudaStream>) -> CudaFunction {
        stream
            .context()
            .load_module(Ptx::from_src(ADD_SEVEN_PTX))
            .unwrap()
            .load_function("add_seven")
            .unwrap()
    }

    /// Queue one `add_seven` over the session's buffer, if it has one.
    fn launch_add_seven(session: &mut Session<'_, CudaBackend>, function: &CudaFunction) {
        let Some((backend, buffer)) = session.write_parts() else {
            return; // Empty sessions have no buffer and never reach a kernel.
        };
        let len = buffer.len() as u64;
        let blocks = u32::try_from(len.div_ceil(256)).unwrap();
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = backend.stream.launch_builder(function);
        launch.arg(buffer).arg(&len);
        // SAFETY: the kernel takes a u32 pointer and a u64 length, matching
        // the arguments pushed above. It writes only indices below `len`,
        // which is the buffer's element count, and runs on the buffer's stream.
        unsafe { launch.launch(config) }.unwrap();
    }

    fn input(len: usize) -> Vec<u32> {
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

    #[test]
    fn repeated_kernels_share_one_buffer_with_one_upload_and_download() {
        let mut memory = GpuMemory::open(MemoryConfig::default()).unwrap();

        for len in LENGTHS {
            let mut values = input(len);
            let expected: Vec<u32> = values.iter().map(|v| v.wrapping_add(56)).collect();
            let before = memory.stats();

            let mut session = memory.begin_mut(&mut values).unwrap();
            let function = session
                .inner_mut()
                .write_parts()
                .map(|(backend, _)| add_seven(&backend.stream));
            if let Some(function) = function {
                for _ in 0..8 {
                    launch_add_seven(session.inner_mut(), &function);
                }
            }
            session.finish().unwrap();

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
    fn dropping_a_session_discards_gpu_writes() {
        let mut memory = GpuMemory::open(MemoryConfig::default()).unwrap();
        let mut values = input(257);
        let original = values.clone();

        let mut session = memory.begin_mut(&mut values).unwrap();
        let (backend, _) = session.inner_mut().write_parts().unwrap();
        let function = add_seven(&backend.stream);
        launch_add_seven(session.inner_mut(), &function);
        drop(session);

        assert_eq!(values, original);
        assert_eq!(memory.stats().downloads, 0);

        // The manager stays usable after a discarded session.
        memory.begin_mut(&mut values).unwrap().finish().unwrap();
    }

    #[test]
    fn a_later_session_sees_cpu_changes_made_in_between() {
        let mut memory = GpuMemory::open(MemoryConfig::default()).unwrap();
        let mut values = input(1025);

        let mut session = memory.begin_mut(&mut values).unwrap();
        let (backend, _) = session.inner_mut().write_parts().unwrap();
        let function = add_seven(&backend.stream);
        launch_add_seven(session.inner_mut(), &function);
        session.finish().unwrap();

        // Change the same allocation in place; the next session must upload it.
        for value in &mut values {
            *value ^= 0xA5A5_A5A5;
        }
        let expected: Vec<u32> = values.iter().map(|v| v.wrapping_add(7)).collect();
        let mut session = memory.begin_mut(&mut values).unwrap();
        launch_add_seven(session.inner_mut(), &function);
        session.finish().unwrap();

        assert_eq!(values, expected);
        assert_eq!(memory.stats().uploads, 2);
        assert_eq!(memory.stats().downloads, 2);
    }
}
