use crate::MemoryError;

/// Private boundary between ownership logic and a GPU or test backend.
///
/// Copies may be queued: the caller must retain source/destination storage
/// until synchronize establishes completion. An error alone does not prove
/// that no queued work still uses that storage.
///
/// Lengths are `u32` element counts. Transfers require equal source and
/// destination lengths. Allocation failure and invalid lengths return errors.
pub(crate) trait MemoryBackend {
    type Buffer;

    fn allocate(&mut self, len: usize) -> Result<Self::Buffer, MemoryError>;
    fn upload(&mut self, buffer: &mut Self::Buffer, input: &[u32]) -> Result<(), MemoryError>;
    fn download(&mut self, buffer: &Self::Buffer, output: &mut [u32]) -> Result<(), MemoryError>;
    fn synchronize(&mut self) -> Result<(), MemoryError>;
}

/// Backend behind the public `GpuMemory` type.
#[cfg(feature = "cuda")]
pub(crate) type DefaultBackend = crate::cuda::CudaBackend;
#[cfg(not(feature = "cuda"))]
pub(crate) type DefaultBackend = NoBackend;

/// Stands in for a GPU when the `cuda` feature is off. It has no values, so a
/// manager using it can never be constructed and its methods never run.
#[cfg(not(feature = "cuda"))]
pub(crate) enum NoBackend {}

#[cfg(not(feature = "cuda"))]
impl MemoryBackend for NoBackend {
    type Buffer = std::convert::Infallible;

    fn allocate(&mut self, _len: usize) -> Result<Self::Buffer, MemoryError> {
        match *self {}
    }

    fn upload(&mut self, _buffer: &mut Self::Buffer, _input: &[u32]) -> Result<(), MemoryError> {
        match *self {}
    }

    fn download(&mut self, _buffer: &Self::Buffer, _output: &mut [u32]) -> Result<(), MemoryError> {
        match *self {}
    }

    fn synchronize(&mut self) -> Result<(), MemoryError> {
        match *self {}
    }
}
