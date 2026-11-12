use crate::{Grid, MemoryError};

/// A kernel argument with its buffer resolved to device storage. `None` is
/// an empty buffer, passed as a null pointer.
// Without the `cuda` feature only tests read the values.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) enum Value<'b, Buf> {
    Buffer(Option<&'b Buf>),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
}

/// Private boundary between ownership logic and a GPU or test backend.
///
/// Copies and launches may be queued: the caller must retain source and
/// destination storage until synchronize establishes completion. An error
/// alone does not prove that no queued work still uses that storage.
///
/// Sizes are byte counts. Transfers require equal source and destination
/// lengths. Buffers are never empty: empty data needs no device storage.
pub(crate) trait MemoryBackend {
    type Buffer;
    type Kernel;

    /// Zero-filled device storage.
    fn allocate(&mut self, bytes: usize) -> Result<Self::Buffer, MemoryError>;
    fn upload(&mut self, buffer: &mut Self::Buffer, input: &[u8]) -> Result<(), MemoryError>;
    fn download(&mut self, buffer: &Self::Buffer, output: &mut [u8]) -> Result<(), MemoryError>;
    /// Queue zero-filling `buffer`, after earlier work.
    fn zero(&mut self, buffer: &mut Self::Buffer) -> Result<(), MemoryError>;
    fn load(&mut self, ptx: &str, name: &str) -> Result<Self::Kernel, MemoryError>;
    /// Queue `kernel` over a nonempty `grid`, after earlier work.
    fn launch(
        &mut self,
        kernel: &Self::Kernel,
        grid: Grid,
        args: &[Value<'_, Self::Buffer>],
    ) -> Result<(), MemoryError>;
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
    type Kernel = std::convert::Infallible;

    fn allocate(&mut self, _bytes: usize) -> Result<Self::Buffer, MemoryError> {
        match *self {}
    }

    fn upload(&mut self, _buffer: &mut Self::Buffer, _input: &[u8]) -> Result<(), MemoryError> {
        match *self {}
    }

    fn download(&mut self, _buffer: &Self::Buffer, _output: &mut [u8]) -> Result<(), MemoryError> {
        match *self {}
    }

    fn zero(&mut self, _buffer: &mut Self::Buffer) -> Result<(), MemoryError> {
        match *self {}
    }

    fn load(&mut self, _ptx: &str, _name: &str) -> Result<Self::Kernel, MemoryError> {
        match *self {}
    }

    fn launch(
        &mut self,
        _kernel: &Self::Kernel,
        _grid: Grid,
        _args: &[Value<'_, Self::Buffer>],
    ) -> Result<(), MemoryError> {
        match *self {}
    }

    fn synchronize(&mut self) -> Result<(), MemoryError> {
        match *self {}
    }
}
