//! CUDA backend built on cudarc. Not implemented yet: it makes no CUDA calls
//! and creates no GPU resources.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};

use crate::{backend::MemoryBackend, MemoryConfig, MemoryError};

pub(crate) struct CudaBackend {
    pub(crate) context: Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
}

impl CudaBackend {
    pub(crate) fn open(_config: MemoryConfig) -> Result<Self, MemoryError> {
        // Research the loader's error behavior before constructing a context.
        Err(MemoryError::NotImplemented("CudaBackend::open"))
    }
}

impl MemoryBackend for CudaBackend {
    type Buffer = CudaSlice<u32>;

    fn allocate(&mut self, _len: usize) -> Result<Self::Buffer, MemoryError> {
        // Start with initialized storage; avoid an unsafe uninitialized buffer.
        Err(MemoryError::NotImplemented("CudaBackend::allocate"))
    }

    fn upload(&mut self, _buffer: &mut Self::Buffer, _input: &[u32]) -> Result<(), MemoryError> {
        Err(MemoryError::NotImplemented("CudaBackend::upload"))
    }

    fn download(&mut self, _buffer: &Self::Buffer, _output: &mut [u32]) -> Result<(), MemoryError> {
        Err(MemoryError::NotImplemented("CudaBackend::download"))
    }

    fn synchronize(&mut self) -> Result<(), MemoryError> {
        Err(MemoryError::NotImplemented("CudaBackend::synchronize"))
    }
}
