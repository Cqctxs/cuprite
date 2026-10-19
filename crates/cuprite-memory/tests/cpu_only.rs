#![cfg(not(feature = "cuda"))]

use cuprite_memory::{GpuMemory, MemoryConfig, MemoryError};

#[test]
fn opening_without_cuda_returns_an_error_instead_of_emulating_a_gpu() {
    let result = GpuMemory::open(MemoryConfig::default());
    assert!(matches!(result, Err(MemoryError::Unavailable(_))));
}
