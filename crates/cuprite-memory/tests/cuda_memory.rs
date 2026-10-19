#![cfg(feature = "cuda")]

use cuprite_memory::{GpuMemory, MemoryConfig};

// Run explicitly on a machine with a working NVIDIA CUDA setup.
// This is the first hardware test, not proof of GPU writes or safe cleanup.
#[test]
#[ignore = "requires the CUDA backend, which is not implemented yet"]
fn uploading_an_unchanged_slice_preserves_values() {
    let mut memory = GpuMemory::open(MemoryConfig::default()).unwrap();
    let mut values = [0, 1, u32::MAX, 7];

    memory.begin_mut(&mut values).unwrap().finish().unwrap();

    assert_eq!(values, [0, 1, u32::MAX, 7]);
    let stats = memory.stats();
    assert_eq!(stats.device_allocations, 1);
    assert_eq!(stats.uploads, 1);
    assert_eq!(stats.uploaded_bytes, 16);
    assert_eq!(stats.downloads, 0, "unchanged data needs no copy-back");
}
