#![cfg(feature = "cuda")]

//! Public-API tests on a real GPU. Enabling the `cuda` feature declares this
//! a GPU test host, so these fail rather than skip when CUDA is unusable.
//! Kernel-write tests live next to the CUDA backend, which can reach the
//! crate-private device buffer.

use cuprite_memory::{GpuMemory, MemoryConfig, MemoryError};

#[test]
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

#[test]
fn consecutive_sessions_work_across_lengths() {
    let mut memory = GpuMemory::open(MemoryConfig::default()).unwrap();

    for len in [0, 1, 255, 256, 257, 1025] {
        let mut values: Vec<u32> = (0..len as u32).map(|i| u32::MAX - i).collect();
        let original = values.clone();

        memory.begin_mut(&mut values).unwrap().finish().unwrap();

        assert_eq!(values, original, "length {len}");
    }
    assert_eq!(
        memory.stats().uploads,
        5,
        "the empty session transfers nothing"
    );
}

#[test]
fn a_request_over_the_budget_is_rejected_before_allocation() {
    let config = MemoryConfig {
        max_device_bytes: 1024,
        ..MemoryConfig::default()
    };
    let mut memory = GpuMemory::open(config).unwrap();
    let mut values = vec![1u32; 257];

    let error = memory.begin_mut(&mut values).err().unwrap();

    assert_eq!(
        error,
        MemoryError::BudgetExceeded {
            requested: 1028,
            limit: 1024,
        }
    );
    assert_eq!(memory.stats().device_allocations, 0);
    memory
        .begin_mut(&mut values[..256])
        .unwrap()
        .finish()
        .unwrap();
}

#[test]
fn an_invalid_device_ordinal_is_unavailable() {
    let config = MemoryConfig {
        device_ordinal: 4096,
        ..MemoryConfig::default()
    };

    let result = GpuMemory::open(config);

    assert!(matches!(result, Err(MemoryError::Unavailable(_))));
}
