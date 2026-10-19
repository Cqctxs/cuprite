/// Settings for one memory manager. The budget covers owned device buffers,
/// including idle pooled buffers; it does not include CUDA context overhead.
#[derive(Clone, Copy, Debug)]
pub struct MemoryConfig {
    pub device_ordinal: usize,
    pub max_device_bytes: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            device_ordinal: 0,
            max_device_bytes: 128 * 1024 * 1024,
        }
    }
}
