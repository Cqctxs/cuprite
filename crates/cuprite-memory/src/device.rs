/// The device a [`GpuMemory`](crate::GpuMemory) runs on, with the versions
/// that decide its performance. Measurements taken on one device are valid
/// only while all of these stay the same.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    pub name: String,
    pub compute_capability: (i32, i32),
    /// As the driver reports it: 13000 for CUDA 13.0.
    pub driver_version: i32,
    /// A hash of the kernel source, which changes whenever a kernel does.
    pub kernels: u64,
}

/// 64-bit FNV-1a: small, and stable across Rust versions, unlike
/// `DefaultHasher`.
#[cfg(any(feature = "cuda", test))]
pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a_matches_the_reference_values() {
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a(b"foobar"), 0x8594_4171_f739_67e8);
    }
}
