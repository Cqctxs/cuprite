//! Element types buffers can hold. Data moves between the CPU and the GPU
//! as raw bytes, so only types for which any bytes are a valid value qualify.

/// A plain value type: no padding, and every bit pattern is a valid value.
/// GPU results can then be copied into it byte for byte, whatever a kernel
/// wrote.
///
/// # Safety
/// Implement it only for types with no padding bytes and no invalid bit
/// patterns, such as `#[repr(C)]` structs of the numeric types below with
/// no gaps between fields.
pub unsafe trait Element: Copy + Send + Sync + 'static {}

macro_rules! elements {
    ($($ty:ty),*) => {
        // SAFETY: primitive integers and floats have no padding, and every
        // bit pattern is a valid value.
        $(unsafe impl Element for $ty {})*
    };
}
elements!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

pub(crate) fn bytes<T: Element>(values: &[T]) -> &[u8] {
    // SAFETY: `Element` types have no padding, so every byte is initialized,
    // and `u8` needs no alignment.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), size_of_val(values)) }
}

pub(crate) fn bytes_mut<T: Element>(values: &mut [T]) -> &mut [u8] {
    // SAFETY: as in `bytes`; also any bytes written leave valid values,
    // since every bit pattern of an `Element` type is valid.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), size_of_val(values)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn views_cover_every_byte_in_memory_order() {
        let values = [0x0403_0201u32, 0x0807_0605];
        assert_eq!(bytes(&values), &[1, 2, 3, 4, 5, 6, 7, 8]);

        let mut floats = [0.0f32; 2];
        bytes_mut(&mut floats)[..4].copy_from_slice(&1.5f32.to_ne_bytes());
        assert_eq!(floats, [1.5, 0.0]);
    }
}
