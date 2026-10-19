use std::mem::size_of;

use crate::{MemoryConfig, MemoryError};

/// Convert a `u32` element count into bytes without overflowing.
///
/// # Errors
/// Returns [`MemoryError::SizeOverflow`] if the byte count does not fit in `usize`.
pub fn checked_bytes(len: usize) -> Result<usize, MemoryError> {
    size_of::<u32>()
        .checked_mul(len)
        .ok_or(MemoryError::SizeOverflow)
}

/// Return the byte count if this request fits the configured device budget.
///
/// This checks one request, not total active or pooled storage. The manager
/// must account for those allocations separately.
///
/// # Errors
/// Returns [`MemoryError::SizeOverflow`] before checking the budget if the byte
/// count overflows, or [`MemoryError::BudgetExceeded`] if the request is too large.
pub fn validate_len(len: usize, config: MemoryConfig) -> Result<usize, MemoryError> {
    let requested_bytes = checked_bytes(len)?;
    if requested_bytes > config.max_device_bytes {
        return Err(MemoryError::BudgetExceeded {
            requested: requested_bytes,
            limit: config.max_device_bytes,
        });
    }
    Ok(requested_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_length_needs_no_bytes() {
        assert_eq!(checked_bytes(0), Ok(0));
    }

    #[test]
    fn counts_u32_storage_in_bytes() {
        assert_eq!(checked_bytes(1), Ok(4));
        assert_eq!(checked_bytes(257), Ok(1028));
    }

    #[test]
    fn detects_byte_count_overflow() {
        assert_eq!(checked_bytes(usize::MAX), Err(MemoryError::SizeOverflow));
    }

    #[test]
    fn accepts_the_largest_representable_byte_count() {
        assert_eq!(checked_bytes(usize::MAX / 4), Ok(usize::MAX - 3));
    }

    #[test]
    fn rejects_the_first_overflowing_element_count() {
        assert_eq!(
            checked_bytes(usize::MAX / 4 + 1),
            Err(MemoryError::SizeOverflow)
        );
    }

    #[test]
    fn allows_an_exact_budget_match() {
        let config = MemoryConfig {
            device_ordinal: 0,
            max_device_bytes: 16,
        };
        assert_eq!(validate_len(4, config), Ok(16));
    }

    #[test]
    fn rejects_a_request_above_the_budget() {
        let config = MemoryConfig {
            device_ordinal: 0,
            max_device_bytes: 16,
        };
        assert_eq!(
            validate_len(5, config),
            Err(MemoryError::BudgetExceeded {
                requested: 20,
                limit: 16,
            })
        );
    }

    #[test]
    fn preserves_overflow_instead_of_reporting_a_budget_error() {
        assert_eq!(
            validate_len(usize::MAX, MemoryConfig::default()),
            Err(MemoryError::SizeOverflow)
        );
    }
}
