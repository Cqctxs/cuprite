//! Public size-validation behavior; no GPU is required.

use cuprite_memory::{checked_bytes, validate_len, MemoryConfig, MemoryError};

fn config_with_budget(max_device_bytes: usize) -> MemoryConfig {
    MemoryConfig {
        max_device_bytes,
        ..MemoryConfig::default()
    }
}

#[test]
fn empty_input_fits_a_zero_budget() {
    assert_eq!(validate_len(0, config_with_budget(0)), Ok(0));
}

#[test]
fn nonempty_input_is_rejected_by_a_zero_budget() {
    assert_eq!(
        validate_len(1, config_with_budget(0)),
        Err(MemoryError::BudgetExceeded {
            requested: 4,
            limit: 0,
        })
    );
}

#[test]
fn budget_is_measured_in_bytes_without_rounding_up() {
    let config = config_with_budget(17);

    assert_eq!(validate_len(4, config), Ok(16));
    assert_eq!(
        validate_len(5, config),
        Err(MemoryError::BudgetExceeded {
            requested: 20,
            limit: 17,
        })
    );
}

#[test]
fn overflow_precedes_both_small_and_large_budgets() {
    for budget in [0, usize::MAX] {
        assert_eq!(
            validate_len(usize::MAX / 4 + 1, config_with_budget(budget)),
            Err(MemoryError::SizeOverflow)
        );
    }
}

#[test]
fn largest_byte_count_fits_an_unrestricted_budget() {
    let len = usize::MAX / 4;

    assert_eq!(checked_bytes(len), Ok(usize::MAX - 3));
    assert_eq!(
        validate_len(len, config_with_budget(usize::MAX)),
        Ok(usize::MAX - 3)
    );
}
