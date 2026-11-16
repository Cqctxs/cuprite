//! A library that knows nothing about Cuprite.

/// Applies one step of the simulation to every value.
///
/// ```
/// let mut values = [1, 2];
/// fixture_lib::step(&mut values);
/// assert_eq!(values, [6, 11]);
/// ```
pub fn step(values: &mut [u32]) {
    for value in values.iter_mut() {
        *value = value.wrapping_mul(5).wrapping_add(1);
    }
}

/// Mixes a value's high bits into its low ones. Neither it nor its helper
/// is generic or inline, so rustc keeps their MIR in this crate's metadata
/// only when told to (`-Zalways-encode-mir`), and callers in other crates
/// can inline them into kernels only then.
#[inline(never)]
pub fn fold(value: u32) -> u32 {
    value ^ shifted(value)
}

#[inline(never)]
fn shifted(value: u32) -> u32 {
    value >> 13
}

/// Runs `passes` steps.
pub fn simulate(values: &mut [u32], passes: u32) {
    for _ in 0..passes {
        step(values);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn simulate_repeats_step() {
        let mut once = [3, u32::MAX];
        super::step(&mut once);
        super::step(&mut once);
        let mut twice = [3, u32::MAX];
        super::simulate(&mut twice, 2);
        assert_eq!(once, twice);
    }
}
