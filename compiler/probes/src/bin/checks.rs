//! Functions whose checks can fail (bounds, zero divisors, overflow,
//! assertions), each run on inputs that pass every check and on inputs that
//! fail one. Built through the driver with `CUPRITE_AUTO=1`, the GPU makes
//! the same checks: when they pass its results are used; when one fails it
//! discards them and the CPU runs the code, which panics as it always did.
//!
//! Each outcome is compared with a reference closure that does the same
//! work on the CPU: the values when it returns, and the panic message and
//! the values left behind when it panics.
//!
//! Run in debug and release builds: debug builds also check overflow.

use std::panic::{catch_unwind, AssertUnwindSafe};

use cuprite_probes as probes;

/// What a call did: the panic message, if it panicked, and the values.
type Outcome<T> = (Option<String>, Vec<T>);

fn outcome<T: Clone>(input: &[T], call: impl FnOnce(&mut [T])) -> Outcome<T> {
    let mut values = input.to_vec();
    let result = catch_unwind(AssertUnwindSafe(|| call(&mut values)));
    let message = result.err().map(|payload| {
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "(not a string)".into())
    });
    (message, values)
}

/// Floats compare by bits; any NaN matches any NaN, as Rust allows.
trait Same {
    fn same(&self, other: &Self) -> bool;
}

macro_rules! exact {
    ($($t:ty),*) => {
        $(impl Same for $t {
            fn same(&self, other: &Self) -> bool {
                self == other
            }
        })*
    };
}
exact!(u8, u32, i32, i64);

impl Same for f32 {
    fn same(&self, other: &Self) -> bool {
        self.to_bits() == other.to_bits() || self.is_nan() && other.is_nan()
    }
}

/// Runs `function` and `reference` on `input` and requires the same
/// outcome. Returns whether it panicked.
fn check<T: Clone + Same + std::fmt::Debug>(
    name: &str,
    input: &[T],
    function: impl FnOnce(&mut [T]),
    reference: impl FnOnce(&mut [T]),
) -> bool {
    let (got_panic, got) = outcome(input, function);
    let (want_panic, want) = outcome(input, reference);
    // Panics are silenced, so a mismatch reports and exits instead.
    if got_panic != want_panic {
        eprintln!("{name}: panicked with {got_panic:?} but should have with {want_panic:?}");
        std::process::exit(1);
    }
    if let Some(index) = got.iter().zip(&want).position(|(a, b)| !a.same(b)) {
        eprintln!("{name}: index {index} is {:?} but should be {:?}", got[index], want[index]);
        std::process::exit(1);
    }
    got_panic.is_some()
}

const LEN: usize = 100_000;

fn main() {
    // Keep expected panics quiet; the comparisons report real failures.
    std::panic::set_hook(Box::new(|_| {}));
    let mut panicked = 0;
    let mut passed = 0;
    let mut count = |did_panic: bool| {
        if did_panic {
            panicked += 1;
        } else {
            passed += 1;
        }
    };

    // Bounds checks: every index in range, then one out of range.
    let bytes: Vec<u8> = (0..LEN).map(|i| (i * 7 % 256) as u8).collect();
    let full: Vec<u8> = (0..=255u8).rev().collect();
    let short: Vec<u8> = (0..100u8).collect();
    let lookup = |table: &[u8]| {
        let table = table.to_vec();
        move |v: &mut [u8]| {
            for value in v {
                *value = table[*value as usize];
            }
        }
    };
    count(check("lookup", &bytes, |v| probes::lookup(v, &full), lookup(&full)));
    count(check("lookup, short table", &bytes, |v| probes::lookup(v, &short), lookup(&short)));

    // A remainder by the table's length, then its bounds check.
    let floats: Vec<f32> = (0..LEN).map(|i| (i as f32 - 5000.0) * 0.37).collect();
    let weights: Vec<f32> = (0..9).map(|i| i as f32 * 0.5 + 0.25).collect();
    let wrapped = |table: &[f32]| {
        let table = table.to_vec();
        move |v: &mut [f32]| {
            for value in v {
                let index = (value.abs() * 16.0) as usize % table.len();
                *value *= table[index];
            }
        }
    };
    count(check("lookup_wrapped", &floats, |v| probes::lookup_wrapped(v, &weights), wrapped(&weights)));
    count(check("lookup_wrapped, empty table", &floats, |v| probes::lookup_wrapped(v, &[]), wrapped(&[])));

    // Division by a parameter: fine, by zero, and `MIN / -1`.
    let ints: Vec<i32> = (0..LEN as i32).map(|i| i.wrapping_mul(2_654_435_761u32 as i32)).collect();
    let mut with_min = ints.clone();
    with_min[60_000] = i32::MIN;
    for (name, input, divisor) in [
        ("div_mod", &ints, 7),
        ("div_mod, MIN by -7", &with_min, -7),
        ("div_mod, by zero", &ints, 0),
        ("div_mod, MIN by -1", &with_min, -1),
    ] {
        count(check(name, input, |v| probes::div_mod(v, divisor), |v| {
            for value in v {
                *value = *value / divisor + *value % divisor;
            }
        }));
    }
    let words: Vec<u32> = (1..=LEN as u32).collect();
    let mut with_zero = words.clone();
    with_zero[70_000] = 0;
    for (name, input) in [("reciprocal", &words), ("reciprocal, a zero", &with_zero)] {
        count(check(name, input, probes::reciprocal, |v| {
            for value in v {
                *value = 100 / *value;
            }
        }));
    }

    // `f32::clamp` asserts its bounds are in order.
    for (name, low, high) in [("clamp_range", -3.0f32, 4.5), ("clamp_range, reversed", 1.0, -1.0)] {
        count(check(name, &floats, |v| probes::clamp_range(v, low, high), |v| {
            for value in v {
                *value = value.clamp(low, high);
            }
        }));
    }

    // An explicit assertion with a formatted message.
    let evens: Vec<u32> = (0..LEN as u32).map(|i| i * 2).collect();
    let mut with_odd = evens.clone();
    with_odd[12_345] = 77;
    for (name, input) in [("asserted_halves", &evens), ("asserted_halves, an odd value", &with_odd)] {
        count(check(name, input, probes::asserted_halves, |v| {
            for value in v {
                assert!(*value % 2 == 0, "odd value {value}");
                *value /= 2;
            }
        }));
    }

    // Overflow: wraps in release builds, panics in debug builds.
    let mut near_max = bytes.clone();
    near_max[40_000] = 250;
    count(check("add_bytes", &near_max, |v| probes::add_bytes(v, 6), |v| {
        for value in v {
            *value += 6;
        }
    }));
    let wide: Vec<i64> = (0..LEN as i64).map(|i| i * 1_000 - 50_000_000).collect();
    let mut with_huge = wide.clone();
    with_huge[90_000] = 4_000_000_000;
    for (name, input) in [("square_wide", &wide), ("square_wide, a huge value", &with_huge)] {
        count(check(name, input, probes::square_wide, |v| {
            for value in v {
                *value *= *value;
            }
        }));
    }
    let mut at_max = words.clone();
    at_max[30_000] = u32::MAX;
    count(check("checked_increment", &at_max, probes::checked_increment, |v| {
        for value in v {
            *value += 1;
        }
    }));

    // Held out: `num-integer`'s floored division, by a divisor and by zero.
    for (name, divisor) in [("ni_div_floor", -7), ("ni_div_floor, by zero", 0)] {
        count(check(name, &with_min, |v| probes::ni_div_floor(v, divisor), |v| {
            for value in v {
                *value = num_integer::div_floor(*value, divisor);
            }
        }));
    }

    // Held out: `strength_reduce`, whose divisor is reduced before the loop
    // (asserting it isn't zero, and testing for a power of two with
    // `count_ones`), then divides with multiplies and, for powers of two,
    // `trailing_zeros`.
    let spread: Vec<u32> = ints.iter().map(|&i| i as u32).collect();
    for (name, divisor) in [("sr_divide", 7u32), ("sr_divide, a power of two", 64), ("sr_divide, by zero", 0)] {
        count(check(name, &spread, |v| probes::sr_divide(v, divisor), |v| {
            let reduced = strength_reduce::StrengthReducedU32::new(divisor);
            for value in v {
                *value = *value / reduced;
            }
        }));
    }

    // Held out: `num-integer`'s binary GCD, a loop of `trailing_zeros`.
    for (name, other) in [("ni_gcd", 360u32), ("ni_gcd, with zero", 0)] {
        count(check(name, &spread, |v| probes::ni_gcd(v, other), |v| {
            for value in v {
                *value = num_integer::gcd(*value, other);
            }
        }));
    }

    // A closure's checks: a remainder by the captured table's length, which
    // is zero for an empty table, then a bounds check.
    let entries: Vec<u32> = (0..37u32).map(|i| i.wrapping_mul(2_654_435_761)).collect();
    for (name, table) in [("closure_table", &entries[..]), ("closure_table, empty table", &[][..])] {
        count(check(name, &spread, |v| probes::closure_table(v, table), |v| {
            let wrap = |i: u32| table[i as usize % table.len()];
            for value in v {
                *value = wrap(*value).wrapping_add(wrap(*value >> 8));
            }
        }));
    }

    // `palette`'s `mix`, which clamps its factor.
    count(check("palette_mix", &floats, |v| probes::palette_mix(v, 0.9, 0.1, 0.5), |v| {
        use palette::Mix;
        let base = palette::LinSrgb::new(0.2f32, 0.4, 0.6);
        let target = palette::LinSrgb::new(0.9, 0.1, 0.5);
        for value in v {
            *value = base.mix(target, *value).green;
        }
    }));

    println!("{} outcomes match ({passed} returned, {panicked} panicked)", passed + panicked);
}
