//! Entry points that functions rewritten by the Cuprite driver call before
//! their original body.
//!
//! Each entry point returns `true` only when it has completed the work and
//! committed the results. On `false` the caller runs its original CPU body.
//! That is always safe, because a failed GPU session never changes the
//! caller's slice.
//!
//! Environment:
//! - `CUPRITE_GPU=1` enables the GPU. Otherwise every call returns `false`.
//! - `CUPRITE_TRACE=1` prints one line per call to stderr.

use std::{
    env,
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex, OnceLock, TryLockError,
    },
};

use cuprite_memory::{GpuMemory, MapStep, MemoryConfig};

// One instance, in a static: its size does not matter.
#[allow(clippy::large_enum_variant)]
enum Device {
    Untried,
    Ready(GpuMemory),
    Unavailable,
}

/// Opened on first use. Calls that find it busy, on another thread, run on
/// the CPU instead of waiting.
static DEVICE: Mutex<Device> = Mutex::new(Device::Untried);

static GPU_CALLS: AtomicU64 = AtomicU64::new(0);
static CPU_CALLS: AtomicU64 = AtomicU64::new(0);

fn flag(name: &str) -> bool {
    env::var_os(name).is_some_and(|value| value == "1")
}

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| flag("CUPRITE_GPU"))
}

fn trace(len: usize, passes: u32, outcome: &str) {
    static TRACE: OnceLock<bool> = OnceLock::new();
    if *TRACE.get_or_init(|| flag("CUPRITE_TRACE")) {
        eprintln!("cuprite: map_u32 len={len} passes={passes} -> {outcome}");
    }
}

/// `passes` repetitions of `x = x * mul + add` (wrapping) over `values`,
/// kept resident on the GPU between passes.
///
/// Returns `true` if the GPU did the work, `false` if the caller must run its
/// CPU body. `values` is unchanged whenever this returns `false`.
pub fn map_u32(values: &mut [u32], mul: u32, add: u32, passes: u32) -> bool {
    let outcome = run_map(values, MapStep { mul, add }, passes);
    let counter = if outcome.is_ok() {
        &GPU_CALLS
    } else {
        &CPU_CALLS
    };
    counter.fetch_add(1, Ordering::Relaxed);
    match outcome {
        Ok(()) => {
            trace(values.len(), passes, "gpu");
            true
        }
        Err(reason) => {
            trace(values.len(), passes, &format!("cpu ({reason})"));
            false
        }
    }
}

fn run_map(values: &mut [u32], step: MapStep, passes: u32) -> Result<(), String> {
    if !enabled() {
        return Err("CUPRITE_GPU is not 1".into());
    }
    if values.is_empty() || passes == 0 {
        return Err("no work".into());
    }
    let mut device = match DEVICE.try_lock() {
        Ok(device) => device,
        Err(TryLockError::WouldBlock) => return Err("GPU busy".into()),
        Err(TryLockError::Poisoned(_)) => return Err("runtime poisoned".into()),
    };
    if let Device::Untried = *device {
        *device = match GpuMemory::open(MemoryConfig::default()) {
            Ok(memory) => Device::Ready(memory),
            Err(error) => {
                trace(values.len(), passes, &format!("cpu (open: {error})"));
                Device::Unavailable
            }
        };
    }
    let Device::Ready(memory) = &mut *device else {
        return Err("GPU unavailable".into());
    };

    let mut session = memory.begin_mut(values).map_err(|e| e.to_string())?;
    for _ in 0..passes {
        session.map(step).map_err(|e| e.to_string())?;
    }
    session.finish().map_err(|e| e.to_string())
}

/// Calls that ran on the GPU and calls left to the CPU, since start.
pub fn counts() -> (u64, u64) {
    (
        GPU_CALLS.load(Ordering::Relaxed),
        CPU_CALLS.load(Ordering::Relaxed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // `CUPRITE_GPU` is read once per process, and test binaries do not set
    // it, so these exercise the disabled path. The driver's integration
    // tests cover the GPU path in a real rewritten program.
    #[test]
    fn disabled_runtime_leaves_values_for_the_cpu() {
        let mut values = vec![1, 2, 3];
        assert!(!map_u32(&mut values, 3, 7, 2));
        assert_eq!(values, [1, 2, 3]);
        assert!(counts().1 >= 1);
    }
}
