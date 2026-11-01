//! Entry points that functions rewritten by the Cuprite driver call before
//! their original body.
//!
//! [`map_u32`] returns [`DONE`] only when it has completed the work and
//! committed the results. Otherwise the caller runs its original CPU body,
//! which is always safe, because a failed GPU session never changes the
//! caller's slice, and then reports to [`map_u32_finished`] with the value it
//! got, so that the policy can learn what the CPU body takes.
//!
//! Environment:
//! - `CUPRITE_GPU` chooses the mode:
//!   - unset or `auto`: the measured policy. A call runs on the GPU when the
//!     profile in `CUPRITE_PROFILE` predicts it is faster there, counting
//!     the GPU's startup ([`policy::Policy`]). Without a valid profile for
//!     this machine, everything runs on the CPU.
//!   - `0`: always the CPU; CUDA is never initialized.
//!   - `1`: always the GPU when it works, for tests and measurements.
//! - `CUPRITE_PROFILE`: a profile written by `cuprite-calibrate`.
//! - `CUPRITE_TRACE=1` prints one line per call to stderr, with the side
//!   that ran and why.

pub mod calibrate;
pub mod policy;

use std::{
    cell::RefCell,
    env, fmt,
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, OnceLock, TryLockError,
    },
    time::Instant,
};

use cuprite_memory::{GpuMemory, MapStep, MemoryConfig};

use policy::{Choice, HostKey, Policy, Profile};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Cpu,
    Auto,
    Always,
}

// One instance, in a static: its size does not matter.
#[allow(clippy::large_enum_variant)]
enum Device {
    Untried,
    Ready(GpuMemory),
    /// Why the GPU is not used for the rest of the process.
    Unavailable(Arc<str>),
}

struct State {
    device: Device,
    policy: Policy,
}

/// Opened on first use. Calls that find it busy, on another thread, run on
/// the CPU instead of waiting.
static STATE: Mutex<State> = Mutex::new(State {
    device: Device::Untried,
    policy: Policy::new(),
});

static GPU_CALLS: AtomicU64 = AtomicU64::new(0);
static CPU_CALLS: AtomicU64 = AtomicU64::new(0);

fn mode() -> &'static Result<Mode, String> {
    static MODE: OnceLock<Result<Mode, String>> = OnceLock::new();
    MODE.get_or_init(|| match env::var("CUPRITE_GPU") {
        Err(env::VarError::NotPresent) => Ok(Mode::Auto),
        Ok(value) => match value.as_str() {
            "auto" => Ok(Mode::Auto),
            "0" => Ok(Mode::Cpu),
            "1" => Ok(Mode::Always),
            other => Err(format!("CUPRITE_GPU={other} is not 0, 1 or auto")),
        },
        Err(env::VarError::NotUnicode(_)) => Err("CUPRITE_GPU is not 0, 1 or auto".into()),
    })
}

/// The profile for this machine, loaded and checked once.
fn profile() -> &'static Result<Profile, String> {
    static PROFILE: OnceLock<Result<Profile, String>> = OnceLock::new();
    PROFILE.get_or_init(|| {
        let path = env::var_os("CUPRITE_PROFILE").ok_or("no profile; set CUPRITE_PROFILE")?;
        let profile = Profile::load(Path::new(&path))?;
        profile.check_host(&HostKey::current())?;
        Ok(profile)
    })
}

fn tracing() -> bool {
    static TRACE: OnceLock<bool> = OnceLock::new();
    *TRACE.get_or_init(|| env::var_os("CUPRITE_TRACE").is_some_and(|v| v == "1"))
}

/// Why a call is left to the CPU. Formatted only for the trace: a declined
/// call should cost little more than the CPU body it leads to.
enum Decline {
    /// `CUPRITE_GPU=0`, or an invalid value.
    Mode(&'static str),
    NoWork,
    /// No profile, or one that cannot be used.
    Profile(&'static str),
    Policy(Choice),
    Busy,
    Poisoned,
    /// The GPU failed to open, or is not the one profiled.
    Unavailable(Arc<str>),
    /// A GPU session failed before committing; the slice is untouched.
    Failed(String),
}

impl fmt::Display for Decline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mode(reason) | Self::Profile(reason) => f.write_str(reason),
            Self::NoWork => f.write_str("no work"),
            Self::Policy(choice) => write!(f, "{choice}"),
            Self::Busy => f.write_str("GPU busy"),
            Self::Poisoned => f.write_str("runtime poisoned"),
            Self::Unavailable(reason) => f.write_str(reason),
            Self::Failed(reason) => f.write_str(reason),
        }
    }
}

/// Returned by [`map_u32`] when the GPU did the work.
pub const DONE: u64 = 0;
/// Returned by [`map_u32`] when the caller must run its CPU body, and nobody
/// times it.
pub const UNTIMED: u64 = 1;

/// A CPU body being timed, for the policy's corrections.
struct Timing {
    token: u64,
    start: Instant,
    len: usize,
    passes: u32,
    step: MapStep,
}

thread_local! {
    /// Bodies running on this thread, innermost last.
    static TIMINGS: RefCell<Vec<Timing>> = const { RefCell::new(Vec::new()) };
}

/// Tokens for timed bodies; 0 and 1 are `DONE` and `UNTIMED`.
static NEXT_TOKEN: AtomicU64 = AtomicU64::new(2);

/// `passes` repetitions of `x = x * mul + add` (wrapping) over `values`,
/// kept resident on the GPU between passes.
///
/// Returns [`DONE`] if the GPU did the work. Anything else means the caller
/// must run its CPU body, then pass the value to [`map_u32_finished`].
/// `values` is unchanged unless this returns `DONE`.
pub fn map_u32(values: &mut [u32], mul: u32, add: u32, passes: u32) -> u64 {
    let step = MapStep { mul, add };
    let outcome = dispatch(values, step, passes);
    let counter = if outcome.is_ok() {
        &GPU_CALLS
    } else {
        &CPU_CALLS
    };
    counter.fetch_add(1, Ordering::Relaxed);
    if tracing() {
        let len = values.len();
        match &outcome {
            Ok(None) => eprintln!("cuprite: map_u32 len={len} passes={passes} -> gpu"),
            Ok(Some(choice)) => {
                eprintln!("cuprite: map_u32 len={len} passes={passes} -> gpu ({choice})");
            }
            Err(decline) => {
                eprintln!("cuprite: map_u32 len={len} passes={passes} -> cpu ({decline})");
            }
        }
    }
    match outcome {
        Ok(_) => DONE,
        // A close call: time the CPU body, so later decisions use what this
        // function's code actually takes.
        Err(Decline::Policy(choice)) if Policy::worth_timing(choice.prediction()) => {
            let token = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed);
            TIMINGS.with_borrow_mut(|timings| {
                timings.push(Timing {
                    token,
                    start: Instant::now(),
                    len: values.len(),
                    passes,
                    step,
                })
            });
            token
        }
        Err(_) => UNTIMED,
    }
}

/// Called by a rewritten function when the CPU body it ran after
/// [`map_u32`] returned `token` has finished.
///
/// A body that unwinds never reports, so entries above `token`, from such
/// bodies called inside this one, are discarded.
pub fn map_u32_finished(token: u64) {
    if token == DONE || token == UNTIMED {
        return;
    }
    let end = Instant::now();
    let timing = TIMINGS.with_borrow_mut(|timings| {
        while timings.last().is_some_and(|timing| timing.token > token) {
            timings.pop();
        }
        match timings.last() {
            Some(timing) if timing.token == token => timings.pop(),
            _ => None,
        }
    });
    let (Some(timing), Ok(profile)) = (timing, profile()) else {
        return;
    };
    if let Ok(mut state) = STATE.try_lock() {
        let micros = end.duration_since(timing.start).as_secs_f64() * 1e6;
        let step = (timing.step.mul, timing.step.add);
        state
            .policy
            .record(profile, timing.len, timing.passes, step, false, micros);
    }
}

/// Runs the call on the GPU if the mode and policy say so: `Ok` with the
/// policy's choice (`None` when the GPU is forced), or `Err` with why the
/// CPU must run it.
fn dispatch(values: &mut [u32], step: MapStep, passes: u32) -> Result<Option<Choice>, Decline> {
    let mode = match mode() {
        Ok(mode) => *mode,
        Err(reason) => return Err(Decline::Mode(reason)),
    };
    if mode == Mode::Cpu {
        return Err(Decline::Mode("CUPRITE_GPU=0"));
    }
    if values.is_empty() || passes == 0 {
        return Err(Decline::NoWork);
    }
    let profile = match (mode, profile()) {
        (Mode::Auto, Ok(profile)) => Some(profile),
        (Mode::Auto, Err(reason)) => return Err(Decline::Profile(reason)),
        _ => None,
    };
    let mut state = match STATE.try_lock() {
        Ok(state) => state,
        Err(TryLockError::WouldBlock) => return Err(Decline::Busy),
        Err(TryLockError::Poisoned(_)) => return Err(Decline::Poisoned),
    };
    let State { device, policy } = &mut *state;
    if let Device::Unavailable(reason) = device {
        return Err(Decline::Unavailable(reason.clone()));
    }
    let choice = match profile {
        None => None,
        Some(profile) => {
            let open = matches!(device, Device::Ready(_));
            let key = (step.mul, step.add);
            let choice = policy.choose(profile, values.len(), passes, key, open);
            if !choice.uses_gpu() {
                return Err(Decline::Policy(choice));
            }
            Some(choice)
        }
    };
    let opening = matches!(device, Device::Untried);
    if opening {
        *device = open_device(profile);
    }
    let memory = match device {
        Device::Ready(memory) => memory,
        Device::Unavailable(reason) => return Err(Decline::Unavailable(reason.clone())),
        Device::Untried => unreachable!("opened above"),
    };
    let start = Instant::now();
    run_region(memory, values, step, passes).map_err(Decline::Failed)?;
    // The first region after opening also allocates for the first time,
    // which the profile already counts as startup.
    if let (Some(profile), false) = (profile, opening) {
        let micros = start.elapsed().as_secs_f64() * 1e6;
        let key = (step.mul, step.add);
        policy.record(profile, values.len(), passes, key, true, micros);
    }
    Ok(choice)
}

/// Opens the GPU; under the policy, only if it is the one profiled.
fn open_device(profile: Option<&Profile>) -> Device {
    let memory = match GpuMemory::open(MemoryConfig::default()) {
        Ok(memory) => memory,
        Err(error) => return Device::Unavailable(format!("open: {error}").into()),
    };
    match profile.map(|profile| profile.check_device(memory.device())) {
        Some(Err(reason)) => Device::Unavailable(reason.into()),
        _ => Device::Ready(memory),
    }
}

/// One complete region: upload, `passes` maps, download and commit.
pub(crate) fn run_region(
    memory: &mut GpuMemory,
    values: &mut [u32],
    step: MapStep,
    passes: u32,
) -> Result<(), String> {
    let mut session = memory.begin_mut(values).map_err(|e| e.to_string())?;
    for _ in 0..passes {
        session.map(step).map_err(|e| e.to_string())?;
    }
    session.finish().map_err(|e| e.to_string())
}

/// The CPU side of the computation, as calibration measures it.
pub(crate) fn cpu_map(values: &mut [u32], step: MapStep, passes: u32) {
    for _ in 0..passes {
        for value in values.iter_mut() {
            *value = value.wrapping_mul(step.mul).wrapping_add(step.add);
        }
    }
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

    // The mode and profile are read once per process, and test binaries set
    // neither, so this exercises the policy without a profile. The driver's
    // integration tests cover the GPU paths in real rewritten programs.
    #[test]
    fn without_a_profile_values_are_left_for_the_cpu() {
        let mut values = vec![1, 2, 3];
        assert_eq!(map_u32(&mut values, 3, 7, 2), UNTIMED);
        assert_eq!(values, [1, 2, 3]);
        assert!(counts().1 >= 1);
        map_u32_finished(UNTIMED);
    }

    fn timing(token: u64) -> Timing {
        Timing {
            token,
            start: Instant::now(),
            len: 1,
            passes: 1,
            step: MapStep { mul: 3, add: 7 },
        }
    }

    fn pending() -> Vec<u64> {
        TIMINGS.with_borrow(|timings| timings.iter().map(|t| t.token).collect())
    }

    /// Bodies finish innermost first. One that unwound never reported, and
    /// its entry is dropped when an enclosing body finishes.
    #[test]
    fn finished_bodies_match_their_tokens() {
        TIMINGS.with_borrow_mut(|timings| {
            timings.extend([timing(5), timing(7), timing(9), timing(11)]);
        });
        map_u32_finished(11);
        assert_eq!(pending(), [5, 7, 9]);
        map_u32_finished(7); // 9 unwound.
        assert_eq!(pending(), [5]);
        map_u32_finished(6); // Unknown: nothing to pop.
        assert_eq!(pending(), [5]);
        map_u32_finished(DONE);
        map_u32_finished(5);
        assert_eq!(pending(), Vec::<u64>::new());
    }

    #[test]
    fn the_cpu_reference_applies_every_pass() {
        let mut values = vec![0, 1, u32::MAX];
        cpu_map(&mut values, MapStep { mul: 3, add: 7 }, 2);
        let expected: Vec<u32> = [0u32, 1, u32::MAX]
            .iter()
            .map(|v| {
                let once = v.wrapping_mul(3).wrapping_add(7);
                once.wrapping_mul(3).wrapping_add(7)
            })
            .collect();
        assert_eq!(values, expected);
    }
}
