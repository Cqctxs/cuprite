//! Measured CPU/GPU selection (M4): a cost model for each side, fitted to
//! complete-call measurements, a profile file that records them with what
//! they were measured on, and the rule that chooses between the two.

use std::fmt;

mod cost;
mod profile;

pub use cost::{fit, worst_error, Cost, Row, Sample};
pub use profile::{HostKey, Profile, COMPUTATION};

/// A duration in microseconds, for messages.
pub fn show(micros: f64) -> String {
    if micros < 1_000.0 {
        format!("{micros:.1} us")
    } else if micros < 1_000_000.0 {
        format!("{:.2} ms", micros / 1e3)
    } else {
        format!("{:.2} s", micros / 1e6)
    }
}

/// Predicted microseconds of one call on each side.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prediction {
    pub gpu: f64,
    pub cpu: f64,
}

/// A decision, with the numbers behind it. It is formatted only for the
/// trace, so that declined calls stay cheap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Choice {
    /// The CPU is predicted faster.
    Cpu(Prediction),
    /// The GPU is predicted faster, but its startup is not covered yet.
    Wait {
        predicted: Prediction,
        cold_us: f64,
        forgone_us: f64,
    },
    Gpu(Prediction),
    /// Open the GPU, then use it: waiting has now cost as much as opening.
    Open(Prediction),
}

impl Choice {
    pub fn uses_gpu(&self) -> bool {
        matches!(self, Self::Gpu(_) | Self::Open(_))
    }

    pub fn prediction(&self) -> Prediction {
        let (Self::Cpu(p) | Self::Wait { predicted: p, .. } | Self::Gpu(p) | Self::Open(p)) = self;
        *p
    }
}

impl fmt::Display for Choice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (Self::Cpu(p) | Self::Wait { predicted: p, .. } | Self::Gpu(p) | Self::Open(p)) = self;
        write!(f, "predicted gpu {} vs cpu {}", show(p.gpu), show(p.cpu))?;
        match self {
            Self::Wait {
                cold_us,
                forgone_us,
                ..
            } => write!(
                f,
                "; waiting to cover {} startup, {} so far",
                show(*cold_us),
                show(*forgone_us)
            ),
            Self::Open(_) => write!(f, "; startup now covered"),
            Self::Cpu(_) | Self::Gpu(_) => Ok(()),
        }
    }
}

/// Chooses a side for each call from a profile.
///
/// While the GPU is open, a call uses it when it is predicted faster. Before
/// that, opening costs `cold_us` once, which only later calls can repay, and
/// whether there will be any is unknown. So each call the GPU would win
/// still runs on the CPU and adds what it would have saved; the GPU opens
/// once that total covers `cold_us`. This is the ski-rental rule: it spends
/// at most twice what the best choice made with hindsight would, since
/// startup is paid only after as much has been lost waiting.
///
/// The profile is corrected by what calls actually take ([`Policy::record`]).
/// Calibration measures one multiply-add kernel and its CPU loop; a
/// generated kernel and the function it came from can be several times
/// faster or slower on either side, and specialized to their constants
/// (`x * 3` becomes shifts and adds). So each correction applies to one
/// kernel, one side and one range of lengths.
#[derive(Debug, Default)]
pub struct Policy {
    forgone_us: f64,
    /// Measured over predicted time, by side, kernel and length range. Few
    /// entries, so a list.
    corrections: Vec<(Key, f64)>,
}

/// What a correction applies to: a kernel, by its description's hash.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Key {
    Gpu { kernel: u64, range: u32 },
    Cpu { kernel: u64, range: u32 },
}

/// Lengths within a factor of two share a correction.
fn range(len: usize) -> u32 {
    usize::BITS - len.leading_zeros()
}

/// How much one measurement moves a correction.
const LEARNING_RATE: f64 = 0.25;

/// A call is worth timing when the prediction is within this factor of a
/// tie; anything further is decided the same way even if the profile is
/// off by that much.
const CLOSE: f64 = 4.0;

impl Policy {
    pub const fn new() -> Self {
        Self {
            forgone_us: 0.0,
            corrections: Vec::new(),
        }
    }

    fn correction(&self, key: Key) -> f64 {
        self.corrections
            .iter()
            .find(|(k, _)| *k == key)
            .map_or(1.0, |&(_, ratio)| ratio)
    }

    /// The profile's prediction, corrected by measurements so far.
    pub fn predict(&self, profile: &Profile, len: usize, passes: u32, kernel: u64) -> Prediction {
        let range = range(len);
        Prediction {
            gpu: profile.gpu.predict(len, passes) * self.correction(Key::Gpu { kernel, range }),
            cpu: profile.cpu.predict(len, passes) * self.correction(Key::Cpu { kernel, range }),
        }
    }

    /// True if timing this call could change later decisions.
    pub fn worth_timing(predicted: Prediction) -> bool {
        (1.0 / CLOSE..CLOSE).contains(&(predicted.gpu / predicted.cpu))
    }

    /// Learns from a call that took `micros` on one side. One outlier (a
    /// context switch, say) counts as at most a factor of `CLOSE`, and
    /// moves the correction only a quarter of the way there.
    pub fn record(
        &mut self,
        profile: &Profile,
        len: usize,
        passes: u32,
        kernel: u64,
        gpu: bool,
        micros: f64,
    ) {
        let range = range(len);
        let (key, base) = if gpu {
            (Key::Gpu { kernel, range }, profile.gpu.predict(len, passes))
        } else {
            (Key::Cpu { kernel, range }, profile.cpu.predict(len, passes))
        };
        if !(base > 0.0 && micros > 0.0) {
            return;
        }
        // Every correction starts at 1, trusting the profile, and moves part
        // of the way each time. A side that one unlucky call made look slow
        // is no longer chosen, so it could never be measured again: one
        // call must not decide that alone.
        let measured = (micros / base).clamp(1.0 / CLOSE, CLOSE);
        let ratio = match self.corrections.iter_mut().find(|(k, _)| *k == key) {
            Some((_, ratio)) => ratio,
            None => {
                self.corrections.push((key, 1.0));
                &mut self.corrections.last_mut().unwrap().1
            }
        };
        *ratio += LEARNING_RATE * (measured - *ratio);
    }

    pub fn choose(
        &mut self,
        profile: &Profile,
        len: usize,
        passes: u32,
        kernel: u64,
        open: bool,
    ) -> Choice {
        let predicted = self.predict(profile, len, passes, kernel);
        if predicted.gpu >= predicted.cpu {
            return Choice::Cpu(predicted);
        }
        if open {
            return Choice::Gpu(predicted);
        }
        self.forgone_us += predicted.cpu - predicted.gpu;
        if self.forgone_us >= profile.cold_us {
            Choice::Open(predicted)
        } else {
            Choice::Wait {
                predicted,
                cold_us: profile.cold_us,
                forgone_us: self.forgone_us,
            }
        }
    }
}

#[cfg(test)]
mod tests;
