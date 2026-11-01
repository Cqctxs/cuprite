//! Measured CPU/GPU selection (M4): a cost model for each side, fitted to
//! complete-call measurements, a profile file that records them with what
//! they were measured on, and the rule that chooses between the two.

use std::{fmt, fs, path::Path};

use cuprite_memory::DeviceInfo;

/// The measured cost at one length: `fixed + per_pass·p` microseconds for
/// `p` passes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Row {
    pub len: usize,
    pub fixed: f64,
    pub per_pass: f64,
}

impl Row {
    fn at(&self, passes: f64) -> f64 {
        self.fixed + self.per_pass * passes
    }
}

/// Predicted time of one complete call, from measurements at a few lengths.
///
/// At each measured length the time is linear in the passes: on the GPU,
/// `fixed` is the session and its transfers and `per_pass` a launch and its
/// work; on the CPU nearly all of it is `per_pass`. Between lengths the time
/// is interpolated linearly in the length, and beyond them extrapolated from
/// the nearest two. A single global formula cannot follow the CPU from cache
/// into main memory, where each element costs 2.5 times as much.
#[derive(Clone, Debug, PartialEq)]
pub struct Cost {
    /// At least two, by strictly increasing length.
    pub rows: Vec<Row>,
}

impl Cost {
    pub fn predict(&self, len: usize, passes: u32) -> f64 {
        let (n, p) = (len as f64, f64::from(passes));
        let rows = &self.rows;
        // The segment containing `n`, or the first or last one outside them.
        let i = rows
            .partition_point(|row| row.len as f64 <= n)
            .clamp(1, rows.len() - 1);
        let (lo, hi) = (&rows[i - 1], &rows[i]);
        let t = (n - lo.len as f64) / (hi.len - lo.len) as f64;
        (lo.at(p) + t * (hi.at(p) - lo.at(p))).max(0.0)
    }
}

/// One complete call: `len` elements and `passes` passes took `micros`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub len: usize,
    pub passes: u32,
    pub micros: f64,
}

/// Fits a [`Cost`] with one row per measured length. Returns `None` unless
/// there are at least two lengths, each with two pass counts, and every
/// time is positive.
pub fn fit(samples: &[Sample]) -> Option<Cost> {
    if samples
        .iter()
        .any(|s| !(s.micros > 0.0 && s.micros.is_finite()))
    {
        return None;
    }
    let mut lengths: Vec<usize> = samples.iter().map(|s| s.len).collect();
    lengths.sort_unstable();
    lengths.dedup();
    let rows: Vec<Row> = lengths
        .into_iter()
        .map(|len| {
            let points: Vec<(f64, f64)> = samples
                .iter()
                .filter(|s| s.len == len)
                .map(|s| (f64::from(s.passes), s.micros))
                .collect();
            fit_line(&points).map(|(fixed, per_pass)| Row {
                len,
                fixed,
                per_pass,
            })
        })
        .collect::<Option<_>>()?;
    (rows.len() >= 2).then_some(Cost { rows })
}

/// `y = a + b·x` with `a, b ≥ 0`, by least squares on relative error, so
/// that one pass, where the choice is closest, counts as much as many.
fn fit_line(points: &[(f64, f64)]) -> Option<(f64, f64)> {
    // Weighted sums with weights 1/y².
    let sum = |f: &dyn Fn(f64, f64) -> f64| points.iter().map(|&(x, y)| f(x, y) / (y * y)).sum();
    let (s1, sx, sxx): (f64, f64, f64) = (sum(&|_, _| 1.0), sum(&|x, _| x), sum(&|x, _| x * x));
    let (sy, sxy): (f64, f64) = (sum(&|_, y| y), sum(&|x, y| x * y));
    let det = s1 * sxx - sx * sx;
    if det <= 1e-12 * s1 * sxx {
        return None; // Fewer than two distinct pass counts.
    }
    let a = (sxx * sy - sx * sxy) / det;
    let b = (s1 * sxy - sx * sy) / det;
    Some(if a < 0.0 {
        (0.0, sxy / sxx)
    } else if b < 0.0 {
        (sy / s1, 0.0)
    } else {
        (a, b)
    })
}

/// The largest relative difference between a model and its samples.
pub fn worst_error(cost: &Cost, samples: &[Sample]) -> f64 {
    samples
        .iter()
        .map(|s| (cost.predict(s.len, s.passes) - s.micros).abs() / s.micros)
        .fold(0.0, f64::max)
}

/// What the CPU side of a profile was measured with. Checked when the
/// profile is loaded, before any GPU work.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostKey {
    /// The computation and its numerical contract.
    pub computation: String,
    pub runtime: String,
    /// The compiler that built the runtime, whose CPU loop was measured.
    pub rustc: String,
    pub cpu: String,
}

pub const COMPUTATION: &str = "map_u32: x * mul + add, wrapping u32";

impl HostKey {
    pub fn current() -> Self {
        Self {
            computation: COMPUTATION.into(),
            runtime: env!("CARGO_PKG_VERSION").into(),
            rustc: env!("CUPRITE_RUSTC_VERSION").into(),
            cpu: cpu_name(),
        }
    }
}

/// The processor's brand string, such as "Intel(R) Core(TM) Ultra 9 275HX".
fn cpu_name() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::__cpuid;
        // `__cpuid` is safe on newer toolchains and `unsafe` on older ones;
        // CPUID is always available on x86_64.
        #[allow(unused_unsafe)]
        let brand: Vec<u8> = unsafe {
            if __cpuid(0x8000_0000).eax < 0x8000_0004 {
                return "x86_64".into();
            }
            (0x8000_0002..=0x8000_0004u32)
                .flat_map(|leaf| {
                    let r = __cpuid(leaf);
                    [r.eax, r.ebx, r.ecx, r.edx]
                })
                .flat_map(u32::to_le_bytes)
                .collect()
        };
        String::from_utf8_lossy(&brand)
            .trim_matches(|c: char| c == '\0' || c.is_whitespace())
            .to_owned()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        std::env::consts::ARCH.to_owned()
    }
}

/// Measurements of both sides for one computation on one machine.
#[derive(Clone, Debug, PartialEq)]
pub struct Profile {
    pub host: HostKey,
    /// Checked when the GPU is opened.
    pub device: DeviceInfo,
    /// Extra time of the first GPU call in a process: opening the device and
    /// compiling the kernels.
    pub cold_us: f64,
    pub gpu: Cost,
    pub cpu: Cost,
}

const FORMAT: &str = "1";

impl Profile {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = fs::read_to_string(path)
            .map_err(|error| format!("cannot read profile {}: {error}", path.display()))?;
        Self::parse(&text).map_err(|error| format!("profile {}: {error}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        let mut fields = std::collections::HashMap::new();
        for line in text.lines().map(str::trim) {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| format!("malformed line `{line}`"))?;
            if fields.insert(key.trim(), value.trim()).is_some() {
                return Err(format!("`{}` appears twice", key.trim()));
            }
        }
        let mut take = |key: &str| fields.remove(key).ok_or_else(|| format!("missing `{key}`"));
        if take("format")? != FORMAT {
            return Err(format!("not format {FORMAT}"));
        }
        let number = |key: &str, value: &str| -> Result<f64, String> {
            value
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite() && *v >= 0.0)
                .ok_or_else(|| format!("`{key}` is not a non-negative number"))
        };
        let cost = |key: &str, value: &str| -> Result<Cost, String> {
            let rows: Vec<Row> = value
                .split(';')
                .map(|row| match row.split_whitespace().collect::<Vec<_>>()[..] {
                    [len, fixed, per_pass] => Ok(Row {
                        len: len
                            .parse()
                            .map_err(|_| format!("`{key}` has a bad length `{len}`"))?,
                        fixed: number(key, fixed)?,
                        per_pass: number(key, per_pass)?,
                    }),
                    _ => Err(format!("`{key}` rows are `length fixed per_pass`")),
                })
                .collect::<Result<_, _>>()?;
            if rows.len() < 2 || rows.windows(2).any(|pair| pair[0].len >= pair[1].len) {
                return Err(format!(
                    "`{key}` needs two or more rows by increasing length"
                ));
            }
            Ok(Cost { rows })
        };
        let host = HostKey {
            computation: take("computation")?.into(),
            runtime: take("runtime")?.into(),
            rustc: take("rustc")?.into(),
            cpu: take("cpu")?.into(),
        };
        let capability = take("compute_capability")?;
        let compute_capability = capability
            .split_once('.')
            .and_then(|(major, minor)| Some((major.parse().ok()?, minor.parse().ok()?)))
            .ok_or("`compute_capability` is not major.minor")?;
        let device = DeviceInfo {
            name: take("device")?.into(),
            compute_capability,
            driver_version: take("driver")?
                .parse()
                .map_err(|_| "`driver` is not a number")?,
            kernels: u64::from_str_radix(take("kernels")?, 16)
                .map_err(|_| "`kernels` is not a hexadecimal hash")?,
        };
        let profile = Self {
            host,
            device,
            cold_us: number("cold_us", take("cold_us")?)?,
            gpu: cost("gpu_us", take("gpu_us")?)?,
            cpu: cost("cpu_us", take("cpu_us")?)?,
        };
        if let Some(key) = fields.keys().next() {
            return Err(format!("unknown key `{key}`"));
        }
        Ok(profile)
    }

    /// Fails with what differs if this profile was not measured with `host`.
    pub fn check_host(&self, host: &HostKey) -> Result<(), String> {
        let fields = [
            ("computation", &self.host.computation, &host.computation),
            ("runtime", &self.host.runtime, &host.runtime),
            ("rustc", &self.host.rustc, &host.rustc),
            ("cpu", &self.host.cpu, &host.cpu),
        ];
        mismatch(&fields)
    }

    /// Fails with what differs if this profile was not measured on `device`.
    pub fn check_device(&self, device: &DeviceInfo) -> Result<(), String> {
        let ours = &self.device;
        let fields = [
            ("device", &ours.name, &device.name),
            (
                "compute capability",
                &format!(
                    "{}.{}",
                    ours.compute_capability.0, ours.compute_capability.1
                ),
                &format!(
                    "{}.{}",
                    device.compute_capability.0, device.compute_capability.1
                ),
            ),
            (
                "driver",
                &ours.driver_version.to_string(),
                &device.driver_version.to_string(),
            ),
            (
                "kernels",
                &format!("{:016x}", ours.kernels),
                &format!("{:016x}", device.kernels),
            ),
        ];
        mismatch(&fields)
    }
}

fn mismatch(fields: &[(&str, &String, &String)]) -> Result<(), String> {
    match fields.iter().find(|(_, profile, actual)| profile != actual) {
        Some((name, profile, actual)) => {
            Err(format!("profile is for {name} `{profile}`, not `{actual}`"))
        }
        None => Ok(()),
    }
}

impl fmt::Display for Profile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cost = |c: &Cost| {
            c.rows
                .iter()
                .map(|row| format!("{} {} {}", row.len, row.fixed, row.per_pass))
                .collect::<Vec<_>>()
                .join("; ")
        };
        writeln!(
            f,
            "# Cuprite performance profile, written by cuprite-calibrate."
        )?;
        writeln!(
            f,
            "# Costs are rows of `length fixed per_pass`: a call with"
        )?;
        writeln!(
            f,
            "# that many elements takes fixed + per_pass * passes us."
        )?;
        writeln!(f, "format = {FORMAT}")?;
        writeln!(f, "computation = {}", self.host.computation)?;
        writeln!(f, "runtime = {}", self.host.runtime)?;
        writeln!(f, "rustc = {}", self.host.rustc)?;
        writeln!(f, "cpu = {}", self.host.cpu)?;
        writeln!(f, "device = {}", self.device.name)?;
        let (major, minor) = self.device.compute_capability;
        writeln!(f, "compute_capability = {major}.{minor}")?;
        writeln!(f, "driver = {}", self.device.driver_version)?;
        writeln!(f, "kernels = {:016x}", self.device.kernels)?;
        writeln!(f, "cold_us = {}", self.cold_us)?;
        writeln!(f, "gpu_us = {}", cost(&self.gpu))?;
        writeln!(f, "cpu_us = {}", cost(&self.cpu))
    }
}

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
/// Calibration measures a general multiply-add, but the compiled CPU body
/// of a function is specialized to its constants (`x * 3` becomes shifts and
/// adds) and can be twice as fast; how much depends on the constants and on
/// whether the slice fits in cache. So each correction applies to one map
/// and one range of lengths.
#[derive(Debug, Default)]
pub struct Policy {
    forgone_us: f64,
    /// Measured over predicted time, by side, map and length range. Few
    /// entries, so a list.
    corrections: Vec<(Key, f64)>,
}

/// What a correction applies to. The GPU kernel's cost does not depend on
/// the constants, the CPU's does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Key {
    Gpu { range: u32 },
    Cpu { step: (u32, u32), range: u32 },
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
    pub fn predict(
        &self,
        profile: &Profile,
        len: usize,
        passes: u32,
        step: (u32, u32),
    ) -> Prediction {
        let range = range(len);
        Prediction {
            gpu: profile.gpu.predict(len, passes) * self.correction(Key::Gpu { range }),
            cpu: profile.cpu.predict(len, passes) * self.correction(Key::Cpu { step, range }),
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
        step: (u32, u32),
        gpu: bool,
        micros: f64,
    ) {
        let range = range(len);
        let (key, base) = if gpu {
            (Key::Gpu { range }, profile.gpu.predict(len, passes))
        } else {
            (Key::Cpu { step, range }, profile.cpu.predict(len, passes))
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
        step: (u32, u32),
        open: bool,
    ) -> Choice {
        let predicted = self.predict(profile, len, passes, step);
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
mod tests {
    use super::*;

    const STEP: (u32, u32) = (3, 7);

    const LENGTHS: [usize; 6] = [1 << 10, 1 << 14, 1 << 17, 1 << 20, 1 << 22, 1 << 24];

    /// `fixed + per_element·n + per_pass·p + per_element_pass·n·p`, as rows
    /// at `LENGTHS`. Linear in `n`, so it is exact between rows too.
    fn cost(fixed: f64, per_element: f64, per_pass: f64, per_element_pass: f64) -> Cost {
        let row = |len: usize| Row {
            len,
            fixed: fixed + per_element * len as f64,
            per_pass: per_pass + per_element_pass * len as f64,
        };
        Cost {
            rows: LENGTHS.into_iter().map(row).collect(),
        }
    }

    fn profile() -> Profile {
        Profile {
            host: HostKey {
                computation: COMPUTATION.into(),
                runtime: "0.1.0".into(),
                rustc: "rustc 1.0.0-nightly".into(),
                cpu: "Test CPU".into(),
            },
            device: DeviceInfo {
                name: "Test GPU".into(),
                compute_capability: (12, 0),
                driver_version: 13000,
                kernels: 0x0123_4567_89ab_cdef,
            },
            cold_us: 100_000.0,
            // Warm GPU calls: 80 us, plus transfers, launches and work.
            gpu: cost(80.0, 1e-3, 8.0, 2e-5),
            cpu: cost(0.0, 0.0, 0.0, 1e-4),
        }
    }

    fn samples(micros: impl Fn(usize, u32, usize) -> f64) -> Vec<Sample> {
        let mut samples = Vec::new();
        for len in LENGTHS {
            for passes in [1, 4, 16, 64] {
                let micros = micros(len, passes, samples.len());
                samples.push(Sample {
                    len,
                    passes,
                    micros,
                });
            }
        }
        samples
    }

    #[test]
    fn fitting_recovers_the_model_that_produced_the_samples() {
        let truth = profile().gpu;
        let samples = samples(|len, passes, _| truth.predict(len, passes));
        let fitted = fit(&samples).unwrap();
        assert!(worst_error(&fitted, &samples) < 1e-9, "{fitted:?}");
        for (a, b) in fitted.rows.iter().zip(&truth.rows) {
            assert_eq!(a.len, b.len);
            assert!((a.fixed - b.fixed).abs() <= 1e-9 * b.fixed, "{a:?} {b:?}");
            assert!(
                (a.per_pass - b.per_pass).abs() <= 1e-9 * b.per_pass,
                "{a:?} {b:?}"
            );
        }
    }

    /// The CPU's cost per element grows from cache into main memory; the
    /// rows follow it where one formula cannot.
    #[test]
    fn fitting_follows_a_cost_that_changes_with_length() {
        let per_element_pass = |len: usize| if len <= 1 << 17 { 1e-4 } else { 2.5e-4 };
        let samples =
            samples(|len, passes, _| per_element_pass(len) * len as f64 * f64::from(passes));
        let fitted = fit(&samples).unwrap();
        assert!(worst_error(&fitted, &samples) < 1e-9, "{fitted:?}");
    }

    #[test]
    fn fitting_keeps_every_term_non_negative() {
        // Pure per-pass work, with noise that would make `fixed` negative.
        let samples = samples(|len, passes, i| {
            1e-4 * len as f64 * f64::from(passes) * (1.0 + 0.05 * (i % 3) as f64)
        });
        let fitted = fit(&samples).unwrap();
        assert!(
            fitted
                .rows
                .iter()
                .all(|r| r.fixed >= 0.0 && r.per_pass >= 0.0),
            "{fitted:?}"
        );
        assert!(worst_error(&fitted, &samples) < 0.1, "{fitted:?}");
    }

    #[test]
    fn fitting_needs_usable_samples() {
        assert_eq!(fit(&[]), None);
        let one = |len, passes, micros| Sample {
            len,
            passes,
            micros,
        };
        // Zero time, one length, and one pass count are all unusable.
        assert_eq!(fit(&[one(1, 1, 0.0), one(1, 2, 1.0)]), None);
        assert_eq!(fit(&[one(1, 1, 1.0), one(1, 2, 2.0)]), None);
        assert_eq!(fit(&[one(1, 1, 1.0), one(2, 1, 2.0)]), None);
    }

    #[test]
    fn predictions_interpolate_between_lengths_and_extrapolate_beyond() {
        let cost = Cost {
            rows: vec![
                Row {
                    len: 100,
                    fixed: 10.0,
                    per_pass: 1.0,
                },
                Row {
                    len: 200,
                    fixed: 20.0,
                    per_pass: 3.0,
                },
            ],
        };
        assert_eq!(cost.predict(100, 2), 12.0);
        assert_eq!(cost.predict(150, 2), 19.0);
        assert_eq!(cost.predict(300, 2), 40.0);
        assert_eq!(cost.predict(0, 2), 0.0, "clamped at zero");
    }

    #[test]
    fn profiles_round_trip_through_text() {
        let profile = profile();
        assert_eq!(Profile::parse(&profile.to_string()).unwrap(), profile);
    }

    #[test]
    fn malformed_profiles_are_refused() {
        let text = profile().to_string();
        let cases = [
            (text.replace("format = 1", "format = 2"), "not format 1"),
            (
                text.replace("cold_us = 100000", "cold_us = -1"),
                "`cold_us`",
            ),
            (text.replace("gpu_us = 1024 ", "gpu_us = "), "rows are"),
            (
                text.replace("gpu_us = 1024 ", "gpu_us = 99999999 "),
                "increasing length",
            ),
            (text.replace("driver = 13000\n", ""), "missing `driver`"),
            (text.clone() + "speed = 11\n", "unknown key `speed`"),
            (text.clone() + "cpu = Other\n", "appears twice"),
            (text.replace("kernels = ", "kernels = xyz"), "hexadecimal"),
        ];
        for (text, expected) in cases {
            let error = Profile::parse(&text).unwrap_err();
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn profiles_apply_only_where_they_were_measured() {
        let profile = profile();
        assert_eq!(profile.check_host(&profile.host), Ok(()));
        assert_eq!(profile.check_device(&profile.device), Ok(()));

        let mut host = profile.host.clone();
        host.cpu = "Other CPU".into();
        assert_eq!(
            profile.check_host(&host).unwrap_err(),
            "profile is for cpu `Test CPU`, not `Other CPU`"
        );
        let mut device = profile.device.clone();
        device.driver_version = 13010;
        assert_eq!(
            profile.check_device(&device).unwrap_err(),
            "profile is for driver `13000`, not `13010`"
        );
        device = profile.device.clone();
        device.kernels = 1;
        assert!(profile.check_device(&device).is_err());
    }

    #[test]
    fn calls_the_gpu_loses_stay_on_the_cpu() {
        let profile = profile();
        let mut policy = Policy::new();
        for _ in 0..1000 {
            // 1000 elements, 8 passes: 0.8 us on the CPU, ~150 us on the GPU.
            assert!(matches!(
                policy.choose(&profile, 1000, 8, STEP, false),
                Choice::Cpu(_)
            ));
            assert!(matches!(
                policy.choose(&profile, 1000, 8, STEP, true),
                Choice::Cpu(_)
            ));
        }
        assert_eq!(policy.forgone_us, 0.0);
    }

    #[test]
    fn the_gpu_opens_once_forgone_savings_cover_its_startup() {
        let profile = profile();
        let mut policy = Policy::new();
        let (len, passes) = (1 << 20, 64);
        let saving = profile.cpu.predict(len, passes) - profile.gpu.predict(len, passes);
        let calls_to_wait = (profile.cold_us / saving).ceil() as usize;
        assert!(calls_to_wait > 1, "the test needs a few calls");
        for call in 1..calls_to_wait {
            let choice = policy.choose(&profile, len, passes, STEP, false);
            assert!(
                matches!(choice, Choice::Wait { .. }),
                "call {call}: {choice:?}"
            );
            assert!(!choice.uses_gpu());
            let reason = choice.to_string();
            assert!(
                reason.contains("; waiting to cover 100.00 ms startup"),
                "{reason}"
            );
        }
        let choice = policy.choose(&profile, len, passes, STEP, false);
        assert!(matches!(choice, Choice::Open(_)), "{choice:?}");
        assert!(choice.uses_gpu());
        assert!(
            choice.to_string().ends_with("; startup now covered"),
            "{choice}"
        );
        assert!(matches!(
            policy.choose(&profile, len, passes, STEP, true),
            Choice::Gpu(_)
        ));
    }

    #[test]
    fn one_call_large_enough_opens_the_gpu_at_once() {
        let profile = profile();
        let mut policy = Policy::new();
        // ~1.7 s on the CPU: the saving alone covers startup.
        assert!(matches!(
            policy.choose(&profile, 1 << 24, 1024, STEP, false),
            Choice::Open(_)
        ));
    }

    /// The profile says the CPU loses this call; the program's own CPU body
    /// turns out twice as fast, which makes it the winner.
    #[test]
    fn measurements_correct_the_profile_for_one_map_and_length_range() {
        let profile = profile();
        let mut policy = Policy::new();
        // 1M elements, 16 passes: the profile favors the GPU by less than 2x.
        let (len, passes) = (1 << 20, 16);
        let before = policy.predict(&profile, len, passes, STEP);
        assert!(
            before.gpu < before.cpu && before.cpu < 2.0 * before.gpu,
            "{before:?}"
        );
        assert!(Policy::worth_timing(before));

        for _ in 0..20 {
            policy.record(&profile, len, passes, STEP, false, before.cpu / 2.0);
        }
        let after = policy.predict(&profile, len, passes, STEP);
        assert!(
            (after.cpu / (before.cpu / 2.0) - 1.0).abs() < 0.01,
            "{after:?}"
        );
        assert_eq!(after.gpu, before.gpu);
        assert!(matches!(
            policy.choose(&profile, len, passes, STEP, false),
            Choice::Cpu(_)
        ));

        // Other constants, other lengths, and the GPU side keep the profile.
        assert_eq!(policy.predict(&profile, len, passes, (5, 1)), before);
        let far = policy.predict(&profile, len * 4, passes, STEP);
        assert_eq!(far.cpu, profile.cpu.predict(len * 4, passes));
        // Lengths within a factor of two share the correction.
        let near = policy.predict(&profile, len + 1000, passes, STEP);
        assert!(near.cpu < 0.6 * profile.cpu.predict(len + 1000, passes));
    }

    #[test]
    fn one_outlier_moves_a_correction_only_so_far() {
        let profile = profile();
        let mut policy = Policy::new();
        let (len, passes) = (1 << 20, 16);
        let predicted = profile.gpu.predict(len, passes);
        policy.record(&profile, len, passes, STEP, true, predicted * 1000.0);
        let after = policy.predict(&profile, len, passes, STEP);
        // Counted as 4x, and moved a quarter of the way from 1.
        assert_eq!(after.gpu, predicted * 1.75);
        // Later ordinary calls pull it back.
        for _ in 0..40 {
            policy.record(&profile, len, passes, STEP, true, predicted);
        }
        let settled = policy.predict(&profile, len, passes, STEP);
        assert!((settled.gpu / predicted - 1.0).abs() < 0.01, "{settled:?}");
    }

    #[test]
    fn only_close_calls_are_worth_timing() {
        let at = |gpu, cpu| Policy::worth_timing(Prediction { gpu, cpu });
        assert!(at(1.0, 1.0));
        assert!(at(3.9, 1.0));
        assert!(at(1.0, 3.9));
        assert!(!at(72.0, 0.8), "small calls are never timed");
        assert!(!at(1.0, 40.0));
    }

    #[test]
    fn choices_explain_themselves() {
        let predicted = Prediction {
            gpu: 72.0,
            cpu: 0.8,
        };
        assert_eq!(
            Choice::Cpu(predicted).to_string(),
            "predicted gpu 72.0 us vs cpu 0.8 us"
        );
        let wait = Choice::Wait {
            predicted,
            cold_us: 120_000.0,
            forgone_us: 3_500.0,
        };
        assert_eq!(
            wait.to_string(),
            "predicted gpu 72.0 us vs cpu 0.8 us; waiting to cover 120.00 ms startup, 3.50 ms so far"
        );
    }

    #[test]
    fn durations_read_naturally() {
        assert_eq!(show(12.34), "12.3 us");
        assert_eq!(show(4_560.0), "4.56 ms");
        assert_eq!(show(2_500_000.0), "2.50 s");
    }
}
