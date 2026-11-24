//! Calibration profile format and checks against the current host and device.

use std::{fmt, fs, path::Path};

use cuprite_memory::DeviceInfo;

use super::cost::{Cost, Row};

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
            cold_us: parse_nonnegative("cold_us", take("cold_us")?)?,
            gpu: parse_cost("gpu_us", take("gpu_us")?)?,
            cpu: parse_cost("cpu_us", take("cpu_us")?)?,
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

fn parse_nonnegative(key: &str, value: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite() && *v >= 0.0)
        .ok_or_else(|| format!("`{key}` is not a non-negative number"))
}

fn parse_cost(key: &str, value: &str) -> Result<Cost, String> {
    let rows: Vec<Row> = value
        .split(';')
        .map(|row| match row.split_whitespace().collect::<Vec<_>>()[..] {
            [len, fixed, per_pass] => Ok(Row {
                len: len
                    .parse()
                    .map_err(|_| format!("`{key}` has a bad length `{len}`"))?,
                fixed: parse_nonnegative(key, fixed)?,
                per_pass: parse_nonnegative(key, per_pass)?,
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
