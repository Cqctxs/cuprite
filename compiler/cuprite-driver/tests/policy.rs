//! M4: with `CUPRITE_GPU` unset, the runtime chooses each call's side from
//! a measured profile. `repeat_demo`, rewritten automatically, keeps small
//! work on the CPU, moves large repeated work to the GPU once that has paid
//! for the GPU's startup, and ignores profiles measured elsewhere.

mod support;

use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
    process::Command,
    sync::{Mutex, MutexGuard, OnceLock},
};

use support::{run_with, runtime, scratch, text, Probes};

/// Tests in this file run one at a time: calibration measures timings, and
/// a build or demo running beside it would distort them.
fn serial() -> MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn demo() -> &'static Path {
    static DEMO: OnceLock<PathBuf> = OnceLock::new();
    DEMO.get_or_init(|| {
        let probes = Probes::new("policy", true)
            .with_runtime()
            .env("CUPRITE_AUTO", "1");
        probes.build(&["build", "--bin", "repeat_demo"]);
        probes.exe("repeat_demo")
    })
}

/// A profile of this machine, measured once per test process.
fn calibrated() -> &'static Path {
    static PROFILE: OnceLock<PathBuf> = OnceLock::new();
    PROFILE.get_or_init(|| {
        let path = scratch("policy-profile.txt");
        let output = Command::new(&runtime().calibrate)
            .arg(&path)
            .output()
            .unwrap();
        assert!(output.status.success(), "{}", text(&output.stderr));
        path
    })
}

/// A copy of `profile` with one line replaced.
fn edited(profile: &Path, name: &str, key: &str, value: &str) -> PathBuf {
    let text = std::fs::read_to_string(profile).unwrap();
    let edited: Vec<String> = text
        .lines()
        .map(|line| match line.split_once(" = ") {
            Some((k, _)) if k == key => format!("{key} = {value}"),
            _ => line.to_owned(),
        })
        .collect();
    let path = scratch(name);
    std::fs::write(&path, edited.join("\n")).unwrap();
    path
}

/// Runs `repeat_demo <elements> <passes> <samples>` and returns the trace of
/// its `repeat` calls, one per sample plus the cold one. When one runs on
/// the CPU, its original body also calls the rewritten `transform` once per
/// pass; those lines are left out.
fn run_demo(profile: Option<&Path>, gpu: Option<&str>, args: &[&str; 3]) -> Vec<String> {
    let mut env: Vec<(&str, &OsStr)> = Vec::new();
    if let Some(profile) = profile {
        env.push(("CUPRITE_PROFILE", profile.as_os_str()));
    }
    if let Some(gpu) = gpu {
        env.push(("CUPRITE_GPU", OsStr::new(gpu)));
    }
    let trace = run_with(demo(), &env, args);
    let region = format!("len={} passes={} ", args[0], args[1]);
    let lines: Vec<String> = trace
        .lines()
        .filter(|line| line.contains(&region))
        .map(str::to_owned)
        .collect();
    let calls: usize = args[2].parse::<usize>().unwrap() + 1;
    assert_eq!(lines.len(), calls, "{trace}");
    lines
}

fn all_contain(lines: &[String], reason: &str) {
    assert!(!lines.is_empty());
    for line in lines {
        assert!(line.contains(reason), "{line}\nexpected: {reason}");
    }
}

/// Without a usable profile, nothing is measured, so everything stays on
/// the CPU, and the reason says what is wrong. No GPU needed.
#[test]
fn without_a_valid_profile_every_call_stays_on_the_cpu() {
    let _serial = serial();
    let args = ["4096", "4", "2"];
    all_contain(
        &run_demo(None, None, &args),
        "-> cpu (no profile; set CUPRITE_PROFILE)",
    );
    all_contain(
        &run_demo(None, Some("auto"), &args),
        "-> cpu (no profile; set CUPRITE_PROFILE)",
    );
    all_contain(&run_demo(None, Some("0"), &args), "-> cpu (CUPRITE_GPU=0)");
    all_contain(
        &run_demo(None, Some("yes"), &args),
        "-> cpu (CUPRITE_GPU=yes is not 0, 1 or auto)",
    );

    let missing = scratch("policy-missing.txt");
    all_contain(
        &run_demo(Some(&missing), None, &args),
        "cannot read profile",
    );

    let garbage = scratch("policy-garbage.txt");
    std::fs::write(&garbage, "format = 1\nspeed = fast\n").unwrap();
    all_contain(
        &run_demo(Some(&garbage), None, &args),
        "missing `computation`",
    );
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "needs an NVIDIA GPU; enable the gpu-tests feature"
)]
fn small_work_stays_on_the_cpu() {
    let _serial = serial();
    // 1000 elements, 8 passes: microseconds on the CPU, ~70 us on the GPU.
    let lines = run_demo(Some(calibrated()), None, &["1000", "8", "20"]);
    for line in &lines {
        assert!(line.contains("-> cpu (predicted gpu"), "{line}");
        assert!(!line.contains("waiting"), "{line}");
    }
}

/// 16M elements, 32 passes: about 4x faster on the GPU, so the forgone
/// savings cover startup within a call or two, and every later call runs on
/// the GPU.
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "needs an NVIDIA GPU; enable the gpu-tests feature"
)]
fn large_repeated_work_moves_to_the_gpu_once_startup_is_covered() {
    let _serial = serial();
    let lines = run_demo(Some(calibrated()), None, &["16777216", "32", "5"]);
    let first_gpu = lines
        .iter()
        .position(|line| line.contains("-> gpu"))
        .unwrap_or_else(|| panic!("never used the GPU: {lines:#?}"));
    assert!(first_gpu <= 2, "{lines:#?}");
    for line in &lines[..first_gpu] {
        assert!(line.contains("-> cpu (predicted gpu"), "{line}");
        assert!(line.contains("waiting to cover"), "{line}");
    }
    assert!(
        lines[first_gpu].contains("startup now covered"),
        "{lines:#?}"
    );
    for line in &lines[first_gpu..] {
        assert!(line.contains("-> gpu (predicted gpu"), "{line}");
    }
}

/// A copy of `profile` with every CPU cost multiplied by `factor`.
fn scaled_cpu(profile: &Path, name: &str, factor: f64) -> PathBuf {
    let text = std::fs::read_to_string(profile).unwrap();
    let line = text.lines().find(|l| l.starts_with("cpu_us = ")).unwrap();
    let rows: Vec<String> = line["cpu_us = ".len()..]
        .split("; ")
        .map(|row| {
            let [len, fixed, per_pass] = row.split(' ').collect::<Vec<_>>()[..] else {
                panic!("bad row {row}");
            };
            let scale = |v: &str| v.parse::<f64>().unwrap() * factor;
            format!("{len} {} {}", scale(fixed), scale(per_pass))
        })
        .collect();
    edited(profile, name, "cpu_us", &rows.join("; "))
}

/// The CPU prediction in a trace line, in microseconds.
fn predicted_cpu(line: &str) -> f64 {
    let after = line
        .split(" vs cpu ")
        .nth(1)
        .unwrap_or_else(|| panic!("{line}"));
    let mut words = after.split([' ', ';', ')']);
    let value: f64 = words.next().unwrap().parse().unwrap();
    match words.next().unwrap() {
        "us" => value,
        "ms" => value * 1e3,
        "s" => value * 1e6,
        unit => panic!("unit {unit} in {line}"),
    }
}

/// The profile claims the CPU is 3x slower than it is, so the policy starts
/// out favoring the GPU (waiting to cover its startup). Timing the CPU body
/// of these close calls pulls the CPU prediction down toward what the body
/// really takes. How far depends on the core the process runs on, which on
/// this laptop changes the CPU's speed by up to 3x, so the test checks the
/// direction and a clear amount, not where it ends.
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "needs an NVIDIA GPU; enable the gpu-tests feature"
)]
fn timing_the_cpu_body_corrects_a_wrong_profile() {
    let _serial = serial();
    let slow_cpu = scaled_cpu(calibrated(), "policy-slow-cpu.txt", 3.0);
    // 64K elements, 32 passes: ~0.1 ms on the CPU, ~0.3 ms on the GPU.
    let lines = run_demo(Some(&slow_cpu), None, &["65536", "32", "29"]);
    assert!(lines[0].contains("waiting to cover"), "{lines:#?}");
    // Far from covering startup in 30 calls, so the GPU stays closed.
    assert!(
        lines
            .iter()
            .all(|line| line.contains("-> cpu (predicted gpu")),
        "{lines:#?}"
    );
    let first = predicted_cpu(&lines[0]);
    let last = predicted_cpu(lines.last().unwrap());
    assert!(last < 0.75 * first, "{first} us -> {last} us\n{lines:#?}");
}

/// A profile from another GPU, driver or kernel is refused when the GPU
/// opens, and from another CPU or compiler when it loads.
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "needs an NVIDIA GPU; enable the gpu-tests feature"
)]
fn profiles_measured_elsewhere_are_refused() {
    let _serial = serial();
    let args = ["16777216", "32", "3"];
    let other_gpu = edited(calibrated(), "policy-gpu.txt", "device", "Other GPU");
    let lines = run_demo(Some(&other_gpu), None, &args);
    assert!(
        lines.iter().all(|line| line.contains("-> cpu")),
        "{lines:#?}"
    );
    assert!(
        lines
            .last()
            .unwrap()
            .contains("profile is for device `Other GPU`"),
        "{lines:#?}"
    );

    let other_driver = edited(calibrated(), "policy-driver.txt", "driver", "1");
    let lines = run_demo(Some(&other_driver), None, &args);
    assert!(
        lines.last().unwrap().contains("profile is for driver `1`"),
        "{lines:#?}"
    );

    let other_cpu = edited(calibrated(), "policy-cpu.txt", "cpu", "Other CPU");
    all_contain(
        &run_demo(Some(&other_cpu), None, &args),
        "-> cpu (profile is for cpu `Other CPU`",
    );
}
