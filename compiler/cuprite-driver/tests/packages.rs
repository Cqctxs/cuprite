//! `cargo cuprite` builds an application, a library outside its workspace,
//! and their crates.io dependencies through the driver (M5). The fixtures
//! in `tests/fixtures` are copied first into a path with spaces and
//! non-ASCII characters.

mod support;

use std::{
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::OnceLock,
};

use support::{scratch, text, CUPRITE_VARS, NO_PROFILE};

const LEN: u32 = 1 << 20;
const PASSES: u32 = 40;

/// Other variables that would change how Cargo builds the fixtures.
const CARGO_VARS: [&str; 5] = [
    "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
    "CARGO_TARGET_DIR",
    "RUSTDOCFLAGS",
    "CARGO_ENCODED_RUSTDOCFLAGS",
];

fn copy_dir(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            if entry.file_name() != "target" {
                copy_dir(&entry.path(), &target);
            }
        } else {
            fs::copy(entry.path(), target).unwrap();
        }
    }
}

/// A fresh copy of the fixtures, made once per test process.
fn fixtures() -> &'static Path {
    static FIXTURES: OnceLock<PathBuf> = OnceLock::new();
    FIXTURES.get_or_init(|| {
        let root = scratch("m5 packages é");
        let _ = fs::remove_dir_all(&root);
        copy_dir(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures"),
            &root,
        );
        root
    })
}

fn manifest(package: &str) -> PathBuf {
    fixtures().join(package).join("Cargo.toml")
}

/// `cargo cuprite <args>`, run from the fixtures' root, as Cargo runs it.
fn cargo_cuprite(args: &[&OsStr], env: &[(&str, &OsStr)]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_cargo-cuprite"));
    command.current_dir(fixtures()).arg("cuprite").args(args);
    for name in CUPRITE_VARS.iter().chain(&CARGO_VARS) {
        command.env_remove(name);
    }
    command.envs(env.iter().copied());
    command.output().unwrap()
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "cargo cuprite failed:\n{}{}",
        text(&output.stdout),
        text(&output.stderr)
    );
}

struct App {
    report: PathBuf,
    exe: PathBuf,
}

/// The fixture application, built once with `cargo cuprite build --release`.
fn app() -> &'static App {
    static APP: OnceLock<App> = OnceLock::new();
    APP.get_or_init(|| {
        let report = fixtures().join("report");
        let manifest = manifest("app");
        let output = cargo_cuprite(
            &[
                "build".as_ref(),
                "--release".as_ref(),
                "--offline".as_ref(),
                "--manifest-path".as_ref(),
                manifest.as_os_str(),
            ],
            &[("CUPRITE_REPORT", report.as_os_str())],
        );
        assert_success(&output);
        App {
            report,
            exe: fixtures().join("app/target/cuprite/release").join(format!(
                "cuprite-fixture-app{}",
                std::env::consts::EXE_SUFFIX
            )),
        }
    })
}

/// Runs the application with tracing on; returns its stdout and stderr.
fn run_app(env: &[(&str, &OsStr)]) -> (String, String) {
    let mut command = Command::new(&app().exe);
    for name in CUPRITE_VARS {
        command.env_remove(name);
    }
    command
        .args([LEN.to_string(), PASSES.to_string()])
        .env("CUPRITE_TRACE", "1")
        .envs(env.iter().copied());
    let output = command.output().unwrap();
    assert!(output.status.success(), "{}", text(&output.stderr));
    (text(&output.stdout), text(&output.stderr))
}

/// What the application prints, computed independently: the library's
/// `x * 5 + 1` for `PASSES` passes, then the application's `x * 7 + 42`,
/// summed; then 7 from the build script.
fn expected() -> String {
    let checksum: u64 = (0..LEN)
        .map(|mut x| {
            for _ in 0..PASSES {
                x = x.wrapping_mul(5).wrapping_add(1);
            }
            u64::from(x.wrapping_mul(7).wrapping_add(42))
        })
        .sum();
    format!("{checksum} 7")
}

/// The calls the application makes: the library's region, then its own map.
fn calls(side: &str) -> [String; 2] {
    [
        format!("cuprite: map_u32 len={LEN} passes={PASSES} -> {side}"),
        format!("cuprite: map_u32 len={LEN} passes=1 -> {side}"),
    ]
}

#[test]
fn a_package_and_its_libraries_are_rewritten() {
    let app = app();
    let report = |name: &str| fs::read_to_string(app.report.join(format!("{name}.txt")));
    let has = |name: &str, line: &str| {
        let report = report(name).unwrap();
        assert!(report.lines().any(|l| l == line), "{name}:\n{report}");
    };
    has("fixture_lib", "fixture_lib::step: candidate: x * 5 + 1");
    has(
        "fixture_lib",
        "fixture_lib::simulate: candidate: x * 5 + 1, repeated `passes` times",
    );
    has(
        "cuprite_fixture_app",
        "cuprite_fixture_app::mix: candidate: x * 7 + 42",
    );
    // crates.io dependencies pass through the driver too.
    for name in ["rayon", "rayon_core", "either", "crossbeam_utils"] {
        report(name).unwrap_or_else(|error| panic!("{name}: {error}"));
    }
    // Build scripts and procedural macros don't.
    for name in ["build_script_build", "fixture_macros"] {
        assert!(report(name).is_err(), "{name} was analyzed");
    }
    // Ordinary builds are left alone.
    assert!(!fixtures().join("app/target/release").exists());

    let (stdout, stderr) = run_app(&[]);
    assert_eq!(stdout.trim(), expected());
    let region = format!("cuprite: map_u32 len={LEN} passes={PASSES} {NO_PROFILE}");
    assert_eq!(stderr.lines().next(), Some(region.as_str()));
    // The region's CPU body calls the rewritten `step` once per pass.
    let [_, map] = calls("cpu");
    let maps = stderr.lines().filter(|line| line.starts_with(&map)).count();
    assert_eq!(maps, PASSES as usize + 1, "{stderr}");
}

#[cfg(feature = "gpu-tests")]
#[test]
fn the_package_and_its_library_run_on_the_gpu() {
    let (stdout, stderr) = run_app(&[("CUPRITE_GPU", "1".as_ref())]);
    assert_eq!(stdout.trim(), expected());
    assert_eq!(stderr.lines().collect::<Vec<_>>(), calls("gpu"));
}

#[cfg(feature = "gpu-tests")]
#[test]
fn a_calibrated_profile_lets_the_policy_decide() {
    let profile = fixtures().join("profile.txt");
    let output = cargo_cuprite(&["calibrate".as_ref(), profile.as_os_str()], &[]);
    assert_success(&output);
    let (stdout, stderr) = run_app(&[("CUPRITE_PROFILE", profile.as_os_str())]);
    assert_eq!(stdout.trim(), expected());
    let decisions: Vec<_> = stderr.lines().filter(|l| l.contains("passes=")).collect();
    assert!(!decisions.is_empty());
    assert!(
        decisions
            .iter()
            .all(|line| line.contains(" (predicted gpu ")),
        "{stderr}"
    );
}

#[test]
fn tests_and_doctests_go_through_the_layer() {
    let manifest = manifest("lib");
    let output = cargo_cuprite(
        &[
            "test".as_ref(),
            "--offline".as_ref(),
            "--manifest-path".as_ref(),
            manifest.as_os_str(),
        ],
        &[],
    );
    assert_success(&output);
    let stdout = text(&output.stdout);
    assert!(stdout.contains("test tests::simulate_repeats_step ... ok"));
    // The doctest links the rewritten library, so it needs the runtime.
    assert!(
        stdout
            .lines()
            .any(|line| line.contains(" - step (line ") && line.ends_with("... ok")),
        "{stdout}"
    );
}

#[test]
fn wrappers_that_would_replace_the_driver_are_refused() {
    let output = cargo_cuprite(
        &["build".as_ref()],
        &[("RUSTC_WRAPPER", "sccache".as_ref())],
    );
    assert_eq!(output.status.code(), Some(2));
    assert!(text(&output.stderr).contains("RUSTC_WRAPPER is set"));
}
