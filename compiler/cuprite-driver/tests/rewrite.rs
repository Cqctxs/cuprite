//! M1 questions 4 and 5: the driver rewrites a named function to call
//! `cuprite_runtime::map_u32` before its original body, and links the
//! runtime into a crate that never declares it. When the runtime declines,
//! the original CPU body still runs.

use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::OnceLock,
};

const REPEAT: &str = "cuprite_probes::repeat=3,7,arg2";

fn workspace() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap()
}

fn scratch(name: &str) -> PathBuf {
    Path::new(env!("CARGO_TARGET_TMPDIR")).join(name)
}

fn cargo() -> Command {
    let mut command = Command::new(env!("CARGO"));
    command.current_dir(workspace());
    command
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// The runtime's `.rlib`/`.rmeta` files and the directories holding its
/// dependencies, built once per test process.
struct Runtime {
    files: OsString,
    dirs: OsString,
}

fn runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        let output = cargo()
            .args([
                "build",
                "--release",
                "-p",
                "cuprite-runtime",
                "--message-format=json",
            ])
            .arg("--target-dir")
            .arg(scratch("runtime"))
            .output()
            .unwrap();
        assert!(output.status.success(), "{}", text(&output.stderr));

        let mut files = Vec::new();
        let mut dirs = Vec::new();
        for line in text(&output.stdout).lines() {
            let message: serde_json::Value = serde_json::from_str(line).unwrap();
            if message["reason"] != "compiler-artifact" {
                continue;
            }
            let runtime = message["target"]["name"] == "cuprite_runtime";
            for file in message["filenames"].as_array().unwrap() {
                let file = PathBuf::from(file.as_str().unwrap());
                let extension = file.extension().and_then(|e| e.to_str());
                if !matches!(extension, Some("rlib" | "rmeta")) {
                    continue;
                }
                dirs.push(file.parent().unwrap().to_owned());
                if runtime {
                    files.push(file);
                }
            }
        }
        assert!(!files.is_empty(), "no runtime artifacts reported");
        Runtime {
            files: std::env::join_paths(files).unwrap(),
            dirs: std::env::join_paths(dirs).unwrap(),
        }
    })
}

/// `cargo <args> -p cuprite-probes --release` through the driver, with
/// `spec` as the replacement list, in its own target directory.
fn through_driver(name: &str, spec: &str, link_runtime: bool, args: &[&str]) -> Output {
    let mut command = cargo();
    command
        .args(args)
        .args(["--release", "-p", "cuprite-probes", "--target-dir"])
        .arg(scratch(name).join("target"))
        .env(
            "RUSTC_WORKSPACE_WRAPPER",
            env!("CARGO_BIN_EXE_cuprite-driver"),
        )
        .env("CUPRITE_REPLACE", spec)
        .env("CUPRITE_DUMP_MIR", scratch(name).join("mir"))
        .env_remove("CUPRITE_RUNTIME")
        .env_remove("CUPRITE_RUNTIME_DEPS");
    if link_runtime {
        let runtime = runtime();
        command
            .env("CUPRITE_RUNTIME", &runtime.files)
            .env("CUPRITE_RUNTIME_DEPS", &runtime.dirs);
    }
    command.output().unwrap()
}

fn build_demo(name: &str) -> PathBuf {
    let output = through_driver(name, REPEAT, true, &["build", "--bin", "repeat_demo"]);
    assert!(output.status.success(), "{}", text(&output.stderr));
    scratch(name)
        .join("target")
        .join("release")
        .join(format!("repeat_demo{}", std::env::consts::EXE_SUFFIX))
}

/// Runs the demo, which panics if `repeat` returns a wrong result.
fn run_demo(exe: &Path, gpu: bool, args: &[&str]) -> String {
    let mut command = Command::new(exe);
    command.args(args).env("CUPRITE_TRACE", "1");
    if gpu {
        command.env("CUPRITE_GPU", "1");
    } else {
        command.env_remove("CUPRITE_GPU");
    }
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "demo failed:\n{}{}",
        text(&output.stdout),
        text(&output.stderr)
    );
    text(&output.stderr)
}

#[test]
fn only_the_named_function_is_rewritten_and_the_cpu_body_still_runs() {
    let exe = build_demo("cpu");

    let trace = run_demo(&exe, false, &["1000", "4", "2"]);
    assert_eq!(trace.matches("-> cpu (CUPRITE_GPU is not 1)").count(), 3);

    let mir = |function: &str| {
        let path = scratch("cpu")
            .join("mir")
            .join("cuprite_probes")
            .join(format!("{function}.optimized.mir"));
        std::fs::read_to_string(path).unwrap()
    };
    assert!(mir("repeat")
        .contains("cuprite_runtime::map_u32(copy _1, const 3_u32, const 7_u32, copy _2)"));
    for untouched in ["transform", "add_into", "prefix_sum"] {
        assert!(!mir(untouched).contains("cuprite_runtime"), "{untouched}");
    }
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "needs an NVIDIA GPU; enable the gpu-tests feature"
)]
fn the_rewritten_function_runs_on_the_gpu() {
    let exe = build_demo("gpu");
    for (len, passes) in [("1", "1"), ("257", "3"), ("100000", "8")] {
        let trace = run_demo(&exe, true, &[len, passes, "2"]);
        assert_eq!(trace.matches("-> gpu").count(), 3, "{trace}");
    }
    // Zero passes is no work, so the runtime declines and the CPU body
    // (an empty loop) runs.
    let trace = run_demo(&exe, true, &["10", "0", "1"]);
    assert_eq!(trace.matches("-> cpu (no work)").count(), 2, "{trace}");
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "needs an NVIDIA GPU; enable the gpu-tests feature"
)]
fn the_probe_tests_pass_with_repeat_on_the_gpu() {
    let mut command = cargo();
    let runtime = runtime();
    let output = command
        .args([
            "test",
            "--release",
            "-p",
            "cuprite-probes",
            "--lib",
            "--target-dir",
        ])
        .arg(scratch("probe-tests").join("target"))
        .env(
            "RUSTC_WORKSPACE_WRAPPER",
            env!("CARGO_BIN_EXE_cuprite-driver"),
        )
        .env("CUPRITE_REPLACE", REPEAT)
        .env("CUPRITE_RUNTIME", &runtime.files)
        .env("CUPRITE_RUNTIME_DEPS", &runtime.dirs)
        .env("CUPRITE_GPU", "1")
        .env("CUPRITE_TRACE", "1")
        // Let the runtime's trace through libtest's output capture.
        .args(["--", "--nocapture"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", text(&output.stderr));
    assert!(text(&output.stderr).contains("-> gpu"));
}

#[test]
fn a_function_with_the_wrong_signature_is_refused() {
    let output = through_driver(
        "wrong-signature",
        "cuprite_probes::wrapping_sum=3,7,1",
        true,
        &["build", "--lib"],
    );
    assert!(!output.status.success());
    assert!(text(&output.stderr).contains("the first parameter must be `&mut [u32]`"));
}

#[test]
fn rewriting_without_the_runtime_is_refused() {
    let output = through_driver("no-runtime", REPEAT, false, &["build", "--lib"]);
    assert!(!output.status.success());
    assert!(text(&output.stderr).contains("`cuprite_runtime::map_u32` is not loaded"));
}
