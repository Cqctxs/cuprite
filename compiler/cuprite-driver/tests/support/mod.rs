//! Shared helpers: build the runtime, build the probe crate through the
//! driver, and run its binaries. Each test file uses only some of them.
#![allow(dead_code)]

use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::OnceLock,
};

/// Every variable the driver or runtime reads, cleared before each build or
/// run so a developer's shell cannot change a test.
const CUPRITE_VARS: [&str; 9] = [
    "CUPRITE_DUMP_MIR",
    "CUPRITE_REPORT",
    "CUPRITE_AUTO",
    "CUPRITE_REPLACE",
    "CUPRITE_RUNTIME",
    "CUPRITE_RUNTIME_DEPS",
    "CUPRITE_GPU",
    "CUPRITE_TRACE",
    "RUSTFLAGS",
];

pub fn workspace() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap()
}

pub fn scratch(name: &str) -> PathBuf {
    Path::new(env!("CARGO_TARGET_TMPDIR")).join(name)
}

pub fn cargo() -> Command {
    let mut command = Command::new(env!("CARGO"));
    command.current_dir(workspace());
    for name in CUPRITE_VARS {
        command.env_remove(name);
    }
    command
}

pub fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// The runtime's `.rlib`/`.rmeta` files and the directories holding its
/// dependencies, built once per test process.
pub struct Runtime {
    pub files: OsString,
    pub dirs: OsString,
}

pub fn runtime() -> &'static Runtime {
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

/// One configuration of the probe crate, built through the driver in its
/// own target directory under `scratch(name)`.
pub struct Probes {
    pub name: &'static str,
    pub release: bool,
    pub link_runtime: bool,
    pub env: Vec<(&'static str, String)>,
}

impl Probes {
    pub fn new(name: &'static str, release: bool) -> Self {
        Self {
            name,
            release,
            link_runtime: false,
            env: Vec::new(),
        }
    }

    pub fn with_runtime(mut self) -> Self {
        self.link_runtime = true;
        self
    }

    pub fn env(mut self, name: &'static str, value: &str) -> Self {
        self.env.push((name, value.to_owned()));
        self
    }

    fn target(&self) -> PathBuf {
        scratch(self.name).join("target")
    }

    pub fn mir_dir(&self) -> PathBuf {
        scratch(self.name).join("mir").join("cuprite_probes")
    }

    pub fn report(&self) -> String {
        let path = scratch(self.name).join("report").join("cuprite_probes.txt");
        std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
    }

    pub fn mir(&self, function: &str, stage: &str) -> String {
        let path = self.mir_dir().join(format!("{function}.{stage}.mir"));
        std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
    }

    /// Arguments after `--` stay last, for the test binary.
    fn command(&self, args: &[&str]) -> Command {
        let split = args
            .iter()
            .position(|&arg| arg == "--")
            .unwrap_or(args.len());
        let (cargo_args, test_args) = args.split_at(split);
        let mut command = cargo();
        command.args(cargo_args);
        if self.release {
            command.arg("--release");
        }
        command
            .args(["-p", "cuprite-probes", "--target-dir"])
            .arg(self.target())
            .args(test_args);
        command
    }

    /// `cargo <args> -p cuprite-probes` through the driver, with the MIR
    /// dump and report enabled. The probe crate is cleaned first: dumps and
    /// reports are written only while compiling.
    pub fn cargo(&self, args: &[&str]) -> Output {
        let clean = self.command(&["clean"]).status().unwrap();
        assert!(clean.success());
        let output_dir = scratch(self.name);
        for dir in ["mir", "report"] {
            let _ = std::fs::remove_dir_all(output_dir.join(dir));
        }

        let mut command = self.command(args);
        command
            .env(
                "RUSTC_WORKSPACE_WRAPPER",
                env!("CARGO_BIN_EXE_cuprite-driver"),
            )
            .env("CUPRITE_DUMP_MIR", output_dir.join("mir"))
            .env("CUPRITE_REPORT", output_dir.join("report"));
        if self.link_runtime {
            let runtime = runtime();
            command
                .env("CUPRITE_RUNTIME", &runtime.files)
                .env("CUPRITE_RUNTIME_DEPS", &runtime.dirs);
        }
        for (name, value) in &self.env {
            command.env(name, value);
        }
        command.output().unwrap()
    }

    /// Like `cargo`, but the command must succeed.
    pub fn build(&self, args: &[&str]) -> Output {
        let output = self.cargo(args);
        assert!(
            output.status.success(),
            "cargo {args:?} through the driver failed:\n{}",
            text(&output.stderr)
        );
        output
    }

    pub fn exe(&self, bin: &str) -> PathBuf {
        let profile = if self.release { "release" } else { "debug" };
        self.target()
            .join(profile)
            .join(format!("{bin}{}", std::env::consts::EXE_SUFFIX))
    }
}

/// Runs a probe binary, which must succeed; returns its stderr, where the
/// runtime's trace goes.
pub fn run(exe: &Path, gpu: bool, args: &[&str]) -> String {
    let mut command = Command::new(exe);
    for name in CUPRITE_VARS {
        command.env_remove(name);
    }
    command.args(args).env("CUPRITE_TRACE", "1");
    if gpu {
        command.env("CUPRITE_GPU", "1");
    }
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{} failed:\n{}{}",
        exe.display(),
        text(&output.stdout),
        text(&output.stderr)
    );
    text(&output.stderr)
}
