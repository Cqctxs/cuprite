//! `cargo cuprite <cargo command> [args]` runs a Cargo command with every
//! crate compiled by the Cuprite driver. The package and the libraries it
//! uses are analyzed, their candidates are rewritten (`CUPRITE_AUTO=1`), and
//! the result is linked with the Cuprite runtime.
//!
//! `cargo cuprite calibrate <profile path>` measures this machine for the
//! runtime's policy (`CUPRITE_PROFILE`).
//!
//! Builds go to `<target dir>/cuprite`, apart from ordinary builds: Cargo
//! might otherwise reuse artifacts that plain rustc built. The runtime is
//! built once, by the same driver, in Cuprite's own target directory.

use std::{
    env,
    ffi::OsString,
    path::{Path, PathBuf},
    process::{Command, ExitCode, ExitStatus, Stdio},
};

use serde_json::Value;

/// The toolchain that built this program, whose rustc library the driver
/// links (`build.rs`).
const SYSROOT: &str = env!("CUPRITE_SYSROOT");

/// Where the loader looks for the driver's rustc library, which is in the
/// sysroot's `bin` on Windows and its `lib` elsewhere.
const LIBRARY_PATH_VAR: &str = if cfg!(windows) {
    "PATH"
} else {
    "LD_LIBRARY_PATH"
};

/// Variables the driver reads, cleared for the runtime's own build.
const DRIVER_VARS: [&str; 7] = [
    "CUPRITE_DUMP_MIR",
    "CUPRITE_CENSUS",
    "CUPRITE_REPORT",
    "CUPRITE_AUTO",
    "CUPRITE_REPLACE",
    "CUPRITE_RUNTIME",
    "CUPRITE_RUNTIME_DEPS",
];

const USAGE: &str = "\
usage: cargo cuprite <cargo command> [args...]
       cargo cuprite calibrate <profile path>

Runs the Cargo command with every crate compiled by the Cuprite driver, and
builds into <target dir>/cuprite.";

fn main() -> ExitCode {
    let mut args: Vec<OsString> = env::args_os().skip(1).collect();
    // Cargo runs `cargo-cuprite cuprite <args>` for `cargo cuprite <args>`.
    if args.first().is_some_and(|arg| arg == "cuprite") {
        args.remove(0);
    }
    match run(&args) {
        Ok(code) => code,
        Err(message) => {
            eprintln!("cargo-cuprite: {message}");
            ExitCode::from(2)
        }
    }
}

fn run(args: &[OsString]) -> Result<ExitCode, String> {
    let Some(first) = args.first() else {
        eprintln!("{USAGE}");
        return Ok(ExitCode::from(2));
    };
    if first == "--help" || first == "-h" {
        println!("{USAGE}");
        return Ok(ExitCode::SUCCESS);
    }
    for name in ["RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER"] {
        if env::var_os(name).is_some_and(|value| !value.is_empty()) {
            return Err(format!(
                "{name} is set, and would wrap the Cuprite driver; unset it"
            ));
        }
    }

    let toolchain = Toolchain::find()?;
    let runtime = toolchain.build_runtime()?;
    let mut command = if first == "calibrate" {
        let mut command = Command::new(&runtime.calibrate);
        command.args(&args[1..]);
        command
    } else {
        toolchain.package_command(args, &runtime)?
    };
    let status = command
        .status()
        .map_err(|error| format!("cannot run {:?}: {error}", command.get_program()))?;
    Ok(exit_code(status))
}

fn exit_code(status: ExitStatus) -> ExitCode {
    match status.code() {
        Some(code) => ExitCode::from(u8::try_from(code).unwrap_or(1)),
        None => ExitCode::FAILURE,
    }
}

fn executable(name: &str) -> String {
    format!("{name}{}", env::consts::EXE_SUFFIX)
}

fn join(paths: &[PathBuf]) -> Result<OsString, String> {
    env::join_paths(paths).map_err(|error| error.to_string())
}

/// The runtime's artifacts, for linking into rewritten crates.
struct Runtime {
    /// Its `.rlib` and `.rmeta` files.
    files: Vec<PathBuf>,
    /// The directories holding its dependencies.
    dirs: Vec<PathBuf>,
    /// The `cuprite-calibrate` executable.
    calibrate: PathBuf,
}

struct Toolchain {
    sysroot: PathBuf,
    driver: PathBuf,
    /// The loader's search path, with the driver's library directory first.
    library_path: OsString,
}

impl Toolchain {
    fn find() -> Result<Self, String> {
        let sysroot = PathBuf::from(SYSROOT);
        let exe = env::current_exe().map_err(|error| format!("cannot locate itself: {error}"))?;
        let toolchain = Self {
            driver: exe.with_file_name(executable("cuprite-driver")),
            library_path: env::join_paths(
                [sysroot.join(if cfg!(windows) { "bin" } else { "lib" })]
                    .into_iter()
                    .chain(env::split_paths(
                        &env::var_os(LIBRARY_PATH_VAR).unwrap_or_default(),
                    )),
            )
            .map_err(|error| error.to_string())?,
            sysroot,
        };
        for path in [&toolchain.driver, &toolchain.tool("cargo")] {
            if !path.is_file() {
                return Err(format!("{} is missing", path.display()));
            }
        }
        Ok(toolchain)
    }

    fn tool(&self, name: &str) -> PathBuf {
        self.sysroot.join("bin").join(executable(name))
    }

    /// The toolchain's own Cargo, compiling with the driver.
    fn cargo(&self) -> Command {
        let mut command = Command::new(self.tool("cargo"));
        command
            .env(LIBRARY_PATH_VAR, &self.library_path)
            .env("RUSTC", &self.driver)
            .env("RUSTDOC", self.tool("rustdoc"))
            // Empty values also override wrappers set in Cargo's config.
            .env("RUSTC_WRAPPER", "")
            .env("RUSTC_WORKSPACE_WRAPPER", "");
        command
    }

    /// Builds the runtime with the driver, so it comes from the same
    /// compiler as the crates it is linked into. It is built from Cuprite's
    /// workspace, without the package's Cargo configuration.
    fn build_runtime(&self) -> Result<Runtime, String> {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the driver is in Cuprite's compiler workspace");
        let mut command = self.cargo();
        command
            .current_dir(workspace)
            .args([
                "build",
                "--release",
                "--locked",
                "--message-format=json-render-diagnostics",
                "-p",
                "cuprite-runtime",
                "--target-dir",
            ])
            .arg(workspace.join("target").join("cuprite-runtime"))
            .stderr(Stdio::inherit());
        for name in DRIVER_VARS {
            command.env_remove(name);
        }
        let output = command
            .output()
            .map_err(|error| format!("cannot run Cargo: {error}"))?;
        if !output.status.success() {
            return Err("building the Cuprite runtime failed".into());
        }

        let mut files = Vec::new();
        let mut dirs = Vec::new();
        let mut calibrate = None;
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            let Ok(message) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            if message["reason"] != "compiler-artifact" {
                continue;
            }
            let target = &message["target"]["name"];
            if target == "cuprite-calibrate" {
                calibrate = message["executable"].as_str().map(PathBuf::from);
            }
            let filenames = message["filenames"].as_array().into_iter().flatten();
            for file in filenames.filter_map(Value::as_str).map(PathBuf::from) {
                let extension = file.extension().and_then(|e| e.to_str());
                if !matches!(extension, Some("rlib" | "rmeta")) {
                    continue;
                }
                let dir = file.parent().expect("artifacts are in a directory");
                if !dirs.iter().any(|known| known == dir) {
                    dirs.push(dir.to_owned());
                }
                if target == "cuprite_runtime" {
                    files.push(file);
                }
            }
        }
        match (files.is_empty(), calibrate) {
            (false, Some(calibrate)) => Ok(Runtime {
                files,
                dirs,
                calibrate,
            }),
            _ => Err("Cargo did not report the runtime's artifacts".into()),
        }
    }

    /// The package's Cargo command, compiling with the driver and linking
    /// the runtime.
    fn package_command(&self, args: &[OsString], runtime: &Runtime) -> Result<Command, String> {
        let target = self.target_dir(args)?.join("cuprite");
        let mut command = self.cargo();
        command
            .args(args)
            .env("CARGO_TARGET_DIR", target)
            .env("CUPRITE_RUNTIME", join(&runtime.files)?)
            .env("CUPRITE_RUNTIME_DEPS", join(&runtime.dirs)?)
            .env("CARGO_ENCODED_RUSTDOCFLAGS", rustdoc_flags(&runtime.dirs)?);
        if env::var_os("CUPRITE_AUTO").is_none() {
            command.env("CUPRITE_AUTO", "1");
        }
        Ok(command)
    }

    /// The package's ordinary target directory.
    fn target_dir(&self, args: &[OsString]) -> Result<PathBuf, String> {
        let mut command = self.cargo();
        command
            .args(["metadata", "--format-version", "1", "--no-deps"])
            .stderr(Stdio::inherit());
        if let Some(manifest) = manifest_path(args) {
            command.arg("--manifest-path").arg(manifest);
        }
        let output = command
            .output()
            .map_err(|error| format!("cannot run Cargo: {error}"))?;
        if !output.status.success() {
            return Err("cannot read the package's metadata".into());
        }
        let metadata: Value = serde_json::from_slice(&output.stdout)
            .map_err(|error| format!("cannot parse the package's metadata: {error}"))?;
        metadata["target_directory"]
            .as_str()
            .map(PathBuf::from)
            .ok_or_else(|| "the package's metadata has no target directory".into())
    }
}

/// The `--manifest-path` among Cargo's own arguments, which end at `--`.
fn manifest_path(args: &[OsString]) -> Option<OsString> {
    let mut args = args.iter().take_while(|arg| *arg != "--");
    while let Some(arg) = args.next() {
        if arg == "--manifest-path" {
            return args.next().cloned();
        }
        if let Some(path) = arg
            .to_str()
            .and_then(|arg| arg.strip_prefix("--manifest-path="))
        {
            return Some(path.into());
        }
    }
    None
}

/// Doctests link the library they test, so rustdoc needs the runtime's
/// directories too. Cargo's encoded form, separated by `\x1f`, keeps paths
/// with spaces whole. Flags the user already set come first.
fn rustdoc_flags(dirs: &[PathBuf]) -> Result<String, String> {
    let mut flags: Vec<String> = match env::var("CARGO_ENCODED_RUSTDOCFLAGS") {
        Ok(flags) if !flags.is_empty() => flags.split('\x1f').map(String::from).collect(),
        _ => env::var("RUSTDOCFLAGS")
            .unwrap_or_default()
            .split_whitespace()
            .map(String::from)
            .collect(),
    };
    for dir in dirs {
        let dir = dir
            .to_str()
            .ok_or_else(|| format!("runtime path {} is not UTF-8", dir.display()))?;
        flags.push(format!("-Ldependency={dir}"));
    }
    Ok(flags.join("\x1f"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(args: &[&str]) -> Vec<OsString> {
        args.iter().map(OsString::from).collect()
    }

    #[test]
    fn the_manifest_path_is_found_in_either_form_before_the_separator() {
        let path = |list: &[&str]| manifest_path(&args(list));
        assert_eq!(
            path(&["build", "--manifest-path", "a b/Cargo.toml"]),
            Some("a b/Cargo.toml".into())
        );
        assert_eq!(
            path(&["run", "--manifest-path=x/Cargo.toml", "--release"]),
            Some("x/Cargo.toml".into())
        );
        assert_eq!(path(&["run", "--", "--manifest-path", "y"]), None);
        assert_eq!(path(&["build"]), None);
    }
}
