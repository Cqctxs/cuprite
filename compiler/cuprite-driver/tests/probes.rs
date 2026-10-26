//! Builds the probe crate through the driver, as Cargo's workspace wrapper,
//! and checks the MIR it records. Cargo puts the toolchain's library
//! directory on `PATH` for tests, so the nested driver finds rustc's DLL.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

struct Build {
    mir: PathBuf,
}

impl Build {
    /// Runs `cargo <args> -p cuprite-probes` through the driver in its own
    /// target directory, so it never shares artifacts with plain rustc.
    fn run(name: &str, args: &[&str]) -> Self {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let scratch = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
        let mir = scratch.join("mir");
        let target = scratch.join("target");
        if mir.exists() {
            fs::remove_dir_all(&mir).unwrap();
        }

        // Dumps are written only while compiling, so an up-to-date build
        // would write nothing. Dependencies stay built.
        let profile = if args.contains(&"--release") {
            "release"
        } else {
            "dev"
        };
        let clean = Command::new(env!("CARGO"))
            .current_dir(workspace)
            .args([
                "clean",
                "-p",
                "cuprite-probes",
                "--profile",
                profile,
                "--target-dir",
            ])
            .arg(&target)
            .status()
            .unwrap();
        assert!(clean.success());

        let output = Command::new(env!("CARGO"))
            .current_dir(workspace)
            .args(args)
            .args(["-p", "cuprite-probes", "--target-dir"])
            .arg(&target)
            .env(
                "RUSTC_WORKSPACE_WRAPPER",
                env!("CARGO_BIN_EXE_cuprite-driver"),
            )
            .env("CUPRITE_DUMP_MIR", &mir)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "cargo {args:?} through the driver failed:\n{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        Self {
            mir: mir.join("cuprite_probes"),
        }
    }

    fn mir(&self, function: &str, stage: &str) -> String {
        let path = self.mir.join(format!("{function}.{stage}.mir"));
        fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
    }
}

const OVERFLOW: &str = "which would overflow";
const BOUNDS: &str = "index out of bounds";

#[test]
fn debug_build_passes_the_probe_tests_and_records_mir() {
    let build = Build::run("debug", &["test"]);

    // Built MIR keeps library calls as calls, so they can be recognized.
    let transform = build.mir("transform", "built");
    assert!(transform.contains("<std::slice::IterMut<'_, u32> as std::iter::Iterator>::next"));
    assert!(transform.contains("core::num::<impl u32>::wrapping_mul"));
    assert!(!transform.contains("assert("));

    assert!(build.mir("transform_indexed", "built").contains(BOUNDS));
    assert!(build.mir("checked_increment", "built").contains(OVERFLOW));

    // Closures are separate bodies; the caller passes them to Rayon.
    assert!(build
        .mir("transform_rayon", "built")
        .contains("rayon::iter::ParallelIterator>::for_each"));
    assert!(build
        .mir("transform_rayon...closure#0.", "built")
        .contains("wrapping_add"));
}

#[test]
fn release_build_changes_what_mir_contains() {
    let build = Build::run("release", &["build", "--release"]);

    // No overflow checks in release, so `+=` cannot panic there.
    assert!(!build.mir("checked_increment", "built").contains(OVERFLOW));

    // Optimized MIR inlines the iterator into raw pointer operations, so
    // library calls are no longer visible as calls.
    let optimized = build.mir("transform", "optimized");
    assert!(!optimized.contains("Iterator>::next("));
    assert!(optimized.contains("Offset("));

    // MIR optimizations do not remove bounds checks; LLVM does later.
    assert!(build.mir("transform_indexed", "optimized").contains(BOUNDS));
}
