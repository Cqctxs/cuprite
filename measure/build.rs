//! Compiles `kernels/grid.rs` to PTX with rustc's nvptx backend. The result
//! is embedded as `GRID_PTX`, and the driver compiles it for the GPU when it
//! is loaded.

use std::{env, path::PathBuf, process::Command};

fn main() {
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("grid.ptx");
    let status = Command::new(rustc)
        .args([
            "--edition",
            "2021",
            "--crate-type",
            "cdylib",
            "--target",
            "nvptx64-nvidia-cuda",
            "-C",
            "opt-level=3",
            "--emit",
            "asm",
            "-o",
        ])
        .arg(&out)
        .arg("kernels/grid.rs")
        .status()
        .expect("rustc runs");
    assert!(
        status.success(),
        "compiling kernels/grid.rs for nvptx64-nvidia-cuda failed"
    );
    println!("cargo:rerun-if-changed=kernels");
}
