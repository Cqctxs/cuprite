//! Records the sysroot of the toolchain building the driver. The driver
//! links that toolchain's rustc library, so `cargo cuprite` puts its
//! directory on the loader's search path and runs that toolchain's Cargo.

use std::{env, process::Command};

fn main() {
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let output = Command::new(rustc)
        .args(["--print", "sysroot"])
        .output()
        .expect("rustc runs");
    assert!(output.status.success(), "rustc --print sysroot failed");
    let sysroot = String::from_utf8(output.stdout).expect("the sysroot is UTF-8");
    println!("cargo:rustc-env=CUPRITE_SYSROOT={}", sysroot.trim());
    println!("cargo:rerun-if-changed=build.rs");
}
