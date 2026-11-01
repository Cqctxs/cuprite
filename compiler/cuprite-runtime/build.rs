//! Records the compiler version, which a profile's CPU measurements depend
//! on (`policy::HostKey`).

use std::{env, process::Command};

fn main() {
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let version = Command::new(rustc)
        .arg("-V")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=CUPRITE_RUSTC_VERSION={version}");
    println!("cargo:rerun-if-changed=build.rs");
}
