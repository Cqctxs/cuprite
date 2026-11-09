//! Has a map the driver would rewrite, but build scripts are compiled as
//! plain Rust: they run while building, not in the program. It also links
//! the library, which was rewritten and so depends on the runtime.

fn bump(values: &mut [u32]) {
    for value in values.iter_mut() {
        *value = value.wrapping_mul(3).wrapping_add(1);
    }
}

fn main() {
    let mut values = [1, 2];
    bump(&mut values);
    fixture_lib::step(&mut values);
    println!("cargo:rustc-env=FIXTURE_BUILT={}", values[1]);
    println!("cargo:rerun-if-changed=build.rs");
}
