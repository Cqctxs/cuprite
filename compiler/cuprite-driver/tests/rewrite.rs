//! M1 questions 4 and 5: the driver rewrites a named function to call
//! `cuprite_runtime::map_u32` before its original body, and links the
//! runtime into a crate that never declares it. When the runtime declines,
//! the original CPU body still runs.

mod support;

use support::{run, text, Probes};

const REPEAT: &str = "cuprite_probes::repeat=3,7,arg2";

fn demo(name: &'static str) -> Probes {
    let probes = Probes::new(name, true)
        .with_runtime()
        .env("CUPRITE_REPLACE", REPEAT);
    probes.build(&["build", "--bin", "repeat_demo"]);
    probes
}

#[test]
fn only_the_named_function_is_rewritten_and_the_cpu_body_still_runs() {
    let probes = demo("cpu");

    let trace = run(&probes.exe("repeat_demo"), false, &["1000", "4", "2"]);
    assert_eq!(trace.matches("-> cpu (CUPRITE_GPU is not 1)").count(), 3);

    assert!(probes
        .mir("repeat", "optimized")
        .contains("cuprite_runtime::map_u32(copy _1, const 3_u32, const 7_u32, copy _2)"));
    for untouched in ["transform", "add_into", "prefix_sum"] {
        assert!(
            !probes
                .mir(untouched, "optimized")
                .contains("cuprite_runtime"),
            "{untouched}"
        );
    }
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "needs an NVIDIA GPU; enable the gpu-tests feature"
)]
fn the_rewritten_function_runs_on_the_gpu() {
    let exe = demo("gpu").exe("repeat_demo");
    for (len, passes) in [("1", "1"), ("257", "3"), ("100000", "8")] {
        let trace = run(&exe, true, &[len, passes, "2"]);
        assert_eq!(trace.matches("-> gpu").count(), 3, "{trace}");
    }
    // Zero passes is no work, so the runtime declines and the CPU body
    // (an empty loop) runs.
    let trace = run(&exe, true, &["10", "0", "1"]);
    assert_eq!(trace.matches("-> cpu (no work)").count(), 2, "{trace}");
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "needs an NVIDIA GPU; enable the gpu-tests feature"
)]
fn the_probe_tests_pass_with_repeat_on_the_gpu() {
    let output = Probes::new("probe-tests", true)
        .with_runtime()
        .env("CUPRITE_REPLACE", REPEAT)
        .env("CUPRITE_GPU", "1")
        .env("CUPRITE_TRACE", "1")
        // Let the runtime's trace through libtest's output capture.
        .build(&["test", "--lib", "--", "--nocapture"]);
    assert!(text(&output.stderr).contains("-> gpu"));
}

#[test]
fn a_function_with_the_wrong_signature_is_refused() {
    let output = Probes::new("wrong-signature", true)
        .with_runtime()
        .env("CUPRITE_REPLACE", "cuprite_probes::wrapping_sum=3,7,1")
        .cargo(&["build", "--lib"]);
    assert!(!output.status.success());
    assert!(text(&output.stderr).contains("the first parameter must be `&mut [u32]`"));
}

#[test]
fn rewriting_without_the_runtime_is_refused() {
    let output = Probes::new("no-runtime", true)
        .env("CUPRITE_REPLACE", REPEAT)
        .cargo(&["build", "--lib"]);
    assert!(!output.status.success());
    assert!(text(&output.stderr).contains("`cuprite_runtime::map_u32` is not loaded"));
}
