//! M2: the recognizer finds the probe crate's maps on its own, rejects
//! everything else with a reason, and `CUPRITE_AUTO=1` rewrites exactly the
//! accepted functions, with results identical to the CPU's.

mod support;

use support::{run, text, Probes};

/// Verdicts that do not depend on the build profile.
const COMMON: &[(&str, &str)] = &[
    ("add_into", "rejected: takes 3 parameters; only `fn(&mut [u32])` is supported"),
    ("affine_chain", "candidate: x * 15 + 33"),
    ("apply", "rejected: takes 2 parameters; only `fn(&mut [u32])` is supported"),
    ("clamp", "rejected: uses `Gt`, which is not supported"),
    ("counted", "rejected: calls `std::sync::atomic::Atomic::<u32>::fetch_add`, which has no GPU summary"),
    ("counted_calls", "rejected: takes 0 parameters; only `fn(&mut [u32])` is supported"),
    ("divide", "rejected: takes 2 parameters; only `fn(&mut [u32])` is supported"),
    ("double", "candidate: x * 2 + 0"),
    ("fill", "candidate: x * 0 + 42"),
    ("minimum", "rejected: takes `&[u32]`, not `&mut [u32]`"),
    ("numbered", "rejected: uses a value computed outside the loop body"),
    ("outside_factor", "candidate: x * 3 + 0"),
    ("prefix_sum", "rejected: calls `core::slice::<impl [u32]>::len`, which has no GPU summary"),
    ("reciprocal", "rejected: may panic: division by zero"),
    ("repeat", "rejected: takes 2 parameters; only `fn(&mut [u32])` is supported"),
    ("reversed", "rejected: calls `<std::slice::IterMut<'_, u32> as std::iter::Iterator>::rev`, which has no GPU summary"),
    ("smooth", "rejected: takes 2 parameters; only `fn(&mut [u32])` is supported"),
    ("square", "rejected: is not a multiply-add map (multiplies the element by itself)"),
    ("transform", "candidate: x * 3 + 7"),
    ("transform_indexed", "rejected: calls `core::slice::<impl [u32]>::len`, which has no GPU summary"),
    ("transform_iter_mut", "candidate: x * 3 + 7"),
    ("transform_rayon", "rejected: calls `<[u32] as rayon::iter::IntoParallelRefMutIterator<'_>>::par_iter_mut`, which has no GPU summary"),
    ("triple", "rejected: takes `u32`, not `&mut [u32]`"),
    ("two_loops", "rejected: contains more than one loop"),
    ("two_stores", "candidate: x * 2 + 2"),
    ("until_zero", "rejected: uses `Eq`, which is not supported"),
    ("with_helper", "rejected: calls `triple`, which has no GPU summary"),
    ("wrapping_sum", "rejected: takes `&[u32]`, not `&mut [u32]`"),
];

/// Plain `+` and `*` panic on overflow in debug builds and wrap in release.
const DEBUG: &[(&str, &str)] = &[
    (
        "checked_increment",
        "rejected: may panic: `Add` can overflow",
    ),
    ("plain_ops", "rejected: may panic: `Mul` can overflow"),
];
const RELEASE: &[(&str, &str)] = &[
    ("checked_increment", "candidate: x * 1 + 1"),
    ("plain_ops", "candidate: x * 3 + 7"),
];

fn expected_report(profile: &[(&str, &str)]) -> String {
    let mut lines: Vec<String> = COMMON
        .iter()
        .chain(profile)
        .map(|(function, verdict)| format!("cuprite_probes::{function}: {verdict}"))
        .collect();
    lines.sort();
    lines.join("\n") + "\n"
}

fn candidates(profile: &[(&'static str, &str)]) -> Vec<&'static str> {
    COMMON
        .iter()
        .chain(profile)
        .filter(|(_, verdict)| verdict.starts_with("candidate"))
        .map(|&(function, _)| function)
        .collect()
}

#[test]
fn the_report_matches_in_debug_and_release() {
    for (name, release, profile) in [
        ("report-debug", false, DEBUG),
        ("report-release", true, RELEASE),
    ] {
        let probes = Probes::new(name, release);
        probes.build(&["build", "--lib"]);
        assert_eq!(probes.report(), expected_report(profile), "{name}");
    }
}

/// With `CUPRITE_AUTO=1`, exactly the candidates call the runtime.
#[test]
fn automatic_rewriting_touches_only_candidates() {
    for (name, release, profile) in [
        ("auto-debug", false, DEBUG),
        ("auto-release", true, RELEASE),
    ] {
        let probes = Probes::new(name, release)
            .with_runtime()
            .env("CUPRITE_AUTO", "1");
        probes.build(&["build", "--lib"]);
        let accepted = candidates(profile);
        for (function, _) in COMMON.iter().chain(profile) {
            let rewritten = probes
                .mir(function, "optimized")
                .contains("cuprite_runtime::map_u32");
            assert_eq!(rewritten, accepted.contains(function), "{name}: {function}");
        }
    }
}

fn differential(name: &'static str) -> std::path::PathBuf {
    let probes = Probes::new(name, true)
        .with_runtime()
        .env("CUPRITE_AUTO", "1");
    probes.build(&["build", "--bin", "candidates"]);
    probes.exe("candidates")
}

/// 9 functions times 9 non-empty lengths go to the runtime; empty slices
/// are left to the CPU.
const GPU_CALLS: usize = 81;

#[test]
fn rewritten_candidates_match_the_cpu_when_the_runtime_declines() {
    let trace = run(&differential("differential-cpu"), false, &[]);
    assert_eq!(
        trace.matches("-> cpu (CUPRITE_GPU is not 1)").count(),
        GPU_CALLS + 9
    );
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "needs an NVIDIA GPU; enable the gpu-tests feature"
)]
fn rewritten_candidates_match_the_cpu_on_the_gpu() {
    let trace = run(&differential("differential-gpu"), true, &[]);
    assert_eq!(trace.matches("-> gpu").count(), GPU_CALLS, "{trace}");
    assert_eq!(trace.matches("-> cpu (no work)").count(), 9, "{trace}");
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "needs an NVIDIA GPU; enable the gpu-tests feature"
)]
fn the_probe_tests_pass_with_every_candidate_on_the_gpu() {
    let output = Probes::new("auto-probe-tests", true)
        .with_runtime()
        .env("CUPRITE_AUTO", "1")
        .env("CUPRITE_GPU", "1")
        .env("CUPRITE_TRACE", "1")
        .build(&["test", "--lib", "--", "--nocapture"]);
    assert!(text(&output.stderr).contains("-> gpu"));
}
