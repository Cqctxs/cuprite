//! M2 and M3: the recognizer finds the probe crate's maps and repeated maps
//! on its own, rejects everything else with a reason, and `CUPRITE_AUTO=1`
//! rewrites exactly the accepted functions, with results identical to the
//! CPU's.

mod support;

use support::{run, text, Probes, NO_PROFILE};

/// Verdicts that do not depend on the build profile.
const COMMON: &[(&str, &str)] = &[
    ("add_into", "rejected: takes `&[u32]` as `a`, which a kernel can't take"),
    ("affine_chain", "candidate: map over `[u32]`"),
    ("apply", "rejected: takes `fn(u32) -> u32` as `f`, which a kernel can't take"),
    ("clamp", "candidate: map over `[u32]`"),
    ("collatz_steps", "candidate: map over `[u32]`"),
    ("counted", "rejected: calls `std::sync::atomic::Atomic::<u32>::fetch_add`, which has no GPU summary"),
    ("counted_calls", "rejected: takes no parameters; the first must be a `&mut` slice of numbers"),
    ("damp", "candidate: map over `[f32]`"),
    ("divide", "rejected: may panic: division by zero"),
    ("double", "candidate: map over `[u32]`"),
    ("fill", "candidate: map over `[u32]`"),
    ("float_rem", "rejected: uses `%` on floats, which has no exact GPU version yet"),
    ("hash_bytes", "candidate: map over `[u8]` reading `seed`"),
    ("invert", "rejected: takes `&mut [bool]`, not a `&mut` slice of numbers"),
    ("levels", "candidate: map over `[u8]`"),
    ("minimum", "rejected: takes `&[u32]`, not a `&mut` slice of numbers"),
    ("mix_wide", "candidate: map over `[i64]` reading `shift`, `bias`"),
    ("numbered", "rejected: uses a value computed outside the loop body"),
    ("outside_factor", "candidate: map over `[u32]`"),
    ("prefix_sum", "rejected: calls `core::slice::<impl [u32]>::len`, which has no GPU summary"),
    ("quantize", "candidate: map over `[i16]` reading `step`"),
    ("reassigned", "rejected: reassigns its parameter `values`"),
    ("reciprocal", "rejected: may panic: division by zero"),
    ("relu", "candidate: map over `[f32]`"),
    ("repeat", "candidate: `transform` over `[u32]`, repeated `passes` times"),
    ("repeat_after_first", "rejected: computes values in the loop body"),
    ("repeat_composed", "candidate: `transform` then `double` over `[u32]`, repeated `passes` times"),
    ("repeat_damp", "candidate: `damp` then `relu` over `[f32]`, repeated `passes` times"),
    ("repeat_fixed", "candidate: `transform` over `[u32]`, repeated 4 times"),
    ("repeat_from_one", "rejected: counts passes from 1, not 0"),
    ("repeat_helper", "rejected: calls `with_helper`, which is not a recognized map"),
    ("repeat_named", "candidate: `affine_chain` over `[u32]`, repeated `passes` times"),
    ("repeat_nested", "rejected: calls `repeat`, which is not a recognized map"),
    ("repeat_reassigned", "rejected: reassigns its parameter `passes`"),
    ("reversed", "rejected: calls `<std::slice::IterMut<'_, u32> as std::iter::Iterator>::rev`, which has no GPU summary"),
    ("scale_offset", "candidate: map over `[f32]` reading `scale`, `offset`"),
    ("smooth", "rejected: takes `&[u32]`, not a `&mut` slice of numbers"),
    ("soft_root", "candidate: map over `[f64]`"),
    ("square", "candidate: map over `[u32]`"),
    ("transform", "candidate: map over `[u32]`"),
    ("transform_indexed", "rejected: calls `core::slice::<impl [u32]>::len`, which has no GPU summary"),
    ("transform_iter_mut", "candidate: map over `[u32]`"),
    ("transform_rayon", "rejected: calls `<[u32] as rayon::iter::IntoParallelRefMutIterator<'_>>::par_iter_mut`, which has no GPU summary"),
    ("triple", "rejected: takes `u32`, not a `&mut` slice of numbers"),
    ("two_loops", "rejected: contains more than one loop"),
    ("two_stores", "candidate: map over `[u32]`"),
    ("until_zero", "rejected: leaves the loop early"),
    ("with_helper", "rejected: calls `triple`, which has no GPU summary"),
    ("wrapping_sum", "rejected: takes `&[u32]`, not a `&mut` slice of numbers"),
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
    ("checked_increment", "candidate: map over `[u32]`"),
    ("plain_ops", "candidate: map over `[u32]`"),
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
                .contains("cuprite_runtime::map_slice");
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

/// Runtime calls the differential program makes with `passes` passes, for
/// each of its 10 lengths (`probes/src/bin/candidates.rs`): 4 repeated maps
/// with 13 passes (3 over `u32`, `repeat_damp` over `f32`), and
/// `repeat_fixed` with 4.
const REGIONS: [(u32, usize); 2] = [(13, 4), (4, 1)];
const LENGTHS: usize = 10;

#[test]
fn rewritten_candidates_match_the_cpu_when_the_runtime_declines() {
    let trace = run(&differential("differential-cpu"), false, &[]);
    assert!(
        trace
            .lines()
            .all(|line| line.ends_with(NO_PROFILE) || line.ends_with("-> cpu (no work)")),
        "{trace}"
    );
    // The declined repeated maps fall back to their CPU bodies, which call
    // the (rewritten) single maps, so only the counted passes are exact.
    for (passes, functions) in REGIONS {
        let calls = trace.matches(&format!("passes={passes} ")).count();
        assert_eq!(calls, functions * LENGTHS, "{passes} passes");
    }
}

/// Every non-empty call runs on the GPU, and each repeated map is one call
/// with all its passes: its CPU body, which would call the per-pass map
/// once per pass, never runs.
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "needs an NVIDIA GPU; enable the gpu-tests feature"
)]
fn rewritten_candidates_match_the_cpu_on_the_gpu() {
    let trace = run(&differential("differential-gpu"), true, &[]);
    let non_empty = LENGTHS - 1;
    // 9 `u32` multiply-add maps; 3 repeated maps with 1 and 13 passes;
    // `repeat_fixed`; 7 general maps, `mix_wide` with 5 parameter choices,
    // and `repeat_damp` with 1 and 13 passes.
    let gpu_calls = 9 * non_empty + 3 * 2 * non_empty + non_empty + (7 + 5) * non_empty + 2 * non_empty;
    assert_eq!(trace.matches("-> gpu").count(), gpu_calls, "{trace}");
    for (passes, functions) in REGIONS {
        let calls = trace.matches(&format!("passes={passes} -> gpu")).count();
        assert_eq!(calls, functions * non_empty, "{passes} passes");
    }
    assert!(
        trace
            .lines()
            .all(|line| line.ends_with("-> gpu") || line.ends_with("-> cpu (no work)")),
        "{trace}"
    );
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
