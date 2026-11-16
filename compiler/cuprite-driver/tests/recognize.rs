//! M2 and M3: the recognizer finds the probe crate's maps and repeated maps
//! on its own, rejects everything else with a reason, and `CUPRITE_AUTO=1`
//! rewrites exactly the accepted functions, with results identical to the
//! CPU's.

mod support;

use support::{run, text, Probes, NO_PROFILE};

/// Verdicts that do not depend on the build profile.
const COMMON: &[(&str, &str)] = &[
    ("clamp_unit", "rejected: takes `&mut f32`, not a `&mut` slice of numbers"),
    ("each_affine", "candidate: map over `[u32]`"),
    ("each_counter", "rejected: passes `for_each` a closure that changes what it captures, which carries a value from one element to the next"),
    ("each_function", "candidate: map over `[f32]`"),
    ("each_saxpy", "candidate: map over `[f32]` zipped with `x` reading `a`"),
    ("each_setup", "candidate: map over `[f64]` reading `scale`"),
    ("each_then_more", "rejected: uses `core::slice::<impl [u32]>::first_mut` after the loop"),
    ("par_saxpy", "candidate: map over `[f32]` zipped with `x` reading `a`"),
    ("apply_once", "rejected: is generic"),
    ("apply_twice", "rejected: is generic"),
    ("halve", "rejected: takes `f32`, not a `&mut` slice of numbers"),
    ("closure_affine", "candidate: map over `[f32]` reading `scale`, `offset`"),
    ("closure_counter", "candidate: map over `[u32]`"),
    ("closure_generic", "candidate: map over `[f32]` reading `add`"),
    ("closure_in_body", "candidate: map over `[u32]` reading `step`"),
    ("closure_move", "candidate: map over `[i32]` reading `bias`"),
    ("closure_once", "candidate: map over `[u32]` reading `mask`"),
    ("closure_pair", "candidate: map over `[u32]` reading `other`"),
    ("closure_table", "candidate: map over `[u32]` indexing `table`"),
    ("function_item", "candidate: map over `[f32]`"),
    ("<Adjust as std::clone::Clone>::clone", "rejected: takes `&Adjust`, not a `&mut` slice of numbers"),
    ("Scale::apply", "rejected: takes `&Scale`, not a `&mut` slice of numbers"),
    ("add_bytes", "candidate: map over `[u8]` reading `amount`"),
    ("add_into", "candidate: map over `[u32]` zipped with `a`, `b`"),
    ("add_pairs", "candidate: map over `[u32]` zipped with `a`, `b`"),
    ("adjust", "candidate: map over `[f32]` reading `settings`"),
    ("adjust_by_value", "candidate: map over `[f32]` reading `settings`, `extra`"),
    ("affine_chain", "candidate: map over `[u32]`"),
    ("apply", "rejected: takes `fn(u32) -> u32` as `f`, which a kernel can't take"),
    ("asserted_halves", "candidate: map over `[u32]`"),
    ("bump", "rejected: takes `&mut f32`, not a `&mut` slice of numbers"),
    ("bump_twice", "candidate: map over `[f32]`"),
    ("checked_increment", "candidate: map over `[u32]`"),
    ("clamp", "candidate: map over `[u32]`"),
    ("clamp_range", "candidate: map over `[f32]` reading `low`, `high`"),
    ("collatz_steps", "candidate: map over `[u32]`"),
    ("conditional_scale", "candidate: map over `[f32]` reading `double`"),
    ("count_halvings", "rejected: calls `halvings`, which calls `halvings` recursively"),
    ("counted", "rejected: uses a reference to constant data (`&std::sync::atomic::Atomic<u32>`)"),
    ("counted_calls", "rejected: takes no parameters; the first must be a `&mut` slice of numbers"),
    ("cross_dot", "candidate: map over `[f32]`"),
    ("damp", "candidate: map over `[f32]`"),
    ("div_mod", "candidate: map over `[i32]` reading `divisor`"),
    ("divide", "candidate: map over `[u32]` reading `divisor`"),
    ("double", "candidate: map over `[u32]`"),
    ("fill", "candidate: map over `[u32]`"),
    ("float_rem", "rejected: uses `%` on floats, which has no exact GPU version yet"),
    ("generic_twice", "candidate: map over `[f32]`"),
    ("halves", "rejected: takes `u32`, not a `&mut` slice of numbers"),
    ("halvings", "rejected: takes `u32`, not a `&mut` slice of numbers"),
    ("hash_bytes", "candidate: map over `[u8]` reading `seed`"),
    ("invert", "rejected: takes `&mut [bool]`, not a `&mut` slice of numbers"),
    ("levels", "candidate: map over `[u8]`"),
    ("lookup", "candidate: map over `[u8]` indexing `table`"),
    ("lookup_wrapped", "candidate: map over `[f32]` indexing `table`"),
    ("minimum", "rejected: takes `&[u32]`, not a `&mut` slice of numbers"),
    ("mix_wide", "candidate: map over `[i64]` reading `shift`, `bias`"),
    ("ni_div_floor", "candidate: map over `[i32]` reading `divisor`"),
    ("ni_gcd", "candidate: map over `[u32]` reading `other`"),
    ("normalize_range", "candidate: map over `[f32]` reading `low`, `high`"),
    ("normalized_y", "candidate: map over `[f32]`"),
    ("numbered", "rejected: carries a value from one element to the next"),
    ("outside_factor", "candidate: map over `[u32]`"),
    ("palette_blend", "candidate: map over `[f32]` zipped with `others`"),
    ("palette_mix", "candidate: map over `[f32]` reading `red`, `green`, `blue`"),
    ("plain_ops", "candidate: map over `[u32]`"),
    ("prefix_sum", "rejected: calls `core::slice::<impl [u32]>::len`, which has no GPU summary"),
    ("quantize", "candidate: map over `[i16]` reading `step`"),
    ("reassigned", "rejected: reassigns its parameter `values`"),
    ("reciprocal", "candidate: map over `[u32]`"),
    ("relu", "candidate: map over `[f32]`"),
    ("repeat", "candidate: `transform` over `[u32]`, repeated `passes` times"),
    ("repeat_after_first", "rejected: computes values in the loop body"),
    ("repeat_composed", "candidate: `transform` then `double` over `[u32]`, repeated `passes` times"),
    ("repeat_damp", "candidate: `damp` then `relu` over `[f32]`, repeated `passes` times"),
    ("repeat_fixed", "candidate: `transform` over `[u32]`, repeated 4 times"),
    ("repeat_from_one", "rejected: counts passes from 1, not 0"),
    ("repeat_helper", "candidate: `with_helper` over `[u32]`, repeated `passes` times"),
    ("repeat_named", "candidate: `affine_chain` over `[u32]`, repeated `passes` times"),
    ("repeat_nested", "rejected: calls `repeat`, which is not a recognized map"),
    ("repeat_reassigned", "rejected: reassigns its parameter `passes`"),
    ("reversed", "rejected: iterates over something other than the slice parameter"),
    ("rounded_box", "candidate: map over `[f32]` reading `size`, `rounding`"),
    ("saxpy", "candidate: map over `[f32]` zipped with `x` reading `a`"),
    ("scale_offset", "candidate: map over `[f32]` reading `scale`, `offset`"),
    ("scaled", "candidate: map over `[f64]`"),
    ("scene_distances", "candidate: map over `[f32]`"),
    ("simd_roots", "rejected: calls `std::arch::x86_64::_mm_set_ss`, which builds an array"),
    ("smooth", "rejected: takes `&[u32]`, not a `&mut` slice of numbers"),
    ("soft_root", "candidate: map over `[f64]`"),
    ("square", "candidate: map over `[u32]`"),
    ("square_wide", "candidate: map over `[i64]`"),
    ("sr_div_rem", "rejected: calls `strength_reduce::StrengthReducedU64::new`, which calls `strength_reduce::long_division::divide_128_max_by_64`, which calls `strength_reduce::long_division::divide_128_by_64_preshifted`, which uses an unsupported operation"),
    ("sr_divide", "candidate: map over `[u32]` reading `divisor`"),
    ("swap_halves", "candidate: map over `[u32]`"),
    ("transform", "candidate: map over `[u32]`"),
    ("transform_indexed", "rejected: calls `core::slice::<impl [u32]>::len`, which has no GPU summary"),
    ("transform_iter_mut", "candidate: map over `[u32]`"),
    ("transform_rayon", "candidate: map over `[u32]`"),
    ("triple", "rejected: takes `u32`, not a `&mut` slice of numbers"),
    ("twice", "rejected: is generic"),
    ("two_loops", "rejected: contains more than one loop"),
    ("two_stores", "candidate: map over `[u32]`"),
    ("until_zero", "rejected: leaves the loop early"),
    ("uv_cross_dot", "candidate: map over `[f32]`"),
    ("uv_lengths", "candidate: map over `[f32]`"),
    ("uv_normalized_y", "candidate: map over `[f32]`"),
    ("vector_lengths", "candidate: map over `[f32]`"),
    ("weighted", "candidate: map over `[f64]` zipped with `weights`"),
    ("with_helper", "candidate: map over `[u32]`"),
    ("wrapping_sum", "rejected: takes `&[u32]`, not a `&mut` slice of numbers"),
];

/// rustc inlines differently in debug and release builds, which changes
/// where some refusals are found.
const DEBUG: &[(&str, &str)] = &[
    ("ni_sqrt", "rejected: calls `<u64 as num_integer::Roots>::sqrt`, which calls `<u64 as num_integer::Roots>::sqrt::go`, which uses an unsupported operation"),
    ("palette_linear", "rejected: calls `palette::rgb::Rgb::into_linear::<f32>`, which calls `<palette::encoding::Srgb as palette::encoding::IntoLinear<f32, f32>>::into_linear`, which calls `<bool as palette::bool_mask::LazySelect<f32>>::lazy_select::<{closure@<palette::encoding::Srgb as palette::encoding::IntoLinear<f32, f32>>::into_linear::{closure#0}}, {closure@<palette::encoding::Srgb as palette::encoding::IntoLinear<f32, f32>>::into_linear::{closure#1}}>`, which calls `<{closure@<palette::encoding::Srgb as palette::encoding::IntoLinear<f32, f32>>::into_linear::{closure#1}} as std::ops::FnOnce<()>>::call_once`, which calls `<f32 as palette::num::Powf>::powf`, which calls `std::f32::<impl f32>::powf`, which uses the intrinsic `powf32`"),
    ("palette_saturation", "rejected: calls `<palette::Hsv as palette::FromColor<palette::rgb::Rgb>>::from_color`, which calls `<palette::Hsv as palette::convert::FromColorUnclamped<palette::rgb::Rgb>>::from_color_unclamped`, which calls `std::any::TypeId::of::<bool>`, which uses a value of type `[*const (); 2]`"),
];
const RELEASE: &[(&str, &str)] = &[
    ("ni_sqrt", "candidate: map over `[u64]`"),
    ("palette_linear", "rejected: calls `palette::rgb::Rgb::into_linear::<f32>`, which calls `<palette::encoding::Srgb as palette::encoding::IntoLinear<f32, f32>>::into_linear`, which calls `<bool as palette::bool_mask::LazySelect<f32>>::lazy_select::<{closure@<palette::encoding::Srgb as palette::encoding::IntoLinear<f32, f32>>::into_linear::{closure#0}}, {closure@<palette::encoding::Srgb as palette::encoding::IntoLinear<f32, f32>>::into_linear::{closure#1}}>`, which calls `<{closure@<palette::encoding::Srgb as palette::encoding::IntoLinear<f32, f32>>::into_linear::{closure#1}} as std::ops::FnOnce<()>>::call_once`, which calls `<f32 as palette::num::Powf>::powf`, which uses the intrinsic `powf32`"),
    ("palette_saturation", "rejected: calls `<palette::Hsv as palette::FromColor<palette::rgb::Rgb>>::from_color`, which calls `<palette::Hsv as palette::convert::FromColorUnclamped<palette::rgb::Rgb>>::from_color_unclamped`, which uses an unsupported operation"),
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

/// The checks program (`probes/src/bin/checks.rs`), built in one profile.
fn checks(name: &'static str, release: bool) -> std::path::PathBuf {
    let probes = Probes::new(name, release)
        .with_runtime()
        .env("CUPRITE_AUTO", "1");
    probes.build(&["build", "--bin", "checks"]);
    probes.exe("checks")
}

const FAILED_CHECK: &str = "-> cpu (a check failed on the GPU)";

/// Runtime calls the checks program makes.
const CHECK_CALLS: usize = 28;

/// Calls whose checks pass keep the GPU's results; calls where one fails
/// fall back to the CPU, which panics with the same message and leaves the
/// same values as plain Rust. Debug builds also check overflow, so 3 more
/// calls fail there. Includes held-out `num-integer` (floored division,
/// by a divisor and by zero, and GCDs) and `strength_reduce` (by a
/// divisor, a power of two and zero), and a closure indexing a captured
/// table, which may be empty.
#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "needs an NVIDIA GPU; enable the gpu-tests feature"
)]
fn failed_checks_on_the_gpu_behave_like_the_cpu() {
    for (name, release, failed) in [("checks-debug", false, 13), ("checks-release", true, 10)] {
        let trace = run(&checks(name, release), true, &[]);
        let calls = trace.lines().filter(|line| line.starts_with("cuprite: ")).count();
        assert_eq!(calls, CHECK_CALLS, "{name}:\n{trace}");
        assert_eq!(trace.matches(FAILED_CHECK).count(), failed, "{name}:\n{trace}");
        assert_eq!(trace.matches("-> gpu").count(), CHECK_CALLS - failed, "{name}:\n{trace}");
    }
}

/// Without a GPU decision, every call runs the CPU code, with the same
/// outcomes.
#[test]
fn checks_match_the_cpu_when_the_runtime_declines() {
    let trace = run(&checks("checks-cpu", true), false, &[]);
    assert!(trace.lines().all(|line| line.ends_with(NO_PROFILE)), "{trace}");
}

fn differential(name: &'static str) -> std::path::PathBuf {
    let probes = Probes::new(name, true)
        .with_runtime()
        .env("CUPRITE_AUTO", "1");
    probes.build(&["build", "--bin", "candidates"]);
    probes.exe("candidates")
}

/// Runtime calls the differential program makes with `passes` passes, for
/// each of its 10 lengths (`probes/src/bin/candidates.rs`): 5 repeated maps
/// with 13 passes (3 multiply-adds, `repeat_helper` and `repeat_damp`),
/// and `repeat_fixed` with 4.
const REGIONS: [(u32, usize); 2] = [(13, 5), (4, 1)];
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
    // 8 maps that call functions, 3 held-out `ultraviolet` maps, and
    // `repeat_helper` and `repeat_damp` with 1 and 13 passes. Then 10 maps
    // reading data, 2 of them twice; and 2 zipped with an input 3 shorter,
    // which have no work below 4 elements. Then 9 maps with closures, and
    // `ni_sqrt` (accepted in release builds). Then 5 maps with `for_each`,
    // and `par_saxpy`, zipped with an input 3 shorter.
    let with_work = LENGTHS - 3;
    let gpu_calls = 9 * non_empty + 3 * 2 * non_empty + non_empty + (7 + 5 + 8 + 3) * non_empty
        + 2 * 2 * non_empty
        + 10 * non_empty
        + 2 * with_work
        + (9 + 1) * non_empty
        + 5 * non_empty
        + with_work;
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
