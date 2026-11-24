use super::*;
use cuprite_memory::DeviceInfo;

const STEP: u64 = 37;

const LENGTHS: [usize; 6] = [1 << 10, 1 << 14, 1 << 17, 1 << 20, 1 << 22, 1 << 24];

/// `fixed + per_element·n + per_pass·p + per_element_pass·n·p`, as rows
/// at `LENGTHS`. Linear in `n`, so it is exact between rows too.
fn cost(fixed: f64, per_element: f64, per_pass: f64, per_element_pass: f64) -> Cost {
    let row = |len: usize| Row {
        len,
        fixed: fixed + per_element * len as f64,
        per_pass: per_pass + per_element_pass * len as f64,
    };
    Cost {
        rows: LENGTHS.into_iter().map(row).collect(),
    }
}

fn profile() -> Profile {
    Profile {
        host: HostKey {
            computation: COMPUTATION.into(),
            runtime: "0.1.0".into(),
            rustc: "rustc 1.0.0-nightly".into(),
            cpu: "Test CPU".into(),
        },
        device: DeviceInfo {
            name: "Test GPU".into(),
            compute_capability: (12, 0),
            driver_version: 13000,
            kernels: 0x0123_4567_89ab_cdef,
        },
        cold_us: 100_000.0,
        // Warm GPU calls: 80 us, plus transfers, launches and work.
        gpu: cost(80.0, 1e-3, 8.0, 2e-5),
        cpu: cost(0.0, 0.0, 0.0, 1e-4),
    }
}

fn samples(micros: impl Fn(usize, u32, usize) -> f64) -> Vec<Sample> {
    let mut samples = Vec::new();
    for len in LENGTHS {
        for passes in [1, 4, 16, 64] {
            let micros = micros(len, passes, samples.len());
            samples.push(Sample {
                len,
                passes,
                micros,
            });
        }
    }
    samples
}

#[test]
fn fitting_recovers_the_model_that_produced_the_samples() {
    let truth = profile().gpu;
    let samples = samples(|len, passes, _| truth.predict(len, passes));
    let fitted = fit(&samples).unwrap();
    assert!(worst_error(&fitted, &samples) < 1e-9, "{fitted:?}");
    for (a, b) in fitted.rows.iter().zip(&truth.rows) {
        assert_eq!(a.len, b.len);
        assert!((a.fixed - b.fixed).abs() <= 1e-9 * b.fixed, "{a:?} {b:?}");
        assert!(
            (a.per_pass - b.per_pass).abs() <= 1e-9 * b.per_pass,
            "{a:?} {b:?}"
        );
    }
}

/// The CPU's cost per element grows from cache into main memory; the
/// rows follow it where one formula cannot.
#[test]
fn fitting_follows_a_cost_that_changes_with_length() {
    let per_element_pass = |len: usize| if len <= 1 << 17 { 1e-4 } else { 2.5e-4 };
    let samples = samples(|len, passes, _| per_element_pass(len) * len as f64 * f64::from(passes));
    let fitted = fit(&samples).unwrap();
    assert!(worst_error(&fitted, &samples) < 1e-9, "{fitted:?}");
}

#[test]
fn fitting_keeps_every_term_non_negative() {
    // Pure per-pass work, with noise that would make `fixed` negative.
    let samples = samples(|len, passes, i| {
        1e-4 * len as f64 * f64::from(passes) * (1.0 + 0.05 * (i % 3) as f64)
    });
    let fitted = fit(&samples).unwrap();
    assert!(
        fitted
            .rows
            .iter()
            .all(|r| r.fixed >= 0.0 && r.per_pass >= 0.0),
        "{fitted:?}"
    );
    assert!(worst_error(&fitted, &samples) < 0.1, "{fitted:?}");
}

#[test]
fn fitting_needs_usable_samples() {
    assert_eq!(fit(&[]), None);
    let one = |len, passes, micros| Sample {
        len,
        passes,
        micros,
    };
    // Zero time, one length, and one pass count are all unusable.
    assert_eq!(fit(&[one(1, 1, 0.0), one(1, 2, 1.0)]), None);
    assert_eq!(fit(&[one(1, 1, 1.0), one(1, 2, 2.0)]), None);
    assert_eq!(fit(&[one(1, 1, 1.0), one(2, 1, 2.0)]), None);
}

#[test]
fn predictions_interpolate_between_lengths_and_extrapolate_beyond() {
    let cost = Cost {
        rows: vec![
            Row {
                len: 100,
                fixed: 10.0,
                per_pass: 1.0,
            },
            Row {
                len: 200,
                fixed: 20.0,
                per_pass: 3.0,
            },
        ],
    };
    assert_eq!(cost.predict(100, 2), 12.0);
    assert_eq!(cost.predict(150, 2), 19.0);
    assert_eq!(cost.predict(300, 2), 40.0);
    assert_eq!(cost.predict(0, 2), 0.0, "clamped at zero");
}

#[test]
fn profiles_round_trip_through_text() {
    let profile = profile();
    assert_eq!(Profile::parse(&profile.to_string()).unwrap(), profile);
}

#[test]
fn malformed_profiles_are_refused() {
    let text = profile().to_string();
    let cases = [
        (text.replace("format = 1", "format = 2"), "not format 1"),
        (
            text.replace("cold_us = 100000", "cold_us = -1"),
            "`cold_us`",
        ),
        (text.replace("gpu_us = 1024 ", "gpu_us = "), "rows are"),
        (
            text.replace("gpu_us = 1024 ", "gpu_us = 99999999 "),
            "increasing length",
        ),
        (text.replace("driver = 13000\n", ""), "missing `driver`"),
        (text.clone() + "speed = 11\n", "unknown key `speed`"),
        (text.clone() + "cpu = Other\n", "appears twice"),
        (text.replace("kernels = ", "kernels = xyz"), "hexadecimal"),
    ];
    for (text, expected) in cases {
        let error = Profile::parse(&text).unwrap_err();
        assert!(error.contains(expected), "{error}");
    }
}

#[test]
fn profiles_apply_only_where_they_were_measured() {
    let profile = profile();
    assert_eq!(profile.check_host(&profile.host), Ok(()));
    assert_eq!(profile.check_device(&profile.device), Ok(()));

    let mut host = profile.host.clone();
    host.cpu = "Other CPU".into();
    assert_eq!(
        profile.check_host(&host).unwrap_err(),
        "profile is for cpu `Test CPU`, not `Other CPU`"
    );
    let mut device = profile.device.clone();
    device.driver_version = 13010;
    assert_eq!(
        profile.check_device(&device).unwrap_err(),
        "profile is for driver `13000`, not `13010`"
    );
    device = profile.device.clone();
    device.kernels = 1;
    assert!(profile.check_device(&device).is_err());
}

#[test]
fn calls_the_gpu_loses_stay_on_the_cpu() {
    let profile = profile();
    let mut policy = Policy::new();
    for _ in 0..1000 {
        // 1000 elements, 8 passes: 0.8 us on the CPU, ~150 us on the GPU.
        assert!(matches!(
            policy.choose(&profile, 1000, 8, STEP, false),
            Choice::Cpu(_)
        ));
        assert!(matches!(
            policy.choose(&profile, 1000, 8, STEP, true),
            Choice::Cpu(_)
        ));
    }
    assert_eq!(policy.forgone_us, 0.0);
}

#[test]
fn the_gpu_opens_once_forgone_savings_cover_its_startup() {
    let profile = profile();
    let mut policy = Policy::new();
    let (len, passes) = (1 << 20, 64);
    let saving = profile.cpu.predict(len, passes) - profile.gpu.predict(len, passes);
    let calls_to_wait = (profile.cold_us / saving).ceil() as usize;
    assert!(calls_to_wait > 1, "the test needs a few calls");
    for call in 1..calls_to_wait {
        let choice = policy.choose(&profile, len, passes, STEP, false);
        assert!(
            matches!(choice, Choice::Wait { .. }),
            "call {call}: {choice:?}"
        );
        assert!(!choice.uses_gpu());
        let reason = choice.to_string();
        assert!(
            reason.contains("; waiting to cover 100.00 ms startup"),
            "{reason}"
        );
    }
    let choice = policy.choose(&profile, len, passes, STEP, false);
    assert!(matches!(choice, Choice::Open(_)), "{choice:?}");
    assert!(choice.uses_gpu());
    assert!(
        choice.to_string().ends_with("; startup now covered"),
        "{choice}"
    );
    assert!(matches!(
        policy.choose(&profile, len, passes, STEP, true),
        Choice::Gpu(_)
    ));
}

#[test]
fn one_call_large_enough_opens_the_gpu_at_once() {
    let profile = profile();
    let mut policy = Policy::new();
    // ~1.7 s on the CPU: the saving alone covers startup.
    assert!(matches!(
        policy.choose(&profile, 1 << 24, 1024, STEP, false),
        Choice::Open(_)
    ));
}

/// The profile says the CPU loses this call; the program's own CPU body
/// turns out twice as fast, which makes it the winner.
#[test]
fn measurements_correct_the_profile_for_one_map_and_length_range() {
    let profile = profile();
    let mut policy = Policy::new();
    // 1M elements, 16 passes: the profile favors the GPU by less than 2x.
    let (len, passes) = (1 << 20, 16);
    let before = policy.predict(&profile, len, passes, STEP);
    assert!(
        before.gpu < before.cpu && before.cpu < 2.0 * before.gpu,
        "{before:?}"
    );
    assert!(Policy::worth_timing(before));

    for _ in 0..20 {
        policy.record(&profile, len, passes, STEP, false, before.cpu / 2.0);
    }
    let after = policy.predict(&profile, len, passes, STEP);
    assert!(
        (after.cpu / (before.cpu / 2.0) - 1.0).abs() < 0.01,
        "{after:?}"
    );
    assert_eq!(after.gpu, before.gpu);
    assert!(matches!(
        policy.choose(&profile, len, passes, STEP, false),
        Choice::Cpu(_)
    ));

    // Other kernels, other lengths, and the GPU side keep the profile.
    assert_eq!(policy.predict(&profile, len, passes, 51), before);
    let far = policy.predict(&profile, len * 4, passes, STEP);
    assert_eq!(far.cpu, profile.cpu.predict(len * 4, passes));
    // Lengths within a factor of two share the correction.
    let near = policy.predict(&profile, len + 1000, passes, STEP);
    assert!(near.cpu < 0.6 * profile.cpu.predict(len + 1000, passes));
}

#[test]
fn one_outlier_moves_a_correction_only_so_far() {
    let profile = profile();
    let mut policy = Policy::new();
    let (len, passes) = (1 << 20, 16);
    let predicted = profile.gpu.predict(len, passes);
    policy.record(&profile, len, passes, STEP, true, predicted * 1000.0);
    let after = policy.predict(&profile, len, passes, STEP);
    // Counted as 4x, and moved a quarter of the way from 1.
    assert_eq!(after.gpu, predicted * 1.75);
    // Later ordinary calls pull it back.
    for _ in 0..40 {
        policy.record(&profile, len, passes, STEP, true, predicted);
    }
    let settled = policy.predict(&profile, len, passes, STEP);
    assert!((settled.gpu / predicted - 1.0).abs() < 0.01, "{settled:?}");
}

#[test]
fn only_close_calls_are_worth_timing() {
    let at = |gpu, cpu| Policy::worth_timing(Prediction { gpu, cpu });
    assert!(at(1.0, 1.0));
    assert!(at(3.9, 1.0));
    assert!(at(1.0, 3.9));
    assert!(!at(72.0, 0.8), "small calls are never timed");
    assert!(!at(1.0, 40.0));
}

#[test]
fn choices_explain_themselves() {
    let predicted = Prediction {
        gpu: 72.0,
        cpu: 0.8,
    };
    assert_eq!(
        Choice::Cpu(predicted).to_string(),
        "predicted gpu 72.0 us vs cpu 0.8 us"
    );
    let wait = Choice::Wait {
        predicted,
        cold_us: 120_000.0,
        forgone_us: 3_500.0,
    };
    assert_eq!(
        wait.to_string(),
        "predicted gpu 72.0 us vs cpu 0.8 us; waiting to cover 120.00 ms startup, 3.50 ms so far"
    );
}

#[test]
fn durations_read_naturally() {
    assert_eq!(show(12.34), "12.3 us");
    assert_eq!(show(4_560.0), "4.56 ms");
    assert_eq!(show(2_500_000.0), "2.50 s");
}
