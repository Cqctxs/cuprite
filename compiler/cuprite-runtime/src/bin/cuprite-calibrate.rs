//! Measures this machine and writes a profile for `CUPRITE_PROFILE`.
//!
//! Usage: `cuprite-calibrate <profile path>`

use std::{env, fs, process::ExitCode};

use cuprite_runtime::{
    calibrate::calibrate,
    policy::{show, worst_error},
};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let [path] = &args[..] else {
        eprintln!("usage: cuprite-calibrate <profile path>");
        return ExitCode::FAILURE;
    };
    let calibration = match calibrate(|line| eprintln!("{line}")) {
        Ok(calibration) => calibration,
        Err(error) => {
            eprintln!("cuprite-calibrate: {error}");
            return ExitCode::FAILURE;
        }
    };
    let profile = &calibration.profile;
    for (side, cost, samples) in [
        ("gpu", &profile.gpu, &calibration.gpu),
        ("cpu", &profile.cpu, &calibration.cpu),
    ] {
        let rows: Vec<String> = cost
            .rows
            .iter()
            .map(|row| {
                format!(
                    "{}: {} + {}/pass",
                    row.len,
                    show(row.fixed),
                    show(row.per_pass)
                )
            })
            .collect();
        eprintln!(
            "{side}: {}; worst fit error {:.0}%",
            rows.join(", "),
            worst_error(cost, samples) * 100.0
        );
    }
    if let Err(error) = fs::write(path, profile.to_string()) {
        eprintln!("cuprite-calibrate: cannot write {path}: {error}");
        return ExitCode::FAILURE;
    }
    eprintln!("cold start {}; wrote {path}", show(profile.cold_us));
    ExitCode::SUCCESS
}
