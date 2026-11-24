//! `coverage-summary <dir> [count]` sums the coverage census files the
//! driver writes (`CUPRITE_COVERAGE=<dir>`, one per crate): how many
//! functions were translated, refused and skipped, and the refusals' root
//! causes ranked by how many functions each blocks.
//!
//! A refusal lists the calls leading to it, followed by the root cause.
//! Causes are grouped with the text between
//! backticks masked (`[u8; 256]` and `[f32; 3]` are both arrays), except an
//! intrinsic's name, and listed exactly too.

use std::{collections::HashMap, env, fs, path::Path, process::ExitCode};

/// The cause at the end of a chain of calls.
fn root(reason: &str) -> &str {
    let mut rest = reason;
    while let Some(after) = rest.strip_prefix("calls `") {
        match after.find("`, which ") {
            Some(end) => rest = &after[end + "`, which ".len()..],
            None => break,
        }
    }
    rest
}

/// `reason` with the text between backticks masked, except an intrinsic's
/// name, which names the gap itself.
fn general(reason: &str) -> String {
    if reason.starts_with("uses the intrinsic `") {
        return reason.split_once(", which").map_or(reason, |(name, _)| name).to_string();
    }
    let mut out = String::new();
    let mut inside = false;
    for c in reason.chars() {
        if c == '`' {
            if inside {
                out.push('…');
            }
            out.push(c);
            inside = !inside;
        } else if !inside {
            out.push(c);
        }
    }
    out
}

fn ranked(counts: HashMap<String, usize>, count: usize) -> Vec<(String, usize)> {
    let mut list: Vec<_> = counts.into_iter().collect();
    list.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    list.truncate(count);
    list
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let Some(dir) = args.first() else {
        eprintln!("usage: coverage-summary <dir> [count]");
        return ExitCode::from(2);
    };
    let count: usize = args.get(1).and_then(|n| n.parse().ok()).unwrap_or(40);
    let Ok(entries) = fs::read_dir(Path::new(dir)) else {
        eprintln!("coverage-summary: cannot read {dir}");
        return ExitCode::from(2);
    };
    // A crate compiled twice with different features (for build scripts and
    // for the program) has a file per compilation, `<crate>-<id>.txt`; the
    // one with the most functions counts, so no function counts twice.
    let mut files: HashMap<String, String> = HashMap::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "txt") {
            continue;
        }
        let name = path.file_stem().unwrap_or_default().to_string_lossy().into_owned();
        let krate = name.rsplit_once('-').map_or(name.as_str(), |(krate, _)| krate).to_string();
        let text = fs::read_to_string(&path).unwrap_or_default();
        let longer = |kept: &String| (text.lines().count(), &text) > (kept.lines().count(), kept);
        if files.get(&krate).is_none_or(longer) {
            files.insert(krate, text);
        }
    }
    let crates = files.len();
    let (mut translated, mut refused, mut generic, mut skipped) = (0, 0, 0, 0);
    let mut causes: HashMap<String, usize> = HashMap::new();
    let mut exact: HashMap<String, usize> = HashMap::new();
    let mut skips: HashMap<String, usize> = HashMap::new();
    for text in files.values() {
        for line in text.lines() {
            // Paths contain `: ` only inside generic arguments, which come
            // before the verdict; the verdict follows the last `: ` before
            // "translated", "refused" or "skipped".
            let verdict = ["translated", "refused: ", "skipped: "]
                .iter()
                .filter_map(|marker| line.rfind(&format!(": {marker}")).map(|at| &line[at + 2..]))
                .min_by_key(|verdict| verdict.len());
            match verdict {
                Some("translated") => translated += 1,
                Some(v) if v.starts_with("refused: ") => {
                    refused += 1;
                    let cause = root(&v["refused: ".len()..]);
                    *causes.entry(general(cause)).or_default() += 1;
                    *exact.entry(cause.to_string()).or_default() += 1;
                }
                Some("skipped: generic") => generic += 1,
                Some(v) if v.starts_with("skipped: ") => {
                    skipped += 1;
                    *skips.entry(general(&v["skipped: ".len()..])).or_default() += 1;
                }
                _ => {}
            }
        }
    }
    let attempted = translated + refused;
    let percent = |n: usize| if attempted == 0 { 0.0 } else { 100.0 * n as f64 / attempted as f64 };
    println!("{crates} crates: {attempted} functions over element data attempted");
    println!("  translated  {translated:>6}  ({:.1}%)", percent(translated));
    println!("  refused     {refused:>6}  ({:.1}%)", percent(refused));
    println!("  skipped     {generic:>6} generic, {skipped} not over element data");
    println!("\nRefusal causes, grouped (functions blocked):");
    for (cause, n) in ranked(causes, count) {
        println!("  {n:>6}  {cause}");
    }
    println!("\nRefusal causes, exact:");
    for (cause, n) in ranked(exact, count) {
        println!("  {n:>6}  {cause}");
    }
    println!("\nSkip reasons:");
    for (reason, n) in ranked(skips, count.min(15)) {
        println!("  {n:>6}  {reason}");
    }
    ExitCode::SUCCESS
}
