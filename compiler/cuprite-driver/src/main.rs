//! A rustc driver: compiles exactly like rustc, and can additionally
//! - dump MIR (`dump`, `CUPRITE_DUMP_MIR=<dir>`),
//! - report which functions are GPU candidates (`analyze`,
//!   `CUPRITE_REPORT=<dir>`),
//! - rewrite functions to dispatch to the Cuprite runtime (`replace`):
//!   every candidate with `CUPRITE_AUTO=1`, or those named in
//!   `CUPRITE_REPLACE`.
//!
//! `cargo cuprite` (`src/bin/cargo-cuprite.rs`) runs it as Cargo's `RUSTC`,
//! so every crate passes through it. It can also be Cargo's workspace
//! wrapper, so that only workspace members do:
//!
//! ```text
//! RUSTC_WORKSPACE_WRAPPER=path/to/cuprite-driver CUPRITE_REPORT=out cargo build -p cuprite-probes
//! ```
//!
//! Build scripts, procedural macros and Cuprite's own crates always compile
//! as with plain rustc.

#![feature(rustc_private)]

extern crate rustc_attr_ir;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;

mod analyze;
mod dump;
mod replace;

use std::{
    env,
    path::{Path, PathBuf},
    process::ExitCode,
    sync::OnceLock,
};

use rustc_data_structures::steal::Steal;
use rustc_driver::{Callbacks, Compilation};
use rustc_interface::interface;
use rustc_middle::{mir::Body, ty::TyCtxt};
use rustc_session::{config::ErrorOutputType, EarlyDiagCtxt};
use rustc_span::{def_id::LocalDefId, Symbol};

/// Every setting the driver reads, recorded as build inputs.
const TRACKED: [&str; 6] = [
    dump::VAR,
    analyze::REPORT_VAR,
    analyze::AUTO_VAR,
    replace::VAR,
    replace::RUNTIME_VAR,
    replace::RUNTIME_DEPS_VAR,
];

/// Cuprite's own crates, which are never analyzed or linked to the runtime.
const CUPRITE_CRATES: [&str; 2] = [replace::RUNTIME_CRATE, "cuprite_memory"];

#[derive(Default)]
pub(crate) struct Settings {
    pub(crate) dump: Option<PathBuf>,
    pub(crate) report: Option<PathBuf>,
    pub(crate) auto: bool,
    pub(crate) replacements: Vec<replace::Replacement>,
    /// The runtime files this crate was given, which it is rebuilt after.
    pub(crate) runtime: Vec<String>,
}

impl Settings {
    fn analyzes(&self) -> bool {
        self.report.is_some() || self.auto
    }

    fn active(&self) -> bool {
        self.dump.is_some() || self.analyzes() || !self.replacements.is_empty()
    }
}

/// Query providers are plain function pointers, so they read the settings
/// from here. The driver compiles one crate per process.
static SETTINGS: OnceLock<Settings> = OnceLock::new();

pub(crate) fn settings() -> &'static Settings {
    SETTINGS.get().expect("settings are read before compiling")
}

/// Wraps rustc's own `mir_built`: hands the body to the dump and the
/// recognizer, then returns it unchanged.
fn mir_built<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> &'tcx Steal<Body<'tcx>> {
    let built = (rustc_interface::DEFAULT_QUERY_PROVIDERS.queries.mir_built)(tcx, def_id);
    let settings = settings();
    {
        let body = built.borrow();
        if settings.dump.is_some() {
            dump::record(tcx, def_id, &body);
        }
        if settings.analyzes() {
            analyze::record(tcx, def_id, &body);
        }
    }
    built
}

struct Driver;

impl Callbacks for Driver {
    fn config(&mut self, config: &mut interface::Config) {
        // Cargo does not know the driver reads these, so record them in
        // rustc's dependency info: changing one then triggers a rebuild.
        // The runtime's files are inputs too: a rebuilt runtime must be
        // linked again.
        config.track_state = Some(Box::new(|session| {
            let mut depinfo = session.env_depinfo.borrow_mut();
            for name in TRACKED {
                let value = env::var(name).ok().map(|value| Symbol::intern(&value));
                depinfo.insert((Symbol::intern(name), value));
            }
            let mut files = session.file_depinfo.borrow_mut();
            for file in &settings().runtime {
                files.insert(Symbol::intern(file));
            }
        }));
        if settings().active() {
            // Incremental compilation would reuse cached query results and
            // skip the overrides below.
            config.opts.incremental = None;
            config.override_queries = Some(|_, providers| {
                providers.queries.mir_built = mir_built;
                providers.queries.optimized_mir = replace::optimized_mir;
            });
        }
    }

    fn after_analysis<'tcx>(
        &mut self,
        _compiler: &interface::Compiler,
        tcx: TyCtxt<'tcx>,
    ) -> Compilation {
        let settings = settings();
        if let Some(dir) = &settings.dump {
            dump::write_all(tcx, dir);
        }
        if let Some(dir) = &settings.report {
            analyze::write_report(tcx, dir);
        }
        Compilation::Continue
    }
}

fn arg_value<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].as_str())
}

/// Whether this invocation compiles as plain rustc: Cargo's information
/// queries, which compile nothing; build scripts and procedural macros,
/// which run while building rather than in the program; and Cuprite's own
/// crates, which must never call themselves.
fn bypasses(args: &[String]) -> bool {
    let info_query = args.iter().any(|arg| {
        matches!(arg.as_str(), "-vV" | "-V" | "--version") || arg.starts_with("--print")
    });
    let crate_name = arg_value(args, "--crate-name").unwrap_or_default();
    info_query
        || crate_name.starts_with("build_script_")
        || CUPRITE_CRATES.contains(&crate_name)
        || arg_value(args, "--crate-type") == Some("proc-macro")
}

/// Makes the runtime crate visible to the crate being compiled without the
/// crate declaring it, and lets the final link find its dependencies.
/// Returns the runtime files.
fn add_runtime_args(early_dcx: &EarlyDiagCtxt, args: &mut Vec<String>) -> Vec<String> {
    // rustc's arguments are UTF-8, so other paths can't be passed on.
    let utf8 = |path: PathBuf| match path.into_os_string().into_string() {
        Ok(path) => path,
        Err(path) => early_dcx.early_fatal(format!(
            "cuprite: runtime path {} is not UTF-8",
            Path::new(&path).display()
        )),
    };
    let Some(runtime) = env::var_os(replace::RUNTIME_VAR) else {
        return Vec::new();
    };
    args.push("-Zunstable-options".into());
    // Cargo may store metadata in a separate `.rmeta` next to a stub
    // `.rlib`; rustc needs both, so the variable is a path list.
    let files: Vec<String> = env::split_paths(&runtime).map(utf8).collect();
    for file in &files {
        args.push("--extern".into());
        args.push(format!("force:{}={file}", replace::RUNTIME_CRATE));
    }
    if let Some(deps) = env::var_os(replace::RUNTIME_DEPS_VAR) {
        for dir in env::split_paths(&deps) {
            args.push("-L".into());
            args.push(format!("dependency={}", utf8(dir)));
        }
    }
    files
}

fn read_settings(early_dcx: &EarlyDiagCtxt, runtime: Vec<String>) -> Settings {
    let replacements = match env::var(replace::VAR).map(|spec| replace::parse(&spec)) {
        Ok(Ok(replacements)) => replacements,
        Ok(Err(message)) => early_dcx.early_fatal(message),
        Err(_) => Vec::new(),
    };
    Settings {
        dump: env::var_os(dump::VAR).map(PathBuf::from),
        report: env::var_os(analyze::REPORT_VAR).map(PathBuf::from),
        auto: env::var_os(analyze::AUTO_VAR).is_some_and(|value| value == "1"),
        replacements,
        runtime,
    }
}

fn main() -> ExitCode {
    let early_dcx = EarlyDiagCtxt::new(ErrorOutputType::default());
    rustc_driver::init_rustc_env_logger(&early_dcx);
    rustc_driver::install_ice_hook("https://github.com/Cqctxs/cuprite/issues", |_| ());

    rustc_driver::catch_with_exit_code(move || {
        let mut args = rustc_driver::args::raw_args(&early_dcx);
        // As a Cargo wrapper, the second argument is the rustc being wrapped.
        if args.get(1).map(Path::new).and_then(Path::file_stem) == Some("rustc".as_ref()) {
            args.remove(1);
        }

        let settings = if bypasses(&args) {
            Settings::default()
        } else {
            let runtime = add_runtime_args(&early_dcx, &mut args);
            read_settings(&early_dcx, runtime)
        };
        if SETTINGS.set(settings).is_err() {
            unreachable!("the driver compiles one crate per process");
        }
        rustc_driver::compiler_entrypoint(&args, &mut Driver);
        ExitCode::SUCCESS
    })
}
