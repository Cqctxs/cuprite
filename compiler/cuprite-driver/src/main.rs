//! A rustc driver: compiles exactly like rustc, and can additionally dump
//! MIR (`dump`) and rewrite named functions to dispatch to the Cuprite
//! runtime (`replace`).
//!
//! Use it as Cargo's workspace wrapper so only workspace members, not
//! dependencies, pass through Cuprite:
//!
//! ```text
//! RUSTC_WORKSPACE_WRAPPER=path/to/cuprite-driver CUPRITE_DUMP_MIR=out cargo build -p cuprite-probes
//! ```

#![feature(rustc_private)]

extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;

mod dump;
mod replace;

use std::{
    env,
    path::{Path, PathBuf},
    process::ExitCode,
};

use rustc_driver::{Callbacks, Compilation};
use rustc_interface::interface;
use rustc_middle::ty::TyCtxt;
use rustc_session::{config::ErrorOutputType, EarlyDiagCtxt};
use rustc_span::Symbol;

/// Every setting the driver reads, recorded as build inputs.
const TRACKED: [&str; 4] = [
    dump::VAR,
    replace::VAR,
    replace::RUNTIME_VAR,
    replace::RUNTIME_DEPS_VAR,
];

struct Driver {
    dump: Option<PathBuf>,
    replace: bool,
}

impl Callbacks for Driver {
    fn config(&mut self, config: &mut interface::Config) {
        // Cargo does not know the driver reads these, so record them in
        // rustc's dependency info: changing one then triggers a rebuild.
        config.track_state = Some(Box::new(|session| {
            let mut depinfo = session.env_depinfo.borrow_mut();
            for name in TRACKED {
                let value = env::var(name).ok().map(|value| Symbol::intern(&value));
                depinfo.insert((Symbol::intern(name), value));
            }
        }));
        if self.replace {
            // Incremental compilation would reuse cached query results that
            // do not know about the rewrite.
            config.opts.incremental = None;
        }
        config.override_queries = match (self.dump.is_some(), self.replace) {
            (false, false) => None,
            (true, false) => Some(|_, providers| {
                providers.queries.mir_built = dump::mir_built;
            }),
            (false, true) => Some(|_, providers| {
                providers.queries.optimized_mir = replace::optimized_mir;
            }),
            (true, true) => Some(|_, providers| {
                providers.queries.mir_built = dump::mir_built;
                providers.queries.optimized_mir = replace::optimized_mir;
            }),
        };
    }

    fn after_analysis<'tcx>(
        &mut self,
        _compiler: &interface::Compiler,
        tcx: TyCtxt<'tcx>,
    ) -> Compilation {
        if let Some(dir) = &self.dump {
            dump::write_all(tcx, dir);
        }
        Compilation::Continue
    }
}

fn arg_value<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].as_str())
}

/// Makes the runtime crate visible to the crate being compiled without the
/// crate declaring it, and lets the final link find its dependencies.
fn add_runtime_args(args: &mut Vec<String>) {
    let Some(runtime) = env::var_os(replace::RUNTIME_VAR) else {
        return;
    };
    if arg_value(args, "--crate-name") == Some(replace::RUNTIME_CRATE) {
        return; // Never make the runtime depend on itself.
    }
    args.push("-Zunstable-options".into());
    // Cargo may store metadata in a separate `.rmeta` next to a stub
    // `.rlib`; rustc needs both, so the variable is a path list.
    for file in env::split_paths(&runtime) {
        args.push("--extern".into());
        args.push(format!(
            "force:{}={}",
            replace::RUNTIME_CRATE,
            file.display()
        ));
    }
    if let Some(deps) = env::var_os(replace::RUNTIME_DEPS_VAR) {
        for dir in env::split_paths(&deps) {
            args.push("-L".into());
            args.push(format!("dependency={}", dir.display()));
        }
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

        // Cargo's information queries compile nothing, so they get plain
        // rustc behavior.
        let info_query = args
            .iter()
            .any(|arg| arg == "-vV" || arg.starts_with("--print"));
        if info_query {
            rustc_driver::compiler_entrypoint(
                &args,
                &mut Driver {
                    dump: None,
                    replace: false,
                },
            );
            return ExitCode::SUCCESS;
        }

        let replacements = match env::var(replace::VAR).map(|spec| replace::parse(&spec)) {
            Ok(Ok(replacements)) => replacements,
            Ok(Err(message)) => {
                early_dcx.early_fatal(message);
            }
            Err(_) => Vec::new(),
        };
        let replace = !replacements.is_empty();
        replace::REPLACEMENTS
            .set(replacements)
            .expect("the driver compiles one crate per process");
        add_runtime_args(&mut args);

        let dump = env::var_os(dump::VAR).map(PathBuf::from);
        rustc_driver::compiler_entrypoint(&args, &mut Driver { dump, replace });
        ExitCode::SUCCESS
    })
}
