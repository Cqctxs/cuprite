//! MIR dumps for inspection: `CUPRITE_DUMP_MIR=<dir>` writes each function
//! body as `<dir>/<crate>/<path>.built.mir` (right after MIR construction)
//! and `<path>.optimized.mir` (what codegen consumes, which depends on the
//! optimization level and includes any Cuprite rewrite).

use std::{collections::BTreeMap, fs, path::Path, sync::Mutex};

use rustc_data_structures::steal::Steal;
use rustc_hir::def::DefKind;
use rustc_middle::{
    mir::{pretty::MirWriter, Body},
    ty::TyCtxt,
};
use rustc_span::def_id::{LocalDefId, LOCAL_CRATE};

pub(crate) const VAR: &str = "CUPRITE_DUMP_MIR";

/// Built MIR captured by the query override, keyed by function path.
/// Query providers are plain function pointers, so this has to be global.
static BUILT: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());

fn pretty<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> String {
    let mut out = Vec::new();
    MirWriter::new(tcx)
        .write_mir_fn(body, &mut out)
        .expect("writing to a Vec cannot fail");
    String::from_utf8(out).expect("MIR pretty-printing produces UTF-8")
}

fn is_function(tcx: TyCtxt<'_>, def_id: LocalDefId) -> bool {
    matches!(
        tcx.def_kind(def_id),
        DefKind::Fn | DefKind::AssocFn | DefKind::Closure
    )
}

/// Wraps rustc's own `mir_built`: records the body, then returns it
/// unchanged.
pub(crate) fn mir_built<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> &'tcx Steal<Body<'tcx>> {
    let built = (rustc_interface::DEFAULT_QUERY_PROVIDERS.queries.mir_built)(tcx, def_id);
    if is_function(tcx, def_id) {
        let text = pretty(tcx, &built.borrow());
        BUILT.lock().unwrap().insert(tcx.def_path_str(def_id), text);
    }
    built
}

/// Writes both stages for every function in the crate being compiled.
pub(crate) fn write_all(tcx: TyCtxt<'_>, dir: &Path) {
    let crate_dir = dir.join(tcx.crate_name(LOCAL_CRATE).as_str());
    fs::create_dir_all(&crate_dir).expect("cannot create the MIR dump directory");
    let mut built = BUILT.lock().unwrap();
    for def_id in tcx.hir_body_owners() {
        if !is_function(tcx, def_id) {
            continue;
        }
        let path = tcx.def_path_str(def_id);
        let optimized = pretty(tcx, tcx.optimized_mir(def_id));
        write(&crate_dir, &path, "optimized", &optimized);
        if let Some(text) = built.remove(&path) {
            write(&crate_dir, &path, "built", &text);
        }
    }
}

fn write(dir: &Path, path: &str, stage: &str, text: &str) {
    // Function paths contain `::`, `{closure#0}` and similar; keep them
    // readable but valid as Windows file names.
    let name: String = path
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '#' {
                c
            } else {
                '.'
            }
        })
        .collect();
    fs::write(dir.join(format!("{name}.{stage}.mir")), text).expect("cannot write a MIR dump");
}
