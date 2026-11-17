//! What the translator does with every compiler intrinsic of the pinned
//! toolchain (`core::intrinsics`, including its SIMD and reflection
//! modules), as a table. A unit test reads the toolchain's library source
//! and fails if an intrinsic is missing here or listed here but gone, so a
//! toolchain update can't add one silently.
//!
//! MIR's own constructs (statements, operations, casts, projections,
//! terminators) are covered by exhaustive matches in `translate`, which the
//! compiler checks the same way; `docs/coverage.md` lists both.

use std::{fs, panic, path::Path};

use rustc_hir::def::DefKind;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::{DefId, LOCAL_CRATE};

use crate::translate;

/// How the translator treats an intrinsic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Status {
    /// Translated into an exact kernel operation.
    Supported,
    /// rustc replaces calls with MIR constructs in optimized MIR (named
    /// here), so callees reach the translator as those constructs. Direct
    /// calls in a loop's own function, whose built MIR is translated, are
    /// planned.
    Lowered(&'static str),
    /// Not translated yet; an exact GPU version is possible, as described.
    Planned(&'static str),
    /// Only under a relaxed numerical contract: the GPU can't reproduce the
    /// CPU's result bit for bit, for the reason given.
    Relaxed(&'static str),
    /// Refused by design, for the reason given.
    Refused(&'static str),
}

use Status::*;

const ATOMIC: Status = Refused("shares memory between threads, while a map's elements are independent");
const ADDRESS: Status = Refused("uses CPU addresses, which mean nothing on the GPU");
const VOLATILE: Status = Refused("accesses memory as volatile");
const HOOK: Status = Refused("is a compiler or debugger hook");
const CONST_ONLY: Status = Refused("exists only during constant evaluation");
const FOREIGN: Status = Refused("belongs to another kind of target");
const REFLECTION: Status = Lowered("a constant, evaluated at compile time");
const PLATFORM_MATH: Status =
    Relaxed("the CPU calls the platform's math library, whose results the GPU can't reproduce bit for bit");
const FAST_MATH: Status =
    Relaxed("lets the compiler reorder or approximate float math, so the CPU's result depends on its code generation");
const WIDE_FLOATS: Status = Planned("`f16` and `f128` numbers, which the kernel doesn't have yet");
const LANES: Status = Planned("SIMD vectors, as one value per lane");

/// Every intrinsic, by name. Names in more than one module appear once.
pub(crate) const INTRINSICS: &[(&str, Status)] = &[
    // Atomics and fences.
    ("atomic_cxchg", ATOMIC),
    ("atomic_cxchgweak", ATOMIC),
    ("atomic_load", ATOMIC),
    ("atomic_store", ATOMIC),
    ("atomic_xchg", ATOMIC),
    ("atomic_xadd", ATOMIC),
    ("atomic_xsub", ATOMIC),
    ("atomic_and", ATOMIC),
    ("atomic_nand", ATOMIC),
    ("atomic_or", ATOMIC),
    ("atomic_xor", ATOMIC),
    ("atomic_max", ATOMIC),
    ("atomic_min", ATOMIC),
    ("atomic_umin", ATOMIC),
    ("atomic_umax", ATOMIC),
    ("atomic_fence", ATOMIC),
    ("atomic_singlethreadfence", ATOMIC),
    // Hints and hooks.
    ("prefetch_read_data", Planned("a cache hint with no effect on results: nothing")),
    ("prefetch_write_data", Planned("a cache hint with no effect on results: nothing")),
    ("prefetch_read_instruction", Planned("a cache hint with no effect on results: nothing")),
    ("prefetch_write_instruction", Planned("a cache hint with no effect on results: nothing")),
    ("breakpoint", HOOK),
    ("codeview_annotation", HOOK),
    ("rustc_peek", HOOK),
    ("abort_immediate", Planned("stop the thread and send the call back to the CPU, which aborts")),
    ("unreachable", Lowered("an `Unreachable` terminator")),
    ("assume", Supported),
    ("cold_path", Supported),
    ("select_unpredictable", Planned("a select")),
    ("assert_inhabited", Lowered("nothing, or a panic, after rustc's checks")),
    ("assert_zero_valid", Lowered("nothing, or a panic, after rustc's checks")),
    ("assert_mem_uninitialized_valid", Lowered("nothing, or a panic, after rustc's checks")),
    ("caller_location", Refused("points to source-location data; only panic paths use it, and the CPU takes those")),
    ("forget", Lowered("nothing")),
    // `mem::transmute` is the intrinsic itself, so a loop body, translated
    // before rustc lowers it to a `Transmute` cast, calls it.
    ("transmute", Supported),
    ("transmute_unchecked", Lowered("a `Transmute` cast")),
    ("needs_drop", Lowered("a constant")),
    // Pointers and memory.
    ("offset", Lowered("an `Offset` operation")),
    ("arith_offset", ADDRESS),
    ("slice_get_unchecked", Planned("an element of a read-only slice, as indexing without the check")),
    ("ptr_mask", ADDRESS),
    ("volatile_copy_nonoverlapping_memory", VOLATILE),
    ("volatile_copy_memory", VOLATILE),
    ("volatile_set_memory", VOLATILE),
    ("volatile_load", VOLATILE),
    ("volatile_store", VOLATILE),
    ("unaligned_volatile_load", VOLATILE),
    ("unaligned_volatile_store", VOLATILE),
    ("read_via_copy", Lowered("a copy")),
    ("write_via_move", Lowered("an assignment")),
    ("nontemporal_store", Refused("is a store hint for CPU caches with no GPU meaning")),
    ("ptr_offset_from", ADDRESS),
    ("ptr_offset_from_unsigned", ADDRESS),
    ("ptr_guaranteed_cmp", ADDRESS),
    ("raw_eq", Planned("compares the bytes of two values")),
    ("compare_bytes", ADDRESS),
    ("typed_swap_nonoverlapping", Planned("swaps two values held as slots")),
    ("aggregate_raw_ptr", Lowered("an `Aggregate` of a raw pointer")),
    ("ptr_metadata", Lowered("a `PtrMetadata` operation")),
    ("copy_nonoverlapping", Lowered("a `CopyNonOverlapping` statement")),
    ("copy", Planned("copies between values held as slots")),
    ("write_bytes", Planned("fills a value held as slots")),
    // Floats.
    ("sqrtf16", WIDE_FLOATS),
    ("sqrtf32", Supported),
    ("sqrtf64", Supported),
    ("sqrtf128", WIDE_FLOATS),
    ("powif16", WIDE_FLOATS),
    ("powif32", Relaxed("LLVM multiplies in an order that depends on optimization")),
    ("powif64", Relaxed("LLVM multiplies in an order that depends on optimization")),
    ("powif128", WIDE_FLOATS),
    ("sin", PLATFORM_MATH),
    ("cos", PLATFORM_MATH),
    ("powf16", WIDE_FLOATS),
    ("powf32", PLATFORM_MATH),
    ("powf64", PLATFORM_MATH),
    ("powf128", WIDE_FLOATS),
    ("exp", PLATFORM_MATH),
    ("exp2", PLATFORM_MATH),
    ("log", PLATFORM_MATH),
    ("log10", PLATFORM_MATH),
    ("log2", PLATFORM_MATH),
    ("fmaf16", WIDE_FLOATS),
    ("fmaf32", Supported),
    ("fmaf64", Supported),
    ("fmaf128", WIDE_FLOATS),
    ("fmuladdf16", WIDE_FLOATS),
    ("fmuladdf32", Relaxed("fuses the multiply and add or not, as the CPU's code generation chooses")),
    ("fmuladdf64", Relaxed("fuses the multiply and add or not, as the CPU's code generation chooses")),
    ("fmuladdf128", WIDE_FLOATS),
    ("floorf16", WIDE_FLOATS),
    ("floorf32", Supported),
    ("floorf64", Supported),
    ("floorf128", WIDE_FLOATS),
    ("ceilf16", WIDE_FLOATS),
    ("ceilf32", Supported),
    ("ceilf64", Supported),
    ("ceilf128", WIDE_FLOATS),
    ("truncf16", WIDE_FLOATS),
    ("truncf32", Supported),
    ("truncf64", Supported),
    ("truncf128", WIDE_FLOATS),
    ("round_ties_even_f16", WIDE_FLOATS),
    ("round_ties_even_f32", Planned("exact: round half to even")),
    ("round_ties_even_f64", Planned("exact: round half to even")),
    ("round_ties_even_f128", WIDE_FLOATS),
    ("roundf16", WIDE_FLOATS),
    ("roundf32", Planned("exact: round half away from zero")),
    ("roundf64", Planned("exact: round half away from zero")),
    ("roundf128", WIDE_FLOATS),
    ("fadd_fast", FAST_MATH),
    ("fsub_fast", FAST_MATH),
    ("fmul_fast", FAST_MATH),
    ("fdiv_fast", FAST_MATH),
    ("frem_fast", FAST_MATH),
    ("float_to_int_unchecked", Planned("a float-to-integer cast, undefined out of range")),
    ("fadd_algebraic", FAST_MATH),
    ("fsub_algebraic", FAST_MATH),
    ("fmul_algebraic", FAST_MATH),
    ("fdiv_algebraic", FAST_MATH),
    ("frem_algebraic", FAST_MATH),
    ("minimum_number_nsz_f16", WIDE_FLOATS),
    ("minimum_number_nsz_f32", Supported),
    ("minimum_number_nsz_f64", Supported),
    ("minimum_number_nsz_f128", WIDE_FLOATS),
    ("minimumf16", WIDE_FLOATS),
    ("minimumf32", Planned("exact: the minimum, NaN if either is NaN")),
    ("minimumf64", Planned("exact: the minimum, NaN if either is NaN")),
    ("minimumf128", WIDE_FLOATS),
    ("maximum_number_nsz_f16", WIDE_FLOATS),
    ("maximum_number_nsz_f32", Supported),
    ("maximum_number_nsz_f64", Supported),
    ("maximum_number_nsz_f128", WIDE_FLOATS),
    ("maximumf16", WIDE_FLOATS),
    ("maximumf32", Planned("exact: the maximum, NaN if either is NaN")),
    ("maximumf64", Planned("exact: the maximum, NaN if either is NaN")),
    ("maximumf128", WIDE_FLOATS),
    ("fabs", Supported),
    ("copysignf16", WIDE_FLOATS),
    ("copysignf32", Planned("exact: the sign bit")),
    ("copysignf64", Planned("exact: the sign bit")),
    ("copysignf128", WIDE_FLOATS),
    // Integers.
    ("integer_min", Planned("exact")),
    ("integer_max", Planned("exact")),
    ("ctpop", Supported),
    ("ctlz", Supported),
    ("ctlz_nonzero", Supported),
    ("cttz", Supported),
    ("cttz_nonzero", Supported),
    ("bswap", Planned("exact: a byte permutation")),
    ("bitreverse", Planned("exact: `brev`")),
    ("three_way_compare", Lowered("a `Cmp` operation")),
    ("disjoint_bitor", Planned("an `or`")),
    ("add_with_overflow", Lowered("an `AddWithOverflow` operation")),
    ("sub_with_overflow", Lowered("a `SubWithOverflow` operation")),
    ("mul_with_overflow", Lowered("a `MulWithOverflow` operation")),
    ("carrying_mul_add", Planned("a double-width product (`mul.hi`)")),
    ("exact_div", Supported),
    ("unchecked_div", Supported),
    ("unchecked_rem", Supported),
    ("unchecked_shl", Supported),
    ("unchecked_shr", Supported),
    ("unchecked_add", Supported),
    ("unchecked_sub", Supported),
    ("unchecked_mul", Supported),
    ("rotate_left", Supported),
    ("rotate_right", Supported),
    ("wrapping_add", Supported),
    ("wrapping_sub", Supported),
    ("wrapping_mul", Supported),
    ("saturating_add", Planned("exact: compare and select")),
    ("saturating_sub", Planned("exact: compare and select")),
    ("unchecked_funnel_shl", Planned("exact: a funnel shift")),
    ("unchecked_funnel_shr", Planned("exact: a funnel shift")),
    ("carryless_mul", Planned("exact: shifts and xors")),
    ("discriminant_value", Lowered("a `Discriminant` operation")),
    // Control and evaluation.
    ("catch_unwind", Refused("catches unwinding, which a kernel never does")),
    ("black_box", Planned("the identity")),
    ("const_eval_select", Planned("calls its run-time function")),
    ("is_val_statically_known", Planned("`false`, which its contract allows")),
    ("ub_checks", Lowered("a `RuntimeChecks` operand")),
    ("overflow_checks", Lowered("a `RuntimeChecks` operand")),
    ("const_allocate", CONST_ONLY),
    ("const_deallocate", CONST_ONLY),
    ("const_make_global", CONST_ONLY),
    ("contract_check_requires", Planned("a check, as `assert!` is")),
    ("contract_check_ensures", Planned("a check, as `assert!` is")),
    ("vtable_size", Refused("reads a trait object's vtable")),
    ("vtable_align", Refused("reads a trait object's vtable")),
    // Layout and reflection, constant after monomorphization.
    ("size_of", Lowered("a constant")),
    ("align_of", Lowered("a constant")),
    ("offset_of", Lowered("a constant")),
    ("field_offset", REFLECTION),
    ("variant_count", Lowered("a constant")),
    ("size_of_val", Planned("a constant, or the length times the element size for a read-only slice")),
    ("align_of_val", Lowered("a constant")),
    ("type_name", REFLECTION),
    ("field_representing_type_actual_type_id", REFLECTION),
    ("field_representing_type_name", REFLECTION),
    ("field_representing_type_offset", REFLECTION),
    ("type_id", REFLECTION),
    ("type_id_type_of", REFLECTION),
    ("type_id_eq", REFLECTION),
    ("type_id_vtable", REFLECTION),
    ("type_id_is_signed", REFLECTION),
    ("type_id_array_len", REFLECTION),
    ("type_id_element_ty", REFLECTION),
    ("type_id_size_of", REFLECTION),
    ("type_id_variants", REFLECTION),
    ("type_id_variant_name", REFLECTION),
    ("type_id_variant_non_exhaustive", REFLECTION),
    ("type_id_fields", REFLECTION),
    ("type_id_non_exhaustive", REFLECTION),
    ("type_id_generics", REFLECTION),
    ("type_id_field_representing_type", REFLECTION),
    ("type_id_points_to", REFLECTION),
    ("type_id_points_mutably", REFLECTION),
    ("type_id_function_ptr", REFLECTION),
    // Other targets and compiler extensions.
    ("autodiff", FOREIGN),
    ("offload", FOREIGN),
    ("offload_get_num_devices", FOREIGN),
    ("gpu_launch_sized_workgroup_mem", FOREIGN),
    ("amdgpu_dispatch_ptr", FOREIGN),
    ("sve_tuple_create2", FOREIGN),
    ("sve_tuple_create3", FOREIGN),
    ("sve_tuple_create4", FOREIGN),
    ("sve_tuple_get", FOREIGN),
    ("sve_tuple_set", FOREIGN),
    ("va_arg", Refused("reads C variadic arguments")),
    ("va_copy", Refused("reads C variadic arguments")),
    ("va_end", Refused("reads C variadic arguments")),
    ("return_address", ADDRESS),
    // SIMD.
    ("simd_insert", LANES),
    ("simd_extract", LANES),
    ("simd_insert_dyn", LANES),
    ("simd_extract_dyn", LANES),
    ("simd_splat", LANES),
    ("simd_add", LANES),
    ("simd_sub", LANES),
    ("simd_mul", LANES),
    ("simd_div", LANES),
    ("simd_rem", LANES),
    ("simd_shl", LANES),
    ("simd_shr", LANES),
    ("simd_funnel_shl", LANES),
    ("simd_funnel_shr", LANES),
    ("simd_carryless_mul", LANES),
    ("simd_and", LANES),
    ("simd_or", LANES),
    ("simd_xor", LANES),
    ("simd_cast", LANES),
    ("simd_as", LANES),
    ("simd_neg", LANES),
    ("simd_fabs", LANES),
    ("simd_minimum_number_nsz", LANES),
    ("simd_maximum_number_nsz", LANES),
    ("simd_eq", LANES),
    ("simd_ne", LANES),
    ("simd_lt", LANES),
    ("simd_le", LANES),
    ("simd_gt", LANES),
    ("simd_ge", LANES),
    ("simd_shuffle", LANES),
    ("simd_gather", ADDRESS),
    ("simd_scatter", ADDRESS),
    ("simd_masked_load", ADDRESS),
    ("simd_masked_store", ADDRESS),
    ("simd_saturating_add", LANES),
    ("simd_saturating_sub", LANES),
    ("simd_reduce_add_ordered", LANES),
    ("simd_reduce_add_unordered", FAST_MATH),
    ("simd_reduce_mul_ordered", LANES),
    ("simd_reduce_mul_unordered", FAST_MATH),
    ("simd_reduce_all", LANES),
    ("simd_reduce_any", LANES),
    ("simd_reduce_max", LANES),
    ("simd_reduce_min", LANES),
    ("simd_reduce_and", LANES),
    ("simd_reduce_or", LANES),
    ("simd_reduce_xor", LANES),
    ("simd_bitmask", LANES),
    ("simd_select", LANES),
    ("simd_select_bitmask", LANES),
    ("simd_arith_offset", ADDRESS),
    ("simd_cast_ptr", ADDRESS),
    ("simd_expose_provenance", ADDRESS),
    ("simd_with_exposed_provenance", ADDRESS),
    ("simd_bswap", LANES),
    ("simd_bitreverse", LANES),
    ("simd_ctlz", LANES),
    ("simd_ctpop", LANES),
    ("simd_cttz", LANES),
    ("simd_ceil", LANES),
    ("simd_floor", LANES),
    ("simd_round", LANES),
    ("simd_round_ties_even", LANES),
    ("simd_trunc", LANES),
    ("simd_fsqrt", LANES),
    ("simd_fma", LANES),
    ("simd_relaxed_fma", Relaxed("fuses the multiply and add or not, as the CPU's code generation chooses")),
    ("simd_fsin", PLATFORM_MATH),
    ("simd_fcos", PLATFORM_MATH),
    ("simd_fexp", PLATFORM_MATH),
    ("simd_fexp2", PLATFORM_MATH),
    ("simd_flog10", PLATFORM_MATH),
    ("simd_flog2", PLATFORM_MATH),
    ("simd_flog", PLATFORM_MATH),
];

/// The status of the intrinsic `name`.
pub(crate) fn status(name: &str) -> Option<Status> {
    INTRINSICS.iter().find(|(known, _)| *known == name).map(|&(_, status)| status)
}

/// Why a call to the intrinsic `name` isn't translated, as a message
/// fragment after "uses the intrinsic `name`".
pub(crate) fn reason(name: &str) -> String {
    match status(name) {
        Some(Supported) => String::new(),
        Some(Lowered(construct)) => format!(", which rustc lowers to {construct} only in optimized MIR"),
        Some(Planned(how)) => format!(", which isn't translated yet ({how})"),
        Some(Relaxed(why)) => format!(", which isn't exact on the GPU: {why}"),
        Some(Refused(why)) => format!(", which {why}"),
        None => ", which the coverage table doesn't list".into(),
    }
}

/// The coverage census: `CUPRITE_COVERAGE=<dir>` writes
/// `<dir>/<crate>-<id>.txt` with one line per function and closure of the
/// crate. The id tells apart compilations of one crate with different
/// features (for build scripts and for the program), which both count:
///
/// ```text
/// path: translated
/// path: refused: <reason>
/// path: skipped: <reason>
/// ```
///
/// Non-generic functions over element data (`translate::element_data`) are
/// translated whole, as a kernel function of their parameters, inlining
/// every call; the others are skipped. Summed over many crates, the refusal
/// reasons rank what the translator lacks.
pub(crate) const VAR: &str = "CUPRITE_COVERAGE";

pub(crate) fn write_census(tcx: TyCtxt<'_>, dir: &Path) {
    let krate = tcx.crate_name(LOCAL_CRATE);
    let mut lines: Vec<String> = tcx
        .hir_body_owners()
        .filter(|&def_id| matches!(tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn | DefKind::Closure))
        .map(|def_id| format!("{krate}::{}: {}", tcx.def_path_str(def_id), census_line(tcx, def_id.to_def_id())))
        .collect();
    lines.sort();
    // Build scripts compile throwaway crates, such as `autocfg`'s probes.
    if lines.is_empty() {
        return;
    }
    fs::create_dir_all(dir).expect("cannot create the coverage directory");
    let id = tcx.stable_crate_id(LOCAL_CRATE).as_u64();
    fs::write(dir.join(format!("{krate}-{id:016x}.txt")), lines.join("\n") + "\n").expect("cannot write the coverage census");
}

fn census_line(tcx: TyCtxt<'_>, def_id: DefId) -> String {
    // A closure has synthetic generic parameters of its own; it is generic
    // only if the function it is in is.
    if tcx.generics_of(tcx.typeck_root_def_id(def_id)).requires_monomorphization(tcx) {
        return "skipped: generic".into();
    }
    if !tcx.is_mir_available(def_id) {
        return "skipped: no MIR".into();
    }
    let body = tcx.optimized_mir(def_id);
    let signature = (0..=body.arg_count).map(rustc_middle::mir::Local::from_usize);
    for local in signature {
        if let Err(reason) = translate::element_data(tcx, body.local_decls[local].ty, 0) {
            return format!("skipped: {reason}");
        }
    }
    // A translator bug shows up as a refusal instead of ending the census.
    match panic::catch_unwind(panic::AssertUnwindSafe(|| translate::check_function(tcx, def_id, body))) {
        Ok(Ok(())) => "translated".into(),
        Ok(Err(reason)) => format!("refused: {reason}"),
        Err(_) => "refused: crashed the translator".into(),
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, fs, path::Path};

    use super::*;

    /// The intrinsics the toolchain's `core` declares: each function after
    /// a `#[rustc_intrinsic]` attribute, outside comments.
    fn declared() -> BTreeSet<String> {
        let dir = Path::new(env!("CUPRITE_SYSROOT")).join("lib/rustlib/src/rust/library/core/src/intrinsics");
        let mut files = Vec::new();
        let mut pending = vec![dir];
        while let Some(dir) = pending.pop() {
            for entry in fs::read_dir(&dir).unwrap_or_else(|error| panic!("{}: {error}", dir.display())) {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    pending.push(path);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    files.push(path);
                }
            }
        }
        let mut names = BTreeSet::new();
        for file in files {
            let text = fs::read_to_string(&file).unwrap();
            let mut after_attribute = false;
            for line in text.lines().map(str::trim) {
                if line.starts_with("//") {
                    continue;
                }
                if line == "#[rustc_intrinsic]" {
                    after_attribute = true;
                    continue;
                }
                if after_attribute {
                    if let Some(rest) = line.split("fn ").nth(1).filter(|_| !line.starts_with('#')) {
                        let name: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
                        names.insert(name);
                        after_attribute = false;
                    }
                }
            }
        }
        names
    }

    #[test]
    fn every_intrinsic_of_the_toolchain_is_classified() {
        let declared = declared();
        assert!(declared.len() > 250, "found only {} intrinsics", declared.len());
        let listed: BTreeSet<String> = INTRINSICS.iter().map(|(name, _)| name.to_string()).collect();
        let missing: Vec<_> = declared.difference(&listed).collect();
        let gone: Vec<_> = listed.difference(&declared).collect();
        assert!(missing.is_empty(), "intrinsics to classify: {missing:?}");
        assert!(gone.is_empty(), "intrinsics no longer declared: {gone:?}");
        assert_eq!(listed.len(), INTRINSICS.len(), "an intrinsic is listed twice");
    }

    /// The translator maps exactly the intrinsics listed as supported.
    #[test]
    fn the_translator_maps_exactly_the_supported_intrinsics() {
        for &(name, status) in INTRINSICS {
            let mapped = crate::translate::intrinsic(name).is_some();
            assert_eq!(mapped, status == Supported, "{name}: {status:?}");
        }
    }
}
