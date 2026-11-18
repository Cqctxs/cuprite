//! Kernels for iteration shapes, around element functions translated from
//! the program.

use crate::{builder::Builder, layout::Layout, BinOp, Const, Function, Invalid, Kernel, Local, Operand, Rvalue, Special, Ty};

/// An in-place map kernel's parameters before its stages' scalars: the
/// first output slice, its length, the failure flag, a pointer per other
/// output slice and per input slice, and with `repeated` the pass count.
pub fn in_place_map_prefix(outputs: usize, inputs: usize, repeated: bool) -> Vec<Ty> {
    let mut prefix = vec![Ty::Ptr, Ty::U64, Ty::Ptr];
    prefix.extend(std::iter::repeat_n(Ty::Ptr, outputs - 1 + inputs));
    if repeated {
        prefix.push(Ty::U32);
    }
    prefix
}

/// `outputs[..][i] = stages(outputs[..][i], inputs[..][i])` for every
/// `i < len`, applying the stages in order, and with `repeated` doing all
/// of that `passes` times. Each element is independent, so passes can run
/// per element: `passes` launches over the slices and one launch looping
/// per element compute the same.
///
/// Each slice's element holds the numbers its [`Layout`] lists. Each stage
/// takes every output's numbers, in order, then every input's, then
/// scalars of its own, and returns the outputs' new numbers in the same
/// order. The inputs are read-only, and every slice holds at least `len`
/// elements. The kernel's parameters are [`in_place_map_prefix`], then
/// every stage's scalars in order. One thread handles one element, in a
/// `Grid::Linear` launch of at most `u32::MAX` threads.
///
/// Where a stage fails (`Terminator::Fail`), the thread writes 1 to the
/// `u32` failure flag and stops. Earlier stores are harmless: the CPU would
/// have panicked there, so the caller discards every result and runs the
/// original code instead.
pub fn in_place_map(
    name: &str,
    outputs: &[Layout],
    inputs: &[Layout],
    stages: &[&Function],
    repeated: bool,
) -> Result<Kernel, Invalid> {
    if outputs.is_empty() {
        return Err(Invalid("a map writes at least one slice".into()));
    }
    for layout in outputs.iter().chain(inputs) {
        layout.validate()?;
    }
    let written: Vec<Ty> = outputs.iter().flat_map(Layout::types).collect();
    let read: Vec<Ty> = inputs.iter().flat_map(Layout::types).collect();
    let taken = written.len() + read.len();
    for stage in stages {
        let params = stage.params();
        if params.get(..written.len()) != Some(&written[..]) || stage.result_types() != written {
            return Err(Invalid(format!("stage `{}` doesn't map {written:?} to new values", stage.kernel.name)));
        }
        if params.get(written.len()..taken) != Some(&read[..]) {
            return Err(Invalid(format!("stage `{}` doesn't take the inputs {read:?}", stage.kernel.name)));
        }
        if let Some(ty) = params[taken..].iter().find(|ty| !ty.is_passable()) {
            return Err(Invalid(format!("stage `{}` takes a {ty:?}, which can't be passed", stage.kernel.name)));
        }
    }
    let prefix = in_place_map_prefix(outputs.len(), inputs.len(), repeated);
    let mut types = prefix.clone();
    for stage in stages {
        types.extend_from_slice(&stage.params()[taken..]);
    }
    let (mut b, params) = Builder::new(name, &types);
    let (len, flag) = (params[1], params[2]);
    let output_pointers: Vec<Local> = std::iter::once(params[0]).chain(params[3..2 + outputs.len()].iter().copied()).collect();
    let input_pointers = &params[2 + outputs.len()..2 + outputs.len() + inputs.len()];
    let mut scalars = params[prefix.len()..].iter().copied();
    let stage_scalars: Vec<Vec<Operand>> = stages
        .iter()
        .map(|stage| (taken..stage.params().len()).map(|_| scalars.next().unwrap().into()).collect())
        .collect();

    let i = b.special(Special::X);
    let wide = b.cast(i, Ty::U64);
    let inside = b.binary(BinOp::Lt, wide, len);
    let (work, done, failed) = (b.block(), b.block(), b.block());
    b.branch(inside, work, done);

    b.switch_to(failed);
    b.store(flag, Const::int(Ty::U32, 1));
    b.ret();

    b.switch_to(work);
    // Each number's address in this thread's element, and its value.
    let addresses = |b: &mut Builder, pointers: &[Local], layouts: &[Layout]| -> Vec<(Local, Local)> {
        let mut numbers = Vec::new();
        for (&pointer, layout) in pointers.iter().zip(layouts) {
            let element = b.assign(Rvalue::Offset(pointer.into(), wide.into(), layout.size));
            for &(offset, ty) in &layout.leaves {
                let address = b.assign(Rvalue::Offset(element.into(), Const::int(Ty::U64, offset.into()).into(), 1));
                let value = b.load(address, ty);
                numbers.push((address, value));
            }
        }
        numbers
    };
    let values = addresses(&mut b, &output_pointers, outputs);
    let input_values: Vec<Operand> =
        addresses(&mut b, input_pointers, inputs).into_iter().map(|(_, value)| value.into()).collect();
    let apply = |b: &mut Builder| {
        for (stage, scalars) in stages.iter().zip(&stage_scalars) {
            let mut args: Vec<Operand> = values.iter().map(|&(_, value)| value.into()).collect();
            args.extend_from_slice(&input_values);
            args.extend_from_slice(scalars);
            let results = b.inline(stage, &args, Some(failed));
            for (&(_, value), result) in values.iter().zip(results) {
                b.set(value, Rvalue::Use(result.into()));
            }
        }
    };
    if repeated {
        let passes = params[prefix.len() - 1];
        let pass = b.copy(Const::int(Ty::U32, 0));
        let (head, body, exit) = (b.block(), b.block(), b.block());
        b.goto(head);
        b.switch_to(head);
        let more = b.binary(BinOp::Lt, pass, passes);
        b.branch(more, body, exit);
        b.switch_to(body);
        apply(&mut b);
        b.set(pass, Rvalue::Binary(BinOp::Add, pass.into(), Const::int(Ty::U32, 1).into()));
        b.goto(head);
        b.switch_to(exit);
    } else {
        apply(&mut b);
    }
    for &(address, value) in &values {
        b.store(address, value);
    }
    b.goto(done);
    b.switch_to(done);
    b.ret();
    b.finish()
}

/// A PTX entry name for a Rust path: `a::b` becomes `a__b`.
pub fn entry_name(path: &str) -> String {
    path.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}
