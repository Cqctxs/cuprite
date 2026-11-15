//! Kernels for iteration shapes, around element functions translated from
//! the program.

use crate::{builder::Builder, BinOp, Const, Function, Invalid, Kernel, Operand, Rvalue, Special, Ty};

/// The parameters an in-place map kernel takes before its stages' scalars:
/// the data, its length, and with `repeated` the pass count.
pub fn in_place_map_prefix(repeated: bool) -> &'static [Ty] {
    if repeated {
        &[Ty::Ptr, Ty::U64, Ty::U32]
    } else {
        &[Ty::Ptr, Ty::U64]
    }
}

/// `data[i] = stages(data[i])` for every `i < len`, applying the stages in
/// order, and with `repeated` doing all of that `passes` times. Each element
/// is independent, so passes can run per element: `passes` launches over
/// the slice and one launch looping per element compute the same.
///
/// Each stage takes the element and then scalars of its own, and returns
/// the new element. The kernel's parameters are
/// [`in_place_map_prefix`], then every stage's scalars in order. One thread
/// handles one element, in a `Grid::Linear` launch of at most `u32::MAX`
/// threads.
pub fn in_place_map(name: &str, element: Ty, stages: &[&Function], repeated: bool) -> Result<Kernel, Invalid> {
    for stage in stages {
        let params = stage.params();
        if params.first() != Some(&element) || stage.result_ty() != Some(element) {
            return Err(Invalid(format!(
                "stage `{}` doesn't map one {element:?} to another",
                stage.kernel.name
            )));
        }
        if let Some(ty) = params[1..].iter().find(|ty| !ty.is_passable()) {
            return Err(Invalid(format!("stage `{}` takes a {ty:?}, which can't be passed", stage.kernel.name)));
        }
    }
    let mut types = in_place_map_prefix(repeated).to_vec();
    for stage in stages {
        types.extend_from_slice(&stage.params()[1..]);
    }
    let (mut b, params) = Builder::new(name, &types);
    let (data, len) = (params[0], params[1]);
    let mut scalars = params[in_place_map_prefix(repeated).len()..].iter().copied();
    let stage_scalars: Vec<Vec<Operand>> = stages
        .iter()
        .map(|stage| (1..stage.params().len()).map(|_| scalars.next().unwrap().into()).collect())
        .collect();

    let i = b.special(Special::X);
    let wide = b.cast(i, Ty::U64);
    let inside = b.binary(BinOp::Lt, wide, len);
    let (work, done) = (b.block(), b.block());
    b.branch(inside, work, done);

    b.switch_to(work);
    let address = b.offset(data, wide, element);
    let value = b.load(address, element);
    let apply = |b: &mut Builder| {
        for (stage, scalars) in stages.iter().zip(&stage_scalars) {
            let mut args = vec![value.into()];
            args.extend_from_slice(scalars);
            let result = b.inline(stage, &args).expect("checked above");
            b.set(value, Rvalue::Use(result.into()));
        }
    };
    if repeated {
        let passes = params[2];
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
    b.store(address, value);
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
