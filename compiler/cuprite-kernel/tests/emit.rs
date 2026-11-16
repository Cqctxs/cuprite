//! Validation and emitted text, without a GPU.

use cuprite_kernel::{
    builder::Builder, llvm, ptx, BinOp, Block, BlockId, Const, Intrinsic, Kernel, Local, Operand, Rvalue, Special,
    Statement, Terminator, Ty,
};

fn one_block(params: &[Ty], locals: &[Ty], statements: Vec<Statement>, terminator: Terminator) -> Kernel {
    Kernel {
        name: "k".into(),
        params: params.len(),
        locals: params.iter().chain(locals).copied().collect(),
        blocks: vec![Block { statements, terminator }],
    }
}

fn local(i: u32) -> Operand {
    Operand::Local(Local(i))
}

fn rejected(kernel: Kernel, reason: &str) {
    let error = kernel.validate().expect_err(reason).0;
    assert!(error.contains(reason), "expected {reason:?} in {error:?}");
}

#[test]
fn validation_rejects_malformed_kernels() {
    rejected(one_block(&[Ty::U8], &[], vec![], Terminator::Return), "can't be passed");
    rejected(one_block(&[Ty::Bool], &[], vec![], Terminator::Return), "can't be passed");
    let float_rem = Statement::Assign(Local(1), Rvalue::Binary(BinOp::Rem, local(0), local(0)));
    rejected(one_block(&[Ty::F32], &[Ty::F32], vec![float_rem], Terminator::Return), "");
    let mixed = Statement::Assign(Local(2), Rvalue::Binary(BinOp::Add, local(0), local(1)));
    rejected(one_block(&[Ty::U32, Ty::I32], &[Ty::U32], vec![mixed], Terminator::Return), "");
    let wrong_local = Statement::Assign(Local(1), Rvalue::Use(local(0)));
    rejected(one_block(&[Ty::U32], &[Ty::U64], vec![wrong_local], Terminator::Return), "assigned to");
    let unknown = Statement::Assign(Local(1), Rvalue::Use(local(9)));
    rejected(one_block(&[Ty::U32], &[Ty::U32], vec![unknown], Terminator::Return), "unknown local");
    let to_value = Statement::Store(local(0), local(0));
    rejected(one_block(&[Ty::U32], &[], vec![to_value], Terminator::Return), "non-pointer");
    let not_bool = Terminator::If(local(0), BlockId(0), BlockId(0));
    rejected(one_block(&[Ty::U32], &[], vec![], not_bool), "bad terminator");
    rejected(one_block(&[], &[], vec![], Terminator::Goto(BlockId(1))), "bad terminator");
    let bool_to_float = Statement::Assign(Local(1), Rvalue::Cast(Operand::Const(Const::bool(true)), Ty::F32));
    rejected(one_block(&[], &[Ty::F32], vec![bool_to_float], Terminator::Return), "");
    let rotate_by_u8 = Statement::Assign(
        Local(1),
        Rvalue::Call(Intrinsic::RotateLeft, vec![local(0), Operand::Const(Const::int(Ty::U8, 1))]),
    );
    rejected(one_block(&[Ty::U32], &[Ty::U32], vec![rotate_by_u8], Terminator::Return), "");
}

/// `out[i] = a[i] * a[i] + 1.0` over f32.
fn square_plus_one() -> Kernel {
    let (mut b, p) = Builder::new("square_plus_one", &[Ty::Ptr, Ty::Ptr]);
    let i = b.special(Special::X);
    let x = b.index(p[1], i, Ty::F32);
    let square = b.binary(BinOp::Mul, x, x);
    let result = b.binary(BinOp::Add, square, Const::f32(1.0));
    let address = b.offset(p[0], i, Ty::F32);
    b.store(address, result);
    b.finish().unwrap()
}

#[test]
fn direct_ptx_rounds_every_operation_separately() {
    let text = ptx::emit(&square_plus_one());
    for expected in [".visible .entry square_plus_one(", "mul.rn.f32", "add.rn.f32", "ld.global.f32", "st.global.f32"] {
        assert!(text.contains(expected), "{expected:?} missing:\n{text}");
    }
    assert!(!text.contains("fma"), "fused:\n{text}");
}

#[test]
fn llvm_ir_declares_a_kernel_on_global_pointers() {
    let text = llvm::emit(&square_plus_one());
    for expected in ["target triple = \"nvptx64-nvidia-cuda\"", "define ptx_kernel void @square_plus_one(", "ptr addrspace(1)", "fmul float", "fadd float"] {
        assert!(text.contains(expected), "{expected:?} missing:\n{text}");
    }
    assert!(!text.contains("contract") && !text.contains("fast"), "fusion allowed:\n{text}");
}

/// Needs the `llvm-tools` component of the pinned toolchain.
#[test]
fn llvm_tools_keep_operations_unfused() {
    let text = llvm::Llvm::find().unwrap().compile(&square_plus_one()).unwrap();
    for expected in [".entry square_plus_one(", "mul.rn.f32", "add.rn.f32"] {
        assert!(text.contains(expected), "{expected:?} missing:\n{text}");
    }
    assert!(!text.contains("fma"), "fused:\n{text}");
}

/// `x < limit ? x * 2 : limit`, over `u32`, reading `limit`.
fn capped_double() -> cuprite_kernel::Function {
    let (mut b, p) = Builder::new("capped_double", &[Ty::U32, Ty::U32]);
    let (x, limit) = (p[0], p[1]);
    let result = b.local(Ty::U32);
    let (below, above, join) = (b.block(), b.block(), b.block());
    let less = b.binary(BinOp::Lt, x, limit);
    b.branch(less, below, above);
    b.switch_to(below);
    b.set(result, Rvalue::Binary(BinOp::Mul, x.into(), Const::int(Ty::U32, 2).into()));
    b.goto(join);
    b.switch_to(above);
    b.set(result, Rvalue::Use(limit.into()));
    b.goto(join);
    b.switch_to(join);
    b.finish_function(Some(result)).unwrap()
}

#[test]
fn element_functions_inline_into_map_kernels() {
    use cuprite_kernel::shapes::in_place_map;
    let stage = capped_double();
    let kernel = in_place_map("twice", Ty::U32, &[], &[&stage, &stage], true).unwrap();
    // The data, its length, the failure flag, the pass count, then each
    // stage's scalar.
    assert_eq!(&kernel.locals[..kernel.params], [Ty::Ptr, Ty::U64, Ty::Ptr, Ty::U32, Ty::U32, Ty::U32]);
    // Both copies are there, each with its own branch.
    let text = ptx::emit(&kernel);
    assert_eq!(text.matches("setp.lt.u32").count(), 3, "{text}"); // 2 stages + the pass loop
    assert!(llvm::emit(&kernel).contains("define ptx_kernel void @twice("));
    // A stage must map the element type to itself.
    assert!(in_place_map("wrong", Ty::F32, &[], &[&stage], false).is_err());
}
