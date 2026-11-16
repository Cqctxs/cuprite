//! Building kernels block by block.

use crate::{
    BinOp, Block, BlockId, Const, Function, Intrinsic, Invalid, Kernel, Local, Operand, Rvalue, Special,
    Statement, Terminator, Ty, UnOp,
};

impl From<Local> for Operand {
    fn from(local: Local) -> Self {
        Operand::Local(local)
    }
}

impl From<Const> for Operand {
    fn from(constant: Const) -> Self {
        Operand::Const(constant)
    }
}

/// Appends to one block at a time. New blocks end in `Return` until given
/// another terminator.
pub struct Builder {
    kernel: Kernel,
    current: BlockId,
}

impl Builder {
    /// A kernel with these parameters, which are its first locals.
    pub fn new(name: &str, params: &[Ty]) -> (Self, Vec<Local>) {
        let kernel = Kernel {
            name: name.into(),
            params: params.len(),
            locals: params.to_vec(),
            blocks: vec![Block {
                statements: Vec::new(),
                terminator: Terminator::Return,
            }],
        };
        let locals = (0..params.len() as u32).map(Local).collect();
        (
            Self {
                kernel,
                current: BlockId(0),
            },
            locals,
        )
    }

    pub fn ty(&self, operand: impl Into<Operand>) -> Ty {
        self.kernel.ty(operand.into())
    }

    pub fn local(&mut self, ty: Ty) -> Local {
        self.kernel.locals.push(ty);
        Local(self.kernel.locals.len() as u32 - 1)
    }

    pub fn block(&mut self) -> BlockId {
        self.kernel.blocks.push(Block {
            statements: Vec::new(),
            terminator: Terminator::Return,
        });
        BlockId(self.kernel.blocks.len() as u32 - 1)
    }

    pub fn switch_to(&mut self, block: BlockId) {
        self.current = block;
    }

    fn push(&mut self, statement: Statement) {
        self.kernel.blocks[self.current.0 as usize]
            .statements
            .push(statement);
    }

    /// Assigns `rvalue` to an existing local.
    pub fn set(&mut self, local: Local, rvalue: Rvalue) {
        self.push(Statement::Assign(local, rvalue));
    }

    /// A new local holding `rvalue`, typed by it.
    ///
    /// # Panics
    /// If the operands' types don't fit together.
    pub fn assign(&mut self, rvalue: Rvalue) -> Local {
        let ty = self
            .kernel
            .rvalue_ty(&rvalue)
            .unwrap_or_else(|invalid| panic!("{invalid}"));
        let local = self.local(ty);
        self.set(local, rvalue);
        local
    }

    pub fn copy(&mut self, from: impl Into<Operand>) -> Local {
        self.assign(Rvalue::Use(from.into()))
    }

    pub fn binary(&mut self, op: BinOp, a: impl Into<Operand>, b: impl Into<Operand>) -> Local {
        self.assign(Rvalue::Binary(op, a.into(), b.into()))
    }

    pub fn unary(&mut self, op: UnOp, a: impl Into<Operand>) -> Local {
        self.assign(Rvalue::Unary(op, a.into()))
    }

    pub fn cast(&mut self, a: impl Into<Operand>, to: Ty) -> Local {
        self.assign(Rvalue::Cast(a.into(), to))
    }

    pub fn call(&mut self, intrinsic: Intrinsic, args: &[Operand]) -> Local {
        self.assign(Rvalue::Call(intrinsic, args.to_vec()))
    }

    pub fn load(&mut self, pointer: impl Into<Operand>, ty: Ty) -> Local {
        self.assign(Rvalue::Load(pointer.into(), ty))
    }

    /// `pointer + index * ty.bytes()`, with any integer index.
    pub fn offset(&mut self, pointer: impl Into<Operand>, index: impl Into<Operand>, ty: Ty) -> Local {
        let index = index.into();
        let index = if self.kernel.ty(index) == Ty::U64 {
            index
        } else {
            self.cast(index, Ty::U64).into()
        };
        self.assign(Rvalue::Offset(pointer.into(), index, ty.bytes()))
    }

    /// The element of type `ty` at `index` of the buffer `pointer`.
    pub fn index(&mut self, pointer: impl Into<Operand>, index: impl Into<Operand>, ty: Ty) -> Local {
        let address = self.offset(pointer, index, ty);
        self.load(address, ty)
    }

    pub fn special(&mut self, special: Special) -> Local {
        self.assign(Rvalue::Special(special))
    }

    pub fn select(
        &mut self,
        condition: impl Into<Operand>,
        a: impl Into<Operand>,
        b: impl Into<Operand>,
    ) -> Local {
        self.assign(Rvalue::Select(condition.into(), a.into(), b.into()))
    }

    pub fn store(&mut self, pointer: impl Into<Operand>, value: impl Into<Operand>) {
        self.push(Statement::Store(pointer.into(), value.into()));
    }

    fn terminate(&mut self, terminator: Terminator) {
        self.kernel.blocks[self.current.0 as usize].terminator = terminator;
    }

    pub fn goto(&mut self, target: BlockId) {
        self.terminate(Terminator::Goto(target));
    }

    pub fn branch(&mut self, condition: impl Into<Operand>, then: BlockId, otherwise: BlockId) {
        self.terminate(Terminator::If(condition.into(), then, otherwise));
    }

    pub fn ret(&mut self) {
        self.terminate(Terminator::Return);
    }

    /// Ends the block with a failed check.
    pub fn fail(&mut self) {
        self.terminate(Terminator::Fail);
    }

    /// Runs `function` here: binds its parameters to `args`, copies its
    /// blocks with fresh locals, and continues in a new block where it
    /// returns. Where it fails, it jumps to `on_fail`, if given, and fails
    /// otherwise. Returns the local holding its result.
    ///
    /// # Panics
    /// If the number of arguments isn't the number of parameters. Argument
    /// types are checked by [`Builder::finish`].
    pub fn inline(&mut self, function: &Function, args: &[Operand], on_fail: Option<BlockId>) -> Option<Local> {
        let callee = &function.kernel;
        assert_eq!(args.len(), callee.params, "inlining `{}`", callee.name);
        let locals: Vec<Local> = callee.locals.iter().map(|ty| self.local(*ty)).collect();
        let blocks: Vec<BlockId> = callee.blocks.iter().map(|_| self.block()).collect();
        let after = self.block();
        for (param, arg) in locals.iter().zip(args) {
            self.set(*param, Rvalue::Use(*arg));
        }
        self.goto(blocks[0]);
        let rename = |operand: Operand| match operand {
            Operand::Local(local) => Operand::Local(locals[local.0 as usize]),
            constant => constant,
        };
        let block = |id: BlockId| blocks[id.0 as usize];
        for (i, original) in callee.blocks.iter().enumerate() {
            let statements = original
                .statements
                .iter()
                .map(|statement| match statement {
                    Statement::Assign(local, rvalue) => {
                        Statement::Assign(locals[local.0 as usize], rvalue.map_operands(rename))
                    }
                    Statement::Store(pointer, value) => Statement::Store(rename(*pointer), rename(*value)),
                })
                .collect();
            let terminator = match original.terminator {
                Terminator::Goto(target) => Terminator::Goto(block(target)),
                Terminator::If(condition, then, otherwise) => {
                    Terminator::If(rename(condition), block(then), block(otherwise))
                }
                Terminator::Return => Terminator::Goto(after),
                Terminator::Fail => on_fail.map_or(Terminator::Fail, Terminator::Goto),
            };
            self.kernel.blocks[blocks[i].0 as usize] = Block { statements, terminator };
        }
        self.switch_to(after);
        function.result.map(|result| locals[result.0 as usize])
    }

    /// The kernel so far, as an element function returning `result`.
    pub fn finish_function(self, result: Option<Local>) -> Result<Function, Invalid> {
        let function = Function { kernel: self.kernel, result };
        function.validate()?;
        Ok(function)
    }

    pub fn finish(self) -> Result<Kernel, Invalid> {
        self.kernel.validate()?;
        Ok(self.kernel)
    }
}
