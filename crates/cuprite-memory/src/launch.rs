//! Kernels, launch shapes and kernel arguments.

/// A loaded kernel, valid only with the [`GpuMemory`](crate::GpuMemory)
/// that loaded it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Kernel {
    pub(crate) owner: u64,
    pub(crate) index: usize,
}

/// A buffer in a [`Job`](crate::Job), valid only in that job. Kernels
/// receive it as a 64-bit device pointer, which is null for an empty buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Buffer {
    pub(crate) job: u64,
    pub(crate) index: usize,
}

/// How many GPU threads a launch starts. Threads beyond the data must do
/// nothing: each kernel checks its own bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grid {
    /// `n` threads in blocks of [`Grid::LINEAR_BLOCK`], numbered by
    /// `%ctaid.x * %ntid.x + %tid.x`.
    Linear(u64),
    /// `width × height` threads in blocks of [`Grid::PLANE_BLOCK`] squared,
    /// with `x` and `y` numbered like `Linear`'s index.
    Plane { width: u32, height: u32 },
}

impl Grid {
    pub const LINEAR_BLOCK: u32 = 256;
    pub const PLANE_BLOCK: u32 = 16;

    pub fn is_empty(self) -> bool {
        match self {
            Self::Linear(threads) => threads == 0,
            Self::Plane { width, height } => width == 0 || height == 0,
        }
    }
}

/// One kernel argument, in the order of the kernel's parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Arg {
    Buffer(Buffer),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
}
