//! Scoped GPU memory: put CPU data on the GPU, run kernels over it there,
//! and copy results back once.
//!
//! A [`Job`] holds any number of buffers of plain numeric [`Element`]s, each
//! an input, output, in-out or scratch buffer, and runs loaded PTX
//! [`Kernel`]s over them. Its results reach the CPU slices all together at
//! `finish`, or not at all. [`ResidentSlice`] is a job over one `u32` slice
//! for the built-in [`MapStep`], an element-wise wrapping multiply-add.
//!
//! The backend-generic manager and job hold the ownership and cleanup
//! rules; the `cuda` feature connects them to a real GPU through cudarc.
//! Completed device buffers are kept in a small pool for later jobs.

#![deny(unsafe_op_in_unsafe_fn)]

mod backend;
mod config;
mod device;
mod element;
mod error;
mod job;
mod launch;
mod manager;
mod pool;
mod session;
mod size;
mod step;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(test)]
mod test_support;

pub use config::MemoryConfig;
pub use device::DeviceInfo;
pub use element::Element;
pub use error::MemoryError;
pub use job::Job;
pub use launch::{Arg, Buffer, Grid, Kernel};
pub use manager::{GpuMemory, MemoryStats};
pub use session::ResidentSlice;
pub use size::{checked_bytes, validate_len};
pub use step::MapStep;
