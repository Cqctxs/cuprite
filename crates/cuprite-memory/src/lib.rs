//! Scoped GPU memory: upload a slice once, work on it on the GPU, copy it back once.
//!
//! The backend-generic manager and session hold the ownership and cleanup
//! rules; the `cuda` feature connects them to a real GPU through cudarc.
//! Completed device buffers are kept in a one-slot pool for the next
//! session of the same length. The only GPU operation so far is
//! [`MapStep`], an element-wise wrapping multiply-add.

#![deny(unsafe_op_in_unsafe_fn)]

mod backend;
mod config;
mod error;
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
pub use error::MemoryError;
pub use manager::{GpuMemory, MemoryStats};
pub use session::ResidentSlice;
pub use size::{checked_bytes, validate_len};
pub use step::MapStep;
