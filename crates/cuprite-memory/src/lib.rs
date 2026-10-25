//! Scoped GPU memory: upload a slice once, work on it on the GPU, copy it back once.
//!
//! The backend-generic manager and session hold the ownership and cleanup
//! rules; the `cuda` feature connects them to a real GPU through cudarc.
//! The buffer pool is not written yet, so `trim` returns
//! [`MemoryError::NotImplemented`].

#![deny(unsafe_op_in_unsafe_fn)]

mod backend;
mod config;
mod error;
mod manager;
mod session;
mod size;

// Not reachable from the public API yet, so silence unused-code warnings.
#[allow(dead_code)]
mod pool;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(test)]
mod test_support;

pub use config::MemoryConfig;
pub use error::MemoryError;
pub use manager::{GpuMemory, MemoryStats};
pub use session::ResidentSlice;
pub use size::{checked_bytes, validate_len};
