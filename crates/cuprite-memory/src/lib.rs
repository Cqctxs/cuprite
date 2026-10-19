//! Scoped GPU memory: upload a slice once, work on it on the GPU, copy it back once.
//!
//! The backend-generic manager and session are implemented and tested with a
//! fake backend. The CUDA backend, the public `GpuMemory` wiring, and the
//! buffer pool are not written yet; their fallible operations return
//! [`MemoryError::NotImplemented`].

#![deny(unsafe_op_in_unsafe_fn)]

mod config;
mod error;
mod manager;
mod session;
mod size;

// Not reachable from the public API yet, so silence unused-code warnings.
#[allow(dead_code)]
mod backend;
#[allow(dead_code)]
mod pool;

#[cfg(feature = "cuda")]
#[allow(dead_code)]
mod cuda;

#[cfg(test)]
#[allow(dead_code)]
mod test_support;

pub use config::MemoryConfig;
pub use error::MemoryError;
pub use manager::{GpuMemory, MemoryStats};
pub use session::ResidentSlice;
pub use size::{checked_bytes, validate_len};
