use crate::{
    backend::MemoryBackend,
    session::{Resources, Session},
    validate_len, MemoryConfig, MemoryError, ResidentSlice,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryStats {
    pub device_allocations: u64,
    pub pool_hits: u64,
    pub uploads: u64,
    pub downloads: u64,
    pub uploaded_bytes: u64,
    pub downloaded_bytes: u64,
}

/// Public entry point. Not yet connected to a backend, so it cannot open a
/// working GPU session; `Manager<B>` below holds the implemented logic.
#[derive(Debug)]
#[allow(dead_code)]
pub struct GpuMemory {
    pub(crate) config: MemoryConfig,
    pub(crate) stats: MemoryStats,
    pub(crate) active: bool,
    pub(crate) poisoned: bool,
}

impl GpuMemory {
    pub fn open(_config: MemoryConfig) -> Result<Self, MemoryError> {
        #[cfg(not(feature = "cuda"))]
        {
            Err(MemoryError::Unavailable(
                "build with --features cuda to enable the CUDA backend".into(),
            ))
        }
        #[cfg(feature = "cuda")]
        {
            Err(MemoryError::NotImplemented("GpuMemory::open"))
        }
    }

    /// Borrow both the manager and original slice for the session's lifetime.
    pub fn begin_mut<'a>(
        &'a mut self,
        _host: &'a mut [u32],
    ) -> Result<ResidentSlice<'a>, MemoryError> {
        Err(MemoryError::NotImplemented("GpuMemory::begin_mut"))
    }

    pub fn stats(&self) -> MemoryStats {
        self.stats
    }

    pub fn trim(&mut self) -> Result<(), MemoryError> {
        Err(MemoryError::NotImplemented("GpuMemory::trim"))
    }
}

/// The memory manager: owns the backend and enforces the session rules.
///
/// `B` is the backend type, so the same logic runs on any backend: tests use
/// `Manager<FakeBackend>`, and the real GPU will use a CUDA backend.
#[allow(dead_code)] // Only tests use this until `GpuMemory` is connected.
pub(crate) struct Manager<B: MemoryBackend> {
    pub(crate) backend: B,
    pub(crate) config: MemoryConfig,
    pub(crate) stats: MemoryStats,
    /// Reserved for tracking an open session; not used yet.
    pub(crate) active: bool,
    /// Reserved for refusing work after an unrecoverable failure; not used yet.
    pub(crate) poisoned: bool,
}

#[allow(dead_code)] // Only tests use this until `GpuMemory` is connected.
impl<B: MemoryBackend> Manager<B> {
    pub(crate) fn new(backend: B, config: MemoryConfig) -> Self {
        Self {
            backend,
            config,
            stats: MemoryStats::default(),
            active: false,
            poisoned: false,
        }
    }

    /// Open a session over `host`: validate, stage, upload, and wait.
    ///
    /// Empty input returns an inert session that allocates and transfers
    /// nothing. Otherwise the request is checked against the budget before
    /// anything is allocated, `host` is copied into session-owned staging,
    /// and the device buffer is filled from staging, never from `host`.
    ///
    /// Errors return early with `?`; cleanup on those paths is not handled yet.
    pub(crate) fn begin_mut<'a>(
        &'a mut self,
        host: &'a mut [u32],
    ) -> Result<Session<'a, B>, MemoryError> {
        if host.is_empty() {
            return Ok(Session {
                // `self` is already `&'a mut Manager<B>`: the session takes
                // over this borrow, so the caller can't use the manager until
                // the session is gone.
                manager: self,
                host,
                resources: None,
                may_have_writes: false,
            });
        }

        // Reject oversized requests before allocating anything.
        let bytes = validate_len(host.len(), self.config)?;

        // Copy into memory the session owns; transfers never touch `host`.
        // `try_reserve_exact` reports allocation failure instead of aborting,
        // and `map_err` turns its error into a `MemoryError` so `?` can return it.
        let mut staging = Vec::new();
        staging
            .try_reserve_exact(host.len())
            .map_err(|error| MemoryError::HostAllocation(error.to_string()))?;
        staging.extend_from_slice(host);

        let mut device = self.backend.allocate(host.len())?;
        self.stats.device_allocations += 1;

        self.backend.upload(&mut device, &staging)?;
        self.backend.synchronize()?;
        // Count the upload only once completion is confirmed.
        self.stats.uploads += 1;
        self.stats.uploaded_bytes += bytes as u64; // `usize` to `u64`: never implicit in Rust.

        Ok(Session {
            manager: self,
            host,
            resources: Some(Resources { staging, device }),
            may_have_writes: false,
        })
    }

    pub(crate) fn stats(&self) -> MemoryStats {
        self.stats
    }
}
