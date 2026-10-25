use crate::{
    backend::{DefaultBackend, MemoryBackend},
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

/// Public entry point: a [`Manager`] over the CUDA backend.
pub struct GpuMemory {
    inner: Manager<DefaultBackend>,
}

impl GpuMemory {
    /// Open the configured CUDA device.
    ///
    /// # Errors
    /// Returns [`MemoryError::Unavailable`] when built without the `cuda`
    /// feature, or when no usable CUDA driver or device is found.
    pub fn open(config: MemoryConfig) -> Result<Self, MemoryError> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = config;
            Err(MemoryError::Unavailable(
                "build with --features cuda to enable the CUDA backend".into(),
            ))
        }
        #[cfg(feature = "cuda")]
        {
            let backend = crate::cuda::CudaBackend::open(config)?;
            Ok(Self {
                inner: Manager::new(backend, config),
            })
        }
    }

    /// Borrow both the manager and original slice for the session's lifetime.
    pub fn begin_mut<'a>(
        &'a mut self,
        host: &'a mut [u32],
    ) -> Result<ResidentSlice<'a>, MemoryError> {
        self.inner.begin_mut(host).map(ResidentSlice::new)
    }

    pub fn stats(&self) -> MemoryStats {
        self.inner.stats()
    }

    pub fn trim(&mut self) -> Result<(), MemoryError> {
        Err(MemoryError::NotImplemented("GpuMemory::trim"))
    }
}

/// The memory manager: owns the backend and enforces the session rules.
///
/// `B` is the backend type, so the same logic runs on any backend: tests use
/// `Manager<FakeBackend>`, and the real GPU uses `CudaBackend`.
pub(crate) struct Manager<B: MemoryBackend> {
    pub(crate) backend: B,
    pub(crate) config: MemoryConfig,
    pub(crate) stats: MemoryStats,
    /// Set while a nonempty session exists. Still being set when the next
    /// session begins means the last one was forgotten without cleanup.
    pub(crate) active: bool,
    /// Once set, every new session is refused with `Poisoned`.
    pub(crate) poisoned: bool,
    /// Resources whose completion could not be established. They are never
    /// freed or reused; dropping the manager leaks them deliberately.
    pub(crate) quarantined: Option<Resources<B::Buffer>>,
}

impl<B: MemoryBackend> Manager<B> {
    // Only the CUDA build and tests construct a manager.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn new(backend: B, config: MemoryConfig) -> Self {
        Self {
            backend,
            config,
            stats: MemoryStats::default(),
            active: false,
            poisoned: false,
            quarantined: None,
        }
    }

    /// Open a session over `host`: validate, stage, upload, and wait.
    ///
    /// Empty input returns an inert session that allocates and transfers
    /// nothing. Otherwise the request is checked against the budget before
    /// anything is allocated, `host` is copied into session-owned staging,
    /// and the device buffer is filled from staging, never from `host`.
    ///
    /// A failure after the upload is issued releases the session's storage
    /// only once completion is established; otherwise it is quarantined.
    pub(crate) fn begin_mut<'a>(
        &'a mut self,
        host: &'a mut [u32],
    ) -> Result<Session<'a, B>, MemoryError> {
        if self.active {
            // The previous session was forgotten, so its cleanup never ran.
            self.poisoned = true;
        }
        if self.poisoned {
            return Err(MemoryError::Poisoned);
        }

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

        let device = self.backend.allocate(host.len())?;
        self.stats.device_allocations += 1;
        let mut resources = Resources { staging, device };

        // Set before any transfer so a forgotten session is detected.
        self.active = true;
        let uploaded = self
            .backend
            .upload(&mut resources.device, &resources.staging)
            .and_then(|()| self.backend.synchronize());
        if let Err(error) = uploaded {
            // The upload may still be reading staging.
            self.release(resources, true);
            self.active = false;
            return Err(error);
        }
        // Count the upload only once completion is confirmed.
        self.stats.uploads += 1;
        self.stats.uploaded_bytes += bytes as u64; // `usize` to `u64`: never implicit in Rust.

        Ok(Session {
            manager: self,
            host,
            resources: Some(resources),
            may_have_writes: false,
        })
    }

    pub(crate) fn stats(&self) -> MemoryStats {
        self.stats
    }

    /// Free a session's storage once no queued work can still use it.
    ///
    /// `pending` means work may have been queued since the last completed
    /// synchronize. If completion then cannot be established, the storage is
    /// quarantined and the manager is poisoned.
    pub(crate) fn release(&mut self, resources: Resources<B::Buffer>, pending: bool) {
        if pending && self.backend.synchronize().is_err() {
            self.poisoned = true;
            if self.quarantined.is_none() {
                self.quarantined = Some(resources);
            } else {
                // Unreachable while poisoning refuses new sessions; leak
                // rather than free storage that may still be in use.
                std::mem::forget(resources);
            }
        }
    }
}

impl<B: MemoryBackend> Drop for Manager<B> {
    fn drop(&mut self) {
        // Completion was never established, so freeing could release storage
        // the GPU still uses. Leaking is bounded to one bundle per manager.
        if let Some(resources) = self.quarantined.take() {
            std::mem::forget(resources);
        }
    }
}
