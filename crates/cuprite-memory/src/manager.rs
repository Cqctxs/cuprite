use crate::{
    backend::{DefaultBackend, MemoryBackend},
    pool::BufferPool,
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

    /// Release idle pooled device storage and the reusable host staging.
    pub fn trim(&mut self) -> Result<(), MemoryError> {
        self.inner.trim();
        Ok(())
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
    /// Completed, healthy device storage kept for the next session.
    pub(crate) pool: BufferPool<B::Buffer>,
    /// Host staging from the last completed session, kept so the next one
    /// reuses already-touched pages instead of faulting in new ones.
    pub(crate) staging: Vec<u32>,
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
            pool: BufferPool::new(),
            staging: Vec::new(),
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
        // CUDA only says a copy from pageable memory "might be synchronous",
        // so uploading from `host` directly could leave the driver reading it
        // after a failed session returns and releases the borrow.
        // The previous session's staging is reused when it is large enough;
        // `try_reserve_exact` reports allocation failure instead of aborting,
        // and `map_err` turns its error into a `MemoryError` so `?` can return it.
        let mut staging = std::mem::take(&mut self.staging);
        staging.clear();
        staging
            .try_reserve_exact(host.len())
            .map_err(|error| MemoryError::HostAllocation(error.to_string()))?;
        staging.extend_from_slice(host);

        // Reuse the idle buffer only for an exact length match. Any other
        // idle buffer is released first, so active plus idle storage stays
        // within the budget checked above.
        let device = match self.pool.take_matching(host.len()) {
            Some(device) => {
                self.stats.pool_hits += 1;
                device
            }
            None => {
                let device = self.backend.allocate(host.len())?;
                self.stats.device_allocations += 1;
                device
            }
        };
        let mut resources = Resources { staging, device };

        // Set before any transfer so a forgotten session is detected.
        self.active = true;
        let uploaded = self
            .backend
            .upload(&mut resources.device, &resources.staging)
            .and_then(|()| self.backend.synchronize());
        if let Err(error) = uploaded {
            // The upload may still be reading staging. Storage from a failed
            // operation is freed, not pooled.
            drop(self.settle(resources, true));
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

    /// Wait for queued work that may still use `resources`.
    ///
    /// `pending` means work may have been queued since the last completed
    /// synchronize. Returns the resources once completion is established;
    /// otherwise quarantines them, poisons the manager, and returns `None`.
    pub(crate) fn settle(
        &mut self,
        resources: Resources<B::Buffer>,
        pending: bool,
    ) -> Option<Resources<B::Buffer>> {
        if !pending || self.backend.synchronize().is_ok() {
            return Some(resources);
        }
        self.poisoned = true;
        if self.quarantined.is_none() {
            self.quarantined = Some(resources);
        } else {
            // Unreachable while poisoning refuses new sessions; leak
            // rather than free storage that may still be in use.
            std::mem::forget(resources);
        }
        None
    }

    /// Keep a completed session's device buffer and staging for reuse.
    pub(crate) fn recycle(&mut self, resources: Resources<B::Buffer>) {
        let len = resources.staging.len();
        self.pool.put_completed(len, resources.device);
        self.staging = resources.staging;
    }

    pub(crate) fn trim(&mut self) {
        self.pool.clear();
        self.staging = Vec::new();
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
