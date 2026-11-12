use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
    sync::atomic::{AtomicU64, Ordering},
};

use crate::{
    backend::{DefaultBackend, MemoryBackend},
    job::{JobCore, Storage},
    pool::BufferPool,
    session::Resident,
    step, DeviceInfo, Job, Kernel, MemoryConfig, MemoryError, ResidentSlice,
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
    device: DeviceInfo,
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
            let (backend, device) = crate::cuda::CudaBackend::open(config)?;
            Ok(Self {
                inner: Manager::new(backend, config),
                device,
            })
        }
    }

    /// Load the PTX entry `name` from `ptx`, compiling it for this GPU. The
    /// same PTX and name load once; later calls return the same kernel.
    ///
    /// # Errors
    /// Returns [`MemoryError::Kernel`] if the driver rejects the PTX or it
    /// has no such entry.
    pub fn kernel(&mut self, ptx: &str, name: &str) -> Result<Kernel, MemoryError> {
        self.inner.kernel(ptx, name)
    }

    /// Start a job: buffers on the GPU, kernels over them, one commit.
    ///
    /// # Errors
    /// Returns [`MemoryError::Poisoned`] if earlier work could not be
    /// cleaned up safely.
    pub fn job(&mut self) -> Result<Job<'_>, MemoryError> {
        self.inner.job().map(Job::new)
    }

    /// A job over one `u32` slice, uploaded now and committed by `finish`,
    /// for [`MapStep`](crate::MapStep)s.
    pub fn begin_mut<'a>(
        &'a mut self,
        host: &'a mut [u32],
    ) -> Result<ResidentSlice<'a>, MemoryError> {
        Resident::begin(&mut self.inner, host).map(ResidentSlice::new)
    }

    pub fn stats(&self) -> MemoryStats {
        self.inner.stats()
    }

    /// The open device and the versions its performance depends on.
    pub fn device(&self) -> &DeviceInfo {
        &self.device
    }

    /// Release idle pooled device storage and the reusable host staging.
    pub fn trim(&mut self) -> Result<(), MemoryError> {
        self.inner.trim();
        Ok(())
    }
}

/// Distinguishes managers, so a kernel is only used with the one that
/// loaded it.
static NEXT_MANAGER: AtomicU64 = AtomicU64::new(0);

/// How many spare staging buffers are kept at most.
const SPARE_STAGING: usize = 8;

struct Loaded<K> {
    ptx: Box<str>,
    name: Box<str>,
    kernel: K,
}

/// The memory manager: owns the backend and enforces the job rules.
///
/// `B` is the backend type, so the same logic runs on any backend: tests use
/// `Manager<FakeBackend>`, and the real GPU uses `CudaBackend`.
pub(crate) struct Manager<B: MemoryBackend> {
    pub(crate) backend: B,
    pub(crate) config: MemoryConfig,
    pub(crate) stats: MemoryStats,
    /// Set while a job holds storage. Still being set when the next job
    /// begins means the last one was forgotten without cleanup.
    pub(crate) active: bool,
    /// Once set, every new job is refused with `Poisoned`.
    pub(crate) poisoned: bool,
    /// Storage whose completion could not be established. It is never
    /// freed or reused; dropping the manager leaks it deliberately.
    pub(crate) quarantined: Option<Vec<Storage<B::Buffer>>>,
    /// Completed, healthy device storage kept for later jobs.
    pub(crate) pool: BufferPool<B::Buffer>,
    /// Host staging from completed jobs, kept so later ones reuse
    /// already-touched pages instead of faulting in new ones.
    pub(crate) staging: Vec<Vec<u8>>,
    kernels: Vec<Loaded<B::Kernel>>,
    /// Indices into `kernels`, by a hash of their PTX.
    by_hash: HashMap<u64, Vec<usize>>,
    pub(crate) id: u64,
    next_job: u64,
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
            kernels: Vec::new(),
            by_hash: HashMap::new(),
            id: NEXT_MANAGER.fetch_add(1, Ordering::Relaxed),
            next_job: 0,
        }
    }

    pub(crate) fn job(&mut self) -> Result<JobCore<'_, B>, MemoryError> {
        if self.active {
            // The previous job was forgotten, so its cleanup never ran.
            self.poisoned = true;
        }
        if self.poisoned {
            return Err(MemoryError::Poisoned);
        }
        let id = self.next_job;
        self.next_job += 1;
        Ok(JobCore::new(self, id))
    }

    pub(crate) fn kernel(&mut self, ptx: &str, name: &str) -> Result<Kernel, MemoryError> {
        if self.poisoned {
            return Err(MemoryError::Poisoned);
        }
        let mut hasher = DefaultHasher::new();
        ptx.hash(&mut hasher);
        let hash = hasher.finish();
        // The hash only narrows the search: a match must have the same PTX.
        let mut known = self.by_hash.get(&hash).into_iter().flatten();
        if let Some(&index) =
            known.find(|&&i| &*self.kernels[i].ptx == ptx && &*self.kernels[i].name == name)
        {
            return Ok(Kernel {
                owner: self.id,
                index,
            });
        }
        let kernel = self.backend.load(ptx, name)?;
        let index = self.kernels.len();
        self.kernels.push(Loaded {
            ptx: ptx.into(),
            name: name.into(),
            kernel,
        });
        self.by_hash.entry(hash).or_default().push(index);
        Ok(Kernel {
            owner: self.id,
            index,
        })
    }

    /// The kernel for [`MapStep`](crate::MapStep), loaded on first use.
    pub(crate) fn map_kernel(&mut self) -> Result<Kernel, MemoryError> {
        self.kernel(step::MAP_PTX, step::MAP_NAME)
    }

    /// The backend and a kernel it loaded, borrowed together for a launch.
    pub(crate) fn launch_parts(
        &mut self,
        kernel: Kernel,
    ) -> Result<(&mut B, &B::Kernel), MemoryError> {
        if kernel.owner != self.id {
            return Err(MemoryError::Launch(
                "the kernel was loaded by another GpuMemory".into(),
            ));
        }
        Ok((&mut self.backend, &self.kernels[kernel.index].kernel))
    }

    /// Device storage for `bytes` more while a job holds `held`: an idle
    /// buffer of exactly that size, or a new one once enough idle storage is
    /// released to stay within the budget. Returns whether it is new, and
    /// so zero-filled.
    pub(crate) fn device_storage(
        &mut self,
        held: usize,
        bytes: usize,
    ) -> Result<(B::Buffer, bool), MemoryError> {
        if let Some(buffer) = self.pool.take(bytes) {
            self.stats.pool_hits += 1;
            return Ok((buffer, false));
        }
        while held + bytes + self.pool.idle_bytes() > self.config.max_device_bytes
            && self.pool.evict_oldest()
        {}
        let buffer = self.backend.allocate(bytes)?;
        self.stats.device_allocations += 1;
        Ok((buffer, true))
    }

    /// Empty staging that can hold `bytes`. Allocation failure is reported
    /// instead of aborting.
    pub(crate) fn take_staging(&mut self, bytes: usize) -> Result<Vec<u8>, MemoryError> {
        let fitting = self
            .staging
            .iter()
            .position(|spare| spare.capacity() >= bytes);
        let mut staging = match fitting {
            Some(index) => self.staging.swap_remove(index),
            None => self.staging.pop().unwrap_or_default(),
        };
        staging.clear();
        staging
            .try_reserve_exact(bytes)
            .map_err(|error| MemoryError::HostAllocation(error.to_string()))?;
        Ok(staging)
    }

    pub(crate) fn stats(&self) -> MemoryStats {
        self.stats
    }

    /// Wait for queued work that may still use `storage`.
    ///
    /// `pending` means work may have been queued since the last completed
    /// synchronize. Returns the storage once completion is established;
    /// otherwise quarantines it, poisons the manager, and returns `None`.
    pub(crate) fn settle(
        &mut self,
        storage: Vec<Storage<B::Buffer>>,
        pending: bool,
    ) -> Option<Vec<Storage<B::Buffer>>> {
        if !pending || self.backend.synchronize().is_ok() {
            return Some(storage);
        }
        self.poisoned = true;
        if self.quarantined.is_none() {
            self.quarantined = Some(storage);
        } else {
            // Unreachable while poisoning refuses new jobs; leak rather
            // than free storage that may still be in use.
            std::mem::forget(storage);
        }
        None
    }

    /// Keep completed storage for reuse.
    pub(crate) fn recycle(&mut self, storage: Storage<B::Buffer>) {
        if let Some(device) = storage.device {
            self.pool.put_completed(storage.bytes, device);
        }
        if storage.staging.capacity() > 0 {
            if self.staging.len() == SPARE_STAGING {
                self.staging.remove(0);
            }
            self.staging.push(storage.staging);
        }
    }

    pub(crate) fn trim(&mut self) {
        self.pool.clear();
        self.staging = Vec::new();
    }
}

impl<B: MemoryBackend> Drop for Manager<B> {
    fn drop(&mut self) {
        // Completion was never established, so freeing could release storage
        // the GPU still uses. Leaking is bounded to one job per manager.
        if let Some(storage) = self.quarantined.take() {
            std::mem::forget(storage);
        }
    }
}
