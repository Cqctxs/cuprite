//! Jobs: several buffers on the GPU, kernels over them, and one commit.

use crate::{
    backend::{DefaultBackend, MemoryBackend, Value},
    element::{bytes, bytes_mut},
    manager::Manager,
    Arg, Buffer, Element, Grid, Kernel, MemoryError,
};

/// A job exclusively borrows the manager and every CPU slice it writes, so
/// safe code can't touch them while the GPU may:
///
/// ```compile_fail,E0506
/// use cuprite_memory::{GpuMemory, MemoryConfig};
/// let mut memory = GpuMemory::open(MemoryConfig::default()).unwrap();
/// let mut values = [1u32, 2];
/// let mut job = memory.job().unwrap();
/// job.in_out(&mut values).unwrap();
/// values[0] = 9; // The mutable borrow is still in use below.
/// job.finish().unwrap();
/// ```
///
/// Results reach CPU slices only through `finish`, all together. Dropping a
/// job without calling `finish` discards every device change.
#[must_use = "call finish to commit results, or drop the job to discard them"]
pub struct Job<'a> {
    inner: JobCore<'a, DefaultBackend>,
}

impl<'a> Job<'a> {
    pub(crate) fn new(inner: JobCore<'a, DefaultBackend>) -> Self {
        Self { inner }
    }

    /// A buffer holding a copy of `host`, which the job never writes back.
    ///
    /// # Errors
    /// Returns an error if the job would exceed the device budget, or if
    /// allocation or the upload fails. After an upload failure the job can
    /// no longer be committed.
    pub fn input<T: Element>(&mut self, host: &[T]) -> Result<Buffer, MemoryError> {
        self.inner.input(host)
    }

    /// A zero-filled buffer whose contents replace `host` at `finish`, if a
    /// kernel was given it.
    ///
    /// # Errors
    /// As for [`Job::input`], without the upload.
    pub fn output<T: Element>(&mut self, host: &'a mut [T]) -> Result<Buffer, MemoryError> {
        self.inner.output(host)
    }

    /// A buffer holding a copy of `host`, whose contents replace `host` at
    /// `finish`, if a kernel was given it.
    ///
    /// # Errors
    /// As for [`Job::input`].
    pub fn in_out<T: Element>(&mut self, host: &'a mut [T]) -> Result<Buffer, MemoryError> {
        self.inner.in_out(host)
    }

    /// A zero-filled, GPU-only buffer of `len` elements.
    ///
    /// # Errors
    /// As for [`Job::output`].
    pub fn scratch<T: Element>(&mut self, len: usize) -> Result<Buffer, MemoryError> {
        self.inner.scratch::<T>(len)
    }

    /// Queue `kernel` over `grid` with `args`, after earlier work. An empty
    /// grid queues nothing.
    ///
    /// This is safe even for a kernel that misbehaves: every buffer element
    /// type accepts any bit pattern, and kernels can't reach CPU memory. A
    /// kernel can corrupt this job's results or fault the device, and the
    /// job's commit or the manager's next job then fails.
    ///
    /// # Errors
    /// Returns [`MemoryError::Launch`] if an argument belongs to another job
    /// or the kernel to another manager, or if the launch can't be queued.
    /// After a failed launch the job can no longer be committed.
    pub fn launch(&mut self, kernel: Kernel, grid: Grid, args: &[Arg]) -> Result<(), MemoryError> {
        self.inner.launch(kernel, grid, args)
    }

    /// Wait for the job's work and commit results to the CPU slices of
    /// every output and in-out buffer a kernel was given.
    ///
    /// # Errors
    /// Returns an error if any operation failed or the downloads or their
    /// completion fail. No CPU slice is then changed.
    pub fn finish(self) -> Result<(), MemoryError> {
        self.inner.finish()
    }
}

/// Storage one buffer owns. Keeping it together means cleanup moves one
/// value, so nothing can be released twice or left half-released.
pub(crate) struct Storage<Buf> {
    pub(crate) bytes: usize,
    /// Job-owned CPU copy. Transfers touch this, never the caller's slice.
    pub(crate) staging: Vec<u8>,
    /// `None` for an empty buffer, which needs no device storage.
    pub(crate) device: Option<Buf>,
}

/// A buffer's CPU side, which decides its transfers.
enum Host<'a> {
    Input,
    Output(&'a mut [u8]),
    InOut(&'a mut [u8]),
    Scratch,
}

struct Slot<'a, Buf> {
    host: Host<'a>,
    storage: Storage<Buf>,
    /// Set when the buffer is given to a launch, before it is queued.
    written: bool,
}

impl<Buf> Slot<'_, Buf> {
    fn downloads(&self) -> bool {
        self.written
            && self.storage.device.is_some()
            && matches!(self.host, Host::Output(_) | Host::InOut(_))
    }
}

/// Generic counterpart of [`Job`].
pub(crate) struct JobCore<'a, B: MemoryBackend> {
    pub(crate) manager: &'a mut Manager<B>,
    id: u64,
    slots: Vec<Slot<'a, B::Buffer>>,
    /// Device bytes the job holds.
    held: usize,
    /// Work may have been queued since the last completed synchronize.
    pending: bool,
    /// An operation failed to queue, so device contents can't be trusted.
    failed: bool,
    /// Uploads queued, counted in the stats once they complete.
    uploads: u64,
    uploaded_bytes: u64,
}

impl<'a, B: MemoryBackend> JobCore<'a, B> {
    pub(crate) fn new(manager: &'a mut Manager<B>, id: u64) -> Self {
        Self {
            manager,
            id,
            slots: Vec::new(),
            held: 0,
            pending: false,
            failed: false,
            uploads: 0,
            uploaded_bytes: 0,
        }
    }

    // Uploads go from job-owned staging, never from the caller's slice. CUDA
    // only says a copy from pageable memory "might be synchronous", so it
    // could otherwise still be reading after a failed job releases the borrow.

    pub(crate) fn input<T: Element>(&mut self, host: &[T]) -> Result<Buffer, MemoryError> {
        let data = bytes(host);
        let mut staging = self.manager.take_staging(data.len())?;
        staging.extend_from_slice(data);
        self.add_storage(Host::Input, staging, data.len(), true)
    }

    pub(crate) fn output<T: Element>(&mut self, host: &'a mut [T]) -> Result<Buffer, MemoryError> {
        let host = bytes_mut(host);
        let bytes = host.len();
        // Downloads land in staging first.
        let mut staging = self.manager.take_staging(bytes)?;
        staging.resize(bytes, 0);
        self.add_storage(Host::Output(host), staging, bytes, false)
    }

    pub(crate) fn in_out<T: Element>(&mut self, host: &'a mut [T]) -> Result<Buffer, MemoryError> {
        let host = bytes_mut(host);
        let bytes = host.len();
        let mut staging = self.manager.take_staging(bytes)?;
        staging.extend_from_slice(host);
        self.add_storage(Host::InOut(host), staging, bytes, true)
    }

    pub(crate) fn scratch<T: Element>(&mut self, len: usize) -> Result<Buffer, MemoryError> {
        let bytes = size_of::<T>()
            .checked_mul(len)
            .ok_or(MemoryError::SizeOverflow)?;
        // GPU-only: no CPU copy at all.
        self.add_storage(Host::Scratch, Vec::new(), bytes, false)
    }

    /// Adds a buffer of `bytes` with its staging, queueing its upload or,
    /// for a reused buffer that isn't uploaded, zero-filling it.
    fn add_storage(
        &mut self,
        host: Host<'a>,
        staging: Vec<u8>,
        bytes: usize,
        upload: bool,
    ) -> Result<Buffer, MemoryError> {
        // Reject oversized requests before allocating anything.
        let limit = self.manager.config.max_device_bytes;
        let held = self
            .held
            .checked_add(bytes)
            .ok_or(MemoryError::SizeOverflow)?;
        if held > limit {
            self.manager.recycle(Storage {
                bytes: 0,
                staging,
                device: None,
            });
            return Err(MemoryError::BudgetExceeded {
                requested: held,
                limit,
            });
        }
        let (device, fresh) = if bytes == 0 {
            (None, true)
        } else {
            match self.manager.device_storage(self.held, bytes) {
                Ok((device, fresh)) => (Some(device), fresh),
                Err(error) => {
                    self.manager.recycle(Storage {
                        bytes: 0,
                        staging,
                        device: None,
                    });
                    return Err(error);
                }
            }
        };
        // Set before any transfer, so a forgotten job is detected.
        self.manager.active = true;
        self.held = held;
        let index = self.slots.len();
        self.slots.push(Slot {
            host,
            storage: Storage {
                bytes,
                staging,
                device,
            },
            written: false,
        });

        let slot = &mut self.slots[index];
        let Some(device) = slot.storage.device.as_mut() else {
            return Ok(Buffer {
                job: self.id,
                index,
            });
        };
        let backend = &mut self.manager.backend;
        let queued = if upload {
            self.pending = true;
            self.uploads += 1;
            self.uploaded_bytes += bytes as u64;
            backend.upload(device, &slot.storage.staging)
        } else if !fresh {
            // A reused buffer still holds an earlier job's values.
            self.pending = true;
            backend.zero(device)
        } else {
            Ok(())
        };
        if let Err(error) = queued {
            self.failed = true;
            return Err(error);
        }
        Ok(Buffer {
            job: self.id,
            index,
        })
    }

    /// The job is missing work it was meant to do, so it can't be committed.
    pub(crate) fn mark_failed(&mut self) {
        self.failed = true;
    }

    pub(crate) fn launch(
        &mut self,
        kernel: Kernel,
        grid: Grid,
        args: &[Arg],
    ) -> Result<(), MemoryError> {
        for arg in args {
            if let Arg::Buffer(buffer) = arg {
                if buffer.job != self.id || buffer.index >= self.slots.len() {
                    return Err(MemoryError::Launch(
                        "a buffer belongs to another job".into(),
                    ));
                }
            }
        }
        // Checked before marking anything, so a refused launch changes nothing.
        self.manager.launch_parts(kernel)?;
        if grid.is_empty() {
            return Ok(());
        }
        for arg in args {
            if let Arg::Buffer(buffer) = arg {
                self.slots[buffer.index].written = true;
            }
        }
        let values: Vec<Value<'_, B::Buffer>> = args
            .iter()
            .map(|arg| match *arg {
                Arg::Buffer(buffer) => {
                    Value::Buffer(self.slots[buffer.index].storage.device.as_ref())
                }
                Arg::U32(value) => Value::U32(value),
                Arg::I32(value) => Value::I32(value),
                Arg::U64(value) => Value::U64(value),
                Arg::I64(value) => Value::I64(value),
                Arg::F32(value) => Value::F32(value),
                Arg::F64(value) => Value::F64(value),
            })
            .collect();
        self.pending = true;
        let (backend, kernel) = self.manager.launch_parts(kernel)?;
        let result = backend.launch(kernel, grid, &values);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    /// Commit results and consume the job.
    ///
    /// Every buffer to commit is downloaded into staging, and only once all
    /// downloads have completed is staging copied into the CPU slices. A
    /// failure before that leaves every CPU slice unchanged.
    pub(crate) fn finish(mut self) -> Result<(), MemoryError> {
        if self.failed {
            // `Drop` discards the device copies after waiting for queued work.
            return Err(MemoryError::Launch(
                "an operation in this job failed; results discarded".into(),
            ));
        }
        let mut slots = std::mem::take(&mut self.slots);
        let backend = &mut self.manager.backend;
        let mut downloads = 0;
        let mut downloaded_bytes = 0;
        let mut result = Ok(());
        for slot in slots.iter_mut().filter(|slot| slot.downloads()) {
            let device = slot
                .storage
                .device
                .as_ref()
                .expect("checked by `downloads`");
            self.pending = true;
            result = backend.download(device, &mut slot.storage.staging);
            if result.is_err() {
                break;
            }
            downloads += 1;
            downloaded_bytes += slot.storage.bytes as u64;
        }
        if result.is_ok() && self.pending {
            result = backend.synchronize();
        }
        if let Err(error) = result {
            // Storage from a failed operation is freed, not pooled.
            let storage = slots.into_iter().map(|slot| slot.storage).collect();
            drop(self.manager.settle(storage, true));
            return Err(error);
        }
        self.pending = false;
        // Count transfers only once completion is confirmed.
        let stats = &mut self.manager.stats;
        stats.uploads += self.uploads;
        stats.uploaded_bytes += self.uploaded_bytes;
        stats.downloads += downloads;
        stats.downloaded_bytes += downloaded_bytes;

        for slot in slots {
            let downloads = slot.downloads();
            if let (Host::Output(host) | Host::InOut(host), true) = (slot.host, downloads) {
                host.copy_from_slice(&slot.storage.staging);
            }
            self.manager.recycle(slot.storage);
        }
        Ok(())
    }
}

impl<B: MemoryBackend> Drop for JobCore<'_, B> {
    /// Discard the job: no copy-back. Waits for queued work before reusing
    /// storage, and quarantines it if that wait fails. Never panics.
    fn drop(&mut self) {
        let slots = std::mem::take(&mut self.slots);
        if !slots.is_empty() {
            let storage = slots.into_iter().map(|slot| slot.storage).collect();
            // Storage from a failed operation is freed, not pooled.
            if let Some(storage) = self.manager.settle(storage, self.pending) {
                if !self.failed {
                    self.manager.stats.uploads += self.uploads;
                    self.manager.stats.uploaded_bytes += self.uploaded_bytes;
                    for storage in storage {
                        self.manager.recycle(storage);
                    }
                }
            }
        }
        self.manager.active = false;
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use super::*;
    use crate::{
        test_support::{Event, FakeBackend, Operation},
        MemoryConfig, MemoryStats,
    };

    fn fake_manager() -> Manager<FakeBackend> {
        Manager::new(FakeBackend::default(), MemoryConfig::default())
    }

    fn failing_manager(failures: &[Operation]) -> Manager<FakeBackend> {
        Manager::new(FakeBackend::failing(failures), MemoryConfig::default())
    }

    fn budget(bytes: usize) -> Manager<FakeBackend> {
        let config = MemoryConfig {
            max_device_bytes: bytes,
            ..MemoryConfig::default()
        };
        Manager::new(FakeBackend::default(), config)
    }

    /// Adds one to every `u32` of `buffer`.
    fn add_one(job: &mut JobCore<'_, FakeBackend>, buffer: Buffer) {
        let kernel = job.manager.kernel("", "add_one_u32").unwrap();
        job.launch(kernel, Grid::Linear(1), &[Arg::Buffer(buffer)])
            .unwrap();
    }

    fn copy(job: &mut JobCore<'_, FakeBackend>, from: Buffer, to: Buffer) {
        let kernel = job.manager.kernel("", "copy").unwrap();
        job.launch(
            kernel,
            Grid::Linear(1),
            &[Arg::Buffer(from), Arg::Buffer(to)],
        )
        .unwrap();
    }

    fn no_idle() -> Vec<usize> {
        Vec::new()
    }

    #[test]
    fn an_empty_job_creates_no_allocation_or_transfer() {
        let mut manager = fake_manager();
        let mut empty: [u32; 0] = [];

        let mut job = manager.job().unwrap();
        job.in_out(&mut empty).unwrap();
        job.finish().unwrap();

        assert!(manager.backend.events.is_empty());
        assert_eq!(manager.stats(), MemoryStats::default());
    }

    /// Each role decides its transfers: inputs upload only, outputs download
    /// only, in-out buffers do both, and scratch does neither.
    #[test]
    fn buffer_roles_decide_transfers() {
        let mut manager = fake_manager();
        let input = [1u32, 2];
        let mut output = [0u32; 2];
        let mut in_out = [5u32];

        let mut job = manager.job().unwrap();
        let a = job.input(&input).unwrap();
        let b = job.output(&mut output).unwrap();
        let c = job.in_out(&mut in_out).unwrap();
        let scratch = job.scratch::<u32>(2).unwrap();
        copy(&mut job, a, scratch);
        add_one(&mut job, scratch);
        copy(&mut job, scratch, b);
        add_one(&mut job, c);
        job.finish().unwrap();

        assert_eq!(output, [2, 3]);
        assert_eq!(in_out, [6]);
        assert_eq!(
            manager.backend.events,
            [
                Event::Allocate(8),
                Event::Upload(8),
                Event::Allocate(8),
                Event::Allocate(4),
                Event::Upload(4),
                Event::Allocate(8),
                Event::Load("copy".into()),
                Event::Launch("copy".into()),
                Event::Load("add_one_u32".into()),
                Event::Launch("add_one_u32".into()),
                Event::Launch("copy".into()),
                Event::Launch("add_one_u32".into()),
                Event::Download(8),
                Event::Download(4),
                Event::Synchronize,
            ]
        );
        let stats = manager.stats();
        assert_eq!((stats.uploads, stats.uploaded_bytes), (2, 12));
        assert_eq!((stats.downloads, stats.downloaded_bytes), (2, 12));
    }

    /// A writable buffer no kernel was given is left alone: its CPU slice
    /// keeps its contents and nothing is downloaded.
    #[test]
    fn buffers_no_kernel_was_given_are_not_written_back() {
        let mut manager = fake_manager();
        let mut output = [7u32, 8];
        let mut in_out = [9u32];

        let mut job = manager.job().unwrap();
        job.output(&mut output).unwrap();
        job.in_out(&mut in_out).unwrap();
        job.finish().unwrap();

        assert_eq!((output, in_out), ([7, 8], [9]));
        assert_eq!(manager.stats().downloads, 0);
    }

    /// Element types other than `u32` move as their bytes.
    #[test]
    fn any_element_type_round_trips() {
        let mut manager = fake_manager();
        let input = [1.5f64, -0.0, f64::NAN];
        let mut output = [0f64; 3];

        let mut job = manager.job().unwrap();
        let a = job.input(&input).unwrap();
        let b = job.output(&mut output).unwrap();
        copy(&mut job, a, b);
        job.finish().unwrap();

        assert_eq!(output.map(f64::to_bits), input.map(f64::to_bits));
    }

    /// The same buffer may be given twice to one launch.
    #[test]
    fn a_buffer_can_be_passed_twice() {
        let mut manager = fake_manager();
        let mut values = [3u32];

        let mut job = manager.job().unwrap();
        let a = job.in_out(&mut values).unwrap();
        copy(&mut job, a, a);
        job.finish().unwrap();

        assert_eq!(values, [3]);
    }

    /// Results already downloaded still don't reach any CPU slice if the
    /// commit's wait for completion fails.
    #[test]
    fn commits_are_all_or_nothing() {
        let mut manager = failing_manager(&[Operation::Synchronize]);
        let mut first = [1u32];
        let mut second = [2u32];

        let mut job = manager.job().unwrap();
        let a = job.in_out(&mut first).unwrap();
        let b = job.in_out(&mut second).unwrap();
        add_one(&mut job, a);
        add_one(&mut job, b);
        let error = job.finish().unwrap_err();

        assert!(matches!(error, MemoryError::Completion(_)));
        assert_eq!((first, second), ([1], [2]));
        assert_eq!(manager.stats().downloads, 0);
        // The second wait succeeded, so nothing is quarantined.
        assert!(!manager.poisoned);
    }

    /// The budget covers all of a job's buffers and is checked before
    /// allocating the buffer that would exceed it.
    #[test]
    fn the_budget_covers_every_buffer_of_a_job() {
        let mut manager = budget(12);
        let input = [1u32, 2];
        let mut output = [0u32; 2];

        let mut job = manager.job().unwrap();
        job.input(&input).unwrap();
        let error = job.output(&mut output).unwrap_err();
        drop(job);

        assert_eq!(
            error,
            MemoryError::BudgetExceeded {
                requested: 16,
                limit: 12,
            }
        );
        assert_eq!(
            manager.backend.events,
            [Event::Allocate(8), Event::Upload(8), Event::Synchronize]
        );
    }

    /// Dropping without `finish` discards device changes, after waiting for
    /// work that may have been queued, and keeps the storage for reuse.
    #[test]
    fn dropping_a_job_discards_device_changes() {
        let mut manager = fake_manager();
        let mut values = [1u32, 2];

        let mut job = manager.job().unwrap();
        let a = job.in_out(&mut values).unwrap();
        add_one(&mut job, a);
        drop(job);

        assert_eq!(values, [1, 2]);
        assert_eq!(manager.backend.events.last(), Some(&Event::Synchronize));
        assert!(!manager.poisoned);
        assert_eq!(manager.pool.idle_sizes(), [8]);
    }

    /// A panic unwinding through a job discards its changes, and the manager
    /// stays usable.
    #[test]
    fn a_panic_during_a_job_discards_changes() {
        let mut manager = fake_manager();
        let mut values = [1u32, 2];

        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut job = manager.job().unwrap();
            let a = job.in_out(&mut values).unwrap();
            add_one(&mut job, a);
            panic!("simulated failure between GPU operations");
        }));

        assert!(result.is_err());
        assert_eq!(values, [1, 2]);
        assert!(!manager.poisoned);
        manager.job().unwrap().finish().unwrap();
    }

    /// Allocation failure happens before any transfer for that buffer, and
    /// the manager stays usable.
    #[test]
    fn allocation_failure_leaves_the_manager_usable() {
        let mut manager = failing_manager(&[Operation::Allocate]);
        let mut values = [1u32, 2];

        let mut job = manager.job().unwrap();
        let error = job.in_out(&mut values).unwrap_err();
        drop(job);

        assert!(matches!(error, MemoryError::DeviceAllocation(_)));
        assert!(manager.backend.events.is_empty());
        let mut job = manager.job().unwrap();
        job.in_out(&mut values).unwrap();
        job.finish().unwrap();
    }

    /// A failed upload means the job can't be committed. Its storage is
    /// freed after waiting, never pooled, and the manager stays usable.
    #[test]
    fn an_upload_failure_discards_the_job() {
        let mut manager = failing_manager(&[Operation::Upload]);
        let mut values = [1u32, 2];

        let mut job = manager.job().unwrap();
        assert!(matches!(
            job.in_out(&mut values),
            Err(MemoryError::Transfer(_))
        ));
        assert!(job.finish().is_err());

        assert_eq!(values, [1, 2]);
        assert_eq!(
            manager.backend.events,
            [Event::Allocate(8), Event::Synchronize]
        );
        assert!(!manager.poisoned);
        assert_eq!(manager.stats().uploads, 0);
        assert_eq!(manager.pool.idle_sizes(), no_idle());
    }

    /// If completion can't be established at commit, storage is
    /// quarantined, nothing reaches the CPU, and later work is refused.
    #[test]
    fn failed_completion_quarantines_and_poisons() {
        let mut manager = failing_manager(&[Operation::Synchronize, Operation::Synchronize]);
        let mut values = [1u32, 2];

        let mut job = manager.job().unwrap();
        let a = job.in_out(&mut values).unwrap();
        add_one(&mut job, a);
        let error = job.finish().unwrap_err();

        assert!(matches!(error, MemoryError::Completion(_)));
        assert_eq!(values, [1, 2]);
        assert!(manager.poisoned);
        assert!(manager.quarantined.is_some());
        assert_eq!(manager.pool.idle_sizes(), no_idle());
        let events_before = manager.backend.events.len();
        assert_eq!(manager.job().err(), Some(MemoryError::Poisoned));
        assert_eq!(
            manager.kernel("", "copy").err(),
            Some(MemoryError::Poisoned)
        );
        assert_eq!(manager.backend.events.len(), events_before);
    }

    /// Dropping a job whose completion fails quarantines too.
    #[test]
    fn failed_completion_while_discarding_quarantines() {
        let mut manager = fake_manager();
        let mut values = [1u32, 2];

        let mut job = manager.job().unwrap();
        let a = job.in_out(&mut values).unwrap();
        add_one(&mut job, a);
        job.manager.backend.failures = vec![Operation::Synchronize];
        drop(job);

        assert!(manager.poisoned);
        assert!(manager.quarantined.is_some());
        assert_eq!(manager.pool.idle_sizes(), no_idle());
    }

    /// Forgetting a job skips its cleanup. The caller's slices are free
    /// again, the job's storage leaks, and the manager refuses further work.
    #[test]
    fn a_forgotten_job_poisons_the_manager() {
        let mut manager = fake_manager();
        let mut values = vec![1u32, 2];

        let mut job = manager.job().unwrap();
        job.in_out(&mut values[..]).unwrap();
        std::mem::forget(job);
        // The borrow ended, so the caller may change or free its slice.
        values[0] = 9;
        drop(values);

        assert_eq!(manager.job().err(), Some(MemoryError::Poisoned));
        assert_eq!(manager.stats().device_allocations, 1);
    }

    /// A job that holds nothing leaves nothing behind when forgotten.
    #[test]
    fn forgetting_an_empty_job_is_harmless() {
        let mut manager = fake_manager();

        std::mem::forget(manager.job().unwrap());

        manager.job().unwrap().finish().unwrap();
    }

    /// Same-size buffers are reused through the pool, and a reused output
    /// buffer is zero-filled, so an earlier job's values never leak out.
    #[test]
    fn pooled_buffers_are_reused_without_leaking_old_values() {
        let mut manager = fake_manager();
        let mut first = [1u32, 2];
        let mut job = manager.job().unwrap();
        let a = job.in_out(&mut first).unwrap();
        add_one(&mut job, a);
        job.finish().unwrap();

        let mut output = [9u32, 9];
        let mut job = manager.job().unwrap();
        let b = job.output(&mut output).unwrap();
        // Copying a buffer onto itself writes nothing new, so it stays zero.
        copy(&mut job, b, b);
        job.finish().unwrap();

        assert_eq!(output, [0, 0]);
        let stats = manager.stats();
        assert_eq!((stats.device_allocations, stats.pool_hits), (1, 1));
        assert!(manager.backend.events.contains(&Event::Zero(8)));
    }

    /// Buffers of different sizes stay pooled together while they fit the
    /// budget; the oldest idle ones are released to make room.
    #[test]
    fn idle_buffers_are_released_only_to_stay_within_the_budget() {
        let mut manager = budget(24);
        let mut four = [0u32; 4];
        let mut two = [0u32; 2];
        let mut three = [0u32; 3];

        for values in [&mut four[..], &mut two[..], &mut three[..]] {
            let mut job = manager.job().unwrap();
            job.in_out(values).unwrap();
            job.finish().unwrap();
        }

        // 16 + 8 fit together; 12 more needed the 16 released.
        assert_eq!(manager.pool.idle_sizes(), [8, 12]);
        assert_eq!(manager.stats().device_allocations, 3);
    }

    /// Host staging is reused across jobs; `trim` releases it.
    #[test]
    fn host_staging_is_reused_until_trimmed() {
        let mut manager = fake_manager();
        let mut four = [1u32, 2, 3, 4];
        let mut three = [5u32, 6, 7];

        let mut job = manager.job().unwrap();
        job.in_out(&mut four).unwrap();
        job.finish().unwrap();
        let spare = manager.staging[0].as_ptr();
        let mut job = manager.job().unwrap();
        let a = job.in_out(&mut three).unwrap();
        assert_eq!(job.slots[a.index].storage.staging.as_ptr(), spare);
        add_one(&mut job, a);
        job.finish().unwrap();

        assert_eq!(three, [6, 7, 8], "reused staging is resized to the slice");
        manager.trim();
        assert!(manager.staging.is_empty());
        assert_eq!(manager.pool.idle_sizes(), no_idle());
    }

    /// Buffers from failed commits and quarantined jobs are never pooled.
    #[test]
    fn failed_or_quarantined_buffers_never_enter_the_pool() {
        let mut manager = failing_manager(&[Operation::Download]);
        let mut values = [1u32, 2];
        let mut job = manager.job().unwrap();
        let a = job.in_out(&mut values).unwrap();
        add_one(&mut job, a);
        job.finish().unwrap_err();
        assert_eq!(manager.pool.idle_sizes(), no_idle());
        assert!(!manager.poisoned);

        let mut manager = fake_manager();
        let mut job = manager.job().unwrap();
        let a = job.in_out(&mut values).unwrap();
        add_one(&mut job, a);
        job.manager.backend.failures = vec![Operation::Synchronize];
        drop(job);
        assert!(manager.quarantined.is_some());
        assert_eq!(manager.pool.idle_sizes(), no_idle());
    }

    /// A failed launch means the job can't be committed, even if later
    /// launches succeed.
    #[test]
    fn a_failed_launch_discards_the_job() {
        let mut manager = failing_manager(&[Operation::Launch]);
        let mut values = [1u32];

        let mut job = manager.job().unwrap();
        let a = job.in_out(&mut values).unwrap();
        let kernel = job.manager.kernel("", "add_one_u32").unwrap();
        let args = [Arg::Buffer(a)];
        assert!(matches!(
            job.launch(kernel, Grid::Linear(1), &args),
            Err(MemoryError::Launch(_))
        ));
        job.launch(kernel, Grid::Linear(1), &args).unwrap();
        assert!(matches!(job.finish(), Err(MemoryError::Launch(_))));

        assert_eq!(values, [1]);
        assert!(!manager.poisoned);
    }

    /// Buffers of another job and kernels of another manager are refused
    /// before anything is queued or marked as written.
    #[test]
    fn handles_from_elsewhere_are_refused() {
        let mut other = fake_manager();
        let foreign_kernel = other.kernel("", "add_one_u32").unwrap();
        let mut manager = fake_manager();
        let mut values = [1u32];

        let mut job = manager.job().unwrap();
        let a = job.in_out(&mut values).unwrap();
        let foreign_buffer = Buffer {
            job: a.job + 1,
            index: 0,
        };
        let kernel = job.manager.kernel("", "add_one_u32").unwrap();
        let events = job.manager.backend.events.len();

        let error = job.launch(foreign_kernel, Grid::Linear(1), &[Arg::Buffer(a)]);
        assert!(matches!(error, Err(MemoryError::Launch(_))));
        let error = job.launch(kernel, Grid::Linear(1), &[Arg::Buffer(foreign_buffer)]);
        assert!(matches!(error, Err(MemoryError::Launch(_))));
        assert_eq!(job.manager.backend.events.len(), events);
        job.finish().unwrap();

        assert_eq!(values, [1]);
        assert_eq!(manager.stats().downloads, 0);
    }

    /// The same PTX and entry load once; an empty grid queues nothing.
    #[test]
    fn kernels_load_once_and_empty_grids_do_nothing() {
        let mut manager = fake_manager();
        let first = manager.kernel("ptx", "copy").unwrap();
        assert_eq!(manager.kernel("ptx", "copy").unwrap(), first);
        assert_ne!(manager.kernel("ptx", "add_one_u32").unwrap(), first);
        let mut values = [1u32];

        let mut job = manager.job().unwrap();
        let a = job.in_out(&mut values).unwrap();
        let empty = Grid::Plane {
            width: 0,
            height: 4,
        };
        job.launch(first, empty, &[Arg::Buffer(a), Arg::Buffer(a)])
            .unwrap();
        job.finish().unwrap();

        let events = &manager.backend.events;
        let loads = events
            .iter()
            .filter(|event| matches!(event, Event::Load(_)))
            .count();
        assert_eq!(loads, 2);
        assert!(!events.iter().any(|event| matches!(event, Event::Launch(_))));
        assert_eq!(manager.stats().downloads, 0);
    }
}
