use crate::{
    backend::{DefaultBackend, MemoryBackend},
    manager::Manager,
    MapStep, MemoryError,
};

/// A session exclusively borrows the manager and original CPU slice.
///
/// The type signature already prevents conflicting safe CPU access:
///
/// ```compile_fail,E0506
/// use cuprite_memory::{GpuMemory, MemoryConfig};
/// let mut memory = GpuMemory::open(MemoryConfig::default()).unwrap();
/// let mut values = [1u32, 2];
/// let session = memory.begin_mut(&mut values).unwrap();
/// values[0] = 9; // The mutable borrow is still in use below.
/// session.finish().unwrap();
/// ```
///
/// Consuming the session ends that borrow:
///
/// ```no_run
/// # use cuprite_memory::{GpuMemory, MemoryConfig, MemoryError};
/// # fn example() -> Result<(), MemoryError> {
/// let mut memory = GpuMemory::open(MemoryConfig::default())?;
/// let mut values = [1u32, 2];
/// memory.begin_mut(&mut values)?.finish()?;
/// values[0] = 9;
/// # Ok(())
/// # }
/// ```
///
/// Dropping a session without calling `finish` discards device changes.
#[must_use = "call finish to commit results, or discard the session"]
pub struct ResidentSlice<'a> {
    inner: Session<'a, DefaultBackend>,
}

impl<'a> ResidentSlice<'a> {
    pub(crate) fn new(inner: Session<'a, DefaultBackend>) -> Self {
        Self { inner }
    }

    /// Commit device results to the original slice and end the session.
    ///
    /// # Errors
    /// Returns an error if the download or its completion fails; the original
    /// slice is then unchanged.
    pub fn finish(self) -> Result<(), MemoryError> {
        self.inner.finish()
    }

    /// Queue `step` over the GPU copy, after earlier operations.
    ///
    /// # Errors
    /// Returns [`MemoryError::Launch`] if the operation cannot be queued.
    /// The session then cannot be committed: `finish` returns an error and
    /// the original slice is left unchanged.
    pub fn map(&mut self, step: MapStep) -> Result<(), MemoryError> {
        self.inner.map(step)
    }

    #[cfg(all(test, feature = "cuda"))]
    pub(crate) fn inner_mut(&mut self) -> &mut Session<'a, DefaultBackend> {
        &mut self.inner
    }
}

/// Everything a nonempty session owns. `Buf` is the backend's buffer type. Keeping these together means cleanup
/// moves one value, so nothing can be released twice or left half-released.
pub(crate) struct Resources<Buf> {
    /// Session-owned CPU copy. Transfers touch this, never the caller's slice.
    pub(crate) staging: Vec<u32>,
    /// The device working copy; same length as `staging`.
    pub(crate) device: Buf,
}

/// Generic counterpart of [`ResidentSlice`]. Same borrowing shape: `'a` ties
/// the manager and the original slice to this session.
#[must_use = "call finish to commit results, or drop to discard them"]
pub(crate) struct Session<'a, B: MemoryBackend> {
    pub(crate) manager: &'a mut Manager<B>,
    pub(crate) host: &'a mut [u32],
    /// `None` for an empty session, which owns nothing.
    pub(crate) resources: Option<Resources<B::Buffer>>,
    /// Set *before* handing out write access, even if no write happens.
    pub(crate) may_have_writes: bool,
    /// Set when an operation failed to queue. The device copy is then
    /// missing that operation, so `finish` must not commit it.
    pub(crate) failed: bool,
}

// `'_` stands for the session's lifetime, which these methods never need to name.
impl<B: MemoryBackend> Session<'_, B> {
    /// Crate-private write access to the device working copy.
    ///
    /// Lends the buffer out (`&mut`) so the session keeps ownership, and
    /// marks `may_have_writes` so `finish` knows to copy back. An empty
    /// session returns `None`: it has no buffer and must never reach a kernel.
    #[cfg(test)]
    pub(crate) fn device_mut(&mut self) -> Option<&mut B::Buffer> {
        self.write_parts().map(|(_, device)| device)
    }

    /// The backend and device buffer together, for launching work on the
    /// buffer (CUDA needs the stream). Marks possible writes like `device_mut`.
    ///
    /// Callers must queue work only on this backend's stream and must not
    /// keep anything derived from the buffer beyond the borrow.
    pub(crate) fn write_parts(&mut self) -> Option<(&mut B, &mut B::Buffer)> {
        // Borrow the bundle rather than moving it: `finish` still needs it.
        let resources = self.resources.as_mut()?;
        self.may_have_writes = true;
        Some((&mut self.manager.backend, &mut resources.device))
    }

    /// Queue `step` over the device copy. An empty session has nothing to
    /// map. After a failure, `finish` refuses to commit.
    pub(crate) fn map(&mut self, step: MapStep) -> Result<(), MemoryError> {
        let Some((backend, device)) = self.write_parts() else {
            return Ok(());
        };
        let result = backend.map(device, step);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    /// Commit results to the original slice and consume the session.
    ///
    /// With no possible writes the device copy still equals `host`, so
    /// nothing is transferred. Otherwise the device copy is downloaded into
    /// staging, and only once that has completed is staging copied into
    /// `host`. A failed download or synchronize leaves `host` unchanged.
    pub(crate) fn finish(mut self) -> Result<(), MemoryError> {
        if self.failed {
            // `Drop` discards the device copy after waiting for queued work.
            return Err(MemoryError::Launch(
                "an operation in this session failed; results discarded".into(),
            ));
        }
        // Take the bundle; `Drop` then sees `None` and only ends the session.
        let Some(mut resources) = self.resources.take() else {
            return Ok(()); // Empty session: nothing was staged or allocated.
        };
        if !self.may_have_writes {
            // Device copy is unchanged and no work is queued since the
            // upload completed, so the buffer can be reused right away.
            self.manager.recycle(resources);
            return Ok(());
        }
        // Lengths were fixed in begin_mut; copy_from_slice relies on it.
        debug_assert_eq!(resources.staging.len(), self.host.len());

        let backend = &mut self.manager.backend;
        let downloaded = backend
            .download(&resources.device, &mut resources.staging)
            .and_then(|()| backend.synchronize());
        if let Err(error) = downloaded {
            // Storage from a failed operation is freed, not pooled.
            drop(self.manager.settle(resources, true));
            return Err(error);
        }
        // Count the download only once completion is confirmed.
        self.manager.stats.downloads += 1;
        self.manager.stats.downloaded_bytes += (resources.staging.len() * size_of::<u32>()) as u64;

        self.host.copy_from_slice(&resources.staging);
        self.manager.recycle(resources);
        Ok(())
    }
}

impl<B: MemoryBackend> Drop for Session<'_, B> {
    /// Discard the session: no copy-back. Waits for queued work before
    /// reusing storage, and quarantines it if that wait fails. Never panics.
    fn drop(&mut self) {
        if let Some(resources) = self.resources.take() {
            if let Some(resources) = self.manager.settle(resources, self.may_have_writes) {
                self.manager.recycle(resources);
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

    /// Empty input is an inert session: no allocation, transfer, or stats.
    #[test]
    fn empty_input_creates_no_allocation_or_transfer() {
        let mut manager = fake_manager();
        let mut values: [u32; 0] = [];

        manager.begin_mut(&mut values).unwrap().finish().unwrap();

        assert!(manager.backend.events.is_empty());
        assert_eq!(manager.stats(), MemoryStats::default());
    }

    /// A device write reaches the original slice only through `finish`,
    /// with exactly one upload and one download.
    #[test]
    fn finish_copies_device_writes_back_to_the_original_slice() {
        let mut manager = fake_manager();
        let mut values = [1, 2];

        let mut session = manager.begin_mut(&mut values).unwrap();
        session.device_mut().unwrap().copy_from_slice(&[8, 9]);
        // `values` is borrowed by the session until here, so it can only be
        // checked after `finish`.
        session.finish().unwrap();

        assert_eq!(values, [8, 9]);
        assert_eq!(
            manager.backend.events,
            [
                Event::Allocate(2),
                Event::Upload(2),
                Event::Synchronize,
                Event::Download(2),
                Event::Synchronize,
            ]
        );
        let stats = manager.stats();
        assert_eq!((stats.uploads, stats.downloads), (1, 1));
        assert_eq!((stats.uploaded_bytes, stats.downloaded_bytes), (8, 8));
        assert_eq!(stats.device_allocations, 1);
    }

    /// Without write access, `finish` has nothing to copy back.
    #[test]
    fn unwritten_session_finishes_without_downloading() {
        let mut manager = fake_manager();
        let mut values = [1, 2];

        let session = manager.begin_mut(&mut values).unwrap();
        session.finish().unwrap();

        assert_eq!(values, [1, 2]);
        assert_eq!(
            manager.backend.events,
            [Event::Allocate(2), Event::Upload(2), Event::Synchronize]
        );
        assert_eq!(manager.stats().downloads, 0);
    }

    /// Eight operations reuse one device buffer; only the ends transfer.
    #[test]
    fn repeated_writes_reuse_one_buffer_and_download_once() {
        let mut manager = fake_manager();
        let mut values = [1, u32::MAX];

        let mut session = manager.begin_mut(&mut values).unwrap();
        // The buffer's address: if it never changes, no buffer was reallocated.
        let first_buffer = session.device_mut().unwrap().as_ptr();
        for _ in 0..8 {
            let device = session.device_mut().unwrap();
            assert_eq!(device.as_ptr(), first_buffer);
            for value in device.iter_mut() {
                // `*value` writes through the reference; `wrapping_add` wraps
                // past u32::MAX instead of panicking, as GPU arithmetic does.
                *value = value.wrapping_add(7);
            }
        }
        session.finish().unwrap();

        // 1 + 8 * 7 = 57; u32::MAX + 56 wraps around to 55.
        assert_eq!(values, [57, 55]);
        assert_eq!(
            manager.backend.events,
            [
                Event::Allocate(2),
                Event::Upload(2),
                Event::Synchronize,
                Event::Download(2),
                Event::Synchronize,
            ]
        );
    }

    /// The budget check happens before any allocation or transfer.
    #[test]
    fn oversized_input_is_rejected_before_allocation() {
        // `..MemoryConfig::default()` fills every field not listed here.
        let config = MemoryConfig {
            max_device_bytes: 4,
            ..MemoryConfig::default()
        };
        let mut manager = Manager::new(FakeBackend::default(), config);
        let mut values = [1, 2];

        // Expect an error; fail the test if `begin_mut` succeeded instead.
        let Err(error) = manager.begin_mut(&mut values) else {
            panic!("expected begin_mut to reject the request");
        };

        assert_eq!(
            error,
            MemoryError::BudgetExceeded {
                requested: 8,
                limit: 4,
            }
        );
        assert_eq!(values, [1, 2]);
        assert!(manager.backend.events.is_empty());
        assert_eq!(manager.stats(), MemoryStats::default());
    }

    /// Dropping without `finish` discards device changes, after waiting
    /// for work that may have been queued on the buffer.
    #[test]
    fn dropping_a_session_discards_device_changes() {
        let mut manager = fake_manager();
        let mut values = [1, 2];

        let mut session = manager.begin_mut(&mut values).unwrap();
        session.device_mut().unwrap().copy_from_slice(&[8, 9]);
        drop(session); // Give up on the session without finishing it.

        assert_eq!(values, [1, 2]);
        assert_eq!(
            manager.backend.events,
            [
                Event::Allocate(2),
                Event::Upload(2),
                Event::Synchronize,
                Event::Synchronize,
            ]
        );
        assert!(!manager.poisoned);
    }

    /// A panic unwinding through a session discards its changes, and the
    /// manager stays usable.
    #[test]
    fn panic_during_a_session_discards_changes() {
        let mut manager = fake_manager();
        let mut values = [1, 2];

        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut session = manager.begin_mut(&mut values).unwrap();
            session.device_mut().unwrap().copy_from_slice(&[8, 9]);
            panic!("simulated failure between GPU operations");
        }));

        assert!(result.is_err());
        assert_eq!(values, [1, 2]);
        assert!(!manager.poisoned);
        manager.begin_mut(&mut values).unwrap().finish().unwrap();
    }

    /// Allocation failure happens before any transfer, so nothing needs
    /// waiting for and the manager stays usable.
    #[test]
    fn allocation_failure_leaves_the_manager_usable() {
        let mut manager = failing_manager(&[Operation::Allocate]);
        let mut values = [1, 2];

        let error = manager.begin_mut(&mut values).err().unwrap();

        assert!(matches!(error, MemoryError::DeviceAllocation(_)));
        assert!(manager.backend.events.is_empty());
        assert_eq!(manager.stats().device_allocations, 0);
        manager.begin_mut(&mut values).unwrap().finish().unwrap();
    }

    /// A failed upload waits for completion before freeing staging, and the
    /// manager can then accept a new session.
    #[test]
    fn upload_failure_waits_before_releasing_storage() {
        let mut manager = failing_manager(&[Operation::Upload]);
        let mut values = [1, 2];

        let error = manager.begin_mut(&mut values).err().unwrap();

        assert!(matches!(error, MemoryError::Transfer(_)));
        assert_eq!(values, [1, 2]);
        assert_eq!(
            manager.backend.events,
            [Event::Allocate(2), Event::Synchronize]
        );
        assert!(!manager.poisoned);
        assert!(manager.quarantined.is_none());
        assert_eq!(manager.stats().uploads, 0);
        manager.begin_mut(&mut values).unwrap().finish().unwrap();
    }

    /// If completion of the upload cannot be established, the storage is
    /// quarantined and every later session is refused.
    #[test]
    fn failed_upload_completion_quarantines_and_poisons() {
        let mut manager = failing_manager(&[Operation::Synchronize, Operation::Synchronize]);
        let mut values = [1, 2];

        let error = manager.begin_mut(&mut values).err().unwrap();

        assert!(matches!(error, MemoryError::Completion(_)));
        assert_eq!(values, [1, 2]);
        assert!(manager.poisoned);
        assert!(manager.quarantined.is_some());
        assert_eq!(manager.stats().uploads, 0);

        let events_before = manager.backend.events.len();
        let error = manager.begin_mut(&mut values).err().unwrap();
        assert_eq!(error, MemoryError::Poisoned);
        assert_eq!(manager.backend.events.len(), events_before);
    }

    /// A failed download leaves the original slice unchanged.
    #[test]
    fn download_failure_leaves_the_original_slice_unchanged() {
        let mut manager = failing_manager(&[Operation::Download]);
        let mut values = [1, 2];

        let mut session = manager.begin_mut(&mut values).unwrap();
        session.device_mut().unwrap().copy_from_slice(&[8, 9]);
        let error = session.finish().unwrap_err();

        assert!(matches!(error, MemoryError::Transfer(_)));
        assert_eq!(values, [1, 2]);
        assert_eq!(manager.stats().downloads, 0);
        assert!(!manager.poisoned);
    }

    /// Download completion failure leaves `host` unchanged and quarantines
    /// the staging the download may still be writing.
    #[test]
    fn failed_download_completion_quarantines_and_poisons() {
        let mut manager = fake_manager();
        let mut values = [1, 2];

        let mut session = manager.begin_mut(&mut values).unwrap();
        session.device_mut().unwrap().copy_from_slice(&[8, 9]);
        session.manager.backend.failures = vec![Operation::Synchronize, Operation::Synchronize];
        let error = session.finish().unwrap_err();

        assert!(matches!(error, MemoryError::Completion(_)));
        assert_eq!(values, [1, 2]);
        assert!(manager.poisoned);
        assert!(manager.quarantined.is_some());
        assert_eq!(
            manager.begin_mut(&mut values).err(),
            Some(MemoryError::Poisoned)
        );
    }

    /// Dropping a written session whose completion fails quarantines too.
    #[test]
    fn failed_completion_while_discarding_quarantines() {
        let mut manager = fake_manager();
        let mut values = [1, 2];

        let mut session = manager.begin_mut(&mut values).unwrap();
        session.device_mut().unwrap();
        session.manager.backend.failures = vec![Operation::Synchronize];
        drop(session);

        assert!(manager.poisoned);
        assert!(manager.quarantined.is_some());
    }

    /// Forgetting a session skips its cleanup. The caller's slice is free
    /// again, the session's own storage leaks, and the manager refuses
    /// further work without allocating.
    #[test]
    fn forgotten_session_poisons_the_manager() {
        let mut manager = fake_manager();
        let mut values = vec![1, 2];

        let mut session = manager.begin_mut(&mut values).unwrap();
        session.device_mut().unwrap();
        std::mem::forget(session);
        // The borrow ended, so the caller may change or free its slice.
        values[0] = 9;
        drop(values);

        let mut other = [3, 4];
        let error = manager.begin_mut(&mut other).err().unwrap();
        assert_eq!(error, MemoryError::Poisoned);
        assert_eq!(manager.stats().device_allocations, 1);
        assert_eq!(
            manager.begin_mut(&mut []).err(),
            Some(MemoryError::Poisoned)
        );
    }

    /// An empty session owns nothing, so forgetting it leaves nothing behind.
    #[test]
    fn forgetting_an_empty_session_is_harmless() {
        let mut manager = fake_manager();
        let mut empty: [u32; 0] = [];

        std::mem::forget(manager.begin_mut(&mut empty).unwrap());

        let mut values = [1, 2];
        manager.begin_mut(&mut values).unwrap().finish().unwrap();
    }

    /// Two same-length sessions share one allocation through the pool, but
    /// each uploads the current CPU contents.
    #[test]
    fn same_length_sessions_reuse_the_pooled_buffer() {
        let mut manager = fake_manager();
        let mut values = [1, 2];

        manager.begin_mut(&mut values).unwrap().finish().unwrap();
        manager.begin_mut(&mut values).unwrap().finish().unwrap();

        let stats = manager.stats();
        assert_eq!(stats.device_allocations, 1);
        assert_eq!(stats.pool_hits, 1);
        assert_eq!(stats.uploads, 2);
        assert_eq!(
            manager.backend.events,
            [
                Event::Allocate(2),
                Event::Upload(2),
                Event::Synchronize,
                Event::Upload(2),
                Event::Synchronize,
            ]
        );
    }

    /// A pooled buffer that held another slice's GPU results is overwritten
    /// by the next upload; stale device values never reach the CPU.
    #[test]
    fn a_pooled_buffer_never_leaks_stale_device_values() {
        let mut manager = fake_manager();
        let mut first = [1, 2];
        let mut session = manager.begin_mut(&mut first).unwrap();
        session.device_mut().unwrap().copy_from_slice(&[8, 9]);
        session.finish().unwrap();

        let mut second = [3, 4];
        let mut session = manager.begin_mut(&mut second).unwrap();
        assert_eq!(session.device_mut().unwrap(), &[3, 4]);
        session.finish().unwrap();

        assert_eq!(second, [3, 4]);
        assert_eq!(manager.stats().pool_hits, 1);
    }

    /// A different length evicts the idle buffer before allocating, so idle
    /// and active storage never coexist beyond the budget.
    #[test]
    fn a_different_length_evicts_and_allocates() {
        let config = MemoryConfig {
            max_device_bytes: 16,
            ..MemoryConfig::default()
        };
        let mut manager = Manager::new(FakeBackend::default(), config);
        let mut four = [1, 2, 3, 4];
        let mut three = [5, 6, 7];

        manager.begin_mut(&mut four).unwrap().finish().unwrap();
        assert_eq!(manager.pool.idle_len(), Some(4));
        manager.begin_mut(&mut three).unwrap().finish().unwrap();

        assert_eq!(manager.pool.idle_len(), Some(3));
        assert_eq!(manager.stats().device_allocations, 2);
        assert_eq!(manager.stats().pool_hits, 0);
    }

    /// Host staging is reused across sessions, so a session no larger than
    /// the last one allocates no new host memory; `trim` releases it.
    #[test]
    fn host_staging_is_reused_until_trimmed() {
        let mut manager = fake_manager();
        let mut four = [1, 2, 3, 4];
        let mut three = [5, 6, 7];

        let mut session = manager.begin_mut(&mut four).unwrap();
        let first = session.resources.as_ref().unwrap().staging.as_ptr();
        session.device_mut().unwrap();
        session.finish().unwrap();
        let mut session = manager.begin_mut(&mut three).unwrap();
        assert_eq!(session.resources.as_ref().unwrap().staging.as_ptr(), first);
        session.device_mut().unwrap().copy_from_slice(&[0, 0, 9]);
        session.finish().unwrap();

        assert_eq!(three, [0, 0, 9], "reused staging is resized to the slice");
        manager.trim();
        assert_eq!(manager.staging.capacity(), 0);
    }

    /// `trim` releases idle storage, so the next session allocates again.
    #[test]
    fn trim_releases_the_idle_buffer() {
        let mut manager = fake_manager();
        let mut values = [1, 2];

        manager.begin_mut(&mut values).unwrap().finish().unwrap();
        manager.trim();
        assert_eq!(manager.pool.idle_len(), None);
        manager.begin_mut(&mut values).unwrap().finish().unwrap();

        assert_eq!(manager.stats().device_allocations, 2);
        assert_eq!(manager.stats().pool_hits, 0);
    }

    /// A discarded session's buffer is pooled once its work has completed.
    #[test]
    fn a_dropped_session_returns_its_buffer_after_waiting() {
        let mut manager = fake_manager();
        let mut values = [1, 2];

        let mut session = manager.begin_mut(&mut values).unwrap();
        session.device_mut().unwrap();
        drop(session);

        assert_eq!(manager.pool.idle_len(), Some(2));
    }

    /// Buffers from failed operations, and quarantined buffers, are never
    /// pooled.
    #[test]
    fn failed_or_quarantined_buffers_never_enter_the_pool() {
        let mut manager = failing_manager(&[Operation::Download]);
        let mut values = [1, 2];
        let mut session = manager.begin_mut(&mut values).unwrap();
        session.device_mut().unwrap();
        session.finish().unwrap_err();
        assert_eq!(manager.pool.idle_len(), None);

        let mut manager = failing_manager(&[Operation::Upload]);
        manager.begin_mut(&mut values).err().unwrap();
        assert_eq!(manager.pool.idle_len(), None);

        let mut manager = fake_manager();
        let mut session = manager.begin_mut(&mut values).unwrap();
        session.device_mut().unwrap();
        session.manager.backend.failures = vec![Operation::Synchronize];
        drop(session);
        assert!(manager.quarantined.is_some());
        assert_eq!(manager.pool.idle_len(), None);
    }

    /// Consecutive sessions on one manager upload fresh contents each time,
    /// even when the CPU address is unchanged.
    #[test]
    fn a_new_session_uploads_changed_contents() {
        let mut manager = fake_manager();
        let mut values = [1, 2];

        manager.begin_mut(&mut values).unwrap().finish().unwrap();
        values = [5, 6];
        let mut session = manager.begin_mut(&mut values).unwrap();
        assert_eq!(session.device_mut().unwrap(), &[5, 6]);
        session.finish().unwrap();

        assert_eq!(manager.stats().uploads, 2);
    }

    const STEP: MapStep = MapStep { mul: 3, add: 7 };

    /// Maps run in order on the device copy, between one upload and one
    /// download.
    #[test]
    fn maps_apply_in_order_between_one_upload_and_download() {
        let mut manager = fake_manager();
        let mut values = [0, u32::MAX];
        let second = MapStep { mul: 2, add: 1 };

        let mut session = manager.begin_mut(&mut values).unwrap();
        session.map(STEP).unwrap();
        session.map(second).unwrap();
        session.finish().unwrap();

        let expected = [0, u32::MAX].map(|v| second.apply(STEP.apply(v)));
        assert_eq!(values, expected);
        assert_eq!(
            manager.backend.events,
            [
                Event::Allocate(2),
                Event::Upload(2),
                Event::Synchronize,
                Event::Map(2, STEP),
                Event::Map(2, second),
                Event::Download(2),
                Event::Synchronize,
            ]
        );
    }

    #[test]
    fn mapping_an_empty_session_does_nothing() {
        let mut manager = fake_manager();
        let mut values: [u32; 0] = [];

        let mut session = manager.begin_mut(&mut values).unwrap();
        session.map(STEP).unwrap();
        session.finish().unwrap();

        assert!(manager.backend.events.is_empty());
    }

    /// A session missing a failed operation is never committed, even if
    /// later operations succeed, and the manager stays usable.
    #[test]
    fn a_failed_map_discards_the_session() {
        let mut manager = failing_manager(&[Operation::Map]);
        let mut values = [1, 2];

        let mut session = manager.begin_mut(&mut values).unwrap();
        assert!(matches!(session.map(STEP), Err(MemoryError::Launch(_))));
        session.map(STEP).unwrap();
        assert!(matches!(session.finish(), Err(MemoryError::Launch(_))));

        assert_eq!(values, [1, 2]);
        assert_eq!(manager.stats().downloads, 0);
        assert!(!manager.poisoned);
        manager.begin_mut(&mut values).unwrap().finish().unwrap();
    }
}
