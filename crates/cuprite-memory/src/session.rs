use crate::{backend::MemoryBackend, manager::Manager, GpuMemory, MemoryError};

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
/// These examples check API lifetimes only; `GpuMemory::open` cannot succeed yet.
#[must_use = "call finish to commit results, or discard the session"]
#[allow(dead_code)]
pub struct ResidentSlice<'a> {
    pub(crate) manager: &'a mut GpuMemory,
    pub(crate) host: &'a mut [u32],
    pub(crate) staging: Vec<u32>,
    pub(crate) may_have_writes: bool,
}

impl ResidentSlice<'_> {
    pub fn finish(self) -> Result<(), MemoryError> {
        Err(MemoryError::NotImplemented("ResidentSlice::finish"))
    }
}

/// Everything a nonempty session owns. `Buf` is the backend's buffer type. Keeping these together means cleanup
/// moves one value, so nothing can be released twice or left half-released.
#[allow(dead_code)] // Only tests use this until `GpuMemory` is connected.
pub(crate) struct Resources<Buf> {
    /// Session-owned CPU copy. Transfers touch this, never the caller's slice.
    pub(crate) staging: Vec<u32>,
    /// The device working copy; same length as `staging`.
    pub(crate) device: Buf,
}

/// Generic counterpart of [`ResidentSlice`]. Same borrowing shape: `'a` ties
/// the manager and the original slice to this session.
#[must_use = "call finish to commit results, or drop to discard them"]
#[allow(dead_code)] // Only tests use this until `GpuMemory` is connected.
pub(crate) struct Session<'a, B: MemoryBackend> {
    pub(crate) manager: &'a mut Manager<B>,
    pub(crate) host: &'a mut [u32],
    /// `None` for an empty session, which owns nothing.
    pub(crate) resources: Option<Resources<B::Buffer>>,
    /// Set *before* handing out write access, even if no write happens.
    pub(crate) may_have_writes: bool,
}

#[allow(dead_code)] // Only tests use this until `GpuMemory` is connected.
                    // `'_` stands for the session's lifetime, which these methods never need to name.
impl<B: MemoryBackend> Session<'_, B> {
    /// Crate-private write access to the device working copy.
    ///
    /// This is the generic stand-in for the plan's `cuda_write_parts`. It
    /// lends the buffer out (`&mut`) so the session keeps ownership, and it
    /// marks `may_have_writes` so `finish` knows to copy back. An empty
    /// session returns `None`: it has no buffer and must never reach a kernel.
    pub(crate) fn device_mut(&mut self) -> Option<&mut B::Buffer> {
        // Borrow the bundle rather than moving it: `finish` still needs it.
        let Some(resources) = &mut self.resources else {
            return None;
        };
        self.may_have_writes = true;
        Some(&mut resources.device)
    }

    /// Commit results to the original slice and consume the session.
    ///
    /// With no possible writes the device copy still equals `host`, so
    /// nothing is transferred. Otherwise the device copy is downloaded into
    /// staging, and only once that has completed is staging copied into
    /// `host`. A failed download or synchronize leaves `host` unchanged.
    pub(crate) fn finish(self) -> Result<(), MemoryError> {
        // `finish` owns `self`, so it can move the bundle out of the session.
        let Some(mut resources) = self.resources else {
            return Ok(()); // Empty session: nothing was staged or allocated.
        };
        if !self.may_have_writes {
            return Ok(()); // Device copy is unchanged; skip the download.
        }
        // Lengths were fixed in begin_mut; copy_from_slice relies on it.
        debug_assert_eq!(resources.staging.len(), self.host.len());

        self.manager
            .backend
            .download(&resources.device, &mut resources.staging)?;
        self.manager.backend.synchronize()?;
        // Count the download only once completion is confirmed.
        self.manager.stats.downloads += 1;
        self.manager.stats.downloaded_bytes += (resources.staging.len() * size_of::<u32>()) as u64;

        self.host.copy_from_slice(&resources.staging);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        test_support::{Event, FakeBackend},
        MemoryConfig, MemoryStats,
    };

    fn fake_manager() -> Manager<FakeBackend> {
        Manager::new(FakeBackend::default(), MemoryConfig::default())
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

    /// Dropping without `finish` discards device changes.
    #[test]
    fn dropping_a_session_discards_device_changes() {
        let mut manager = fake_manager();
        let mut values = [1, 2];

        let mut session = manager.begin_mut(&mut values).unwrap();
        session.device_mut().unwrap().copy_from_slice(&[8, 9]);
        drop(session); // Give up on the session without finishing it.

        assert_eq!(values, [1, 2]);
        assert!(!manager.backend.events.contains(&Event::Download(2)));
    }
}
