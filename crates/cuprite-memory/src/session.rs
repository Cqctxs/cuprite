use crate::{
    backend::{DefaultBackend, MemoryBackend},
    job::JobCore,
    manager::Manager,
    Arg, Buffer, Grid, MapStep, MemoryError,
};

/// A [`Job`](crate::Job) over one `u32` slice, for [`MapStep`]s. It
/// exclusively borrows the manager and the slice:
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
    inner: Resident<'a, DefaultBackend>,
}

impl<'a> ResidentSlice<'a> {
    pub(crate) fn new(inner: Resident<'a, DefaultBackend>) -> Self {
        Self { inner }
    }

    /// Commit device results to the original slice and end the session.
    ///
    /// # Errors
    /// Returns an error if any operation failed, or the download or its
    /// completion fails; the original slice is then unchanged.
    pub fn finish(self) -> Result<(), MemoryError> {
        self.inner.finish()
    }

    /// Queue `step` over the GPU copy, after earlier operations.
    ///
    /// # Errors
    /// Returns an error if the operation cannot be queued. The session then
    /// cannot be committed: `finish` returns an error and the original slice
    /// is left unchanged.
    pub fn map(&mut self, step: MapStep) -> Result<(), MemoryError> {
        self.inner.map(step)
    }
}

/// Generic counterpart of [`ResidentSlice`].
pub(crate) struct Resident<'a, B: MemoryBackend> {
    job: JobCore<'a, B>,
    buffer: Buffer,
    len: usize,
}

impl<'a, B: MemoryBackend> Resident<'a, B> {
    pub(crate) fn begin(
        manager: &'a mut Manager<B>,
        host: &'a mut [u32],
    ) -> Result<Self, MemoryError> {
        let len = host.len();
        let mut job = manager.job()?;
        let buffer = job.in_out(host)?;
        Ok(Self { job, buffer, len })
    }

    /// Launch the built-in map kernel. An empty slice has nothing to map.
    pub(crate) fn map(&mut self, step: MapStep) -> Result<(), MemoryError> {
        if self.len == 0 {
            return Ok(());
        }
        let kernel = match self.job.manager.map_kernel() {
            Ok(kernel) => kernel,
            Err(error) => {
                // The device copy would be missing this step.
                self.job.mark_failed();
                return Err(error);
            }
        };
        let len = self.len as u64;
        self.job.launch(
            kernel,
            Grid::Linear(len),
            &[
                Arg::Buffer(self.buffer),
                Arg::U64(len),
                Arg::U32(step.mul),
                Arg::U32(step.add),
            ],
        )
    }

    pub(crate) fn finish(self) -> Result<(), MemoryError> {
        self.job.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        test_support::{Event, FakeBackend, Operation},
        MemoryConfig,
    };

    const STEP: MapStep = MapStep { mul: 3, add: 7 };

    fn fake_manager() -> Manager<FakeBackend> {
        Manager::new(FakeBackend::default(), MemoryConfig::default())
    }

    /// Maps run in order on the device copy, between one upload and one
    /// download, and the kernel is loaded once.
    #[test]
    fn maps_apply_in_order_between_one_upload_and_download() {
        let mut manager = fake_manager();
        let mut values = [0, u32::MAX];
        let second = MapStep { mul: 2, add: 1 };

        let mut session = Resident::begin(&mut manager, &mut values).unwrap();
        session.map(STEP).unwrap();
        session.map(second).unwrap();
        session.finish().unwrap();

        let expected = [0, u32::MAX].map(|v| second.apply(STEP.apply(v)));
        assert_eq!(values, expected);
        assert_eq!(
            manager.backend.events,
            [
                Event::Allocate(8),
                Event::Upload(8),
                Event::Load("map_u32".into()),
                Event::Launch("map_u32".into()),
                Event::Launch("map_u32".into()),
                Event::Download(8),
                Event::Synchronize,
            ]
        );
    }

    /// Without a map, the device copy still equals the slice, so nothing is
    /// downloaded.
    #[test]
    fn an_unmapped_session_finishes_without_downloading() {
        let mut manager = fake_manager();
        let mut values = [1, 2];

        Resident::begin(&mut manager, &mut values)
            .unwrap()
            .finish()
            .unwrap();

        assert_eq!(values, [1, 2]);
        assert_eq!(manager.stats().downloads, 0);
        assert_eq!(manager.stats().uploads, 1);
    }

    #[test]
    fn mapping_an_empty_session_does_nothing() {
        let mut manager = fake_manager();
        let mut values: [u32; 0] = [];

        let mut session = Resident::begin(&mut manager, &mut values).unwrap();
        session.map(STEP).unwrap();
        session.finish().unwrap();

        assert!(manager.backend.events.is_empty());
    }

    /// A session missing a failed operation is never committed, even if
    /// later operations succeed, and the manager stays usable.
    #[test]
    fn a_failed_map_discards_the_session() {
        let mut manager = Manager::new(
            FakeBackend::failing(&[Operation::Launch]),
            MemoryConfig::default(),
        );
        let mut values = [1, 2];

        let mut session = Resident::begin(&mut manager, &mut values).unwrap();
        assert!(matches!(session.map(STEP), Err(MemoryError::Launch(_))));
        session.map(STEP).unwrap();
        assert!(matches!(session.finish(), Err(MemoryError::Launch(_))));

        assert_eq!(values, [1, 2]);
        assert_eq!(manager.stats().downloads, 0);
        assert!(!manager.poisoned);
        Resident::begin(&mut manager, &mut values)
            .unwrap()
            .finish()
            .unwrap();
    }

    /// A kernel that can't be loaded leaves the step missing, so the session
    /// can't be committed either.
    #[test]
    fn a_kernel_that_fails_to_load_discards_the_session() {
        let mut manager = Manager::new(
            FakeBackend::failing(&[Operation::Load]),
            MemoryConfig::default(),
        );
        let mut values = [1, 2];

        let mut session = Resident::begin(&mut manager, &mut values).unwrap();
        assert!(matches!(session.map(STEP), Err(MemoryError::Kernel(_))));
        assert!(session.finish().is_err());
        assert_eq!(values, [1, 2]);
    }
}
