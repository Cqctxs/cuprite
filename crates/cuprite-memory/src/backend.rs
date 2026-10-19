use crate::MemoryError;

/// Private boundary between ownership logic and a GPU or test backend.
///
/// Copies may be queued: the caller must retain source/destination storage
/// until synchronize establishes completion. An error alone does not prove
/// that no queued work still uses that storage.
///
/// Lengths are `u32` element counts. Transfers require equal source and
/// destination lengths. Allocation failure and invalid lengths return errors.
pub(crate) trait MemoryBackend {
    type Buffer;

    fn allocate(&mut self, len: usize) -> Result<Self::Buffer, MemoryError>;
    fn upload(&mut self, buffer: &mut Self::Buffer, input: &[u32]) -> Result<(), MemoryError>;
    fn download(&mut self, buffer: &Self::Buffer, output: &mut [u32]) -> Result<(), MemoryError>;
    fn synchronize(&mut self) -> Result<(), MemoryError>;
}
