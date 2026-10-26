//! Test-only backend with separate CPU storage representing device memory.
//!
//! Copies complete immediately; synchronization records an explicit boundary.
//! Successful operations record element counts, not byte counts. Failed
//! operations leave destination contents and the event log unchanged.
//! This backend does not model queued work or real CUDA failure behavior.

use crate::{backend::MemoryBackend, checked_bytes, MapStep, MemoryError};

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Event {
    Allocate(usize),
    Upload(usize),
    Download(usize),
    Map(usize, MapStep),
    Synchronize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Operation {
    Allocate,
    Upload,
    Download,
    Map,
    Synchronize,
}

#[derive(Default)]
pub(crate) struct FakeBackend {
    pub(crate) events: Vec<Event>,
    /// Injected failures: each entry makes the next call of that operation
    /// fail once. List an operation twice to make it fail twice.
    pub(crate) failures: Vec<Operation>,
}

impl FakeBackend {
    pub(crate) fn failing(failures: &[Operation]) -> Self {
        Self {
            failures: failures.to_vec(),
            ..Self::default()
        }
    }

    fn inject(&mut self, operation: Operation) -> Result<(), MemoryError> {
        let Some(index) = self.failures.iter().position(|&f| f == operation) else {
            return Ok(());
        };
        self.failures.remove(index);
        Err(match operation {
            Operation::Allocate => MemoryError::DeviceAllocation("injected".into()),
            Operation::Upload | Operation::Download => MemoryError::Transfer("injected".into()),
            Operation::Map => MemoryError::Launch("injected".into()),
            Operation::Synchronize => MemoryError::Completion("injected".into()),
        })
    }

    fn reserve_event(&mut self) -> Result<(), MemoryError> {
        // Reserve before changing data so recording success cannot then fail.
        self.events
            .try_reserve(1)
            .map_err(|error| MemoryError::HostAllocation(error.to_string()))
    }
}

fn validate_transfer_lengths(source: usize, destination: usize) -> Result<(), MemoryError> {
    if source != destination {
        return Err(MemoryError::Transfer(format!(
            "source has {source} elements but destination has {destination}"
        )));
    }
    Ok(())
}

impl MemoryBackend for FakeBackend {
    type Buffer = Vec<u32>;

    fn allocate(&mut self, len: usize) -> Result<Self::Buffer, MemoryError> {
        checked_bytes(len)?;
        self.inject(Operation::Allocate)?;
        let mut buffer = Vec::new();
        // Treat this allocation as device storage for the backend contract.
        buffer
            .try_reserve_exact(len)
            .map_err(|error| MemoryError::DeviceAllocation(error.to_string()))?;
        buffer.resize(len, 0);
        self.reserve_event()?;
        self.events.push(Event::Allocate(len));
        Ok(buffer)
    }

    fn upload(&mut self, buffer: &mut Self::Buffer, input: &[u32]) -> Result<(), MemoryError> {
        validate_transfer_lengths(input.len(), buffer.len())?;
        self.inject(Operation::Upload)?;
        self.reserve_event()?;
        buffer.copy_from_slice(input);
        self.events.push(Event::Upload(input.len()));
        Ok(())
    }

    fn download(&mut self, buffer: &Self::Buffer, output: &mut [u32]) -> Result<(), MemoryError> {
        validate_transfer_lengths(buffer.len(), output.len())?;
        self.inject(Operation::Download)?;
        self.reserve_event()?;
        output.copy_from_slice(buffer);
        self.events.push(Event::Download(output.len()));
        Ok(())
    }

    fn map(&mut self, buffer: &mut Self::Buffer, step: MapStep) -> Result<(), MemoryError> {
        self.inject(Operation::Map)?;
        self.reserve_event()?;
        for value in buffer.iter_mut() {
            *value = step.apply(*value);
        }
        self.events.push(Event::Map(buffer.len(), step));
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), MemoryError> {
        self.inject(Operation::Synchronize)?;
        self.reserve_event()?;
        self.events.push(Event::Synchronize);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_rejects_byte_count_overflow_without_recording_success() {
        let mut backend = FakeBackend::default();

        assert_eq!(backend.allocate(usize::MAX), Err(MemoryError::SizeOverflow));
        assert!(backend.events.is_empty());
    }

    #[test]
    fn allocation_rejects_unrepresentable_capacity_without_recording_success() {
        let mut backend = FakeBackend::default();
        let len = isize::MAX as usize / size_of::<u32>() + 1;

        assert!(matches!(
            backend.allocate(len),
            Err(MemoryError::DeviceAllocation(_))
        ));
        assert!(backend.events.is_empty());
    }

    #[test]
    fn transfer_error_reports_source_and_destination_lengths() {
        let mut backend = FakeBackend::default();
        let mut buffer = vec![0; 3];

        let error = backend.upload(&mut buffer, &[1, 2]).unwrap_err();

        let MemoryError::Transfer(message) = error else {
            panic!("expected a transfer error");
        };
        assert!(message.contains("2"));
        assert!(message.contains("3"));
    }

    #[test]
    fn allocate_creates_initialized_storage() {
        let mut backend = FakeBackend::default();

        let buffer = backend.allocate(3).unwrap();

        assert_eq!(buffer, vec![0, 0, 0]);
        assert_eq!(backend.events, [Event::Allocate(3)]);
    }

    #[test]
    fn allocate_accepts_zero_elements() {
        let mut backend = FakeBackend::default();

        let buffer = backend.allocate(0).unwrap();

        assert!(buffer.is_empty());
        assert_eq!(backend.events, [Event::Allocate(0)]);
    }

    #[test]
    fn upload_copies_values_into_existing_storage() {
        let mut backend = FakeBackend::default();
        let mut buffer = vec![0, 0, 0];
        let original_pointer = buffer.as_ptr();

        backend.upload(&mut buffer, &[4, 5, 6]).unwrap();

        assert_eq!(buffer, vec![4, 5, 6]);
        assert_eq!(buffer.as_ptr(), original_pointer);
        assert_eq!(backend.events, [Event::Upload(3)]);
    }

    #[test]
    fn upload_rejects_mismatched_lengths_without_changing_storage() {
        for input in [&[4][..], &[4, 5, 6][..]] {
            let mut backend = FakeBackend {
                events: vec![Event::Synchronize],
                ..FakeBackend::default()
            };
            let mut buffer = vec![9, 9];

            let result = backend.upload(&mut buffer, input);

            assert!(matches!(result, Err(MemoryError::Transfer(_))));
            assert_eq!(buffer, vec![9, 9]);
            assert_eq!(backend.events, [Event::Synchronize]);
        }
    }

    #[test]
    fn download_copies_values_into_new_storage() {
        let mut backend = FakeBackend::default();
        let buffer = vec![4, 5, 6];
        let mut output = [0, 0, 0];

        backend.download(&buffer, &mut output).unwrap();

        assert_eq!(output, [4, 5, 6]);
        assert_eq!(buffer, vec![4, 5, 6]);
        assert_eq!(backend.events, [Event::Download(3)]);
    }

    #[test]
    fn download_rejects_mismatched_lengths_without_changing_storage() {
        for buffer in [vec![9], vec![9, 9, 9]] {
            let mut backend = FakeBackend {
                events: vec![Event::Synchronize],
                ..FakeBackend::default()
            };
            let mut output = [4, 5];

            let result = backend.download(&buffer, &mut output);

            assert!(matches!(result, Err(MemoryError::Transfer(_))));
            assert_eq!(output, [4, 5]);
            assert_eq!(backend.events, [Event::Synchronize]);
        }
    }

    #[test]
    fn download_error_reports_source_and_destination_lengths() {
        let mut backend = FakeBackend::default();
        let buffer = vec![1, 2];
        let mut output = [0; 3];

        let error = backend.download(&buffer, &mut output).unwrap_err();

        let MemoryError::Transfer(message) = error else {
            panic!("expected a transfer error");
        };
        assert!(message.contains("2"));
        assert!(message.contains("3"));
    }

    #[test]
    fn transfers_accept_empty_storage() {
        let mut backend = FakeBackend::default();
        let mut buffer = Vec::new();
        let mut output = [];

        backend.upload(&mut buffer, &[]).unwrap();
        backend.download(&buffer, &mut output).unwrap();

        assert!(buffer.is_empty());
        assert_eq!(backend.events, [Event::Upload(0), Event::Download(0)]);
    }

    #[test]
    fn synchronize_records_each_completion_boundary() {
        let mut backend = FakeBackend::default();

        backend.synchronize().unwrap();
        backend.synchronize().unwrap();

        assert_eq!(backend.events, [Event::Synchronize, Event::Synchronize]);
    }

    #[test]
    fn round_trip_preserves_host_input_and_records_operations_in_order() {
        let mut backend = FakeBackend::default();
        let input = [1, 2];
        let mut output = [0, 0];

        let mut buffer = backend.allocate(2).unwrap();
        backend.upload(&mut buffer, &input).unwrap();
        // Simulate writes to the device working copy.
        buffer.copy_from_slice(&[8, 9]);
        backend.download(&buffer, &mut output).unwrap();
        backend.synchronize().unwrap();

        assert_eq!(input, [1, 2]);
        assert_eq!(output, [8, 9]);
        assert_eq!(
            backend.events,
            [
                Event::Allocate(2),
                Event::Upload(2),
                Event::Download(2),
                Event::Synchronize,
            ]
        );
    }

    #[test]
    fn rejected_transfer_does_not_prevent_a_later_successful_transfer() {
        let mut backend = FakeBackend::default();
        let mut buffer = vec![9, 9];

        assert!(matches!(
            backend.upload(&mut buffer, &[4]),
            Err(MemoryError::Transfer(_))
        ));
        backend.upload(&mut buffer, &[4, 5]).unwrap();

        assert_eq!(buffer, vec![4, 5]);
        assert_eq!(backend.events, [Event::Upload(2)]);
    }
}
