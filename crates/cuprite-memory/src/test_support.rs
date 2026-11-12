//! Test-only backend with separate CPU storage representing device memory.
//!
//! Copies and kernels complete immediately; synchronization records an
//! explicit boundary. Successful operations record byte counts or kernel
//! names. Failed operations leave destination contents and the event log
//! unchanged. This backend does not model queued work or real CUDA failure
//! behavior.
//!
//! Kernels are Rust functions looked up by entry name; the PTX is ignored.

use std::cell::RefCell;

use crate::{
    backend::{MemoryBackend, Value},
    step::MAP_NAME,
    Grid, MapStep, MemoryError,
};

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Event {
    Allocate(usize),
    Upload(usize),
    Download(usize),
    Zero(usize),
    Load(String),
    Launch(String),
    Synchronize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Operation {
    Allocate,
    Upload,
    Download,
    Zero,
    Load,
    Launch,
    Synchronize,
}

/// Device storage: launches only get shared references, as with CUDA,
/// where the same buffer may be passed twice.
pub(crate) type FakeBuffer = RefCell<Vec<u8>>;

type Run = fn(&[Value<'_, FakeBuffer>]);

pub(crate) struct FakeKernel {
    name: String,
    run: Run,
}

fn buffer<'b>(value: &Value<'b, FakeBuffer>) -> &'b FakeBuffer {
    match value {
        Value::Buffer(Some(buffer)) => buffer,
        _ => panic!("fake kernel expected a nonempty buffer"),
    }
}

fn u32_at(value: &Value<'_, FakeBuffer>) -> u32 {
    match value {
        Value::U32(value) => *value,
        _ => panic!("fake kernel expected a u32"),
    }
}

fn update_u32s(buffer: &FakeBuffer, mut f: impl FnMut(u32) -> u32) {
    for chunk in buffer.borrow_mut().as_chunks_mut::<4>().0 {
        *chunk = f(u32::from_ne_bytes(*chunk)).to_ne_bytes();
    }
}

/// `map_u32(values, len, mul, add)`, as on the GPU.
fn map_u32(args: &[Value<'_, FakeBuffer>]) {
    let step = MapStep {
        mul: u32_at(&args[2]),
        add: u32_at(&args[3]),
    };
    update_u32s(buffer(&args[0]), |value| step.apply(value));
}

/// `add_one_u32(values)`: every `u32` plus one, wrapping.
fn add_one_u32(args: &[Value<'_, FakeBuffer>]) {
    update_u32s(buffer(&args[0]), |value| value.wrapping_add(1));
}

/// `copy(source, destination)`: as many bytes as both hold.
fn copy(args: &[Value<'_, FakeBuffer>]) {
    let source = buffer(&args[0]).borrow().clone();
    let mut destination = buffer(&args[1]).borrow_mut();
    let len = source.len().min(destination.len());
    destination[..len].copy_from_slice(&source[..len]);
}

const KERNELS: [(&str, Run); 3] = [
    (MAP_NAME, map_u32),
    ("add_one_u32", add_one_u32),
    ("copy", copy),
];

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
            Operation::Upload | Operation::Download | Operation::Zero => {
                MemoryError::Transfer("injected".into())
            }
            Operation::Load => MemoryError::Kernel("injected".into()),
            Operation::Launch => MemoryError::Launch("injected".into()),
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
            "source has {source} bytes but destination has {destination}"
        )));
    }
    Ok(())
}

impl MemoryBackend for FakeBackend {
    type Buffer = FakeBuffer;
    type Kernel = FakeKernel;

    fn allocate(&mut self, bytes: usize) -> Result<Self::Buffer, MemoryError> {
        self.inject(Operation::Allocate)?;
        let mut buffer = Vec::new();
        // Treat this allocation as device storage for the backend contract.
        buffer
            .try_reserve_exact(bytes)
            .map_err(|error| MemoryError::DeviceAllocation(error.to_string()))?;
        buffer.resize(bytes, 0);
        self.reserve_event()?;
        self.events.push(Event::Allocate(bytes));
        Ok(RefCell::new(buffer))
    }

    fn upload(&mut self, buffer: &mut Self::Buffer, input: &[u8]) -> Result<(), MemoryError> {
        let buffer = buffer.get_mut();
        validate_transfer_lengths(input.len(), buffer.len())?;
        self.inject(Operation::Upload)?;
        self.reserve_event()?;
        buffer.copy_from_slice(input);
        self.events.push(Event::Upload(input.len()));
        Ok(())
    }

    fn download(&mut self, buffer: &Self::Buffer, output: &mut [u8]) -> Result<(), MemoryError> {
        let buffer = buffer.borrow();
        validate_transfer_lengths(buffer.len(), output.len())?;
        self.inject(Operation::Download)?;
        self.reserve_event()?;
        output.copy_from_slice(&buffer);
        self.events.push(Event::Download(output.len()));
        Ok(())
    }

    fn zero(&mut self, buffer: &mut Self::Buffer) -> Result<(), MemoryError> {
        self.inject(Operation::Zero)?;
        self.reserve_event()?;
        let buffer = buffer.get_mut();
        buffer.fill(0);
        self.events.push(Event::Zero(buffer.len()));
        Ok(())
    }

    fn load(&mut self, _ptx: &str, name: &str) -> Result<Self::Kernel, MemoryError> {
        self.inject(Operation::Load)?;
        let Some(&(_, run)) = KERNELS.iter().find(|(known, _)| *known == name) else {
            return Err(MemoryError::Kernel(format!("no fake kernel named {name}")));
        };
        self.reserve_event()?;
        self.events.push(Event::Load(name.into()));
        Ok(FakeKernel {
            name: name.into(),
            run,
        })
    }

    fn launch(
        &mut self,
        kernel: &Self::Kernel,
        grid: Grid,
        args: &[Value<'_, Self::Buffer>],
    ) -> Result<(), MemoryError> {
        assert!(!grid.is_empty(), "jobs never launch an empty grid");
        self.inject(Operation::Launch)?;
        self.reserve_event()?;
        (kernel.run)(args);
        self.events.push(Event::Launch(kernel.name.clone()));
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
    fn allocate_creates_zeroed_storage_of_the_byte_count() {
        let mut backend = FakeBackend::default();

        let buffer = backend.allocate(3).unwrap();

        assert_eq!(*buffer.borrow(), vec![0, 0, 0]);
        assert_eq!(backend.events, [Event::Allocate(3)]);
    }

    #[test]
    fn allocation_rejects_unrepresentable_capacity_without_recording_success() {
        let mut backend = FakeBackend::default();

        assert!(matches!(
            backend.allocate(isize::MAX as usize + 1),
            Err(MemoryError::DeviceAllocation(_))
        ));
        assert!(backend.events.is_empty());
    }

    #[test]
    fn transfers_copy_into_existing_storage_and_check_lengths() {
        let mut backend = FakeBackend::default();
        let mut buffer = RefCell::new(vec![0, 0, 0]);
        let original_pointer = buffer.borrow().as_ptr();

        backend.upload(&mut buffer, &[4, 5, 6]).unwrap();
        let mut output = [0; 3];
        backend.download(&buffer, &mut output).unwrap();

        assert_eq!(output, [4, 5, 6]);
        assert_eq!(buffer.borrow().as_ptr(), original_pointer);
        assert_eq!(backend.events, [Event::Upload(3), Event::Download(3)]);

        let error = backend.upload(&mut buffer, &[1, 2]).unwrap_err();
        let MemoryError::Transfer(message) = error else {
            panic!("expected a transfer error");
        };
        assert!(message.contains('2') && message.contains('3'));
        assert_eq!(*buffer.borrow(), vec![4, 5, 6]);
        assert!(backend.download(&buffer, &mut [0; 4]).is_err());
        assert_eq!(backend.events.len(), 2);
    }

    #[test]
    fn kernels_are_found_by_name_and_run_on_shared_buffers() {
        let mut backend = FakeBackend::default();
        let kernel = backend.load("ignored", "add_one_u32").unwrap();
        let buffer = RefCell::new(5u32.to_ne_bytes().to_vec());

        backend
            .launch(&kernel, Grid::Linear(1), &[Value::Buffer(Some(&buffer))])
            .unwrap();

        assert_eq!(*buffer.borrow(), 6u32.to_ne_bytes());
        assert!(matches!(
            backend.load("ignored", "missing"),
            Err(MemoryError::Kernel(_))
        ));
        assert_eq!(
            backend.events,
            [
                Event::Load("add_one_u32".into()),
                Event::Launch("add_one_u32".into())
            ]
        );
    }

    #[test]
    fn injected_failures_happen_once_each() {
        let mut backend = FakeBackend::failing(&[Operation::Synchronize]);

        assert!(backend.synchronize().is_err());
        backend.synchronize().unwrap();

        assert_eq!(backend.events, [Event::Synchronize]);
    }
}
