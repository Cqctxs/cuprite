use std::fmt;

#[derive(Debug, PartialEq, Eq)]
pub enum MemoryError {
    Unavailable(String),
    SizeOverflow,
    BudgetExceeded {
        requested: usize,
        limit: usize,
    },
    HostAllocation(String),
    DeviceAllocation(String),
    Transfer(String),
    Completion(String),
    Poisoned,
    /// Marks an operation that has not been written yet.
    NotImplemented(&'static str),
}

impl fmt::Display for MemoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(message) => write!(formatter, "GPU unavailable: {message}"),
            Self::SizeOverflow => write!(formatter, "slice byte count exceeds usize"),
            Self::BudgetExceeded { requested, limit } => {
                write!(formatter, "requested {requested} bytes; budget is {limit}")
            }
            Self::HostAllocation(message) => write!(formatter, "CPU allocation: {message}"),
            Self::DeviceAllocation(message) => write!(formatter, "GPU allocation: {message}"),
            Self::Transfer(message) => write!(formatter, "memory transfer: {message}"),
            Self::Completion(message) => write!(formatter, "GPU completion: {message}"),
            Self::Poisoned => write!(formatter, "memory manager cannot safely accept more work"),
            Self::NotImplemented(operation) => write!(formatter, "not implemented: {operation}"),
        }
    }
}

impl std::error::Error for MemoryError {}
