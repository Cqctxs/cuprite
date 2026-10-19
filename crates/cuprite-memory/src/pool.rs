struct IdleBuffer<B> {
    len: usize,
    buffer: B,
}

/// One slot of completed owned storage. The caller proves completion before
/// insertion. This pool never associates buffers with CPU addresses.
pub(crate) struct BufferPool<B> {
    idle: Option<IdleBuffer<B>>,
}

impl<B> BufferPool<B> {
    pub(crate) fn new() -> Self {
        Self { idle: None }
    }

    pub(crate) fn take_matching(&mut self, _len: usize) -> Option<B> {
        todo!("BufferPool::take_matching")
    }

    pub(crate) fn put_completed(&mut self, _len: usize, _buffer: B) {
        todo!("BufferPool::put_completed")
    }

    pub(crate) fn clear(&mut self) {
        todo!("BufferPool::clear")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Strings stand in for owned buffers; the tests check the real pool logic.
    #[test]
    #[ignore = "pool not implemented yet"]
    fn returns_a_matching_buffer_only_once() {
        let mut pool = BufferPool::new();
        pool.put_completed(4, String::from("buffer-a"));
        assert_eq!(pool.take_matching(4), Some(String::from("buffer-a")));
        assert_eq!(pool.take_matching(4), None);
    }

    #[test]
    #[ignore = "pool not implemented yet"]
    fn evicts_a_buffer_with_the_wrong_length() {
        let mut pool = BufferPool::new();
        pool.put_completed(4, String::from("buffer-a"));
        assert_eq!(pool.take_matching(5), None);
        assert_eq!(pool.take_matching(4), None);
    }

    #[test]
    #[ignore = "pool not implemented yet"]
    fn trimming_removes_idle_storage() {
        let mut pool = BufferPool::new();
        pool.put_completed(4, String::from("buffer-a"));
        pool.clear();
        assert_eq!(pool.take_matching(4), None);
    }
}
