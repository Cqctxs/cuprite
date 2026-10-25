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

    /// Take the idle buffer if it has exactly `len` elements. A buffer of any
    /// other length is released, so it never counts against the budget
    /// alongside the replacement the caller is about to allocate.
    pub(crate) fn take_matching(&mut self, len: usize) -> Option<B> {
        let idle = self.idle.take()?;
        (idle.len == len).then_some(idle.buffer)
    }

    /// Keep a buffer whose queued work is known to be complete, replacing
    /// any buffer already idle.
    pub(crate) fn put_completed(&mut self, len: usize, buffer: B) {
        self.idle = Some(IdleBuffer { len, buffer });
    }

    pub(crate) fn clear(&mut self) {
        self.idle = None;
    }

    #[cfg(test)]
    pub(crate) fn idle_len(&self) -> Option<usize> {
        self.idle.as_ref().map(|idle| idle.len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Strings stand in for owned buffers; the tests check the real pool logic.
    #[test]
    fn returns_a_matching_buffer_only_once() {
        let mut pool = BufferPool::new();
        pool.put_completed(4, String::from("buffer-a"));
        assert_eq!(pool.take_matching(4), Some(String::from("buffer-a")));
        assert_eq!(pool.take_matching(4), None);
    }

    #[test]
    fn evicts_a_buffer_with_the_wrong_length() {
        let mut pool = BufferPool::new();
        pool.put_completed(4, String::from("buffer-a"));
        assert_eq!(pool.take_matching(5), None);
        assert_eq!(pool.take_matching(4), None);
    }

    #[test]
    fn trimming_removes_idle_storage() {
        let mut pool = BufferPool::new();
        pool.put_completed(4, String::from("buffer-a"));
        pool.clear();
        assert_eq!(pool.take_matching(4), None);
    }

    #[test]
    fn a_new_buffer_replaces_the_idle_one() {
        let mut pool = BufferPool::new();
        pool.put_completed(4, String::from("buffer-a"));
        pool.put_completed(4, String::from("buffer-b"));
        assert_eq!(pool.take_matching(4), Some(String::from("buffer-b")));
    }
}
