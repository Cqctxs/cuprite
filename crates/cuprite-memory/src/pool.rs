/// How many idle buffers the pool keeps at most.
const SLOTS: usize = 8;

struct IdleBuffer<B> {
    bytes: usize,
    buffer: B,
}

/// Completed owned storage kept for reuse, by exact byte length, oldest
/// first. The caller proves completion before insertion. This pool never
/// associates buffers with CPU addresses.
pub(crate) struct BufferPool<B> {
    idle: Vec<IdleBuffer<B>>,
}

impl<B> BufferPool<B> {
    pub(crate) fn new() -> Self {
        Self { idle: Vec::new() }
    }

    /// Take the most recently kept buffer of exactly `bytes`, if any.
    pub(crate) fn take(&mut self, bytes: usize) -> Option<B> {
        let index = self.idle.iter().rposition(|idle| idle.bytes == bytes)?;
        Some(self.idle.remove(index).buffer)
    }

    /// Keep a buffer whose queued work is known to be complete. Beyond the
    /// pool's capacity, the oldest idle buffer is released.
    pub(crate) fn put_completed(&mut self, bytes: usize, buffer: B) {
        if self.idle.len() == SLOTS {
            self.idle.remove(0);
        }
        self.idle.push(IdleBuffer { bytes, buffer });
    }

    pub(crate) fn idle_bytes(&self) -> usize {
        self.idle.iter().map(|idle| idle.bytes).sum()
    }

    /// Release the oldest idle buffer; false if there was none.
    pub(crate) fn evict_oldest(&mut self) -> bool {
        if self.idle.is_empty() {
            return false;
        }
        self.idle.remove(0);
        true
    }

    pub(crate) fn clear(&mut self) {
        self.idle.clear();
    }

    #[cfg(test)]
    pub(crate) fn idle_sizes(&self) -> Vec<usize> {
        self.idle.iter().map(|idle| idle.bytes).collect()
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
        assert_eq!(pool.take(4), Some(String::from("buffer-a")));
        assert_eq!(pool.take(4), None);
    }

    #[test]
    fn keeps_buffers_of_several_sizes_and_matches_exactly() {
        let mut pool = BufferPool::new();
        pool.put_completed(4, String::from("four"));
        pool.put_completed(8, String::from("eight"));
        assert_eq!(pool.take(5), None);
        assert_eq!(pool.take(8), Some(String::from("eight")));
        assert_eq!(pool.idle_sizes(), [4]);
    }

    #[test]
    fn the_most_recent_matching_buffer_is_reused_first() {
        let mut pool = BufferPool::new();
        pool.put_completed(4, String::from("older"));
        pool.put_completed(4, String::from("newer"));
        assert_eq!(pool.take(4), Some(String::from("newer")));
    }

    #[test]
    fn beyond_capacity_the_oldest_buffer_is_released() {
        let mut pool = BufferPool::new();
        for bytes in 1..=SLOTS + 1 {
            pool.put_completed(bytes, bytes.to_string());
        }
        assert_eq!(pool.take(1), None);
        assert_eq!(pool.idle_sizes().len(), SLOTS);
    }

    #[test]
    fn eviction_and_clearing_release_idle_storage() {
        let mut pool = BufferPool::new();
        pool.put_completed(4, String::from("a"));
        pool.put_completed(8, String::from("b"));
        assert_eq!(pool.idle_bytes(), 12);
        assert!(pool.evict_oldest());
        assert_eq!(pool.idle_sizes(), [8]);
        pool.clear();
        assert!(!pool.evict_oldest());
        assert_eq!(pool.idle_bytes(), 0);
    }
}
