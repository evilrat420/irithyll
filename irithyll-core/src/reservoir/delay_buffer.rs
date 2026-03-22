//! Circular delay buffer for time-delay embeddings.
//!
//! [`DelayBuffer`] stores the last `capacity` observations (each a `Vec<f64>`)
//! in a ring buffer, enabling efficient construction of time-delay feature vectors
//! for reservoir computing models like NG-RC.
//!
//! Index 0 is the most recent observation, index 1 is the previous, and so on.

use alloc::vec;
use alloc::vec::Vec;

/// Circular buffer storing the last `capacity` observation vectors.
///
/// Observations are pushed one at a time. Once `capacity` observations have
/// been stored, the buffer is "ready" and older observations are overwritten
/// in FIFO order.
///
/// # Examples
///
/// ```
/// use irithyll_core::reservoir::DelayBuffer;
///
/// let mut buf = DelayBuffer::new(3);
/// buf.push(&[1.0]);
/// buf.push(&[2.0]);
/// buf.push(&[3.0]);
/// assert!(buf.is_ready());
///
/// assert_eq!(buf.get(0).unwrap(), &[3.0]); // most recent
/// assert_eq!(buf.get(1).unwrap(), &[2.0]);
/// assert_eq!(buf.get(2).unwrap(), &[1.0]); // oldest
/// ```
pub struct DelayBuffer {
    /// Ring buffer storage.
    data: Vec<Vec<f64>>,
    /// Capacity (maximum number of stored observations).
    capacity: usize,
    /// Write cursor — the index where the next push will write.
    head: usize,
    /// Number of observations currently stored (saturates at capacity).
    count: usize,
}

impl DelayBuffer {
    /// Create a new delay buffer with the given capacity.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is 0.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "DelayBuffer capacity must be > 0");
        Self {
            data: vec![Vec::new(); capacity],
            capacity,
            head: 0,
            count: 0,
        }
    }

    /// Push a new observation into the buffer.
    ///
    /// If the buffer is full, the oldest observation is overwritten.
    pub fn push(&mut self, obs: &[f64]) {
        self.data[self.head] = obs.to_vec();
        self.head = (self.head + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    /// Get the observation at the given delay index.
    ///
    /// - `delay_idx = 0` returns the most recently pushed observation.
    /// - `delay_idx = 1` returns the previous observation.
    /// - Returns `None` if `delay_idx >= len()`.
    pub fn get(&self, delay_idx: usize) -> Option<&[f64]> {
        if delay_idx >= self.count {
            return None;
        }
        // head points to the NEXT write slot, so most recent is head - 1.
        // delay_idx = 0 → most recent = head - 1
        // delay_idx = 1 → head - 2, etc.
        let idx = (self.head + self.capacity - 1 - delay_idx) % self.capacity;
        Some(&self.data[idx])
    }

    /// Whether the buffer has been filled to capacity at least once.
    ///
    /// NG-RC requires `is_ready()` before building features, because all
    /// `k` delay slots must be populated.
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.count >= self.capacity
    }

    /// Number of observations currently stored (at most `capacity`).
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The maximum number of observations this buffer can hold.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Reset the buffer, discarding all stored observations.
    pub fn reset(&mut self) {
        for slot in &mut self.data {
            slot.clear();
        }
        self.head = 0;
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_push_and_get() {
        let mut buf = DelayBuffer::new(3);
        buf.push(&[1.0, 10.0]);
        buf.push(&[2.0, 20.0]);
        buf.push(&[3.0, 30.0]);

        assert!(buf.is_ready());
        assert_eq!(buf.len(), 3);

        assert_eq!(buf.get(0).unwrap(), &[3.0, 30.0]);
        assert_eq!(buf.get(1).unwrap(), &[2.0, 20.0]);
        assert_eq!(buf.get(2).unwrap(), &[1.0, 10.0]);
        assert!(buf.get(3).is_none());
    }

    #[test]
    fn wraparound_overwrites_oldest() {
        let mut buf = DelayBuffer::new(2);
        buf.push(&[1.0]);
        buf.push(&[2.0]);
        buf.push(&[3.0]); // overwrites [1.0]

        assert_eq!(buf.len(), 2);
        assert_eq!(buf.get(0).unwrap(), &[3.0]);
        assert_eq!(buf.get(1).unwrap(), &[2.0]);
        assert!(buf.get(2).is_none());
    }

    #[test]
    fn not_ready_until_full() {
        let mut buf = DelayBuffer::new(3);
        assert!(!buf.is_ready());
        assert!(buf.is_empty());

        buf.push(&[1.0]);
        assert!(!buf.is_ready());
        assert_eq!(buf.len(), 1);

        buf.push(&[2.0]);
        assert!(!buf.is_ready());
        assert_eq!(buf.len(), 2);

        buf.push(&[3.0]);
        assert!(buf.is_ready());
        assert_eq!(buf.len(), 3);
    }

    #[test]
    fn get_returns_none_for_empty() {
        let buf = DelayBuffer::new(5);
        assert!(buf.get(0).is_none());
    }

    #[test]
    fn get_partial_fill() {
        let mut buf = DelayBuffer::new(5);
        buf.push(&[10.0]);
        buf.push(&[20.0]);

        assert_eq!(buf.get(0).unwrap(), &[20.0]);
        assert_eq!(buf.get(1).unwrap(), &[10.0]);
        assert!(buf.get(2).is_none());
    }

    #[test]
    fn reset_clears_all() {
        let mut buf = DelayBuffer::new(3);
        buf.push(&[1.0]);
        buf.push(&[2.0]);
        buf.push(&[3.0]);
        assert!(buf.is_ready());

        buf.reset();
        assert!(buf.is_empty());
        assert!(!buf.is_ready());
        assert_eq!(buf.len(), 0);
        assert!(buf.get(0).is_none());
    }

    #[test]
    fn capacity_one() {
        let mut buf = DelayBuffer::new(1);
        buf.push(&[42.0]);
        assert!(buf.is_ready());
        assert_eq!(buf.get(0).unwrap(), &[42.0]);

        buf.push(&[99.0]);
        assert_eq!(buf.get(0).unwrap(), &[99.0]);
        assert!(buf.get(1).is_none());
    }

    #[test]
    fn many_pushes_stress_test() {
        let cap = 5;
        let mut buf = DelayBuffer::new(cap);
        for i in 0..100 {
            buf.push(&[i as f64]);
        }
        assert!(buf.is_ready());
        assert_eq!(buf.len(), cap);

        // Most recent should be 99, oldest should be 95.
        for d in 0..cap {
            let expected = (99 - d) as f64;
            assert_eq!(
                buf.get(d).unwrap(),
                &[expected],
                "delay {} should be {}",
                d,
                expected,
            );
        }
    }

    #[test]
    #[should_panic(expected = "capacity must be > 0")]
    fn zero_capacity_panics() {
        let _ = DelayBuffer::new(0);
    }
}
