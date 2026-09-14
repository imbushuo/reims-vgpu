//! Initialization-aware destinations for checked memory readers.

use core::mem::MaybeUninit;

mod sealed {
    pub trait Sealed {}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DestinationSize {
    pub expected: usize,
    pub actual: usize,
}

/// A writer capability, not mutable access to the destination's initialization state.
///
/// Passing this trait object to a producer keeps the concrete buffer with its
/// owner: the producer cannot replace it with another, already-filled buffer.
///
/// ```compile_fail
/// use core::mem::MaybeUninit;
/// use reims_vgpu_memory::{ReadBuffer, ReadDestination};
///
/// fn replace_owner(destination: &mut dyn ReadDestination) {
///     let mut other = [MaybeUninit::uninit(); 8];
///     *destination = ReadBuffer::new(&mut other);
/// }
/// ```
pub trait ReadDestination: sealed::Sealed {
    fn len(&self) -> usize;
    fn initialized_len(&self) -> usize;
    fn initialized(&self) -> Option<&[u8]>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn is_complete(&self) -> bool {
        self.initialized_len() == self.len()
    }

    fn copy_from_slice(&mut self, source: &[u8]) -> Result<(), DestinationSize> {
        if source.len() != self.len() {
            return Err(DestinationSize {
                expected: self.len(),
                actual: source.len(),
            });
        }
        // SAFETY: the slice supplies initialized bytes and both complete spans
        // have the same length. Safe Rust cannot alias the borrowed destination.
        unsafe { self.copy_from_raw(0, source.as_ptr(), source.len()) };
        Ok(())
    }

    /// Copy initialized bytes into a bounded destination subrange.
    ///
    /// # Safety
    ///
    /// `source` must be valid for reading `len` initialized bytes, disjoint from
    /// the destination; `offset + len` must be within this destination.
    ///
    /// Completion tracks a contiguous prefix. Ordered writes covering the whole
    /// destination establish completion; writes beyond a gap do not certify it.
    unsafe fn copy_from_raw(&mut self, offset: usize, source: *const u8, len: usize);
}

pub struct ReadBuffer<'a> {
    bytes: &'a mut [MaybeUninit<u8>],
    initialized: usize,
}

impl<'a> ReadBuffer<'a> {
    pub fn new(bytes: &'a mut [MaybeUninit<u8>]) -> Self {
        Self {
            bytes,
            initialized: 0,
        }
    }
}

impl sealed::Sealed for ReadBuffer<'_> {}

impl ReadDestination for ReadBuffer<'_> {
    fn len(&self) -> usize {
        self.bytes.len()
    }

    fn initialized_len(&self) -> usize {
        self.initialized
    }

    fn initialized(&self) -> Option<&[u8]> {
        if !self.is_complete() {
            return None;
        }
        // SAFETY: only copies from initialized sources advance the prefix;
        // completion proves that every byte in this borrowed buffer was written.
        Some(unsafe {
            core::slice::from_raw_parts(self.bytes.as_ptr().cast::<u8>(), self.bytes.len())
        })
    }

    unsafe fn copy_from_raw(&mut self, offset: usize, source: *const u8, len: usize) {
        // SAFETY: the caller supplies both range bounds and source validity.
        unsafe {
            core::ptr::copy_nonoverlapping(
                source,
                self.bytes.as_mut_ptr().add(offset).cast::<u8>(),
                len,
            );
        }
        if offset <= self.initialized {
            self.initialized = self.initialized.max(offset + len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_requires_the_whole_destination() {
        let mut storage = [MaybeUninit::uninit(); 8];
        let mut output = ReadBuffer::new(&mut storage);
        assert!(output.initialized().is_none());
        assert_eq!(
            output.copy_from_slice(&[1; 7]),
            Err(DestinationSize {
                expected: 8,
                actual: 7
            })
        );
        assert_eq!(output.initialized_len(), 0);
        output.copy_from_slice(&[3; 8]).unwrap();
        assert_eq!(output.initialized(), Some(&[3; 8][..]));
    }

    #[test]
    fn ordered_chunks_prove_complete_initialization() {
        let mut storage = [MaybeUninit::uninit(); 8];
        let mut output = ReadBuffer::new(&mut storage);
        let first = [1, 2, 3];
        let second = [4, 5, 6, 7, 8];
        // SAFETY: both live source arrays fit their disjoint destination ranges.
        unsafe {
            output.copy_from_raw(0, first.as_ptr(), first.len());
            assert_eq!(output.initialized_len(), 3);
            assert!(output.initialized().is_none());
            output.copy_from_raw(3, second.as_ptr(), second.len());
        }
        assert_eq!(output.initialized(), Some(&[1, 2, 3, 4, 5, 6, 7, 8][..]));
    }

    #[test]
    fn a_gap_cannot_certify_unwritten_bytes() {
        let mut storage = [MaybeUninit::uninit(); 8];
        let mut output = ReadBuffer::new(&mut storage);
        let bytes = [4; 4];
        // SAFETY: the source is live and each copy fits the destination. The
        // deliberately out-of-order writes must not claim a complete prefix.
        unsafe {
            output.copy_from_raw(4, bytes.as_ptr(), 4);
            assert_eq!(output.initialized_len(), 0);
            output.copy_from_raw(0, bytes.as_ptr(), 4);
        }
        assert!(!output.is_complete());
        assert!(output.initialized().is_none());
        output.copy_from_slice(&[5; 8]).unwrap();
        assert_eq!(output.initialized(), Some(&[5; 8][..]));
    }

    #[test]
    fn empty_destination_is_complete_without_a_write() {
        let mut storage = [];
        let output = ReadBuffer::new(&mut storage);
        assert!(output.is_complete());
        assert_eq!(output.initialized(), Some(&[][..]));
    }
}
