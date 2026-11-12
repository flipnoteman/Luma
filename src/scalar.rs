//! [`Scalar`]: a host-side value shared between clones, usable wherever a plain number is.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use crate::gpu_type::ScalarType;

/// A number that can be changed between operations: `&x * &s` multiplies by the value `s` holds
/// when the operation is recorded, `s.set(..)` changes what later operations see.
///
/// Like every number operand it is passed to the generated kernel as a uniform, so changing it
/// never recompiles anything: `&x * &s` with `s = 2.0` and with `s = 3.0` run the same kernel.
/// The value is read when the operation is recorded, not when it runs: operations recorded
/// before a `set` keep the old value, also if they are evaluated later. Clones share the value.
///
/// ```no_run
/// use luma::{Array, Scalar};
/// let x = Array::from_slice(&[1.0f32, 2.0], 2)?;
/// let s = Scalar::new(2.0f32);
/// let a = &x * &s; // doubles
/// s.set(3.0);
/// let b = &x * &s; // triples; `a` still doubles
/// assert_eq!(a.to_vec()?, [2.0, 4.0]);
/// assert_eq!(b.to_vec()?, [3.0, 6.0]);
/// # Ok::<(), luma::LumaError>(())
/// ```
#[derive(Clone, Debug)]
pub struct Scalar<T: ScalarType> {
    bits: Arc<AtomicU32>,
    _marker: PhantomData<T>,
}

/// The element types a [`Scalar`] can hold: those stored in one 32-bit word.
#[doc(hidden)]
pub trait Word: ScalarType {
    fn to_word(self) -> u32;
    fn from_word(word: u32) -> Self;
}

impl Word for f32 {
    fn to_word(self) -> u32 {
        self.to_bits()
    }

    fn from_word(word: u32) -> Self {
        f32::from_bits(word)
    }
}

impl Word for u32 {
    fn to_word(self) -> u32 {
        self
    }

    fn from_word(word: u32) -> Self {
        word
    }
}

impl<T: Word> Scalar<T> {
    /// A scalar holding `value`.
    pub fn new(value: T) -> Self {
        Scalar { bits: Arc::new(AtomicU32::new(value.to_word())), _marker: PhantomData }
    }

    /// Changes the value that operations recorded from now on see (in every clone).
    pub fn set(&self, value: T) {
        self.bits.store(value.to_word(), Ordering::Relaxed);
    }

    /// The current value.
    pub fn get(&self) -> T {
        T::from_word(self.bits.load(Ordering::Relaxed))
    }
}
