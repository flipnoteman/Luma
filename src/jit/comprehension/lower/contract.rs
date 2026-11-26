//! G-A: the fused matmul tail for any separable body (W2-R0 §5.3).

use std::panic::Location;
use std::sync::Arc;

use crate::array::{Failure, Storage};
use crate::jit::comprehension::Comprehension;
use crate::jit::structured::emit::Emit;

pub(crate) struct Prepared {
    pub(crate) k: u64,
}

pub(crate) fn prepare(_: &Comprehension) -> Option<Prepared> {
    None
}

impl Prepared {
    pub(crate) fn run(&self, _: &mut Emit<'_>, _: &'static Location<'static>) -> Result<Arc<Storage>, Failure> {
        unreachable!("not yet")
    }
}
