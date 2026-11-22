//! The suites, in the order they run. Each `run` verifies its cases against host references and
//! times them through the [`Harness`](crate::harness::Harness).

pub mod cfd;
pub mod compact;
pub mod compute;
pub mod datasci;
pub mod jit;
pub mod sparse;
pub mod ml;

use crate::harness::Harness;

/// A suite's entry point.
pub type SuiteFn = fn(&mut Harness);

/// Every suite: name and entry point.
pub const ALL: &[(&str, SuiteFn)] = &[
    ("ml", ml::run),
    ("cfd", cfd::run),
    ("datasci", datasci::run),
    ("compute", compute::run),
    ("compact", compact::run),
    ("jit", jit::run),
    ("sparse", sparse::run),
];
