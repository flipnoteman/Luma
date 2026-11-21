//! The suites, in the order they run. Each `run` verifies its cases against host references and
//! times them through the [`Harness`](crate::harness::Harness).

pub mod cfd;
pub mod compute;
pub mod datasci;
pub mod jit;
pub mod ml;

use crate::harness::Harness;

/// Every suite: name and entry point.
pub const ALL: &[(&str, fn(&mut Harness))] = &[
    ("ml", ml::run),
    ("cfd", cfd::run),
    ("datasci", datasci::run),
    ("compute", compute::run),
    ("jit", jit::run),
];
