//! GPU timing of kernel launches. The result types live in `luma_core` (every backend fills
//! them); the timestamp machinery is the backend's.

pub(crate) use luma_core::{FusionSplit, SplitReason};
