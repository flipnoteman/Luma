#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CannyThreshold { Manual, AutoOtsu }
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Kernel {}
pub(crate) fn source(k: Kernel) -> String { match k {} }
