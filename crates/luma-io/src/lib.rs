//! File I/O for file-backed Luma arrays: a positional-I/O engine (`engine`)
//! and the `.npy` / sidecar file formats (`format`). No GPU and no Luma types.

#![deny(missing_docs)]

mod engine;
mod format;

#[cfg(feature = "testing")]
pub use engine::{FailPlan, FailingIo, Fault};
pub use engine::{
    Advice, CHUNK_BYTES, Extent, IoAlign, IoFile, IoTicket, PoolIo, StagePtr, TileIo,
    default_engine, read_blocking, write_blocking,
};
pub use format::{
    Dtype, FormatError, NpyHeader, Sidecar, SidecarState, parse_npy, parse_sidecar,
    write_npy_header, write_sidecar,
};
