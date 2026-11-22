//! Harris and SUSAN corners, difference of Gaussians.
//!
//! [`Array::harris`] and [`Array::susan`] are fused detectors: a workgroup stages a tile of the
//! image with its halo in workgroup memory, computes the corner response of the tile and of its
//! one-pixel border there, decides which pixels are strict local maxima and writes them (in the
//! order of the tile's pixels) to a small per-tile list; a scan of the per-tile counts and one
//! gather kernel then compact the lists, like [`Array::fast`]. The response image itself is never
//! stored. [`Array::dog`] is a difference of two separable Gaussian blurs.
//!
//! Where ArrayFire's CPU and CUDA backends agree the detectors reproduce their features exactly
//! (a pixel is a corner of the same response, a strict maximum of its eight neighbours); the
//! order of the points is the one of the tiles, like ArrayFire's CUDA backend (which also
//! returns the features of its blocks in no fixed order), not the row order of the CPU backend.

mod detect;
mod dog;
mod harris;
mod select;
mod susan;

/// One kernel of this group (part of the key of its cache entry in the context).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Kernel {
    /// The Harris response and its maxima, see [`Array::harris`].
    Harris(harris::Kernel),
    /// The SUSAN response and its maxima, see [`Array::susan`].
    Susan(susan::Kernel),
    /// Compacts the per-tile lists of a detector.
    Gather(detect::Gather),
    /// The difference of Gaussians, see [`Array::dog`].
    Dog(dog::Kernel),
    /// The strongest corners of a list, see [`Array::harris`].
    Select(select::Kernel),
}

/// The WGSL of a kernel of this group.
pub(crate) fn source(kernel: Kernel) -> String {
    match kernel {
        Kernel::Harris(k) => harris::source(k),
        Kernel::Susan(k) => susan::source(k),
        Kernel::Gather(k) => detect::gather_source(k),
        Kernel::Dog(k) => dog::source(k),
        Kernel::Select(k) => select::source(k),
    }
}
