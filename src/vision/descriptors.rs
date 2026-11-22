//! ORB, SIFT and GLOH: oriented keypoints with descriptors on an image pyramid (ArrayFire's
//! `orb`, `sift` and `gloh`, vision.h).
//!
//! * [`Array::orb`](crate::Array::orb): FAST corners ranked by Harris response on a pyramid of
//!   bilinear resizes, orientation by intensity centroid, 256-bit rotated BRIEF (`orb.rs`).
//! * [`Array::sift`](crate::Array::sift) and [`Array::gloh`](crate::Array::gloh): scale-space
//!   extrema, refinement, 36-bin orientation histograms, 128-element SIFT and 272-element GLOH
//!   descriptors (`sift.rs`).
//!
//! The pyramids resize with [`resize`] (ArrayFire's sampling convention). The algorithms follow
//! ArrayFire's CUDA backend; its quirks that decide which points are found are reproduced and
//! documented on the functions.

mod orb;
mod orb_pattern;
mod resize;
mod sift;

/// One kernel of this group (part of the key of its cache entry in the context).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Kernel {
    /// The bilinear resize of the pyramids.
    Resize,
    /// A kernel of ORB.
    Orb(orb::Kernel),
    /// A kernel of SIFT and GLOH.
    Sift(sift::Kernel),
}

pub(crate) fn source(kernel: Kernel) -> String {
    match kernel {
        Kernel::Resize => resize::source(),
        Kernel::Orb(k) => orb::source(k),
        Kernel::Sift(k) => sift::source(k),
    }
}
