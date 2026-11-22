//! ORB, SIFT and GLOH: oriented keypoints with descriptors on an image pyramid (ArrayFire's
//! `orb`, `sift` and `gloh`). See [`Array::orb`](crate::Array::orb).

mod orb;
mod orb_pattern;
mod resize;

/// One kernel of this group (part of the key of its cache entry in the context).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Kernel {
    /// The bilinear resize of the pyramids.
    Resize,
    /// A kernel of ORB.
    Orb(orb::Kernel),
}

pub(crate) fn source(kernel: Kernel) -> String {
    match kernel {
        Kernel::Resize => resize::source(),
        Kernel::Orb(k) => orb::source(k),
    }
}
