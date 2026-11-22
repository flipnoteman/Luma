//! Feature detection, matching and image analysis against ArrayFire's `vision.h` / `image.h`
//! functions. One file per group (see `src/vision/` of luma).

mod corners;
mod descriptors;
mod edges;
mod fast;
mod matching;
mod regions;

use crate::bench::Bench;

pub fn vision(b: &mut Bench) {
    fast::rows(b);
    corners::rows(b);
    matching::rows(b);
    edges::rows(b);
    regions::rows(b);
    descriptors::rows(b);
}
