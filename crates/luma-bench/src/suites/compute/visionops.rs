//! The feature detectors and image analysis of `luma::vision` (`fast`, ...) at 1080p and 4K.
//! Each kind is checked against a host computation before it is timed. One file per group, see
//! `src/vision/`.

use crate::harness::*;

mod corners;
mod descriptors;
mod edges;
mod fast;
mod homography;
mod matching;
mod regions;

pub fn run(h: &mut Harness) {
    fast::run(h);
    homography::run(h);
    corners::run(h);
    matching::run(h);
    edges::run(h);
    regions::run(h);
    descriptors::run(h);
}
