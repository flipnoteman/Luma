//! Structural operations (`tile`, `diag`, `diag_matrix`, `identity`, `lower`/`upper`, `iota`,
//! `replace`, `diff1`/`diff2`): index maps and generators that fuse into what reads them, run on
//! their own here so that each is one kernel writing its output (or, for `diag`, a handful of
//! elements). Every case is checked against a host loop first.

use luma::*;

use super::{check_sampled, host, pick, upload};
use crate::harness::*;

pub fn run(h: &mut Harness) {
    h.guarded("structure", |h| {
        let ctx = h.ctx.clone();
        let cpu = ctx.is_cpu();
        let side: usize = if cpu { pick(h, 512, 2048) } else { pick(h, 1024, 4096) };
        let n = side * side;
        let bytes = 4.0 * n as f64;
        let sample = 61;
        let rel = 0.0;

        // tile: a quarter-side matrix repeated 4 x 4.
        let q = side / 4;
        let src = values(q * q, 301, -1.0, 1.0);
        let m = upload(h, &src, [q, q]);
        let tile_host = |i: usize| f64::from(src[(i % side) % q + q * ((i / side) % q)]);
        check_sampled("structure/tile", &host(&m.tile([4, 4])), tile_host, sample, rel, 0.0);
        h.case("structure/tile", &format!("[{q},{q}] x 4x4"), Work::Bytes(bytes), &mut || m.tile([4, 4]).eval().unwrap());
        let v = upload(h, &src[..1000], 1000);
        let reps = n / 1000;
        check_sampled("structure/tile_vector", &host(&v.tile(reps)), |i| f64::from(src[i % 1000]), sample, rel, 0.0);
        h.case("structure/tile_vector", &format!("1000 x {reps}"), Work::Bytes(4.0 * (reps * 1000) as f64), &mut || v.tile(reps).eval().unwrap());

        // diag: extract (a view of a big matrix: reads one element per row) and create.
        let data = values(n, 302, -1.0, 1.0);
        let a = upload(h, &data, [side, side]);
        check_sampled("structure/diag_extract", &host(&a.diag(1)), |i| f64::from(data[i + side * (i + 1)]), 1, rel, 0.0);
        h.case("structure/diag_extract", &format!("[{side},{side}] k=1"), Work::Bytes(8.0 * (side - 1) as f64), &mut || a.diag(1).eval().unwrap());
        let col = upload(h, &data[..side], side);
        check_sampled("structure/diag_create", &host(&col.diag_matrix(0)), |i| if i % side == i / side { f64::from(data[i % side]) } else { 0.0 }, sample, rel, 0.0);
        h.case("structure/diag_create", &format!("{side} -> [{side},{side}]"), Work::Bytes(bytes), &mut || col.diag_matrix(0).eval().unwrap());

        // identity and the triangles.
        let ident = |i: usize| if i % side == i / side { 1.0 } else { 0.0 };
        check_sampled("structure/identity", &host(&Array::<f32>::identity_on(&ctx, [side, side]).unwrap()), ident, sample, rel, 0.0);
        h.case("structure/identity", &format!("[{side},{side}]"), Work::Bytes(bytes), &mut || Array::<f32>::identity_on(&ctx, [side, side]).unwrap().eval().unwrap());
        check_sampled("structure/lower", &host(&a.lower(false)), |i| if i / side > i % side { 0.0 } else { f64::from(data[i]) }, sample, rel, 0.0);
        h.case("structure/lower", &format!("[{side},{side}]"), Work::Bytes(2.0 * bytes), &mut || a.lower(false).eval().unwrap());
        let upper = |i: usize| match (i / side).cmp(&(i % side)) {
            std::cmp::Ordering::Equal => 1.0,
            std::cmp::Ordering::Less => 0.0,
            std::cmp::Ordering::Greater => f64::from(data[i]),
        };
        check_sampled("structure/upper_unit", &host(&a.upper(true)), upper, sample, rel, 0.0);
        h.case("structure/upper_unit", &format!("[{side},{side}]"), Work::Bytes(2.0 * bytes), &mut || a.upper(true).eval().unwrap());

        // iota: tiled and plain.
        check_sampled("structure/iota", &host(&Array::<f32>::iota_on(&ctx, [q, q], [4, 4]).unwrap()), |i| ((i % side) % q + q * ((i / side) % q)) as f64, sample, rel, 0.0);
        h.case("structure/iota", &format!("[{q},{q}] x 4x4"), Work::Bytes(bytes), &mut || Array::<f32>::iota_on(&ctx, [q, q], [4, 4]).unwrap().eval().unwrap());
        check_sampled("structure/iota_flat", &host(&Array::<f32>::iota_on(&ctx, n, 1).unwrap()), |i| i as f64, sample, rel, 0.0);
        h.case("structure/iota_flat", &format!("{n}"), Work::Bytes(bytes), &mut || Array::<f32>::iota_on(&ctx, n, 1).unwrap().eval().unwrap());

        // replace: `a` where the condition holds, `b` (an array) or a scalar elsewhere.
        let other = values(n, 303, 10.0, 11.0);
        let b = upload(h, &other, n);
        let flat = upload(h, &data, n);
        let cond = flat.gt(0.0);
        cond.eval().unwrap();
        check_sampled("structure/replace", &host(&flat.replace(&cond, &b)), |i| f64::from(if data[i] > 0.0 { data[i] } else { other[i] }), sample, rel, 0.0);
        h.case("structure/replace", &format!("{n}"), Work::Bytes(4.0 * bytes + 0.125 * n as f64), &mut || flat.replace(&cond, &b).eval().unwrap());
        check_sampled("structure/replace_scalar", &host(&flat.replace(&cond, 0.5)), |i| f64::from(if data[i] > 0.0 { data[i] } else { 0.5 }), sample, rel, 0.0);
        h.case("structure/replace_scalar", &format!("{n}"), Work::Bytes(2.0 * bytes + 0.125 * n as f64), &mut || flat.replace(&cond, 0.5).eval().unwrap());

        // differences along the fast axis (a vector) and along a strided axis.
        check_sampled("structure/diff1", &host(&flat.diff1(0)), |i| f64::from(data[i + 1] - data[i]), sample, rel, 0.0);
        h.case("structure/diff1", &format!("{n}"), Work::Bytes(2.0 * bytes), &mut || flat.diff1(0).eval().unwrap());
        let second = |i: usize| f64::from((data[i + 2] - data[i + 1]) - (data[i + 1] - data[i]));
        check_sampled("structure/diff2", &host(&flat.diff2(0)), second, sample, rel, 0.0);
        h.case("structure/diff2", &format!("{n}"), Work::Bytes(2.0 * bytes), &mut || flat.diff2(0).eval().unwrap());
        check_sampled("structure/diff1_axis1", &host(&a.diff1(1)), |i| f64::from(data[i + side] - data[i]), sample, rel, 0.0);
        h.case("structure/diff1_axis1", &format!("[{side},{side}]"), Work::Bytes(2.0 * bytes), &mut || a.diff1(1).eval().unwrap());
    });
}
