//! `tile`, `diag`, `diag_matrix`, `identity`, `lower`/`upper`, `iota`, `replace` and `diff1`/`diff2`
//! against host loops over the same index rules, for every element type.

use std::fmt::Debug;

use luma::{Array, Context, Df32, LumaError, Numeric, ScalarType};

/// An element type of the tests with the host-side values the references need.
trait Elem: ScalarType + Debug + Copy {
    fn make(i: usize) -> Self;
    fn zero() -> Self;
    fn one() -> Self;
    /// Whether the two agree as results of the same operation (bits for the exact types).
    fn same(self, other: Self) -> bool;
}

impl Elem for f32 {
    fn make(i: usize) -> f32 {
        (i as f32) * 0.37 - 5.0
    }
    fn zero() -> f32 {
        0.0
    }
    fn one() -> f32 {
        1.0
    }
    fn same(self, other: f32) -> bool {
        self.to_bits() == other.to_bits() || (self.is_nan() && other.is_nan())
    }
}

impl Elem for i32 {
    fn make(i: usize) -> i32 {
        (i as i32).wrapping_mul(2_654_435) - 1_000_000
    }
    fn zero() -> i32 {
        0
    }
    fn one() -> i32 {
        1
    }
    fn same(self, other: i32) -> bool {
        self == other
    }
}

impl Elem for u32 {
    fn make(i: usize) -> u32 {
        (i as u32).wrapping_mul(2_654_435_761)
    }
    fn zero() -> u32 {
        0
    }
    fn one() -> u32 {
        1
    }
    fn same(self, other: u32) -> bool {
        self == other
    }
}

impl Elem for bool {
    fn make(i: usize) -> bool {
        (i * 7 + i / 3) % 3 != 0
    }
    fn zero() -> bool {
        false
    }
    fn one() -> bool {
        true
    }
    fn same(self, other: bool) -> bool {
        self == other
    }
}

impl Elem for f64 {
    fn make(i: usize) -> f64 {
        (i as f64) * 0.37e-3 + 1.0e9 + 1.0e-7
    }
    fn zero() -> f64 {
        0.0
    }
    fn one() -> f64 {
        1.0
    }
    fn same(self, other: f64) -> bool {
        self.to_bits() == other.to_bits()
    }
}

impl Elem for Df32 {
    fn make(i: usize) -> Df32 {
        Df32::from((i as f64) * 0.37 + 1.0e-9 * i as f64 + 3.0)
    }
    fn zero() -> Df32 {
        Df32 { hi: 0.0, lo: 0.0 }
    }
    fn one() -> Df32 {
        Df32 { hi: 1.0, lo: 0.0 }
    }
    fn same(self, other: Df32) -> bool {
        (self.to_f64() - other.to_f64()).abs() <= 1e-12 * self.to_f64().abs().max(1.0)
    }
}

fn has_f64(ctx: &Context) -> bool {
    ctx.features().contains(luma::Features::SHADER_F64)
}

fn strides(dims: [usize; 4]) -> [usize; 4] {
    [1, dims[0], dims[0] * dims[1], dims[0] * dims[1] * dims[2]]
}

fn coords(dims: [usize; 4], i: usize) -> [usize; 4] {
    let s = strides(dims);
    std::array::from_fn(|k| i / s[k] % dims[k])
}

fn flat(dims: [usize; 4], c: [usize; 4]) -> usize {
    let s = strides(dims);
    (0..4).map(|k| c[k] * s[k]).sum()
}

fn len(dims: [usize; 4]) -> usize {
    dims.iter().product()
}

fn data<T: Elem>(dims: [usize; 4]) -> Vec<T> {
    (0..len(dims)).map(T::make).collect()
}

fn assert_same<T: Elem>(got: &[T], want: &[T], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(g.same(*w), "{what}: element {i}: got {g:?}, want {w:?}");
    }
}

const TILE_CASES: &[([usize; 4], [usize; 4])] = &[
    ([1, 1, 1, 1], [1, 1, 1, 1]),
    ([1, 1, 1, 1], [5, 1, 1, 1]),
    ([3, 1, 1, 1], [1, 1, 1, 1]),
    ([3, 1, 1, 1], [4, 1, 1, 1]),
    ([5, 1, 1, 1], [2, 3, 1, 1]),
    ([4, 3, 1, 1], [1, 2, 1, 1]),
    ([4, 3, 1, 1], [3, 1, 2, 1]),
    ([7, 5, 3, 1], [2, 2, 2, 2]),
    ([2, 3, 2, 3], [3, 2, 3, 2]),
    ([33, 1, 1, 1], [1, 31, 1, 1]),
    ([64, 8, 1, 1], [2, 3, 1, 1]),
];

fn tile_test<T: Elem>(ctx: &Context) {
    for &(dims, reps) in TILE_CASES {
        let host = data::<T>(dims);
        let a = Array::from_slice_on(ctx, &host, dims).unwrap();
        let out: [usize; 4] = std::array::from_fn(|k| dims[k] * reps[k]);
        let tiled = a.tile(reps);
        assert_eq!(tiled.shape().dims(), out);
        let want: Vec<T> = (0..len(out))
            .map(|i| {
                let c = coords(out, i);
                host[flat(dims, std::array::from_fn(|k| c[k] % dims[k]))]
            })
            .collect();
        assert_same(&tiled.to_vec().unwrap(), &want, &format!("tile {dims:?} x {reps:?}"));
    }
}

#[test]
fn tile_matches_the_host() {
    let ctx = Context::new().unwrap();
    tile_test::<f32>(&ctx);
    tile_test::<i32>(&ctx);
    tile_test::<u32>(&ctx);
    tile_test::<bool>(&ctx);
    tile_test::<Df32>(&ctx);
    if has_f64(&ctx) {
        tile_test::<f64>(&ctx);
    }
}

#[test]
fn tile_takes_missing_reps_as_one_and_composes() {
    let ctx = Context::new().unwrap();
    let a = Array::from_slice_on(&ctx, &[1.0f32, 2.0, 3.0], 3).unwrap();
    assert_eq!(a.tile(2).to_vec().unwrap(), [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    assert_eq!(a.tile([1, 2]).shape().dims(), [3, 2, 1, 1]);
    // A tile of a tile reads the original: lengths multiply.
    let b = a.tile(2).tile([2, 2]);
    assert_eq!(b.shape().dims(), [12, 2, 1, 1]);
    let want: Vec<f32> = (0..24).map(|i| [1.0, 2.0, 3.0][i % 12 % 3]).collect();
    assert_eq!(b.to_vec().unwrap(), want);
    // Tiles of expressions and of slices fuse.
    let e = (&a * 2.0 + 1.0).tile(3);
    assert_eq!(e.to_vec().unwrap(), [3.0, 5.0, 7.0, 3.0, 5.0, 7.0, 3.0, 5.0, 7.0]);
    let s = a.slice(0, 1..3).tile([2, 2]);
    assert_eq!(s.to_vec().unwrap(), [2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0]);
}

#[test]
fn tile_keeps_nan_and_inf() {
    let ctx = Context::new().unwrap();
    let host = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0, 1.0];
    let a = Array::from_slice_on(&ctx, &host, 5).unwrap();
    let got = a.tile([3]).to_vec().unwrap();
    let want: Vec<f32> = (0..15).map(|i| host[i % 5]).collect();
    assert_same(&got, &want, "tile");
}

const DIAG_SHAPES: &[[usize; 4]] = &[[1, 1, 1, 1], [5, 1, 1, 1], [1, 5, 1, 1], [4, 4, 1, 1], [6, 3, 1, 1], [3, 6, 1, 1], [7, 5, 2, 3], [33, 65, 1, 1]];

fn diag_test<T: Elem>(ctx: &Context) {
    for &dims in DIAG_SHAPES {
        let host = data::<T>(dims);
        let a = Array::from_slice_on(ctx, &host, dims).unwrap();
        let [m, n, b2, b3] = dims;
        for k in -(m as isize) - 1..=(n as isize) + 1 {
            let shift = k.unsigned_abs();
            let len = if k >= 0 { n.checked_sub(shift).map(|c| c.min(m)) } else { m.checked_sub(shift).map(|r| r.min(n)) };
            let d = a.diag(k);
            let Some(len) = len.filter(|&l| l > 0) else {
                assert!(matches!(d.check(), Err(LumaError::Op { .. })), "diag {k} of {dims:?} must fail");
                continue;
            };
            assert_eq!(d.shape().dims(), [len, 1, b2, b3], "diag {k} of {dims:?}");
            let out = [len, 1, b2, b3];
            let want: Vec<T> = (0..len * b2 * b3)
                .map(|o| {
                    let c = coords(out, o);
                    let (i, j) = if k >= 0 { (c[0], c[0] + shift) } else { (c[0] + shift, c[0]) };
                    host[flat(dims, [i, j, c[2], c[3]])]
                })
                .collect();
            assert_same(&d.to_vec().unwrap(), &want, &format!("diag {k} of {dims:?}"));
        }
    }
}

#[test]
fn diag_extracts_every_diagonal() {
    let ctx = Context::new().unwrap();
    diag_test::<f32>(&ctx);
    diag_test::<i32>(&ctx);
    diag_test::<u32>(&ctx);
    diag_test::<bool>(&ctx);
    diag_test::<Df32>(&ctx);
    if has_f64(&ctx) {
        diag_test::<f64>(&ctx);
    }
}

fn diag_matrix_test<T: Elem>(ctx: &Context) {
    for dims in [[1, 1, 1, 1], [5, 1, 1, 1], [8, 1, 1, 1], [3, 1, 2, 2], [100, 1, 1, 1]] {
        let host = data::<T>(dims);
        let v = Array::from_slice_on(ctx, &host, dims).unwrap();
        let n = dims[0];
        for k in [-3isize, -1, 0, 1, 2, 5] {
            let side = n + k.unsigned_abs();
            let m = v.diag_matrix(k);
            let out = [side, side, dims[2], dims[3]];
            assert_eq!(m.shape().dims(), out);
            let want: Vec<T> = (0..len(out))
                .map(|o| {
                    let c = coords(out, o);
                    let (i, j) = (c[0] as isize, c[1] as isize);
                    if j - i == k { host[flat(dims, [c[0].min(c[1]), 0, c[2], c[3]])] } else { T::zero() }
                })
                .collect();
            assert_same(&m.to_vec().unwrap(), &want, &format!("diag_matrix {k} of {dims:?}"));
            // The diagonal of the matrix is the column again.
            assert_same(&m.diag(k).to_vec().unwrap(), &host, &format!("diag(diag_matrix) {k}"));
        }
    }
}

#[test]
fn diag_matrix_builds_every_diagonal() {
    let ctx = Context::new().unwrap();
    diag_matrix_test::<f32>(&ctx);
    diag_matrix_test::<i32>(&ctx);
    diag_matrix_test::<u32>(&ctx);
    diag_matrix_test::<bool>(&ctx);
    diag_matrix_test::<Df32>(&ctx);
    if has_f64(&ctx) {
        diag_matrix_test::<f64>(&ctx);
    }
}

#[test]
fn diag_keeps_nan_and_the_off_diagonal_is_zero() {
    let ctx = Context::new().unwrap();
    let v = Array::from_slice_on(&ctx, &[f32::NAN, f32::INFINITY, -2.0], 3).unwrap();
    let m = v.diag_matrix(0).to_vec().unwrap();
    assert_same(&m, &[f32::NAN, 0.0, 0.0, 0.0, f32::INFINITY, 0.0, 0.0, 0.0, -2.0], "diag_matrix");
    let m = Array::from_slice_on(&ctx, &[f32::NAN, 1.0, 2.0, f32::INFINITY], [2, 2]).unwrap();
    assert_same(&m.diag(0).to_vec().unwrap(), &[f32::NAN, f32::INFINITY], "diag");
}

#[test]
fn diag_of_expressions_fuses() {
    let ctx = Context::new().unwrap();
    let host = data::<f32>([6, 4, 1, 1]);
    let a = Array::from_slice_on(&ctx, &host, [6, 4]).unwrap();
    let got = (&a * 2.0 + 1.0).diag(1).to_vec().unwrap();
    let want: Vec<f32> = (0..4 - 1).map(|i| host[i + 6 * (i + 1)] * 2.0 + 1.0).collect();
    assert_same(&got, &want, "diag of an expression");
    // diag_matrix of a lazy column, summed with a matrix.
    let v = (&Array::from_slice_on(&ctx, &[1.0f32, 2.0, 3.0], 3).unwrap() * 10.0).diag_matrix(0);
    let sum = (&v + &Array::<f32>::identity_on(&ctx, [3, 3]).unwrap()).to_vec().unwrap();
    assert_eq!(sum, [11.0, 0.0, 0.0, 0.0, 21.0, 0.0, 0.0, 0.0, 31.0]);
}

fn identity_test<T: Elem>(ctx: &Context) {
    for dims in [[1, 1, 1, 1], [1, 4, 1, 1], [4, 1, 1, 1], [5, 5, 1, 1], [6, 3, 1, 1], [3, 6, 1, 1], [4, 4, 2, 3], [33, 33, 1, 1], [128, 128, 1, 1]] {
        let id = Array::<T>::identity_on(ctx, dims).unwrap();
        assert_eq!(id.shape().dims(), dims);
        let want: Vec<T> = (0..len(dims))
            .map(|o| {
                let c = coords(dims, o);
                if c[0] == c[1] { T::one() } else { T::zero() }
            })
            .collect();
        assert_same(&id.to_vec().unwrap(), &want, &format!("identity {dims:?}"));
    }
}

#[test]
fn identity_matches_the_host() {
    let ctx = Context::new().unwrap();
    identity_test::<f32>(&ctx);
    identity_test::<i32>(&ctx);
    identity_test::<u32>(&ctx);
    identity_test::<bool>(&ctx);
    identity_test::<Df32>(&ctx);
    if has_f64(&ctx) {
        identity_test::<f64>(&ctx);
    }
}

#[test]
fn identity_on_the_default_context_and_rejects_empty_shapes() {
    let id = Array::<f32>::identity(3).unwrap();
    assert_eq!(id.to_vec().unwrap(), [1.0, 0.0, 0.0]);
    let id = Array::<f32>::identity([3, 3]).unwrap();
    assert_eq!(id.to_vec().unwrap(), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    assert!(Array::<f32>::identity([0, 3]).is_err());
}

fn triangle_test<T: Elem>(ctx: &Context) {
    for dims in [[1, 1, 1, 1], [1, 5, 1, 1], [5, 1, 1, 1], [5, 5, 1, 1], [6, 3, 1, 1], [3, 6, 1, 1], [4, 5, 2, 3], [65, 33, 1, 1]] {
        let host = data::<T>(dims);
        let a = Array::from_slice_on(ctx, &host, dims).unwrap();
        for unit in [false, true] {
            let (lo, up) = (a.lower(unit), a.upper(unit));
            let reference = |lower: bool| -> Vec<T> {
                (0..len(dims))
                    .map(|o| {
                        let c = coords(dims, o);
                        let (i, j) = (c[0], c[1]);
                        if i == j && unit {
                            T::one()
                        } else if (lower && j > i) || (!lower && j < i) {
                            T::zero()
                        } else {
                            host[o]
                        }
                    })
                    .collect()
            };
            assert_same(&lo.to_vec().unwrap(), &reference(true), &format!("lower({unit}) {dims:?}"));
            assert_same(&up.to_vec().unwrap(), &reference(false), &format!("upper({unit}) {dims:?}"));
        }
    }
}

#[test]
fn lower_and_upper_match_the_host() {
    let ctx = Context::new().unwrap();
    triangle_test::<f32>(&ctx);
    triangle_test::<i32>(&ctx);
    triangle_test::<u32>(&ctx);
    triangle_test::<bool>(&ctx);
    triangle_test::<Df32>(&ctx);
    if has_f64(&ctx) {
        triangle_test::<f64>(&ctx);
    }
}

#[test]
fn triangles_zero_nan_and_fuse() {
    let ctx = Context::new().unwrap();
    let n = f32::NAN;
    let a = Array::from_slice_on(&ctx, &[n, n, n, 1.0, n, n, 2.0, 3.0, n], [3, 3]).unwrap();
    // Column-major: element (i, j) is data[i + 3 j]; NaN above the diagonal is zeroed, NaN on it stays.
    assert_same(&a.lower(false).to_vec().unwrap(), &[n, n, n, 0.0, n, n, 0.0, 0.0, n], "lower");
    assert_same(&a.lower(true).to_vec().unwrap(), &[1.0, n, n, 0.0, 1.0, n, 0.0, 0.0, 1.0], "lower unit");
    assert_same(&a.upper(false).to_vec().unwrap(), &[n, 0.0, 0.0, 1.0, n, 0.0, 2.0, 3.0, n], "upper");
    // `lower + upper` with unit diagonals splits the matrix; a lazy input fuses.
    let x = Array::from_slice_on(&ctx, &[1.0f32, 2.0, 3.0, 4.0], [2, 2]).unwrap();
    let e = (&x * 2.0).lower(false) + (&x * 2.0).upper(true) - Array::<f32>::identity_on(&ctx, [2, 2]).unwrap();
    assert_eq!(e.to_vec().unwrap(), [2.0, 4.0, 6.0, 8.0]);
}

fn iota_test<T: Elem + Numeric>(ctx: &Context, make: impl Fn(&Context, [usize; 4], [usize; 4]) -> Array<T>, from: impl Fn(usize) -> T) {
    for (dims, tiles) in [
        ([1, 1, 1, 1], [1, 1, 1, 1]),
        ([1, 1, 1, 1], [4, 1, 1, 1]),
        ([9, 1, 1, 1], [1, 1, 1, 1]),
        ([9, 1, 1, 1], [3, 1, 1, 1]),
        ([3, 4, 1, 1], [1, 1, 1, 1]),
        ([3, 4, 1, 1], [2, 3, 1, 1]),
        ([3, 1, 4, 1], [1, 5, 1, 2]),
        ([2, 3, 4, 5], [2, 2, 2, 2]),
        ([7, 5, 1, 1], [1, 1, 2, 1]),
    ] {
        let got = make(ctx, dims, tiles);
        let out: [usize; 4] = std::array::from_fn(|k| dims[k] * tiles[k]);
        assert_eq!(got.shape().dims(), out);
        let want: Vec<T> = (0..len(out))
            .map(|o| {
                let c = coords(out, o);
                from(flat(dims, std::array::from_fn(|k| c[k] % dims[k])))
            })
            .collect();
        assert_same(&got.to_vec().unwrap(), &want, &format!("iota {dims:?} x {tiles:?}"));
    }
}

#[test]
fn iota_matches_the_host() {
    let ctx = Context::new().unwrap();
    iota_test::<u32>(&ctx, |c, d, t| Array::iota_on(c, d, t).unwrap(), |i| i as u32);
    iota_test::<i32>(&ctx, |c, d, t| Array::iota_on(c, d, t).unwrap(), |i| i as i32);
    iota_test::<f32>(&ctx, |c, d, t| Array::iota_on(c, d, t).unwrap(), |i| i as f32);
    iota_test::<Df32>(&ctx, |c, d, t| Array::iota_on(c, d, t).unwrap(), |i| Df32::from(i as f64));
    if has_f64(&ctx) {
        iota_test::<f64>(&ctx, |c, d, t| Array::iota_on(c, d, t).unwrap(), |i| i as f64);
    }
}

#[test]
fn iota_counts_past_f32_precision_exactly_where_promised() {
    let ctx = Context::new().unwrap();
    // 2^24 + 3 elements: u32 and Df32 count exactly (f64 too, where the device has it).
    let n = (1usize << 24) + 3;
    let last = Array::<u32>::iota_on(&ctx, n, 1).unwrap().slice(0, n - 4..n).to_vec().unwrap();
    assert_eq!(last, [(n - 4) as u32, (n - 3) as u32, (n - 2) as u32, (n - 1) as u32]);
    let last = Array::<Df32>::iota_on(&ctx, n, 1).unwrap().slice(0, n - 4..n).to_vec().unwrap();
    assert_eq!(last.iter().map(|d| d.to_f64()).collect::<Vec<_>>(), [(n - 4) as f64, (n - 3) as f64, (n - 2) as f64, (n - 1) as f64]);
    let last = Array::<u32>::iota_on(&ctx, [n, 1], [1, 2]).unwrap().slice(0, n - 2..n).slice(1, 1..2).to_vec().unwrap();
    assert_eq!(last, [(n - 2) as u32, (n - 1) as u32]);
    if has_f64(&ctx) {
        let last = Array::<f64>::iota_on(&ctx, n, 1).unwrap().slice(0, n - 4..n).to_vec().unwrap();
        assert_eq!(last, [(n - 4) as f64, (n - 3) as f64, (n - 2) as f64, (n - 1) as f64]);
    }
}

#[test]
fn iota_rejects_empty_shapes_and_fuses() {
    assert!(Array::<u32>::iota([0, 2], 1).is_err());
    assert!(Array::<u32>::iota(2, [0, 2]).is_err());
    let ctx = Context::new().unwrap();
    let x = Array::<f32>::iota_on(&ctx, [4], [2]).unwrap();
    assert_eq!((&x * 2.0).to_vec().unwrap(), [0.0, 2.0, 4.0, 6.0, 0.0, 2.0, 4.0, 6.0]);
}

fn replace_test<T: Elem>(ctx: &Context) {
    for dims in [[1, 1, 1, 1], [9, 1, 1, 1], [5, 4, 1, 1], [3, 4, 2, 2], [64, 33, 1, 1]] {
        let a_host = data::<T>(dims);
        let b_host: Vec<T> = (0..len(dims)).map(|i| T::make(i * 5 + 11)).collect();
        let c_host: Vec<bool> = (0..len(dims)).map(|i| (i * 13 + i / 5) % 4 < 2).collect();
        let a = Array::from_slice_on(ctx, &a_host, dims).unwrap();
        let b = Array::from_slice_on(ctx, &b_host, dims).unwrap();
        let c = Array::from_slice_on(ctx, &c_host, dims).unwrap();
        let want: Vec<T> = (0..len(dims)).map(|i| if c_host[i] { a_host[i] } else { b_host[i] }).collect();
        assert_same(&a.replace(&c, &b).to_vec().unwrap(), &want, &format!("replace {dims:?}"));
        // `a` is unchanged.
        assert_same(&a.to_vec().unwrap(), &a_host, "replace leaves the source alone");
    }
}

#[test]
fn replace_matches_the_host() {
    let ctx = Context::new().unwrap();
    replace_test::<f32>(&ctx);
    replace_test::<i32>(&ctx);
    replace_test::<u32>(&ctx);
    replace_test::<bool>(&ctx);
    replace_test::<Df32>(&ctx);
    if has_f64(&ctx) {
        replace_test::<f64>(&ctx);
    }
}

#[test]
fn replace_with_scalars_conditions_and_nan() {
    let ctx = Context::new().unwrap();
    let a = Array::from_slice_on(&ctx, &[1.0f32, f32::NAN, 3.0, f32::INFINITY], 4).unwrap();
    // The usual use: replace the NaNs.
    let fixed = a.replace(&!a.is_nan(), 0.0);
    assert_eq!(fixed.to_vec().unwrap(), [1.0, 0.0, 3.0, f32::INFINITY]);
    // A word condition (non-zero is true), as `select` takes.
    let cond = Array::from_slice_on(&ctx, &[1u32, 0, 5, 0], 4).unwrap();
    assert_same(&a.replace(&cond, -1.0).to_vec().unwrap(), &[1.0, -1.0, 3.0, -1.0], "word condition");
    let i = Array::from_slice_on(&ctx, &[1i32, 2, 3, 4], 4).unwrap();
    assert_eq!(i.replace(&i.gt(2), 0).to_vec().unwrap(), [1, 2, 3, 4].map(|v| if v > 2 { v } else { 0 }));
    // A broadcast branch.
    let row = Array::from_slice_on(&ctx, &[10.0f32, 20.0], [1, 2]).unwrap();
    let m = Array::from_slice_on(&ctx, &[1.0f32, 2.0, 3.0, 4.0], [2, 2]).unwrap();
    assert_eq!(m.replace(&m.gt(2.0), &row).to_vec().unwrap(), [10.0, 10.0, 3.0, 4.0]);
    // Lazy operands fuse.
    assert_eq!((&m * 2.0).replace(&m.gt(2.0), 0.0).to_vec().unwrap(), [0.0, 0.0, 6.0, 8.0]);
}

#[test]
fn replace_checks_shapes_and_contexts() {
    let ctx = Context::new().unwrap();
    let a = Array::from_slice_on(&ctx, &[1.0f32, 2.0, 3.0, 4.0], 4).unwrap();
    let c = Array::from_slice_on(&ctx, &[true, false], 2).unwrap();
    assert!(matches!(a.replace(&c, 0.0).check(), Err(LumaError::Op { .. })));
    let b = Array::from_slice_on(&ctx, &[1.0f32, 2.0], 2).unwrap();
    assert!(a.replace(&a.gt(1.0), &b).check().is_err());
}

fn diff_test<T: Elem + Numeric>(ctx: &Context, sub: impl Fn(T, T) -> T) {
    for dims in [[2, 1, 1, 1], [3, 1, 1, 1], [10, 1, 1, 1], [257, 1, 1, 1], [5, 4, 1, 1], [3, 4, 2, 3], [33, 17, 1, 1]] {
        let host = data::<T>(dims);
        let a = Array::from_slice_on(ctx, &host, dims).unwrap();
        for axis in 0..4 {
            for order in [1usize, 2] {
                let r = if order == 1 { a.diff1(axis) } else { a.diff2(axis) };
                if dims[axis] <= order {
                    assert!(r.check().is_err(), "diff{order} axis {axis} of {dims:?} must fail");
                    continue;
                }
                let mut out = dims;
                out[axis] -= order;
                assert_eq!(r.shape().dims(), out);
                let want: Vec<T> = (0..len(out))
                    .map(|o| {
                        let c = coords(out, o);
                        let at = |d: usize| {
                            let mut p = c;
                            p[axis] += d;
                            host[flat(dims, p)]
                        };
                        if order == 1 { sub(at(1), at(0)) } else { sub(sub(at(2), at(1)), sub(at(1), at(0))) }
                    })
                    .collect();
                assert_same(&r.to_vec().unwrap(), &want, &format!("diff{order} axis {axis} of {dims:?}"));
            }
        }
    }
}

#[test]
fn diff1_and_diff2_match_the_host() {
    let ctx = Context::new().unwrap();
    diff_test::<f32>(&ctx, |a, b| a - b);
    diff_test::<i32>(&ctx, i32::wrapping_sub);
    diff_test::<u32>(&ctx, u32::wrapping_sub);
    diff_test::<Df32>(&ctx, |a, b| Df32::from(a.to_f64() - b.to_f64()));
    if has_f64(&ctx) {
        diff_test::<f64>(&ctx, |a, b| a - b);
    }
}

#[test]
fn diff_of_special_values_and_expressions() {
    let ctx = Context::new().unwrap();
    let a = Array::from_slice_on(&ctx, &[1.0f32, f32::INFINITY, f32::INFINITY, f32::NAN, 2.0], 5).unwrap();
    assert_same(&a.diff1(0).to_vec().unwrap(), &[f32::INFINITY, f32::NAN, f32::NAN, f32::NAN], "diff1");
    assert_same(&a.diff2(0).to_vec().unwrap(), &[f32::NAN, f32::NAN, f32::NAN], "diff2");
    // diff1(diff1) is diff2; diff of a lazy expression.
    let x = Array::from_slice_on(&ctx, &[0.0f32, 1.0, 4.0, 9.0, 16.0, 25.0], 6).unwrap();
    assert_eq!(x.diff2(0).to_vec().unwrap(), [2.0, 2.0, 2.0, 2.0]);
    assert_eq!(x.diff1(0).diff1(0).to_vec().unwrap(), [2.0, 2.0, 2.0, 2.0]);
    assert_eq!((&x * 2.0).diff1(0).to_vec().unwrap(), [2.0, 6.0, 10.0, 14.0, 18.0]);
    // Unsigned differences wrap.
    let u = Array::from_slice_on(&ctx, &[3u32, 1, 2], 3).unwrap();
    assert_eq!(u.diff1(0).to_vec().unwrap(), [u32::MAX - 1, 1]);
}

#[test]
fn diff_and_structure_errors_poison_the_result() {
    let ctx = Context::new().unwrap();
    let a = Array::from_slice_on(&ctx, &[1.0f32, 2.0, 3.0, 4.0], [2, 2]).unwrap();
    assert!(a.diff1(4).check().is_err());
    assert!(a.diff2(0).check().is_err());
    assert!(a.diff1(2).check().is_err());
    assert!(a.diag(2).check().is_err());
    assert!(a.diag(-2).check().is_err());
    assert!(a.diag_matrix(0).check().is_err());
    assert!(a.tile([0, 1]).check().is_err());
    // A poisoned input poisons with its own error.
    let bad = a.diff1(4);
    assert!(bad.diag(0).check().is_err());
    assert!(bad.tile(2).check().is_err());
    assert!(bad.lower(false).check().is_err());
    assert!(bad.replace(&a.gt(1.0), 0.0).check().is_err());
}
