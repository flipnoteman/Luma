//! Matrix properties (W3-R6): what each exact construction records, how the few operations that keep a pattern carry it, and that
//! nothing else, and no write through a handle, leaves a stale property behind. Read through the doc-hidden
//! `properties_debug`; the flags are in the convention of the dense routines (the row is dimension 1, see `src/props.rs`).

use luma::Array;

fn mat(n: usize) -> Array<f32> {
    let data: Vec<f32> = (0..n * n).map(|i| 1.0 + (i % 7) as f32).collect();
    Array::from_slice(&data, [n, n]).unwrap()
}

fn col(n: usize) -> Array<f32> {
    let data: Vec<f32> = (0..n).map(|i| 1.0 + i as f32).collect();
    Array::from_slice(&data, [n, 1]).unwrap()
}

/// The properties as a set of names (the band separated).
fn has(a: &Array<f32>, names: &[&str]) {
    let text = a.properties_debug();
    for name in names {
        assert!(text.split('|').any(|part| part == *name), "{name} missing from {text}");
    }
}

fn lacks(a: &Array<f32>, names: &[&str]) {
    let text = a.properties_debug();
    for name in names {
        assert!(!text.split('|').any(|part| part == *name), "{name} present in {text}");
    }
}

fn none(a: &Array<f32>) {
    assert_eq!(a.properties_debug(), "NONE");
}

#[test]
fn identity_and_diagonal_matrices_record_their_pattern() {
    let i = Array::<f32>::identity([4, 4]).unwrap();
    has(&i, &["IDENTITY", "DIAGONAL", "SYMMETRIC", "LOWER", "UPPER", "PERMUTATION", "BANDED", "band=(0,0)"]);
    let wide = Array::<f32>::identity([5, 3]).unwrap();
    has(&wide, &["DIAGONAL", "LOWER", "UPPER"]);
    lacks(&wide, &["IDENTITY", "SYMMETRIC", "PERMUTATION"]);
    let d = col(4).diag_matrix(0);
    has(&d, &["DIAGONAL", "SYMMETRIC", "LOWER", "UPPER", "band=(0,0)"]);
    lacks(&d, &["IDENTITY"]);
}

#[test]
fn offset_diagonals_are_banded_on_the_side_of_the_dense_routines() {
    // Element i of the column lands at (dim 0, dim 1) = (i, i + k): below the diagonal of a row-major matrix for k > 0.
    let below = col(4).diag_matrix(1);
    has(&below, &["LOWER", "BANDED", "band=(1,0)"]);
    lacks(&below, &["UPPER", "DIAGONAL"]);
    let above = col(4).diag_matrix(-2);
    has(&above, &["UPPER", "band=(0,2)"]);
    lacks(&above, &["LOWER"]);
    // The pattern is the one that is stored: the claim is that everything else is zero.
    let v = below.to_vec().unwrap();
    let n = 5;
    for r in 0..n {
        for c in 0..n {
            if c > r {
                assert_eq!(v[r * n + c], 0.0, "entry above the diagonal of ({r}, {c})");
            }
        }
    }
}

#[test]
fn triangles_record_the_triangle_they_keep() {
    // `lower` keeps `j <= i` with `i` along dimension 0: an upper triangle of the row-major matrix.
    let l = mat(5).lower(false);
    has(&l, &["UPPER", "band=(0,4)"]);
    lacks(&l, &["LOWER", "UNIT_DIAG"]);
    let u = mat(5).upper(true);
    has(&u, &["LOWER", "UNIT_DIAG", "band=(4,0)"]);
    let v = u.to_vec().unwrap();
    for r in 0..5 {
        for c in 0..5 {
            if c > r {
                assert_eq!(v[r * 5 + c], 0.0);
            }
            if c == r {
                assert_eq!(v[r * 5 + c], 1.0);
            }
        }
    }
}

#[test]
fn factor_outputs_are_triangles_and_the_permutation_a_permutation() {
    let a = mat(6) + Array::<f32>::identity([6, 6]).unwrap() * 20.0;
    let lu = a.lu();
    has(&lu.lower(), &["LOWER", "UNIT_DIAG"]);
    has(&lu.upper(), &["UPPER"]);
    lacks(&lu.upper(), &["UNIT_DIAG"]);
    has(&lu.permutation_matrix(), &["PERMUTATION"]);
    lacks(&lu.permutation_matrix(), &["IDENTITY", "DIAGONAL"]);
    has(&a.qr().r(), &["UPPER"]);
    none(&a.qr().q());
    none(&a);
    // P A = L U
    let pa = lu.permutation_matrix().matmul_f32(&a).to_vec().unwrap();
    let l_u = lu.lower().matmul_f32(&lu.upper()).to_vec().unwrap();
    for (x, y) in pa.iter().zip(&l_u) {
        assert!((x - y).abs() < 1e-3 * (1.0 + x.abs()), "{x} {y}");
    }
}

#[test]
fn transposes_swap_the_triangles_and_keep_the_symmetric_facts() {
    let l = mat(5).upper(false);
    has(&l, &["LOWER"]);
    let t = l.transpose();
    has(&t, &["UPPER", "band=(0,4)"]);
    lacks(&t, &["LOWER"]);
    has(&Array::<f32>::identity([4, 4]).unwrap().transpose(), &["IDENTITY", "SYMMETRIC", "PERMUTATION"]);
    none(&mat(5).transpose());
    // A slice, a reshape and a gather keep nothing.
    none(&l.slice(0, 0..3));
    none(&l.reshape([25]));
}

#[test]
fn sums_and_scalar_multiples_keep_what_both_have() {
    let l1 = mat(5).upper(false);
    let l2 = mat(5).upper(true);
    let sum = &l1 + &l2;
    has(&sum, &["LOWER", "band=(4,0)"]);
    lacks(&sum, &["UNIT_DIAG"]);
    let mixed = &l1 + &mat(5).lower(false);
    lacks(&mixed, &["LOWER", "UPPER", "DIAGONAL"]);
    let d = &col(5).diag_matrix(0) - &col(5).diag_matrix(0);
    has(&d, &["DIAGONAL", "SYMMETRIC"]);
    // A diagonal plus a lower triangle is lower.
    has(&(&col(5).diag_matrix(0) + &l1), &["LOWER"]);
    let i = Array::<f32>::identity([4, 4]).unwrap();
    let scaled = &i * 3.0;
    has(&scaled, &["DIAGONAL", "SYMMETRIC"]);
    lacks(&scaled, &["IDENTITY", "PERMUTATION", "UNIT_DIAG"]);
    has(&(-&l2), &["LOWER", "band=(4,0)"]);
    lacks(&(-&l2), &["UNIT_DIAG"]);
    none(&(&l1 + 1.0));
}

#[test]
fn a_matrix_added_to_or_multiplied_by_its_transpose_is_symmetric() {
    let a = mat(6).cache();
    has(&(&a + &a.transpose()), &["SYMMETRIC"]);
    has(&(&a.transpose() + &a), &["SYMMETRIC"]);
    has(&(&a * &a.transpose()), &["SYMMETRIC"]);
    lacks(&(&a - &a.transpose()), &["SYMMETRIC"]);
    let b = mat(6).cache();
    lacks(&(&a + &b.transpose()), &["SYMMETRIC"]);
    let s = (&a + &a.transpose()).to_vec().unwrap();
    for r in 0..6 {
        for c in 0..6 {
            assert_eq!(s[r * 6 + c].to_bits(), s[c * 6 + r].to_bits());
        }
    }
}

/// Every claim of `a` holds for its values (the zero region of the triangles, the band, the symmetry).
fn sound(a: &Array<f32>) {
    let text = a.properties_debug();
    let [cols, rows, ..] = a.shape().dims();
    let v = a.to_vec().unwrap();
    let at = |r: usize, c: usize| v[r * cols + c];
    let flag = |name: &str| text.split('|').any(|part| part == name);
    let band = text.split('|').find_map(|part| part.strip_prefix("band=(")).map(|rest| {
        let (lo, hi) = rest.trim_end_matches(')').split_once(',').unwrap();
        (lo.parse::<usize>().unwrap(), hi.parse::<usize>().unwrap())
    });
    for r in 0..rows {
        for c in 0..cols {
            if flag("LOWER") {
                assert!(c <= r || at(r, c) == 0.0, "{text}: ({r}, {c}) above the diagonal");
            }
            if flag("UPPER") {
                assert!(c >= r || at(r, c) == 0.0, "{text}: ({r}, {c}) below the diagonal");
            }
            if let Some((lo, hi)) = band {
                assert!(r <= c + lo && c <= r + hi || at(r, c) == 0.0, "{text}: ({r}, {c}) outside the band");
            }
            if flag("SYMMETRIC") {
                assert_eq!(at(r, c).to_bits(), at(c, r).to_bits(), "{text}: ({r}, {c}) not symmetric");
            }
            if flag("UNIT_DIAG") && r == c {
                assert_eq!(at(r, c), 1.0, "{text}");
            }
        }
    }
    if flag("IDENTITY") {
        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(at(r, c), if r == c { 1.0 } else { 0.0 }, "{text}");
            }
        }
    }
}

#[test]
fn products_never_claim_what_is_not_true() {
    // A product is evaluated at the call unless its site is lazy (`LUMA_LAZY`), and an evaluated array keeps no property, so
    // what a product carries is checked against its values, whichever way it was made (`Props::product` has its own unit tests).
    let n = 6;
    let lo = mat(n).upper(false);
    let lo2 = mat(n).upper(true);
    let up = mat(n).lower(false);
    let d = col(n).diag_matrix(0);
    let i = Array::<f32>::identity([n, n]).unwrap();
    let b1 = col(n).diag_matrix(1);
    let products = [
        lo.matmul_f32(&lo2),
        up.matmul_f32(&up),
        lo.matmul_f32(&up),
        lo2.matmul_f32(&lo2),
        d.matmul_f32(&lo),
        lo.matmul_f32(&d),
        d.matmul_f32(&d),
        i.matmul_f32(&lo),
        i.matmul_f32(&i),
        b1.matmul_f32(&b1),
        mat(n).matmul_f32(&mat(n)),
    ];
    for product in &products {
        sound(product);
    }
    for array in [&lo, &lo2, &up, &d, &i, &b1, &lo.transpose(), &(&lo + &lo2), &(&lo * 0.5), &(-&lo2)] {
        sound(array);
    }
    none(&mat(n).matmul_f32(&mat(n)));
}

#[test]
fn nothing_else_keeps_a_property() {
    let l = mat(5).upper(false);
    none(&l.reshape([25]));
    none(&l.slice(1, 0..2));
    none(&l.flip(0));
    none(&l.copy());
    none(&l.abs());
    none(&(&l * &l));
    none(&l.sum());
    none(&l.take(&Array::from_slice(&[0u32, 1, 2], 3).unwrap()));
}

#[test]
fn a_write_through_a_handle_forgets_its_properties() {
    let mut l = mat(4).upper(false);
    has(&l, &["LOWER"]);
    l.write(&[1.0; 16]).unwrap();
    none(&l);

    let mut m = mat(4).upper(false);
    m.slice_mut(0, 3..4).assign(7.0).unwrap();
    none(&m);

    let mut t = mat(4).upper(false);
    let _ = t.transpose_mut();
    none(&t);

    let mut s = mat(4).upper(false);
    s += 1.0;
    none(&s);

    let mut p = mat(4).upper(false);
    p.assign(2.0).unwrap();
    none(&p);

    let mut q = mat(4).upper(false);
    q.put(&Array::from_slice(&[1u32], 1).unwrap(), 5.0).unwrap();
    none(&q);
}

#[test]
fn a_clone_written_through_does_not_leave_a_stale_property_on_the_original() {
    let original = mat(4).upper(false);
    has(&original, &["LOWER"]);
    let mut copy = original.clone();
    copy.write(&[3.0; 16]).unwrap();
    none(&copy);
    none(&original);
    let seen = original.to_vec().unwrap();
    assert!(seen.iter().all(|&x| x == 3.0), "the clone shares the data");

    // The same for an evaluated array.
    let mut valid = mat(4).upper(false).cache();
    let twin = valid.clone();
    valid.write(&[4.0; 16]).unwrap();
    none(&valid);
    none(&twin);
}

#[test]
fn valid_results_carry_properties_until_they_are_shared_or_written() {
    let n = 6;
    let up = mat(n).lower(false);
    // The product is evaluated (or not) as the site says; either way it carries the product rule.
    let product = up.matmul_f32(&up);
    has(&product, &["UPPER"]);
    // Evaluating it keeps them (the handle keeps its representation, or a fresh valid result carries them); `cache` is a share.
    product.to_vec().unwrap();
    has(&product, &["UPPER"]);
    none(&product.cache());
    // `cholesky` of a symmetric positive definite matrix is `LOWER` whatever the representation.
    let spd = {
        let a = mat(n);
        &a.matmul_nt(&a) + &(&Array::<f32>::identity([n, n]).unwrap() * (n as f32))
    };
    let l = spd.cholesky();
    has(&l, &["LOWER"]);
    sound(&l);
    l.to_vec().unwrap();
    has(&l, &["LOWER"]);
    // Sharing forgets, and counts what it forgot.
    let before = Array::<f32>::lost_properties_debug();
    let copy = l.clone();
    none(&copy);
    none(&l);
    if cfg!(debug_assertions) {
        assert_eq!(Array::<f32>::lost_properties_debug(), before + 1);
    }
    let mut written = mat(n).matmul_f32(&mat(n));
    written.to_vec().unwrap();
    written.write(&[1.0; 36]).unwrap();
    none(&written);
}

#[test]
fn a_gram_product_is_symmetric() {
    let x = Array::from_slice(&(0..24).map(|i| (i % 5) as f32 - 1.5).collect::<Vec<_>>(), [4, 6]).unwrap();
    let gram = x.matmul_tn(&x);
    has(&gram, &["SYMMETRIC"]);
    has(&x.matmul_nt(&x), &["SYMMETRIC"]);
    none(&x.matmul_tn(&x.clone().abs()));
    sound_within(&gram);
}

/// Symmetric to a few ulps (the full kernel's two triangles may differ in the last bits on tensor cores).
fn sound_within(a: &Array<f32>) {
    let [n, ..] = a.shape().dims();
    let v = a.to_vec().unwrap();
    for r in 0..n {
        for c in 0..r {
            let (x, y) = (v[r * n + c], v[c * n + r]);
            assert!((x - y).abs() <= 8.0 * f32::EPSILON * x.abs().max(y.abs()), "({r}, {c}): {x} vs {y}");
        }
    }
}

// -- Routing --

use luma::{solve_tridiagonal, Context, ContextOptions, Numerics, Triangle};

fn ctx_with(numerics: Numerics) -> Context {
    Context::with_options(ContextOptions::from_env().numerics(numerics)).unwrap()
}

fn on(ctx: &Context, data: &[f32], dims: impl Into<luma::Shape>) -> Array<f32> {
    Array::from_slice_on(ctx, data, dims).unwrap()
}

/// The bits with the sign of a zero ignored (the properties are about finite values; a `-0` is outside their claims).
fn unsigned_bits(a: &Array<f32>) -> Vec<u32> {
    a.to_vec().unwrap().iter().map(|x| (x + 0.0).to_bits()).collect()
}

fn bits(a: &Array<f32>) -> Vec<u32> {
    a.to_vec().unwrap().iter().map(|x| x.to_bits()).collect()
}

fn dominant_tridiagonal(ctx: &Context, n: usize) -> (Array<f32>, [Array<f32>; 3]) {
    let main: Vec<f32> = (0..n).map(|i| 4.0 + (i % 3) as f32).collect();
    let sub: Vec<f32> = (0..n - 1).map(|i| 1.0 - (i % 4) as f32 * 0.25).collect();
    let sup: Vec<f32> = (0..n - 1).map(|i| 0.5 + (i % 5) as f32 * 0.125).collect();
    let (main, sub, sup) = (on(ctx, &main, [n, 1]), on(ctx, &sub, [n - 1, 1]), on(ctx, &sup, [n - 1, 1]));
    // `diag(1)` is the entry below the diagonal of a row, `diag(-1)` the one above it (see `crate::props`).
    let a = Array::from_diagonals(&[(0, &main), (1, &sub), (-1, &sup)]);
    (a, [main, sub, sup])
}

#[test]
fn from_diagonals_builds_the_band_it_says() {
    let ctx = ctx_with(Numerics::strict());
    let (a, _) = dominant_tridiagonal(&ctx, 7);
    has(&a, &["BANDED", "band=(1,1)"]);
    lacks(&a, &["LOWER", "UPPER", "DIAGONAL"]);
    let v = a.to_vec().unwrap();
    for r in 0usize..7 {
        for c in 0..7 {
            if r.abs_diff(c) > 1 {
                assert_eq!(v[r * 7 + c], 0.0);
            }
        }
    }
    let wrong = Array::from_diagonals(&[(0, &on(&ctx, &[1.0; 4], [4, 1])), (1, &on(&ctx, &[1.0; 4], [4, 1]))]);
    assert!(wrong.to_vec().is_err(), "sizes 4 and 5 do not fit one matrix");
}

#[test]
fn solve_banded_runs_the_tridiagonal_solver_and_agrees_with_the_dense_solve() {
    for numerics in [Numerics::strict(), Numerics::default(), Numerics::algebraic()] {
        let ctx = ctx_with(numerics);
        for (n, k) in [(9usize, 1usize), (64, 1), (64, 5)] {
            let (a, [main, sub, sup]) = dominant_tridiagonal(&ctx, n);
            let data: Vec<f32> = (0..n * k).map(|i| ((i * 7 % 11) as f32) - 5.0).collect();
            let b = on(&ctx, &data, [k, n]);
            let x = a.solve_banded(&b);
            // The line solver called directly on the same diagonals.
            let pad = |v: &Array<f32>, front: bool| {
                let mut host = v.to_vec().unwrap();
                if front { host.insert(0, 0.0) } else { host.push(0.0) }
                on(&ctx, &host, [1, n]).broadcast_to([k, n])
            };
            let (lo, up) = (pad(&sub, true), pad(&sup, false));
            let diag = on(&ctx, &main.to_vec().unwrap(), [1, n]).broadcast_to([k, n]);
            let direct = solve_tridiagonal(1, &lo, &diag, &up, &b);
            assert_eq!(bits(&x), bits(&direct), "n = {n}, k = {k}");
            // The dense solve, on the evaluated matrix (which carries no property): `solve` never routes a band.
            let dense = a.cache().solve(&b);
            let (xs, ds) = (x.to_vec().unwrap(), dense.to_vec().unwrap());
            for (x, d) in xs.iter().zip(&ds) {
                assert!((x - d).abs() <= 1e-3 * (1.0 + d.abs()), "n = {n}: {x} vs {d}");
            }
            assert_eq!(bits(&a.solve(&b)), bits(&a.cache().solve(&b)), "solve on a band is the dense path, whatever the numerics");
        }
    }
}

#[test]
fn solve_banded_without_a_band_is_an_error_and_a_wide_band_is_the_dense_solve() {
    let ctx = ctx_with(Numerics::strict());
    let data: Vec<f32> = (0..16).map(|i| if i % 5 == 0 { 8.0 } else { 1.0 }).collect();
    let a = on(&ctx, &data, [4, 4]);
    let b = on(&ctx, &[1.0, 2.0, 3.0, 4.0], 4);
    assert!(a.solve_banded(&b).to_vec().is_err());
    let main = on(&ctx, &[9.0; 6], [6, 1]);
    let one = on(&ctx, &[1.0; 5], [5, 1]);
    let two = on(&ctx, &[0.5; 4], [4, 1]);
    let wide = Array::from_diagonals(&[(0, &main), (1, &one), (-1, &one), (2, &two), (-2, &two)]);
    has(&wide, &["band=(2,2)"]);
    let rhs = on(&ctx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 6);
    assert_eq!(bits(&wide.solve_banded(&rhs)), bits(&wide.cache().solve(&rhs)));
}

#[test]
fn triangular_solves_route_under_reformulate_and_diagonal_ones_under_exact_finite() {
    let n = 9;
    let off = ctx_with(Numerics::default());
    let on_ = ctx_with(Numerics::algebraic());
    let raw: Vec<f32> = (0..n * n).map(|i| if i % (n + 1) == 0 { 6.0 } else { 0.3 + (i % 5) as f32 * 0.1 }).collect();
    let rhs: Vec<f32> = (0..n * 3).map(|i| (i % 7) as f32 - 3.0).collect();
    for (name, triangle) in [("lower", Triangle::Lower), ("upper", Triangle::Upper)] {
        for ctx in [&off, &on_] {
            let full = on(ctx, &raw, [n, n]);
            let b = on(ctx, &rhs, [3, n]);
            // `Array::lower` keeps the entries of an upper triangle of the dense routines (see `crate::props`).
            let a = if name == "lower" { full.upper(false) } else { full.lower(false) };
            let routed = a.solve(&b);
            let dense = a.cache().solve(&b);
            if ctx.numerics().reformulate {
                let reference = a.cache().solve_triangular(&b, triangle, false, false);
                assert_eq!(bits(&routed), bits(&reference), "{name}: routed to solve_triangular");
            } else {
                assert_eq!(bits(&routed), bits(&dense), "{name}: the dense path without reformulate");
            }
            let (x, d) = (routed.to_vec().unwrap(), dense.to_vec().unwrap());
            for (x, d) in x.iter().zip(&d) {
                assert!((x - d).abs() <= 1e-3 * (1.0 + d.abs()), "{name}: {x} vs {d}");
            }
        }
    }
    // Diagonal.
    let v: Vec<f32> = (0..n).map(|i| 1.0 + i as f32).collect();
    for ctx in [&off, &on_] {
        let a = on(ctx, &v, [n, 1]).diag_matrix(0);
        let b = on(ctx, &rhs, [3, n]);
        let x = a.solve(&b).to_vec().unwrap();
        for r in 0..n {
            for c in 0..3 {
                let want = rhs[r * 3 + c] / v[r];
                assert!((x[r * 3 + c] - want).abs() <= 1e-5 * (1.0 + want.abs()), "diagonal solve ({r}, {c})");
            }
        }
    }
    // The diagonal solve is exact for finite inputs, so it runs under `exact_finite` alone and not under `reformulate` alone.
    let finite = ctx_with(Numerics::parse("default+exact-finite").unwrap());
    let reformulating = ctx_with(Numerics::parse("default+reformulate").unwrap());
    // On the device the same division `b / d`, as an expression (the host's quotient may differ in the last bit from the GPU's).
    let (b_dev, d_dev) = (on(&finite, &rhs, [3, n]), on(&finite, &v, [1, n]).broadcast_to([3, n]));
    let routed = on(&finite, &v, [n, 1]).diag_matrix(0).solve(&b_dev);
    assert_eq!(bits(&routed), bits(&(&b_dev / &d_dev)), "exact_finite: b / d");
    let (a, b) = (on(&reformulating, &v, [n, 1]).diag_matrix(0), on(&reformulating, &rhs, [3, n]));
    assert_eq!(bits(&a.solve(&b)), bits(&a.cache().solve(&b)), "reformulate alone: the dense path");
    // A vector right-hand side.
    let a = on(&on_, &v, [n, 1]).diag_matrix(0);
    let b = on(&on_, &v, n);
    let x = a.solve(&b);
    assert_eq!(x.shape().dims(), [n, 1, 1, 1]);
    assert!(x.to_vec().unwrap().iter().all(|&x| (x - 1.0).abs() < 1e-6));
}

#[test]
fn the_determinant_of_a_triangle_is_the_product_of_its_diagonal() {
    let n = 6;
    let ctx = ctx_with(Numerics::algebraic());
    let raw: Vec<f32> = (0..n * n).map(|i| if i % (n + 1) == 0 { 1.5 } else { 0.25 }).collect();
    let a = on(&ctx, &raw, [n, n]).lower(false);
    let det = a.det().to_vec().unwrap()[0];
    assert!((det - 1.5f32.powi(n as i32)).abs() < 1e-4, "{det}");
    let reference = a.cache().det().to_vec().unwrap()[0];
    assert!((det - reference).abs() < 1e-3 * reference.abs());
}

/// A diagonally dominant matrix with the band `(lo, hi)` (`lo` sub-diagonals in the sense of the dense routines), and the same
/// matrix as a dense array without properties.
fn dominant_band(ctx: &Context, n: usize, lo: usize, hi: usize) -> (Array<f32>, Array<f32>) {
    let vector = |len: usize, salt: usize| -> Array<f32> {
        let data: Vec<f32> = (0..len).map(|i| 0.25 + ((i * 5 + salt * 3) % 7) as f32 * 0.125).collect();
        on(ctx, &data, [len, 1])
    };
    let main = on(ctx, &(0..n).map(|i| 3.0 * (lo + hi) as f32 + 2.0 + (i % 3) as f32).collect::<Vec<_>>(), [n, 1]);
    let mut diagonals: Vec<(isize, Array<f32>)> = vec![(0, main)];
    for k in 1..=lo {
        diagonals.push((k as isize, vector(n - k, k)));
    }
    for k in 1..=hi {
        diagonals.push((-(k as isize), vector(n - k, 10 + k)));
    }
    let refs: Vec<(isize, &Array<f32>)> = diagonals.iter().map(|(k, v)| (*k, v)).collect();
    let band = Array::from_diagonals(&refs);
    let dense = on(ctx, &band.to_vec().unwrap(), [n, n]);
    (band, dense)
}

#[test]
fn solve_banded_with_a_wider_band_runs_the_block_tridiagonal_solver() {
    let ctx = ctx_with(Numerics::strict());
    for (n, lo, hi, k) in [(45usize, 2usize, 2usize, 1usize), (45, 2, 1, 3), (37, 4, 4, 2), (64, 3, 5, 1), (50, 8, 8, 2)] {
        let (a, dense) = dominant_band(&ctx, n, lo, hi);
        has(&a, &["BANDED"]);
        let data: Vec<f32> = (0..n * k).map(|i| ((i * 7 % 11) as f32) - 5.0).collect();
        let b = on(&ctx, &data, [k, n]);
        let x = a.solve_banded(&b).to_vec().unwrap();
        let reference = dense.solve(&b).to_vec().unwrap();
        assert_eq!(x.len(), reference.len());
        for (x, d) in x.iter().zip(&reference) {
            assert!((x - d).abs() <= 2e-3 * (1.0 + d.abs()), "n = {n}, band ({lo}, {hi}): {x} vs {d}");
        }
    }
}

/// The permutation matrix of an LU with real pivoting, and what `P X` should be on the host.
fn permutation_case(ctx: &Context, n: usize, k: usize) -> (Array<f32>, Vec<usize>, Vec<f32>, Array<f32>) {
    let data: Vec<f32> = (0..n * n).map(|i| (((i * 31 + 7) % 13) as f32) - 6.0 + if i % (n + 1) == 0 { 0.5 } else { 0.0 }).collect();
    let lu = on(ctx, &data, [n, n]).lu();
    let perm: Vec<usize> = lu.perm.to_vec().unwrap().iter().map(|&p| p as usize).collect();
    let x: Vec<f32> = (0..n * k).map(|i| (i % 9) as f32 - 4.0).collect();
    (lu.permutation_matrix(), perm, x.clone(), on(ctx, &x, [k, n]))
}

#[test]
fn a_permutation_solves_and_multiplies_as_a_row_gather_under_exact_finite() {
    let (n, k) = (12, 5);
    for numerics in [Numerics::strict(), Numerics::algebraic()] {
        let ctx = ctx_with(numerics);
        let (p, perm, host, x) = permutation_case(&ctx, n, k);
        has(&p, &["PERMUTATION"]);
        // `P X`: row `i` is row `perm[i]` of `X`.
        let product = p.matmul(&x).to_vec().unwrap();
        // `P y = x`: `y[perm[i]] = x[i]`.
        let solved = p.solve(&x).to_vec().unwrap();
        for i in 0..n {
            for c in 0..k {
                assert_eq!(product[i * k + c], host[perm[i] * k + c], "P X at ({i}, {c})");
                assert_eq!(solved[perm[i] * k + c], host[i * k + c], "P y = x at ({i}, {c})");
            }
        }
    }
}

#[test]
fn a_banded_product_is_a_sum_of_shifts_under_exact_finite() {
    // The sum of shifts is taken where it costs less than the GEMM: `60 * width < n` rows.
    let n = 1100;
    // Integers keep every sum exact, so the fused sum and the GEMM agree bit for bit whatever their order.
    let whole = |ctx: &Context, len: usize, salt: usize| on(ctx, &(0..len).map(|i| ((i * 3 + salt) % 5) as f32 - 2.0).collect::<Vec<_>>(), [len, 1]);
    for (lo, hi) in [(0usize, 0usize), (1, 1), (1, 2), (2, 2), (8, 7), (8, 8)] {
        let ctx = ctx_with(Numerics::algebraic());
        let strict = ctx_with(Numerics::strict());
        let build = |ctx: &Context| {
            let mut diagonals = vec![(0isize, whole(ctx, n, 0))];
            for k in 1..=lo {
                diagonals.push((k as isize, whole(ctx, n - k, k)));
            }
            for k in 1..=hi {
                diagonals.push((-(k as isize), whole(ctx, n - k, 7 + k)));
            }
            let refs: Vec<(isize, &Array<f32>)> = diagonals.iter().map(|(k, v)| (*k, v)).collect();
            Array::from_diagonals(&refs)
        };
        let x: Vec<f32> = (0..n * 6).map(|i| ((i * 5) % 9) as f32 - 4.0).collect();
        let a = build(&ctx);
        has(&a, &["BANDED"]);
        ctx.record_decisions(true);
        let fused = a.matmul(&on(&ctx, &x, [6, n]));
        let fused_values = fused.to_vec().unwrap();
        assert!(ctx.take_decisions().iter().any(|d| d.chosen == "banded-shifts"), "band ({lo}, {hi}) takes the sum of shifts");
        let reference = on(&strict, &build(&strict).to_vec().unwrap(), [n, n]).matmul_f32(&on(&strict, &x, [6, n]));
        assert_eq!(fused_values.iter().map(|x| (x + 0.0).to_bits()).collect::<Vec<_>>(), unsigned_bits(&reference), "band ({lo}, {hi})");
    }
    // Too wide for its size (19 diagonals of 1100 rows): the GEMM runs and gives the same (integers).
    let ctx = ctx_with(Numerics::algebraic());
    let mut diagonals = vec![(0isize, whole(&ctx, n, 0))];
    for k in 1..=9 {
        diagonals.push((k, whole(&ctx, n - k as usize, 1)));
        diagonals.push((-k, whole(&ctx, n - k as usize, 2)));
    }
    let refs: Vec<(isize, &Array<f32>)> = diagonals.iter().map(|(k, v)| (*k, v)).collect();
    let wide = Array::from_diagonals(&refs);
    let x = on(&ctx, &(0..n * 2).map(|i| (i % 5) as f32).collect::<Vec<_>>(), [2, n]);
    let dense = on(&ctx, &wide.to_vec().unwrap(), [n, n]).matmul_f32(&x);
    assert_eq!(bits(&wide.matmul_f32(&x)), bits(&dense));
}

#[test]
fn a_banded_matrix_can_be_used_more_than_once() {
    let ctx = ctx_with(Numerics::algebraic());
    let (a, _) = dominant_tridiagonal(&ctx, 16);
    let b = on(&ctx, &[1.0; 32], [2, 16]);
    let first = a.solve_banded(&b).to_vec().unwrap();
    let second = a.solve_banded(&b).to_vec().unwrap();
    assert_eq!(first, second, "a second solve_banded on the same handle still finds its band");
    let _ = a.matmul(&on(&ctx, &[1.0; 32], [2, 16])).to_vec().unwrap();
    assert!(a.solve_banded(&b).to_vec().is_ok());
}

/// `solve_banded` reads the diagonals back from the vectors `from_diagonals` was given: a write to a vector afterwards does not
/// change the matrix, so it must not change the solve either.
#[test]
fn solve_banded_does_not_see_a_write_to_a_diagonal_made_after_the_matrix() {
    let ctx = ctx_with(Numerics::default());
    let (a, [mut main, _, _]) = dominant_tridiagonal(&ctx, 12);
    let b = on(&ctx, &[1.0; 24], [2, 12]);
    let before = a.solve_banded(&b).to_vec().unwrap();
    // The matrix is still the pending expression (the solve above did not evaluate it).
    main.write(&[100.0; 12]).unwrap();
    let after = a.solve_banded(&b).to_vec().unwrap();
    assert_eq!(before, after);
    let dense = a.cache().solve(&b).to_vec().unwrap();
    for (x, d) in after.iter().zip(&dense) {
        assert!((x - d).abs() <= 1e-3 * (1.0 + d.abs()), "{x} vs {d}");
    }
}

fn gram_data(len: usize, salt: usize) -> Vec<f32> {
    (0..len).map(|i| (((i * 2654435761usize + salt * 97) >> 7) % 2001) as f32 / 1000.0 - 1.0).collect()
}

/// `xᵀ x` is bit-symmetric, and its lower triangle is the lower triangle of the full product of two equal copies.
#[test]
fn a_gram_product_is_bit_symmetric_and_its_lower_triangle_is_the_full_products() {
    let ctx = ctx_with(Numerics::strict());
    for (rows, cols) in [(40usize, 33usize), (300, 128), (700, 257), (200, 1000), (64, 96)] {
        let data = gram_data(rows * cols, cols);
        let x = on(&ctx, &data, [cols, rows]);
        let copy = on(&ctx, &data, [cols, rows]);
        for (name, gram, full) in [("tn", x.matmul_tn(&x), x.matmul_tn(&copy)), ("nt", x.transpose().matmul_nt(&x.transpose()), x.transpose().cache().matmul_nt(&copy.transpose().cache()))] {
            let (g, f) = (gram.to_vec().unwrap(), full.to_vec().unwrap());
            let n = if name == "tn" { cols } else { cols };
            assert_eq!(g.len(), n * n, "{name} {rows}x{cols}");
            for r in 0..n {
                for c in 0..n {
                    assert_eq!(g[r * n + c].to_bits(), g[c * n + r].to_bits(), "{name} {rows}x{cols}: ({r}, {c}) is not symmetric");
                    if c <= r {
                        assert_eq!(g[r * n + c].to_bits(), f[r * n + c].to_bits(), "{name} {rows}x{cols}: ({r}, {c}) differs from the full product");
                    }
                }
            }
        }
    }
}

#[test]
fn a_gram_product_runs_the_half_work_strategy_and_a_plain_product_does_not() {
    let ctx = ctx_with(Numerics::strict());
    let data = gram_data(512 * 192, 3);
    let x = on(&ctx, &data, [192, 512]);
    let other = on(&ctx, &data, [192, 512]);
    ctx.record_decisions(true);
    x.matmul_tn(&x).to_vec().unwrap();
    let gram = ctx.take_decisions();
    x.matmul_tn(&other).to_vec().unwrap();
    let plain = ctx.take_decisions();
    let syrk = |decisions: &[luma::Decision]| decisions.iter().any(|d| d.chosen.contains("syrk"));
    if matches!(ctx.backend(), luma::BackendKind::Cpu | luma::BackendKind::Cuda) {
        assert!(syrk(&gram), "{gram:?}");
    }
    assert!(!syrk(&plain), "{plain:?}");
}
