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
