//! Records of arrays (`Fields` and `#[derive(Fields)]`): every operation must equal the same
//! operation on each field.

mod common;

use common::{assert_close, values};
use luma::{Array, Context, Fields, LumaError, PadMode};

#[derive(Fields)]
struct Cons {
    rho: Array<f32>,
    mu: Array<f32>,
    mv: Array<f32>,
    e: Array<f32>,
}

#[derive(Fields)]
struct Both {
    q: Cons,
    t: [Array<f32>; 2],
}

const SHAPE: [usize; 2] = [5, 4];

fn host(seed: u64) -> [Vec<f32>; 4] {
    std::array::from_fn(|k| values(20, seed * 10 + k as u64, -1.0, 1.0))
}

fn cons(ctx: &Context, data: &[Vec<f32>; 4]) -> Cons {
    let f = |k: usize| Array::from_slice_on(ctx, &data[k], SHAPE).unwrap();
    Cons { rho: f(0), mu: f(1), mv: f(2), e: f(3) }
}

fn fields(c: &Cons) -> [Vec<f32>; 4] {
    [c.rho.to_vec().unwrap(), c.mu.to_vec().unwrap(), c.mv.to_vec().unwrap(), c.e.to_vec().unwrap()]
}

#[test]
fn rusanov_style_flux_matches_the_host_per_field() {
    let ctx = Context::new().unwrap();
    let (ha, hb) = (host(1), host(2));
    let hl = values(20, 77, 0.5, 1.5);
    let (a, b) = (cons(&ctx, &ha), cons(&ctx, &hb));
    let lam = Array::from_slice_on(&ctx, &hl, SHAPE).unwrap();

    let flux = 0.5 * (&a + &b) - 0.5 * &lam * (&b - &a);
    let got = fields(&flux);
    for k in 0..4 {
        let want: Vec<f32> =
            (0..20).map(|i| 0.5 * (ha[k][i] + hb[k][i]) - 0.5 * hl[i] * (hb[k][i] - ha[k][i])).collect();
        assert_close(&got[k], &want, 1e-5, 1e-6);
    }
}

#[test]
fn operator_forms_agree() {
    let ctx = Context::new().unwrap();
    let (ha, hb) = (host(3), host(4));
    let hl = values(20, 78, 0.5, 1.5);
    let (a, b) = (cons(&ctx, &ha), cons(&ctx, &hb));
    let lam = Array::from_slice_on(&ctx, &hl, SHAPE).unwrap();

    #[allow(clippy::type_complexity)]
    let cases: Vec<(Cons, Box<dyn Fn(f32, f32, f32) -> f32>)> = vec![
        (&a - &b, Box::new(|x, y, _| x - y)),
        (a.map(|x| x * 1.0) * b.map(|x| x * 1.0), Box::new(|x, y, _| x * y)),
        (&a / 2.0, Box::new(|x, _, _| x / 2.0)),
        (2.0 / (&b + 3.0), Box::new(|_, y, _| 2.0 / (y + 3.0))),
        (&lam - &a, Box::new(|x, _, l| l - x)),
        (&a * &lam, Box::new(|x, _, l| x * l)),
        (&lam / &a.map(|x| x + 3.0), Box::new(|x, _, l| l / (x + 3.0))),
        (-&b, Box::new(|_, y, _| -y)),
    ];
    for (n, (rec, f)) in cases.iter().enumerate() {
        let got = fields(rec);
        for k in 0..4 {
            let want: Vec<f32> = (0..20).map(|i| f(ha[k][i], hb[k][i], hl[i])).collect();
            assert_close(&got[k], &want, 1e-5, 1e-6);
            let _ = n;
        }
    }
}

#[test]
fn dot_sums_the_products_of_the_fields() {
    let ctx = Context::new().unwrap();
    let (ha, hb) = (host(5), host(6));
    let (a, b) = (cons(&ctx, &ha), cons(&ctx, &hb));
    let got = a.dot(&b).to_vec().unwrap();
    let want: Vec<f32> = (0..20).map(|i| (0..4).map(|k| ha[k][i] * hb[k][i]).sum()).collect();
    assert_close(&got, &want, 1e-5, 1e-6);
}

#[test]
fn negation_and_compound_assignment() {
    let ctx = Context::new().unwrap();
    let (ha, hb) = (host(7), host(8));
    let (mut a, b) = (cons(&ctx, &ha), cons(&ctx, &hb));
    let neg = fields(&-&a);
    a += &b;
    let sum = fields(&a);
    a *= 2.0;
    let twice = fields(&a);
    for k in 0..4 {
        let want: Vec<f32> = ha[k].iter().map(|v| -v).collect();
        assert_eq!(neg[k], want);
        let want: Vec<f32> = (0..20).map(|i| ha[k][i] + hb[k][i]).collect();
        assert_close(&sum[k], &want, 1e-6, 1e-6);
        let want: Vec<f32> = want.iter().map(|v| v * 2.0).collect();
        assert_close(&twice[k], &want, 1e-6, 1e-6);
    }
}

#[test]
fn slices_have_the_shape_of_a_slice_of_each_field() {
    let ctx = Context::new().unwrap();
    let ha = host(9);
    let a = cons(&ctx, &ha);
    let s = a.slice(0, 1..3);
    for f in s.arrays() {
        assert_eq!(f.shape().dims(), [2, 4, 1, 1]);
    }
    let got = fields(&s);
    for k in 0..4 {
        let want: Vec<f32> = (0..4).flat_map(|y| (1..3).map(move |x| (x, y))).map(|(x, y)| ha[k][x + 5 * y]).collect();
        assert_eq!(got[k], want);
    }
}

#[test]
fn a_padded_record_takes_its_ghost_cells_by_assignment() {
    let ctx = Context::new().unwrap();
    let ha = host(10);
    let a = cons(&ctx, &ha);
    let w = a.pad([(2, 2), (0, 0)], PadMode::Edge);
    let ghost = w.slice(0, 2..4).flip(0) * -1.0;
    w.slice(0, ..2).assign(&ghost).unwrap();
    let got = fields(&w);
    for k in 0..4 {
        for y in 0..4 {
            // Padded row: ghost 0, 1, then the interior 0..5, then two edge cells.
            assert_eq!(got[k][9 * y], -ha[k][1 + 5 * y]);
            assert_eq!(got[k][9 * y + 1], -ha[k][5 * y]);
            for x in 0..5 {
                assert_eq!(got[k][9 * y + 2 + x], ha[k][x + 5 * y]);
            }
            assert_eq!(got[k][9 * y + 7], ha[k][4 + 5 * y]);
        }
    }
    // The source record is untouched.
    assert_eq!(fields(&a), ha);
}

/// Fields whose padded array is evaluated are written into their buffers, the others are
/// replaced by expressions; the result is the same either way.
#[test]
fn a_record_with_some_fields_evaluated_takes_ghost_cells_like_one_without() {
    let ctx = Context::new().unwrap();
    let ha = host(11);
    let a = cons(&ctx, &ha);
    let ghosted = |evaluate: &[usize]| {
        let padded = |k: usize, f: &Array<f32>| {
            let p = (f * 2.0).pad([(2, 2), (0, 0)], PadMode::Edge);
            if evaluate.contains(&k) { p.cache() } else { p }
        };
        let w = Cons { rho: padded(0, &a.rho), mu: padded(1, &a.mu), mv: padded(2, &a.mv), e: padded(3, &a.e) };
        let ghost = w.slice(0, 2..4).flip(0) * -1.0;
        w.slice(0, ..2).assign(ghost).unwrap();
        fields(&w)
    };
    let lazy = ghosted(&[]);
    assert_eq!(ghosted(&[1]), lazy);
    assert_eq!(ghosted(&[0, 3]), lazy);
    assert_eq!(ghosted(&[0, 1, 2, 3]), lazy);
    for k in 0..4 {
        for y in 0..4 {
            assert_eq!(lazy[k][9 * y], -2.0 * ha[k][1 + 5 * y]);
            assert_eq!(lazy[k][9 * y + 2], 2.0 * ha[k][5 * y]);
        }
    }
}

#[test]
fn nested_records_count_every_array() {
    let ctx = Context::new().unwrap();
    assert_eq!(<Both as Fields>::COUNT, 6);
    assert_eq!(<(Cons, [Array<f32>; 2]) as Fields>::COUNT, 6);
    let ha = host(11);
    let one = Array::from_slice_on(&ctx, &ha[0], SHAPE).unwrap();
    let both = Both { q: cons(&ctx, &ha), t: [one.map(|x| x * 1.0), one.map(|x| x * 3.0)] };
    assert_eq!(both.arrays().len(), 6);
    let twice = &both * 2.0;
    assert_close(&twice.t[1].to_vec().unwrap(), &ha[0].iter().map(|v| v * 6.0).collect::<Vec<_>>(), 1e-6, 1e-6);
    assert_close(&twice.q.e.to_vec().unwrap(), &ha[3].iter().map(|v| v * 2.0).collect::<Vec<_>>(), 1e-6, 1e-6);
}

#[test]
fn put_and_take_work_on_every_field() {
    let ctx = Context::new().unwrap();
    let ha = host(12);
    let a = cons(&ctx, &ha);
    let idx = Array::from_slice_on(&ctx, &[3u32, 0, 19], 3).unwrap();
    let t = a.take(&idx);
    let got = fields(&t);
    for k in 0..4 {
        assert_eq!(got[k], vec![ha[k][3], ha[k][0], ha[k][19]]);
    }
    let dst = cons(&ctx, &host(13)).map(|x| x * 0.0);
    dst.put(&idx, &t).unwrap();
    let got = fields(&dst);
    for k in 0..4 {
        assert_eq!(got[k][3], ha[k][3]);
        assert_eq!(got[k][19], ha[k][19]);
        assert_eq!(got[k][1], 0.0);
    }
}

#[test]
fn a_shape_mismatch_poisons_the_fields_at_the_callers_location() {
    let ctx = Context::new().unwrap();
    let a = cons(&ctx, &host(14));
    let small = Cons {
        rho: Array::zeros_on(&ctx, [2, 2]).unwrap(),
        mu: Array::zeros_on(&ctx, [2, 2]).unwrap(),
        mv: Array::zeros_on(&ctx, [2, 2]).unwrap(),
        e: Array::zeros_on(&ctx, [2, 2]).unwrap(),
    };
    let bad = &a + &small;
    for f in bad.arrays() {
        let error = f.error().expect("poisoned");
        assert!(matches!(error.root(), LumaError::ShapeMismatch { .. }));
        assert!(error.location().unwrap().file().ends_with("records.rs"), "{:?}", error.location());
    }
    let sliced = a.slice(0, 3..9);
    let error = sliced.rho.error().unwrap();
    assert!(error.location().unwrap().file().ends_with("records.rs"));
}
