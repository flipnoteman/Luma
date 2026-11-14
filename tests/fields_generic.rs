//! Records and loops over arrays that are not `f32`: `Df32` (double-float) records match the
//! same arithmetic in `f64` to about 1e-12, `f64` records work where the device has 64-bit
//! shader floats and are refused (not miscompiled) where it has not.

mod common;

use common::values;
use luma::*;

luma::fields! {
    struct Dd {
        u: Array<Df32>,
        v: Array<Df32>,
    }
}

luma::fields! {
    struct Wide {
        a: Array<f64>,
        b: Array<f64>,
    }
}

luma::fields! {
    #[elem = Df32]
    struct Nested {
        d: Dd,
        t: [Array<Df32>; 2],
    }
}

const N: usize = 24;
const SHAPE: [usize; 2] = [6, 4];

fn has_f64(ctx: &Context) -> bool {
    ctx.features().contains(luma::wgpu::Features::SHADER_F64)
}

/// `f64` data in `[0.5, 2]` that is not representable in `f32`.
fn host(seed: u64) -> Vec<f64> {
    values(N, seed, 0.5, 2.0).iter().map(|&v| f64::from(v) * 1.000_000_117).collect()
}

fn df32s(ctx: &Context, data: &[f64]) -> Array<Df32> {
    let data: Vec<Df32> = data.iter().map(|&v| Df32::from(v)).collect();
    Array::from_slice_on(ctx, &data, SHAPE).unwrap()
}

fn dd(ctx: &Context, u: &[f64], v: &[f64]) -> Dd {
    Dd { u: df32s(ctx, u), v: df32s(ctx, v) }
}

fn read(a: &Array<Df32>) -> Vec<f64> {
    a.to_vec().unwrap().into_iter().map(Df32::to_f64).collect()
}

fn assert_f64_close(got: &[f64], want: &[f64], tolerance: f64) {
    assert_eq!(got.len(), want.len(), "length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tolerance * w.abs().max(1.0), "index {i}: got {g}, want {w}");
    }
}

#[test]
fn a_df32_record_does_arithmetic_with_scalars_and_arrays() {
    let ctx = Context::new().unwrap();
    let (hu, hv, hw, hl) = (host(1), host(2), host(3), host(4));
    let (a, b) = (dd(&ctx, &hu, &hv), dd(&ctx, &hw, &hl));
    let lam = df32s(&ctx, &hl);

    // Record with record, element scalar (both sides), array, negation.
    let r = (&a * &b + &a / &b - Df32::from(0.5) * &a) * Df32::from(1.25) - &b * &lam;
    let neg = -&a;
    let mut acc = a.copy();
    acc += &b;
    acc *= Df32::from(0.5);
    acc -= &lam;

    for (got, (x, y, z)) in [(&r.u, (&hu, &hw, &hl)), (&r.v, (&hv, &hl, &hl))] {
        let want: Vec<f64> = (0..N)
            .map(|i| (x[i] * y[i] + x[i] / y[i] - 0.5 * x[i]) * 1.25 - y[i] * z[i])
            .collect();
        assert_f64_close(&read(got), &want, 1e-12);
    }
    let want_neg: Vec<f64> = hu.iter().map(|v| -v).collect();
    assert_f64_close(&read(&neg.u), &want_neg, 1e-13);
    let want_acc: Vec<f64> = (0..N).map(|i| (hu[i] + hw[i]) * 0.5 - hl[i]).collect();
    assert_f64_close(&read(&acc.u), &want_acc, 1e-12);

    // `dot`, `slice`, `map`, `zip` and `select` come from the trait.
    let dot: Vec<f64> = read(&a.dot(&b));
    let want_dot: Vec<f64> = (0..N).map(|i| hu[i] * hw[i] + hv[i] * hl[i]).collect();
    assert_f64_close(&dot, &want_dot, 1e-12);
    let s = a.slice(0, 1..4);
    assert_eq!(s.u.shape().dims(), [3, 4, 1, 1]);
    let sums = a.zip(&b, |x, y| x + y);
    let want: Vec<f64> = (0..N).map(|i| hu[i] + hw[i]).collect();
    assert_f64_close(&read(&sums.u), &want, 1e-13);
    assert_eq!(Dd::COUNT, 2);
    assert_eq!(Nested::COUNT, 4);
}

#[test]
fn records_of_records_and_tuples_keep_their_element_type() {
    let ctx = Context::new().unwrap();
    let (hu, hv) = (host(5), host(6));
    let n = Nested { d: dd(&ctx, &hu, &hv), t: [df32s(&ctx, &hu), df32s(&ctx, &hv)] };
    let m = &n + &n;
    let want: Vec<f64> = hu.iter().map(|v| 2.0 * v).collect();
    assert_f64_close(&read(&m.d.u), &want, 1e-13);
    assert_f64_close(&read(&m.t[0]), &want, 1e-13);

    let pair = (df32s(&ctx, &hu), df32s(&ctx, &hv));
    let flipped = Fields::flip(&pair, 0);
    let got = read(&flipped.1);
    for y in 0..4 {
        for x in 0..6 {
            assert!((got[x + 6 * y] - hv[5 - x + 6 * y]).abs() <= 1e-13);
        }
    }
}

#[test]
fn a_df32_record_is_assigned_through_its_slices() {
    let ctx = Context::new().unwrap();
    let (hu, hv) = (host(7), host(8));
    let mut w = dd(&ctx, &hu, &hv);

    // The mirror ghost rule of a wall: columns 0, 1 take minus columns 3, 2.
    let rhs = w.slice(0, 2..4).flip(0) * Df32::from(-1.0);
    w.slice_mut(0, ..2).assign(rhs).unwrap();
    for (got, h) in [(read(&w.u), &hu), (read(&w.v), &hv)] {
        for y in 0..4 {
            for x in 0..6 {
                let want = match x {
                    0 | 1 => -h[3 - x + 6 * y],
                    _ => h[x + 6 * y],
                };
                assert!((got[x + 6 * y] - want).abs() <= 1e-12, "({x}, {y}): {} vs {want}", got[x + 6 * y]);
            }
        }
    }

    // The same by reference.
    let mut w = dd(&ctx, &hu, &hv);
    let ghost = w.slice(0, 2..4).flip(0) * Df32::from(-1.0);
    w.slice_mut(0, ..2).assign(&ghost).unwrap();
    let got = read(&w.u);
    assert!((got[0] + hu[3]).abs() <= 1e-12);
    assert!((read(&ghost.u)[0] + hu[3]).abs() <= 1e-12, "the right-hand side keeps its values");
}

/// `step`: both states relax towards each other and a decaying source.
fn relax(x: &Array<Df32>, y: &Array<Df32>) -> [Array<Df32>; 2] {
    let (c, d, h) = (Df32::from(0.75), Df32::from(0.25), Df32::from(0.5));
    [x * c + y * d, y * h + x * d - x * y * Df32::from(0.0625)]
}

fn relax_host(x: &[f64], y: &[f64], steps: usize) -> [Vec<f64>; 2] {
    let (mut x, mut y) = (x.to_vec(), y.to_vec());
    for _ in 0..steps {
        let nx: Vec<f64> = (0..N).map(|i| x[i] * 0.75 + y[i] * 0.25).collect();
        let ny: Vec<f64> = (0..N).map(|i| y[i] * 0.5 + x[i] * 0.25 - x[i] * y[i] * 0.0625).collect();
        (x, y) = (nx, ny);
    }
    [x, y]
}

#[test]
fn iterate_all_runs_df32_states() {
    let ctx = Context::new().unwrap();
    let (hu, hv) = (host(9), host(10));
    let (x, y) = (df32s(&ctx, &hu), df32s(&ctx, &hv));
    let calls = std::cell::Cell::new(0);
    let [gx, gy] = iterate_all([&x, &y], 37, |[x, y]| {
        calls.set(calls.get() + 1);
        relax(x, y)
    });
    assert_eq!(calls.get(), 1, "the register loop is f32 only: Df32 traces the step once and replays it");
    let [wx, wy] = relax_host(&hu, &hv, 37);
    assert_f64_close(&read(&gx), &wx, 1e-12);
    assert_f64_close(&read(&gy), &wy, 1e-12);
    // The initial state is not modified.
    assert_f64_close(&read(&x), &hu, 1e-14);

    // A step with shifts and a `cache()` in the middle cannot be a register loop.
    calls.set(0);
    let [sx, sy] = iterate_all([&x, &y], 11, |[x, y]| {
        calls.set(calls.get() + 1);
        let [a, b] = relax(x, y);
        let a = a.cache();
        [&a + a.shift([1, 0]) * Df32::from(0.5) - &a * Df32::from(0.5), b]
    });
    assert_eq!(calls.get(), 1, "traced once");
    let mut want_x = hu.clone();
    let mut want_y = hv.clone();
    for _ in 0..11 {
        let a: Vec<f64> = (0..N).map(|i| want_x[i] * 0.75 + want_y[i] * 0.25).collect();
        let b: Vec<f64> = (0..N).map(|i| want_y[i] * 0.5 + want_x[i] * 0.25 - want_x[i] * want_y[i] * 0.0625).collect();
        want_x = (0..N).map(|i| a[i] + a[(i + 5) % 6 + 6 * (i / 6)] * 0.5 - a[i] * 0.5).collect();
        want_y = b;
    }
    assert_f64_close(&read(&sx), &want_x, 1e-12);
    assert_f64_close(&read(&sy), &want_y, 1e-12);

    // Zero steps give a copy of the states.
    let [zx, _] = iterate_all([&x, &y], 0, |[x, y]| relax(x, y));
    assert_f64_close(&read(&zx), &hu, 1e-14);
}

#[test]
fn iterate_until_all_runs_a_df32_record() {
    let ctx = Context::new().unwrap();
    let (hu, hv) = (host(11), host(12));
    let run = iterate_until_all(
        dd(&ctx, &hu, &hv),
        Until::new(200)
            .record("max", |s: &Dd| s.u.max().to_f32())
            .stop_when(|h| h["max"].last().le(1e-2)),
        |s, _| {
            let [u, v] = relax(&s.u, &s.v);
            Dd { u: u * Df32::from(0.5), v }
        },
    )
    .unwrap();

    // The host loop with the same rule.
    let (mut u, mut v) = (hu.clone(), hv.clone());
    let mut maxes = Vec::new();
    let mut steps = 0;
    loop {
        let [nu, nv] = relax_host(&u, &v, 1);
        u = nu.iter().map(|x| x * 0.5).collect();
        v = nv;
        steps += 1;
        maxes.push(u.iter().copied().fold(f64::MIN, f64::max) as f32);
        if *maxes.last().unwrap() <= 1e-2 || steps == 200 {
            break;
        }
    }
    assert_eq!(run.steps, steps);
    assert_eq!(run.reason, StopReason::Predicate(0));
    assert_f64_close(&read(&run.state.u), &u, 1e-12);
    assert_f64_close(&read(&run.state.v), &v, 1e-12);
    let history = run.history("max").unwrap();
    for (g, w) in history.iter().zip(&maxes) {
        assert!((g - w).abs() <= 1e-6 * w.abs().max(1e-3), "{g} vs {w}");
    }
}

#[test]
fn f64_records_work_where_the_device_has_f64_and_are_refused_elsewhere() {
    let ctx = Context::new().unwrap();
    let (ha, hb) = (host(13), host(14));
    let a = Wide {
        a: Array::from_slice_on(&ctx, &ha, SHAPE).unwrap(),
        b: Array::from_slice_on(&ctx, &hb, SHAPE).unwrap(),
    };
    let r = &a * &a + 0.5 * &a - &a.a;
    if has_f64(&ctx) {
        let got = r.a.to_vec().unwrap();
        for i in 0..N {
            assert_f64_close(&got[i..=i], &[ha[i] * ha[i] + 0.5 * ha[i] - ha[i]], 1e-14);
        }
        let got = r.b.to_vec().unwrap();
        for i in 0..N {
            assert_f64_close(&got[i..=i], &[hb[i] * hb[i] + 0.5 * hb[i] - ha[i]], 1e-14);
        }
        let run = iterate_until_all(
            Wide { a: Array::full_on(&ctx, SHAPE, 1.0).unwrap(), b: Array::full_on(&ctx, SHAPE, 2.0).unwrap() },
            Until::new(50).record("max", |s: &Wide| s.a.max().to_f32()).stop_when(|h| h["max"].last().le(1e-3)),
            |s, _| s * 0.5,
        )
        .unwrap();
        assert_eq!(run.steps, 10);
        assert!(run.state.b.to_vec().unwrap().iter().all(|&v| v == 2.0 * 0.5f64.powi(10)));
    } else {
        let err = r.a.to_vec().unwrap_err();
        assert!(matches!(err.root(), LumaError::Unsupported(_)), "{err}");
    }
}
