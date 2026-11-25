//! Scenarios for trace following (W1-U1b). Prints one bit hash per scenario; the hashes must not
//! depend on `LUMA_TRACE_FOLLOW` (`tools/followcheck` runs it with following on, off and with
//! `+verify` and compares).
use luma::*;

const N: usize = 2048;

fn hash(v: &[f32]) -> u64 {
    v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| (h ^ u64::from(x.to_bits())).wrapping_mul(0x100_0000_01b3))
}

fn data(seed: u32) -> Vec<f32> {
    (0..N).map(|i| 0.5 + ((i as u32).wrapping_mul(2654435761).wrapping_add(seed.wrapping_mul(40503)) % 1000) as f32 * 1e-3).collect()
}

fn run(name: &str, f: impl FnOnce() -> Result<u64>) {
    match f() {
        Ok(h) => println!("{name}: {h:016x}"),
        Err(e) => println!("{name}: error {e}"),
    }
}

fn scalar(i: usize, salt: usize) -> f32 {
    1.0 + i as f32 * 1e-3 + salt as f32 * 1e-4
}

fn chain(x: &Array<f32>, k: usize, salt: usize) -> Array<f32> {
    let mut y = x.clone();
    for i in 0..k {
        let c = scalar(i, salt);
        y = match i % 4 {
            0 => &y * c,
            1 => &y + c * 1e-3,
            2 => y.abs(),
            _ => &y - c * 1e-3,
        };
    }
    y
}

fn main() -> Result<()> {
    let ctx = Context::new()?;
    let xh = data(1);
    let zh = data(2);
    let x = Array::from_slice_on(&ctx, &xh, N)?;
    let z = Array::from_slice_on(&ctx, &zh, N)?;
    let steps = 12;

    run("chain", || {
        let mut h = 0;
        for s in 0..steps {
            let y = chain(&x, 20, s);
            y.eval()?;
            h ^= hash(&y.to_vec()?).rotate_left(s as u32);
            if std::env::var("FC_VERBOSE").is_ok() {
                let v = y.to_vec()?;
                println!("  step {s}: {:016x} {:?}", hash(&v), &v[..4]);
            }
        }
        Ok(h)
    });

    run("chain-no-eval-read", || {
        // The step ends with a read, not with an eval.
        let mut h = 0;
        for s in 0..steps {
            let y = chain(&x, 20, s);
            h ^= hash(&y.to_vec()?).rotate_left(s as u32);
        }
        Ok(h)
    });

    run("wide-shared", || {
        let mut h = 0;
        for s in 0..steps {
            let mut acc: Option<Array<f32>> = None;
            for i in 0..8 {
                let c = scalar(i, s);
                let base = if i % 2 == 0 { &x } else { &z };
                let t = &(&(base * c) + c * 1e-3).abs() * c;
                acc = Some(match acc {
                    None => t,
                    Some(a) => a + t,
                });
            }
            let y = acc.unwrap();
            y.eval()?;
            h ^= hash(&y.to_vec()?).rotate_left(s as u32);
        }
        Ok(h)
    });

    run("op-changes", || {
        let mut h = 0;
        for s in 0..steps {
            let c = scalar(1, s);
            let y = if (s / 4) % 2 == 0 { &(&x * c) + 1.0 } else { &(&x + c) * 2.0 };
            y.eval()?;
            h ^= hash(&y.to_vec()?).rotate_left(s as u32);
        }
        Ok(h)
    });

    run("scalar-specials", || {
        // Scalars that are 1.0 (a rewrite), 0.0, equal to each other, or equal to the length.
        let mut h = 0;
        let vals = [1.5f32, 1.0, 2.0, 2.0, 0.0, 1.0, 1.5, 3.0, 1.0, 1.0, 2.5, 0.0];
        for s in 0..steps {
            let a = vals[s];
            let b = vals[(s + 3) % vals.len()];
            let y = &(&(&x * a) + b) * a;
            y.eval()?;
            h ^= hash(&y.to_vec()?).rotate_left(s as u32);
        }
        Ok(h)
    });

    run("same-buffer-patterns", || {
        let mut h = 0;
        for s in 0..steps {
            let y = if s % 3 == 0 { &x * &x } else { &x * &z };
            let w = &y + 1.0;
            w.eval()?;
            h ^= hash(&w.to_vec()?).rotate_left(s as u32);
        }
        Ok(h)
    });

    run("write-between", || {
        let mut xw = Array::from_slice_on(&ctx, &xh, N)?;
        let mut h = 0;
        for s in 0..steps {
            let y = &(&xw * scalar(2, s)) + 1.0;
            if s % 4 == 3 {
                // A write while the step is pending: the pending expression keeps the old values.
                xw.write(&data(10 + s as u32))?;
            }
            y.eval()?;
            h ^= hash(&y.to_vec()?).rotate_left(s as u32);
        }
        Ok(h)
    });

    run("in-place-op", || {
        let mut xw = Array::from_slice_on(&ctx, &xh, N)?;
        let mut h = 0;
        for s in 0..steps {
            let y = &xw * scalar(3, s);
            xw += 0.5;
            let w = &y + &xw;
            w.eval()?;
            h ^= hash(&w.to_vec()?).rotate_left(s as u32);
        }
        Ok(h)
    });

    run("extra-pending", || {
        let mut h = 0;
        let mut held: Option<Array<f32>> = None;
        for s in 0..steps {
            let y = chain(&x, 10, s);
            if s % 5 == 0 {
                held = Some(&z * scalar(4, s));
            }
            y.eval()?;
            h ^= hash(&y.to_vec()?).rotate_left(s as u32);
            if s % 5 == 4
                && let Some(p) = &held
            {
                h ^= hash(&p.to_vec()?);
            }
        }
        Ok(h)
    });

    run("intermediate-read", || {
        let mut h = 0;
        for s in 0..steps {
            let t = &x * scalar(5, s);
            if s % 3 == 2 {
                h ^= hash(&t.to_vec()?);
            }
            let y = &t + 1.0;
            y.eval()?;
            h ^= hash(&y.to_vec()?).rotate_left(s as u32);
        }
        Ok(h)
    });

    run("intermediate-kept", || {
        // Intermediates are held across the eval and read after it.
        let mut h = 0;
        for s in 0..steps {
            let t = &x * scalar(6, s);
            let u = &t + 1.0;
            let y = &u * 2.0;
            y.eval()?;
            h ^= hash(&y.to_vec()?).rotate_left(s as u32);
            if s % 2 == 0 {
                h ^= hash(&u.to_vec()?);
                h ^= hash(&t.to_vec()?);
            }
        }
        Ok(h)
    });

    run("reduction-in-step", || {
        let mut h = 0;
        for s in 0..steps {
            let t = &x * scalar(7, s);
            let m = t.sum();
            let y = &t / &m;
            y.eval()?;
            h ^= hash(&y.to_vec()?).rotate_left(s as u32);
        }
        Ok(h)
    });

    run("cons-hit", || {
        let mut h = 0;
        for s in 0..steps {
            let c = scalar(8, s);
            let a = &x * c;
            let y = if s % 4 == 3 {
                // The first operand is held, so the second is not the same node.
                let keep = &x * c;
                &(&a + &(&x * c)) + &keep
            } else {
                &a + &(&x * c)
            };
            y.eval()?;
            h ^= hash(&y.to_vec()?).rotate_left(s as u32);
        }
        Ok(h)
    });

    run("state-advance", || {
        let mut cur = Array::from_slice_on(&ctx, &xh, N)?;
        for s in 0..steps {
            let next = &(&cur * 0.99) + scalar(9, 0) * 0.01;
            next.eval()?;
            cur = next;
            let _ = s;
        }
        Ok(hash(&cur.to_vec()?))
    });

    run("state-advance-fixed-input", || {
        let mut cur = Array::from_slice_on(&ctx, &xh, N)?;
        for _ in 0..steps {
            let next = &(&cur * 0.99) + &(&z * 0.01);
            next.eval()?;
            cur = next;
        }
        Ok(hash(&cur.to_vec()?))
    });

    run("stencil-advance", || {
        let mut cur = Array::from_slice_on(&ctx, &xh, N)?;
        for _ in 0..steps {
            let next = &(&cur.shift([1]) + &cur.shift([-1])) * 0.5;
            next.eval()?;
            cur = next;
        }
        Ok(hash(&cur.to_vec()?))
    });

    run("multi-root", || {
        let mut h = 0;
        for s in 0..steps {
            let a = &x * scalar(1, s);
            let b = &z + scalar(2, s);
            let c = &a * &b;
            eval_all(&[&a, &b, &c])?;
            h ^= hash(&c.to_vec()?).rotate_left(s as u32);
        }
        Ok(h)
    });

    run("flush-ends-step", || {
        let mut h = 0;
        for s in 0..steps {
            let y = chain(&x, 12, s);
            ctx.flush()?;
            h ^= hash(&y.to_vec()?).rotate_left(s as u32);
        }
        Ok(h)
    });

    run("other-thread", || {
        let mut h = 0;
        let (xh2, ctx2) = (xh.clone(), ctx.clone());
        let t = std::thread::spawn(move || -> Result<u64> {
            let x = Array::from_slice_on(&ctx2, &xh2, N)?;
            let mut h = 0;
            for s in 0..steps {
                let y = chain(&x, 15, s + 100);
                y.eval()?;
                h ^= hash(&y.to_vec()?).rotate_left(s as u32);
            }
            Ok(h)
        });
        for s in 0..steps {
            let y = chain(&x, 15, s);
            y.eval()?;
            h ^= hash(&y.to_vec()?).rotate_left(s as u32);
        }
        Ok(h ^ t.join().unwrap()?)
    });

    run("two-contexts", || {
        let ctx2 = Context::new()?;
        let x2 = Array::from_slice_on(&ctx2, &xh, N)?;
        let mut h = 0;
        for s in 0..steps {
            let y = chain(&x, 10, s);
            let y2 = chain(&x2, 10, s);
            y.eval()?;
            y2.eval()?;
            h ^= hash(&y.to_vec()?).rotate_left(s as u32) ^ hash(&y2.to_vec()?);
        }
        Ok(h)
    });

    run("drop-without-eval", || {
        let mut h = 0;
        for s in 0..steps {
            let y = chain(&x, 10, s);
            if s % 3 == 0 {
                drop(y);
                continue;
            }
            y.eval()?;
            h ^= hash(&y.to_vec()?).rotate_left(s as u32);
        }
        Ok(h)
    });
    run("deep-chain", || {
        // A chain deeper than the fusion depth: `create` cuts it, and the cut is part of the step.
        let mut h = 0;
        for s in 0..steps {
            let y = chain(&x, 1100, s);
            y.eval()?;
            h ^= hash(&y.to_vec()?).rotate_left(s as u32);
        }
        Ok(h)
    });

    run("assign-in-step", || {
        // Assignment into lazy arrays (the arrays that hold the node are given another one).
        let mut a = Array::from_slice_on(&ctx, &xh, N)?;
        let mut b = Array::from_slice_on(&ctx, &zh, N)?;
        for s in 0..steps {
            let t = &a * scalar(11, s);
            let u = &b + &t;
            a.assign(&u)?;
            b.assign(&(&t - &u))?;
            a.eval()?;
            b.eval()?;
        }
        Ok(hash(&a.to_vec()?) ^ hash(&b.to_vec()?).rotate_left(7))
    });

    run("alternating-steps", || {
        // Two kinds of step, one after the other: never the same as the step before.
        let mut h = 0;
        for s in 0..steps {
            let y = if s % 2 == 0 { chain(&x, 9, s) } else { chain(&z, 14, s) };
            y.eval()?;
            h ^= hash(&y.to_vec()?).rotate_left(s as u32);
        }
        Ok(h)
    });

    run("many-small-steps", || {
        // More steps than the cooldown after a divergence lasts.
        let mut h = 0;
        for s in 0..200 {
            let y = chain(&x, 6 + (s / 50) * 2, s);
            y.eval()?;
            h ^= hash(&y.to_vec()?).rotate_left((s % 60) as u32);
        }
        Ok(h)
    });
    Ok(())
}
