//! Host cost of recording one kernel launch.
//!
//! Every case records `LAUNCHES` dependent launches of a trivial kernel (`a[i] += 1`) on arrays
//! of different sizes. "record" is the host time until the last `launch` returned (the GPU may
//! already be executing what was submitted by then), "total" includes the final synchronization.
//! Launch cost is independent of the problem size as long as the host is the bottleneck.

use luma::*;
use std::time::Instant;

const LAUNCHES: usize = 4000;
const REPEATS: usize = 5;

const INCREMENT: &str = r#"
@group(1) @binding(0) var<storage, read_write> a: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < luma.size.x) { a[gid.x] += 1.0; }
}
"#;

const SCALE: &str = r#"
struct Params { s: f32, n: u32 }
@group(1) @binding(0) var<storage, read_write> a: array<f32>;
@group(1) @binding(1) var<uniform> p: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < p.n) { a[gid.x] = a[gid.x] * p.s + 1.0; }
}
"#;

const COPY: &str = r#"
@group(1) @binding(0) var<storage, read> src: array<f32>;
@group(1) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < luma.size.x) { dst[gid.x] = src[gid.x] + 1.0; }
}
"#;

/// Shaped like cfd-gpu's fused kernel: ten storage buffers and an 80-byte uniform block.
const WIDE: &str = r#"
struct Wide { p0: f32, p1: f32, p2: f32, p3: f32, p4: f32, p5: f32, p6: f32, p7: f32, p8: f32, p9: f32, p10: f32, p11: f32, p12: f32, p13: f32, p14: f32, p15: f32, p16: f32, p17: f32, p18: f32, p19: f32 }
@group(1) @binding(0) var<storage, read_write> o: array<f32>;
@group(1) @binding(1) var<storage, read> i1: array<f32>;
@group(1) @binding(2) var<storage, read> i2: array<f32>;
@group(1) @binding(3) var<storage, read> i3: array<f32>;
@group(1) @binding(4) var<storage, read> i4: array<f32>;
@group(1) @binding(5) var<storage, read> i5: array<f32>;
@group(1) @binding(6) var<storage, read> i6: array<f32>;
@group(1) @binding(7) var<storage, read> i7: array<f32>;
@group(1) @binding(8) var<storage, read> i8: array<f32>;
@group(1) @binding(9) var<storage, read> i9: array<f32>;
@group(1) @binding(10) var<uniform> w: Wide;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= luma.size.x) { return; }
    o[i] = w.p0 * (i1[i] + i2[i] + i3[i] + i4[i] + i5[i] + i6[i] + i7[i] + i8[i] + i9[i]);
}
"#;

#[derive(encase::ShaderType)]
struct Params {
    s: f32,
    n: u32,
}

#[derive(encase::ShaderType)]
struct Wide {
    p0: f32,
    p1: f32,
    p2: f32,
    p3: f32,
    p4: f32,
    p5: f32,
    p6: f32,
    p7: f32,
    p8: f32,
    p9: f32,
    p10: f32,
    p11: f32,
    p12: f32,
    p13: f32,
    p14: f32,
    p15: f32,
    p16: f32,
    p17: f32,
    p18: f32,
    p19: f32,
}

/// Best (lowest) `(record, total)` microseconds per launch of `repeats` runs of `f`.
fn measure(mut f: impl FnMut() -> Result<()>, ctx: &Context) -> Result<(f64, f64)> {
    f()?;
    ctx.sync()?;
    let (mut record, mut total) = (f64::MAX, f64::MAX);
    for _ in 0..REPEATS {
        let start = Instant::now();
        f()?;
        let recorded = start.elapsed();
        ctx.sync()?;
        let all = start.elapsed();
        record = record.min(recorded.as_secs_f64());
        total = total.min(all.as_secs_f64());
    }
    Ok((
        record / LAUNCHES as f64 * 1e6,
        total / LAUNCHES as f64 * 1e6,
    ))
}

fn main() -> Result<()> {
    let ctx = default_context()?;
    let increment = ctx.kernel(INCREMENT).build()?;
    let scale = ctx.kernel(SCALE).build()?;
    let copy = ctx.kernel(COPY).build()?;
    let wide = ctx.kernel(WIDE).build()?;

    println!("Host cost per launch ({LAUNCHES} dependent launches, best of {REPEATS})");
    println!("{}", "=".repeat(78));
    println!(
        "{:<10} {:<34} {:>14} {:>14}",
        "elements", "case", "record us", "total us"
    );
    println!("{}", "-".repeat(78));
    for &n in &[64u32, 5_000, 80_000, 1_000_000] {
        let mut a = Array::<f32>::zeros_on(&ctx, n as usize)?;
        let mut b = Array::<f32>::zeros_on(&ctx, n as usize)?;
        let inputs: Vec<Array<f32>> = (0..9)
            .map(|_| Array::<f32>::zeros_on(&ctx, n as usize))
            .collect::<Result<_>>()?;
        let row = |name: &str, result: (f64, f64)| {
            println!(
                "{:<10} {:<34} {:>14.2} {:>14.2}",
                n, name, result.0, result.1
            );
        };

        row(
            "kernel, 1 storage arg",
            measure(
                || {
                    for _ in 0..LAUNCHES {
                        increment.launch([n], args![&mut a])?;
                    }
                    Ok(())
                },
                &ctx,
            )?,
        );
        row(
            "kernel, storage + uniform",
            measure(
                || {
                    let params = Params { s: 1.0, n };
                    for _ in 0..LAUNCHES {
                        scale.launch([n], args![&mut a, uniform(&params)])?;
                    }
                    Ok(())
                },
                &ctx,
            )?,
        );
        row(
            "kernel, 2 storage args (ping-pong)",
            measure(
                || {
                    for i in 0..LAUNCHES {
                        if i % 2 == 0 {
                            copy.launch([n], args![&a, &mut b])?;
                        } else {
                            copy.launch([n], args![&b, &mut a])?;
                        }
                    }
                    Ok(())
                },
                &ctx,
            )?,
        );
        row(
            "kernel, 10 storage + 80 B uniform",
            measure(
                || {
                    let w = Wide {
                        p0: 0.5,
                        p1: 0.5,
                        p2: 0.5,
                        p3: 0.5,
                        p4: 0.5,
                        p5: 0.5,
                        p6: 0.5,
                        p7: 0.5,
                        p8: 0.5,
                        p9: 0.5,
                        p10: 0.5,
                        p11: 0.5,
                        p12: 0.5,
                        p13: 0.5,
                        p14: 0.5,
                        p15: 0.5,
                        p16: 0.5,
                        p17: 0.5,
                        p18: 0.5,
                        p19: 0.5,
                    };
                    for _ in 0..LAUNCHES {
                        wide.launch(
                            [n],
                            args![
                                &mut a,
                                &inputs[0],
                                &inputs[1],
                                &inputs[2],
                                &inputs[3],
                                &inputs[4],
                                &inputs[5],
                                &inputs[6],
                                &inputs[7],
                                &inputs[8],
                                uniform(&w)
                            ],
                        )?;
                    }
                    Ok(())
                },
                &ctx,
            )?,
        );
        row(
            "built-in `a += 1.0`",
            measure(
                || {
                    for _ in 0..LAUNCHES {
                        a += 1.0;
                    }
                    a.check()
                },
                &ctx,
            )?,
        );
    }
    println!("{}", "=".repeat(78));

    verify(&ctx, &increment)?;
    for n in [5_000u32, 2_000_000] {
        flush_overlap(&ctx, &copy, n)?;
    }
    Ok(())
}

/// The launches really ran, in order: a dependent chain gives an exact result.
fn verify(ctx: &Context, increment: &Kernel) -> Result<()> {
    let mut a = Array::<f32>::zeros_on(ctx, 1000)?;
    for _ in 0..LAUNCHES {
        increment.launch([1000], args![&mut a])?;
    }
    assert!(
        a.to_vec()?.iter().all(|&v| v == LAUNCHES as f32),
        "wrong result"
    );
    println!("verified: {LAUNCHES} dependent launches give exactly {LAUNCHES}");
    Ok(())
}

/// Recording and executing overlap when the batch is flushed early: 200 dependent launches, one
/// sync at the end, flushing after every `every` launches (never: the batch only goes out at 256
/// launches and at the sync). Helps while the host is the bottleneck, not when the GPU is.
fn flush_overlap(ctx: &Context, copy: &Kernel, n: u32) -> Result<()> {
    let mut a = Array::<f32>::zeros_on(ctx, n as usize)?;
    let mut b = Array::<f32>::zeros_on(ctx, n as usize)?;
    let launches = 200;
    println!();
    println!("Flush overlap: {launches} dependent launches over {n} elements, one sync at the end");
    println!("{:<28} {:>12}", "flush", "total us");
    for every in [None, Some(64usize), Some(16), Some(4)] {
        let mut best = f64::MAX;
        for _ in 0..REPEATS + 1 {
            let start = Instant::now();
            for i in 0..launches {
                if i % 2 == 0 {
                    copy.launch([n], args![&a, &mut b])?;
                } else {
                    copy.launch([n], args![&b, &mut a])?;
                }
                if every.is_some_and(|every| (i + 1) % every == 0) {
                    ctx.flush()?;
                }
            }
            ctx.sync()?;
            best = best.min(start.elapsed().as_secs_f64());
        }
        let name = match every {
            None => "none (sync only)".to_owned(),
            Some(every) => format!("every {every} launches"),
        };
        println!("{:<28} {:>12.0}", name, best * 1e6);
    }
    Ok(())
}
