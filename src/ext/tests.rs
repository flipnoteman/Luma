//! Custom ops under the library's internal replay engine (no public trace API).

use crate::ext::{Bind, OutShape, Strategy};
use crate::trace::trace;
use crate::*;

const SCALE: &str = r#"
@group(1) @binding(0) var<storage, read> x: array<f32>;
@group(1) @binding(1) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= luma.size.x) { return; }
    y[gid.x] = 3.0 * x[gid.x] + 1.0;
}
"#;

fn op(ctx: &Context) -> ext::Op {
    ctx.custom_op("replayed")
        .input::<f32>()
        .output::<f32>(OutShape::Same(0))
        .strategy(Strategy::new("plain", |lower| {
            let kernel = lower.kernel(SCALE).build()?;
            let n = lower.inputs()[0].shape.len() as u32;
            lower.launch(&kernel, [n, 1, 1], &[Bind::Input(0), Bind::Output(0)])
        }))
        .build()
        .unwrap()
}

fn step(op: &ext::Op, x: &Array<f32>) -> Array<f32> {
    let y = op.apply1::<f32>(&[&(x * 2.0 - 0.5)], &[]).unwrap();
    (&y * 0.5).sqrt().maximum(&y * 0.1) + x
}

fn data(ctx: &Context, seed: f32) -> Array<f32> {
    Array::from_slice_on(ctx, &(0..2000).map(|i| (i as f32 * 0.37 + seed).sin()).collect::<Vec<_>>(), 2000).unwrap()
}

#[test]
fn a_custom_op_in_a_replayed_step_matches_eager() {
    let ctx = Context::new().unwrap();
    let op = op(&ctx);
    let compiled = {
        let op = op.clone();
        trace(move |inp| step(&op, &inp.get::<f32>(0)).into())
    };
    for seed in [1.0f32, 2.0, 3.0] {
        let x = data(&ctx, seed);
        let replayed = compiled.call1::<f32>(&[&x]).unwrap().to_vec().unwrap();
        let eager = step(&op, &x).to_vec().unwrap();
        assert_eq!(replayed.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), eager.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "seed {seed}");
    }
    let stats = compiled.stats();
    assert_eq!((stats.calls, stats.traces), (3, 1), "{stats:?}");
}
