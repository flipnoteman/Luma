//! Slabs below segments (`LUMA_SLAB_BYTES`, W3-P4): a stage runs its steps over slabs shorter than the segments, binding each by its
//! offset. Every test compares with the same program on a context whose arrays fit one buffer, bit for bit.

use luma::{select, Array, Context, ContextOptions, SplitMode};

/// A context of 4096-byte segments (1024 `f32`), deferring in slabs of `slab` bytes.
fn slabbed(slab: Option<u64>) -> Context {
    Context::with_options(ContextOptions::from_env().chunk_bytes(4096).split(SplitMode::Auto).slab_bytes(slab)).expect("a device is available")
}

/// The same arrays in one segment each.
fn whole() -> Context {
    Context::with_options(ContextOptions::from_env().split(SplitMode::Off)).expect("a device is available")
}

const N: usize = 5 * 1024 + 17;

fn floats() -> Vec<f32> {
    (0..N).map(|i| ((i * 37) % 101) as f32 * 0.125 - 5.0).collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Elementwise programs over `f32`, `u32`, `i32` and `bool`, with scalars and selects, and programs with shifts.
fn programs(ctx: &Context) -> Vec<Vec<u32>> {
    let xs = floats();
    let us: Vec<u32> = (0..N as u32).map(|i| i.wrapping_mul(2_654_435_761)).collect();
    let is: Vec<i32> = us.iter().map(|&u| u as i32 >> 7).collect();
    let x = Array::from_slice_on(ctx, &xs, N).unwrap();
    let u = Array::from_slice_on(ctx, &us, N).unwrap();
    let i = Array::from_slice_on(ctx, &is, N).unwrap();
    let a = select(&x.gt(0.5), &x * 3.0 - 1.0, x.abs() + 0.25).to_vec().unwrap();
    let b = ((&u ^ (&u >> 3u32)) + 7u32).to_vec().unwrap();
    let c = (&i * 3 - (&i >> 2)).to_vec().unwrap();
    let d = (&x * 2.0 + 1.0).sqrt().to_vec().unwrap();
    let flags = x.gt(0.0).to_f32().to_vec().unwrap();
    let halo = (x.shift(1) + x.shift(-1) - &x * 2.0).to_vec().unwrap();
    let far = (x.shift(1500) * 0.5 + x.shift(-4097)).to_vec().unwrap();
    vec![bits(&a), b, c.iter().map(|&v| v as u32).collect(), bits(&d), bits(&flags), bits(&halo), bits(&far)]
}

#[test]
fn slab_size_does_not_change_maps_or_halos() {
    let want = programs(&whole());
    for slab in [None, Some(1 << 8), Some(1 << 10), Some(1 << 14), Some(1 << 20)] {
        let ctx = slabbed(slab);
        assert_eq!(programs(&ctx), want, "slab_bytes {slab:?}");
        assert!(ctx.stream_stats().schedules > 0);
    }
}

#[test]
fn slabs_below_segments_are_counted() {
    // 64-element slabs over six segments of 1024: the stage runs `ceil(N / 64)` passes.
    let ctx = slabbed(Some(256));
    let x = Array::from_slice_on(&ctx, &floats(), N).unwrap();
    let _ = ((&x * 2.0 + 1.0).sqrt() - &x).abs().to_vec().unwrap();
    let stats = ctx.stream_stats();
    assert_eq!(stats.stages, 1, "{stats:?}");
    assert_eq!(stats.slabs, N.div_ceil(64) as u64, "{stats:?}");
    // Transients get a buffer of a slab, not of a segment.
    let long = slabbed(Some(1 << 20));
    let x = Array::from_slice_on(&long, &floats(), N).unwrap();
    let _ = ((&x * 2.0 + 1.0).sqrt() - &x).abs().to_vec().unwrap();
    assert_eq!(long.stream_stats().slabs, N.div_ceil(1024) as u64);
}

#[test]
fn the_slab_decision_is_recorded_once_per_stage() {
    let ctx = slabbed(Some(1 << 10));
    ctx.record_decisions(true);
    let x = Array::from_slice_on(&ctx, &floats(), N).unwrap();
    let _ = (&x * 2.0).abs().to_vec().unwrap();
    let slab: Vec<_> = ctx.take_decisions().into_iter().filter(|d| d.site == luma::DecisionSite::Slab).collect();
    assert_eq!(slab.len(), 1, "{slab:?}");
    assert!(slab[0].chosen.starts_with("L=2^8 "), "{}", slab[0].chosen);
}

/// Arrays of `f32` (1024 to a segment) with `Df32` (512): the grid is in elements, so a slab lies in one segment of each.
fn mixed(ctx: &Context) -> Vec<Vec<u32>> {
    let xs = floats();
    let x = Array::from_slice_on(ctx, &xs, N).unwrap();
    let d = (x.to_df32() * x.to_df32() + x.to_df32()).to_vec_array();
    let wide = (&d + x.to_df32()).to_f32().to_vec().unwrap();
    let back = (&d * x.to_df32()).to_vec().unwrap().iter().map(|v| v.to_f64() as f32).collect::<Vec<_>>();
    let narrow = (&d - &d.abs()).to_f32().to_vec().unwrap();
    let flags = select(&x.gt(0.0), d.to_f32(), x.abs()).to_vec().unwrap();
    vec![bits(&wide), bits(&back), bits(&narrow), bits(&flags)]
}

trait Stored {
    fn to_vec_array(self) -> Array<luma::Df32>;
}

impl Stored for Array<luma::Df32> {
    /// Evaluates the array into storage so that later programs read a stored array of 8-byte elements.
    fn to_vec_array(self) -> Array<luma::Df32> {
        self.eval().unwrap();
        self
    }
}

#[test]
fn mixed_element_sizes_share_a_slab_grid() {
    let want = mixed(&whole());
    for slab in [None, Some(1 << 8), Some(1 << 12)] {
        assert_eq!(mixed(&slabbed(slab)), want, "slab_bytes {slab:?}");
    }
    // Kernel-major launches (no schedules) over the same arrays: the old refusal is gone there too.
    let eager = Context::with_options(ContextOptions::from_env().chunk_bytes(4096).split(SplitMode::Off)).unwrap();
    assert_eq!(mixed(&eager), want);
}

#[test]
fn a_shift_larger_than_a_slab_gets_a_ghost_as_big_as_the_slab() {
    // Slabs of 64 over 1024-element segments, and a shift of 1500: the planner would not shrink slabs below the reach of a shift (L_hard),
    // the test hook does. The program also reads `x` at its own element, so the window is the body and both strips of the circular
    // shift (`i - 1500` and `i + N - 1500`, each a slab long where it exists) are ghost.
    let ctx = slabbed(None);
    ctx.set_slab_len(Some(64));
    let xs = floats();
    let x = Array::from_slice_on(&ctx, &xs, N).unwrap();
    let got = (x.shift(1500) + &x).to_vec().unwrap();
    let want = {
        let off = whole();
        let x = Array::from_slice_on(&off, &xs, N).unwrap();
        (x.shift(1500) + &x).to_vec().unwrap()
    };
    assert_eq!(bits(&got), bits(&want));
    let stats = ctx.stream_stats();
    assert_eq!(stats.slabs, N.div_ceil(64) as u64, "{stats:?}");
    let mut expected = 0;
    for a in (0..N).step_by(64) {
        let b = (a + 64).min(N);
        for delta in [-1500i64, N as i64 - 1500] {
            let (lo, hi) = ((a as i64 + delta).max(0), (b as i64 - 1 + delta).min(N as i64 - 1));
            if lo <= hi {
                expected += (hi - lo + 1) as u64 * 4;
            }
        }
    }
    assert_eq!(stats.ghost_bytes, expected, "{stats:?}");
}

#[test]
fn ghost_strips_are_exact_at_the_array_ends() {
    // 1-D and 2-D circular shifts, larger than a slab or not, at slab sizes of 64, 256 and 1024 elements.
    let xs: Vec<f32> = (0..101 * 99).map(|i| ((i * 13) % 251) as f32 * 0.5).collect();
    let run = |ctx: &Context| -> Vec<Vec<u32>> {
        let flat = Array::from_slice_on(ctx, &floats(), N).unwrap();
        let grid = Array::from_slice_on(ctx, &xs, [101usize, 99]).unwrap();
        let mut out = Vec::new();
        for s in [1isize, -1, 63, 64, 65, 1023, 1025, 2000, -4097, (N / 2) as isize] {
            out.push(bits(&flat.shift(s).to_vec().unwrap()));
        }
        for s in [[1isize, 0], [0, -1], [3, 5], [50, 49], [-100, 98]] {
            out.push(bits(&grid.shift(s).to_vec().unwrap()));
        }
        let stencil = grid.shift([1, 0]) + grid.shift([-1, 0]) + grid.shift([0, 1]) + grid.shift([0, -1]) - &grid * 4.0;
        out.push(bits(&stencil.to_vec().unwrap()));
        out
    };
    let want = run(&whole());
    for slab in [Some(256), Some(1 << 10), Some(1 << 12)] {
        assert_eq!(run(&slabbed(slab)), want, "slab_bytes {slab:?}");
    }
}

#[test]
fn views_with_every_edge_are_exact_over_slabs() {
    use luma::PadMode;
    let run = |ctx: &Context| -> Vec<Vec<u32>> {
        let x = Array::from_slice_on(ctx, &floats(), N).unwrap();
        let mut out = Vec::new();
        for s in [5isize, -300, 600, -2000] {
            out.push(bits(&x.shift_clamp(s).to_vec().unwrap()));
        }
        out.push(bits(&x.shift_fill(5, -2.0).to_vec().unwrap()));
        out.push(bits(&x.shift_fill(-1700, 3.0).to_vec().unwrap()));
        out.push(bits(&x.flip(0).to_vec().unwrap()));
        out.push(bits(&x.slice(0, 100..N - 100).to_vec().unwrap()));
        for mode in [PadMode::Edge, PadMode::Mirror, PadMode::Wrap, PadMode::Zero] {
            out.push(bits(&x.pad([(40, 70)], mode).to_vec().unwrap()));
        }
        // A view of a lazy expression is evaluated first.
        let lazy = (&x * 2.0 + 1.0).abs();
        out.push(bits(&(lazy.slice(0, 1..) - lazy.slice(0, ..N - 1)).to_vec().unwrap()));
        out
    };
    let want = run(&whole());
    for slab in [None, Some(256), Some(1 << 10)] {
        assert_eq!(run(&slabbed(slab)), want, "slab_bytes {slab:?}");
    }
}

#[test]
fn a_pad_and_slice_ghost_is_the_padding_not_a_segment() {
    // `pad(16, 16)` then `slice(0..N)` reads `x[clamp(i - 16)]`: a slab of 64 reads 64 elements, 16 of them before the window.
    for len in [64, 512] {
        pad_ghost(len);
    }
}

fn pad_ghost(len: usize) {
    use luma::PadMode;
    let ctx = slabbed(None);
    ctx.set_slab_len(Some(len));
    let x = Array::from_slice_on(&ctx, &floats(), N).unwrap();
    let got = (x.pad([(16, 16)], PadMode::Edge).slice(0, 0..N) + &x).to_vec().unwrap();
    let off = whole();
    let y = Array::from_slice_on(&off, &floats(), N).unwrap();
    assert_eq!(bits(&got), bits(&(y.pad([(16, 16)], PadMode::Edge).slice(0, 0..N) + &y).to_vec().unwrap()));
    let stats = ctx.stream_stats();
    // The pad (N + 32 elements) and the slice each run a stage.
    let slabs = (N + 32).div_ceil(len) as u64 + N.div_ceil(len) as u64;
    assert_eq!(stats.slabs, slabs, "{stats:?}");
    assert!(stats.ghost_bytes <= slabs * 32 * 4, "{stats:?}");
}

/// A 3-point stencil of an expression: the expression is the producer of the halo edge.
fn stencil_of_expression(ctx: &Context) -> Vec<u32> {
    let x = Array::from_slice_on(ctx, &floats(), N).unwrap();
    // The expression's handle is gone when the program runs: nothing can read the producer's output but the stage.
    let out = {
        let e = (&x * 2.0 + 1.0).abs();
        e.shift(1) + e.shift(-1) - &e * 2.0
    };
    bits(&out.to_vec().unwrap())
}

#[test]
fn expansion_recomputes_only_the_halo() {
    let want = stencil_of_expression(&whole());
    let slabs = N.div_ceil(64) as u64;
    let ctx = slabbed(None);
    ctx.set_slab_len(Some(64));
    assert_eq!(stencil_of_expression(&ctx), want);
    let stats = ctx.stream_stats();
    // One stage: the producer runs over the body and its halos per slab (the wrap strips of the ends make the first and last slab
    // read the far end of the array), at most three regions each.
    assert_eq!((stats.stages, stats.slabs), (1, slabs), "{stats:?}");
    assert!(stats.producer_regions >= slabs && stats.producer_regions <= 3 * slabs, "{stats:?}");
    assert_eq!(stats.transients, 1, "{stats:?}");
    // The cut form (the producer's stage runs over all slabs first) is bit-equal.
    let cut = slabbed(None);
    cut.set_slab_len(Some(64));
    cut.set_slab_expand(false);
    assert_eq!(stencil_of_expression(&cut), want);
    let stats = cut.stream_stats();
    assert_eq!((stats.stages, stats.producer_regions), (2, 0), "{stats:?}");
}

#[test]
fn a_halo_as_far_as_the_slab_is_cut() {
    // A shift by 600 over slabs of 256 reads a strip of its own, as long as the body: recomputing it doubles the producer's work.
    let run = |ctx: &Context| {
        let x = Array::from_slice_on(ctx, &floats(), N).unwrap();
        let out = {
            let e = (&x * 2.0 + 1.0).abs();
            e.shift(600) - &e
        };
        bits(&out.to_vec().unwrap())
    };
    let want = run(&whole());
    let ctx = slabbed(None);
    ctx.set_slab_len(Some(256));
    assert_eq!(run(&ctx), want);
    assert_eq!(ctx.stream_stats().producer_regions, 0);
}

#[test]
fn a_chain_of_stencils_is_cut_not_expanded() {
    let run = |ctx: &Context| {
        let x = Array::from_slice_on(ctx, &floats(), N).unwrap();
        let out = {
            let e = (&x * 2.0 + 1.0).abs();
            let f = e.shift(1) + e.shift(-1);
            f.shift(2) - &f
        };
        bits(&out.to_vec().unwrap())
    };
    let want = run(&whole());
    let ctx = slabbed(None);
    ctx.set_slab_len(Some(64));
    assert_eq!(run(&ctx), want);
}

#[test]
fn an_expression_read_through_a_shrinking_view_is_a_global_edge() {
    let run = |ctx: &Context| {
        let x = Array::from_slice_on(ctx, &floats(), N).unwrap();
        let out = {
            let e = (&x * 2.0 + 1.0).abs();
            e.slice(0, 1..) - e.slice(0, ..N - 1)
        };
        bits(&out.to_vec().unwrap())
    };
    let want = run(&whole());
    let ctx = slabbed(None);
    ctx.set_slab_len(Some(64));
    assert_eq!(run(&ctx), want);
    assert_eq!(ctx.stream_stats().producer_regions, 0);
}
