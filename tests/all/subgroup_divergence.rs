//! Subgroup operations after divergence. A loop whose lanes run different numbers of iterations
//! (`spin`) and then a ballot, reduction, broadcast or shuffle, all inside an outer loop that
//! exits on the ballot: each iteration the operation must see every lane of the subgroup that
//! is still in the outer loop, however the lanes left the inner one. On CUDA a subgroup operation
//! that takes `activemask` as its member mask instead sees only the lanes that have met again so far.

use luma::*;

fn ctx() -> Context {
    default_context().expect("Could not create Context.")
}

const SRC: &str = r#"
@group(1) @binding(0) var<storage, read> spin: array<u32>;
@group(1) @binding(1) var<storage, read> due: array<u32>;
@group(1) @binding(2) var<storage, read_write> out: array<u32>;
@group(1) @binding(3) var<storage, read_write> sink: array<u32>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(subgroup_invocation_id) lane: u32,
    @builtin(subgroup_size) sg: u32,
) {
    let g = gid.x;
    let full = min(sg, 32u);
    var i = 0u;
    var cnt = 0u;
    var add = 0u;
    var mx = 0u;
    var shuf = 0u;
    var br_add = 0u;
    var br_cnt = 0u;
    var acc = g;
    loop {
        // The lanes leave this loop at different iterations.
        var j = 0u;
        let n = spin[g] + i;
        loop {
            if (j >= n) { break; }
            acc = acc * 1664525u + j + 1013904223u;
            j++;
        }
        let on = i >= due[g];
        let b = subgroupBallot(on).x;
        cnt += countOneBits(b);
        add += subgroupAdd(select(0u, lane + 1u, on));
        mx += subgroupMax(select(0u, lane + 1u, on));
        shuf += subgroupShuffle(due[g], (lane + 1u) % sg);
        // Only the lanes with an even `due` are here: the others are on the other side of the branch.
        if ((due[g] & 1u) == 0u) {
            br_add += subgroupAdd(1u);
            br_cnt += countOneBits(subgroupBallot(true).x);
        }
        // The exit depends on the ballot, which is the same in every lane.
        if (countOneBits(b) == full || i >= 5u) { break; }
        i++;
    }
    let first = subgroupBroadcastFirst(due[g]);
    out[g * 10u + 0u] = cnt;
    out[g * 10u + 1u] = add;
    out[g * 10u + 2u] = mx;
    out[g * 10u + 3u] = i;
    out[g * 10u + 4u] = shuf;
    out[g * 10u + 5u] = first;
    out[g * 10u + 6u] = sg;
    out[g * 10u + 7u] = lane;
    out[g * 10u + 8u] = br_add;
    out[g * 10u + 9u] = br_cnt;
    sink[g] = acc;
}
"#;

/// What each invocation must write, from the subgroup (`sg` invocations from `g - lane`) it is in.
fn expected(due: &[u32], g: usize, lane: usize, sg: usize) -> [u32; 8] {
    let base = g - lane;
    let full = sg.min(32);
    let (mut cnt, mut add, mut mx, mut shuf, mut br_add, mut br_cnt) =
        (0u32, 0u32, 0u32, 0u32, 0u32, 0u32);
    let even = |l: usize| due[base + l] & 1 == 0;
    let mut i = 0u32;
    loop {
        let on = |l: usize| i >= due[base + l];
        let c = (0..full).filter(|&l| on(l)).count();
        cnt += c as u32;
        add += (0..sg)
            .filter(|&l| on(l))
            .map(|l| l as u32 + 1)
            .sum::<u32>();
        mx += (0..sg)
            .filter(|&l| on(l))
            .map(|l| l as u32 + 1)
            .max()
            .unwrap_or(0);
        shuf += due[base + (lane + 1) % sg];
        if even(lane) {
            br_add += (0..sg).filter(|&l| even(l)).count() as u32;
            br_cnt += (0..full).filter(|&l| even(l)).count() as u32;
        }
        if c == full || i >= 5 {
            break;
        }
        i += 1;
    }
    [cnt, add, mx, i, shuf, due[base], br_add, br_cnt]
}

fn run(groups: usize, seed: u64, max_spin: u64) {
    let ctx = ctx();
    if !ctx.features().contains(Features::SUBGROUP) {
        eprintln!("skipped: the device has no subgroup operations");
        return;
    }
    eprintln!("device: {}", ctx.device_name());
    let n = groups * 64;
    let mut state = seed;
    let mut next = move |m: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) % m) as u32
    };
    let spin: Vec<u32> = (0..n).map(|_| 1 + next(max_spin)).collect();
    let due: Vec<u32> = (0..n).map(|_| next(6)).collect();
    let spin_a = Array::from_slice(&spin, [n, 1, 1, 1]).unwrap();
    let due_a = Array::from_slice(&due, [n, 1, 1, 1]).unwrap();
    let mut out = Array::<u32>::zeros([n * 10, 1, 1, 1]).unwrap();
    let mut sink = Array::<u32>::zeros([n, 1, 1, 1]).unwrap();
    let kernel = ctx
        .kernel(SRC)
        .label("subgroup_divergence")
        .build()
        .unwrap();
    kernel
        .launch_raw(
            [groups as u32, 1, 1],
            args![&spin_a, &due_a, &mut out, &mut sink],
        )
        .unwrap();
    let out = out.to_vec().unwrap();
    let mut bad = 0;
    for g in 0..n {
        let (sg, lane) = (out[g * 10 + 6] as usize, out[g * 10 + 7] as usize);
        assert!(
            sg.is_power_of_two()
                && (4..=128).contains(&sg)
                && lane < sg
                && g % 64 / sg * sg + lane == g % 64,
            "invocation {g}: subgroup size {sg}, lane {lane}"
        );
        let want = expected(&due, g, lane, sg);
        let got: [u32; 8] = [0, 1, 2, 3, 4, 5, 8, 9].map(|k| out[g * 10 + k]);
        if got != want {
            if bad < 8 {
                eprintln!(
                    "invocation {g} (lane {lane} of {sg}): got {got:?}, want {want:?} (cnt, add, max, iterations, shuffle, first, branch add, branch ballot)"
                );
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{bad} of {n} invocations differ");
}

#[test]
fn subgroup_operations_after_divergent_loops_see_every_lane_small() {
    run(1, 1, 150);
    run(3, 2, 150);
    run(2, 3, 20000);
}

#[test]
fn subgroup_operations_after_divergent_loops_see_every_lane_large() {
    for seed in 0..4 {
        run(4096, 10 + seed, 150);
        run(256, 20 + seed, 3000);
    }
}

/// The shape of a decoupled look-back: the first warp of the workgroup, some lanes of which wait
/// (`spin`) before everyone takes a ballot and a sum.
const LOOKBACK: &str = r#"
@group(1) @binding(0) var<storage, read> spin: array<u32>;
@group(1) @binding(1) var<storage, read> due: array<u32>;
@group(1) @binding(2) var<storage, read_write> out: array<u32>;
@group(1) @binding(3) var<storage, read_write> sink: array<u32>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) t: u32,
    @builtin(subgroup_invocation_id) lane: u32,
    @builtin(subgroup_size) sg: u32,
) {
    let g = gid.x;
    var acc = g;
    var r = vec4<u32>(0u, 0u, 0u, 0u);
    if (t < 32u) {
        if (lane < due[g] + 2u) {
            var j = 0u;
            loop {
                if (j >= spin[g]) { break; }
                acc = acc * 1664525u + j + 1013904223u;
                j++;
            }
        }
        r = vec4<u32>(countOneBits(subgroupBallot((due[g] & 1u) == 0u).x), subgroupAdd(due[g]), subgroupMax(due[g] + lane), sg);
    }
    out[g * 4u + 0u] = r.x;
    out[g * 4u + 1u] = r.y;
    out[g * 4u + 2u] = r.z;
    out[g * 4u + 3u] = r.w;
    sink[g] = acc;
}
"#;

fn run_lookback(groups: usize, seed: u64, max_spin: u64) {
    let ctx = ctx();
    if !ctx.features().contains(Features::SUBGROUP) {
        eprintln!("skipped: the device has no subgroup operations");
        return;
    }
    let n = groups * 64;
    let mut state = seed;
    let mut next = move |m: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) % m) as u32
    };
    let spin: Vec<u32> = (0..n).map(|_| 1 + next(max_spin)).collect();
    let due: Vec<u32> = (0..n).map(|_| next(40)).collect();
    let spin_a = Array::from_slice(&spin, [n, 1, 1, 1]).unwrap();
    let due_a = Array::from_slice(&due, [n, 1, 1, 1]).unwrap();
    let mut out = Array::<u32>::zeros([n * 4, 1, 1, 1]).unwrap();
    let mut sink = Array::<u32>::zeros([n, 1, 1, 1]).unwrap();
    let kernel = ctx
        .kernel(LOOKBACK)
        .label("subgroup_lookback")
        .build()
        .unwrap();
    kernel
        .launch_raw(
            [groups as u32, 1, 1],
            args![&spin_a, &due_a, &mut out, &mut sink],
        )
        .unwrap();
    let out = out.to_vec().unwrap();
    let mut bad = 0;
    for g in 0..n {
        let got = [out[g * 4], out[g * 4 + 1], out[g * 4 + 2], out[g * 4 + 3]];
        let t = g % 64;
        let want = if t < 32 {
            let sg = got[3] as usize;
            assert!(
                sg.is_power_of_two() && (4..=32).contains(&sg),
                "subgroup size {sg}"
            );
            let base = g - t % sg;
            let lanes = base..base + sg;
            [
                lanes.clone().filter(|&m| due[m] & 1 == 0).count() as u32,
                lanes.clone().map(|m| due[m]).sum(),
                lanes
                    .clone()
                    .map(|m| due[m] + (m - base) as u32)
                    .max()
                    .unwrap(),
                sg as u32,
            ]
        } else {
            [0; 4]
        };
        if got != want {
            if bad < 8 {
                eprintln!(
                    "invocation {g}: got {got:?}, want {want:?} (ballot, sum, max, subgroup size)"
                );
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{bad} of {n} invocations differ");
}

#[test]
fn a_subgroup_operation_after_some_lanes_waited_sees_every_lane() {
    run_lookback(1, 5, 150);
    run_lookback(2, 6, 20000);
    for seed in 0..4 {
        run_lookback(2048, 30 + seed, 3000);
    }
}

/// A decoupled look-back, as the radix sort and the set merge kernels once had it: workgroups
/// (tiles, numbered by a ticket in the order they start) publish their count, and the first warp
/// of each adds the counts of the tiles before its own, a warp-wide step at a time. The lanes wait
/// (`spin`) for the statuses of different tiles, and the ballot and the sum that follow take
/// them all: a lane that runs ahead of the others after its wait adds up only part of the tiles.
const LOOK_BACK: &str = r#"
@group(1) @binding(0) var<storage, read> counts: array<u32>;
@group(1) @binding(1) var<storage, read_write> st: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> out: array<u32>;
var<workgroup> ticket: u32;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_index) t: u32,
    @builtin(subgroup_invocation_id) lane: u32,
    @builtin(subgroup_size) sgs: u32,
) {
    let tiles = arrayLength(&counts);
    if (t == 0u) { ticket = atomicAdd(&st[tiles], 1u); }
    TICKET
    let mine = counts[tl];
    if (t == 0u) {
        // Flag 1: the aggregate; flag 2: the inclusive prefix (tile 0 has nothing before it).
        atomicStore(&st[tl], select((1u << 30u) | mine, (2u << 30u) | mine, tl == 0u));
    }
    if (t < 32u) {
        let width = min(sgs, 32u);
        var excl = 0u;
        var j = tl;
        loop {
            if (j == 0u) { break; }
            var w = 0u;
            if (lane < j) {
                w = atomicLoad(&st[j - 1u - lane]);
                while (w == 0u) { w = atomicLoad(&st[j - 1u - lane]); }
            }
            let inclusive = subgroupBallot(lane < j && (w >> 30u) == 2u).x;
            var upto = width;
            if (inclusive != 0u) { upto = firstTrailingBit(inclusive) + 1u; }
            excl += subgroupAdd(select(0u, w & 0x3fffffffu, lane < upto && lane < j));
            if (inclusive != 0u) { break; }
            j -= width;
        }
        if (t == 0u) {
            atomicStore(&st[tl], (2u << 30u) | (excl + mine));
            out[tl] = excl;
        }
    }
}
"#;

fn look_back(tiles: usize, seed: u64, uniform_load: bool) {
    let ctx = ctx();
    if !ctx.features().contains(Features::SUBGROUP) {
        eprintln!("skipped: the device has no subgroup operations");
        return;
    }
    let mut state = seed;
    let counts: Vec<u32> = (0..tiles)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            1 + ((state >> 33) % 1000) as u32
        })
        .collect();
    let counts_a = Array::from_slice(&counts, [tiles, 1, 1, 1]).unwrap();
    let mut st = Array::<u32>::zeros([tiles + 1, 1, 1, 1]).unwrap();
    let mut out = Array::<u32>::zeros([tiles, 1, 1, 1]).unwrap();
    // The ticket is a plain workgroup read (`tl` is not provably the same in every lane) or `workgroupUniformLoad`.
    let ticket = if uniform_load {
        "let tl = workgroupUniformLoad(&ticket);"
    } else {
        "workgroupBarrier();\n    let tl = ticket;"
    };
    let kernel = ctx
        .kernel(&LOOK_BACK.replace("TICKET", ticket))
        .label("subgroup_look_back")
        .build()
        .unwrap();
    kernel
        .launch_raw([tiles as u32, 1, 1], args![&counts_a, &mut st, &mut out])
        .unwrap();
    let out = out.to_vec().unwrap();
    // The tiles are numbered by the order their workgroups start, so `out[tile]` is the sum of the
    // counts before it.
    let mut want = Vec::with_capacity(tiles);
    let mut sum = 0u32;
    for c in &counts {
        want.push(sum);
        sum += c;
    }
    let bad: Vec<usize> = (0..tiles).filter(|&i| out[i] != want[i]).collect();
    assert!(
        bad.is_empty(),
        "{} of {tiles} tiles differ, first {:?} (got {:?}, want {:?})",
        bad.len(),
        &bad[..bad.len().min(4)],
        bad.iter().take(4).map(|&i| out[i]).collect::<Vec<_>>(),
        bad.iter().take(4).map(|&i| want[i]).collect::<Vec<_>>()
    );
}

/// `sizes` tiles at a time; the large ones run beyond what a GPU keeps resident, so that tiles
/// wait for the statuses of others for a while.
fn look_backs(uniform_load: bool, sizes: &[usize]) {
    for (seed, &tiles) in sizes.iter().enumerate() {
        look_back(tiles, 1 + seed as u64, uniform_load);
    }
}

#[test]
fn a_look_back_adds_every_tile_small() {
    look_backs(false, &[1, 2, 33]);
    look_backs(true, &[1, 2, 33]);
}

#[test]
fn a_look_back_adds_every_tile_medium() {
    look_backs(false, &[1200]);
    look_backs(true, &[1200]);
}

#[test]
fn a_look_back_adds_every_tile_large() {
    look_backs(false, &[5000; 4]);
}

#[test]
fn a_look_back_on_a_workgroup_uniform_load_adds_every_tile_large() {
    look_backs(true, &[5000; 4]);
}
