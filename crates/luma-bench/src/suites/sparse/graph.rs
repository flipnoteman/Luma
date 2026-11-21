//! Graph workloads on edge lists: PageRank, breadth-first search, connected components, shortest
//! paths, degree statistics of a bipartite graph and k-hop neighbourhood sizes.
//!
//! The graphs are power-law multigraphs of `N` nodes and `8 N` edges (see `data`), `N` being 1M on a
//! GPU, 256K on the CPU backend and 64K with `--quick`; the k-hop and grid cases use smaller ones.

use luma::*;

use super::data::{self, Csr, Graph};
use super::*;

type A = Array<f32>;
type U = Array<u32>;

struct Dev {
    src: U,
    dst: U,
}

fn upload(h: &Harness, g: &Graph) -> Dev {
    Dev { src: up_u32(h, &g.src, g.m()), dst: up_u32(h, &g.dst, g.m()) }
}

pub fn run(h: &mut Harness) {
    let n = scale(h, 65_536, 262_144, 1_000_000);
    if ["pagerank", "bfs", "cc", "sssp"].iter().any(|g| wants(h, g)) {
        let g = data::power_law_graph(n, 8 * n, 61);
        h.guarded("pagerank", |h| pagerank(h, &g));
        h.guarded("bfs", |h| bfs(h, &g));
        h.guarded("cc", |h| cc_cases(h, "powerlaw", &g));
        h.guarded("sssp", |h| sssp_cases(h, "powerlaw", &g));
    }
    h.guarded("cc", |h| cc_cases(h, "sparse", &data::random_graph(n, n + n / 5, 62)));
    if ["cc", "sssp"].iter().any(|g| wants(h, g)) {
        let grid = data::grid_graph(scale(h, 128, 256, 512), 63);
        h.guarded("cc", |h| cc_cases(h, "grid", &grid));
        h.guarded("sssp", |h| sssp_cases(h, "grid", &data::symmetrize(&grid)));
    }
    h.guarded("degree", degree_cases);
    h.guarded("khop", khop_cases);
}

// -------------------------------------------------------------------------------------------------
// PageRank
// -------------------------------------------------------------------------------------------------

const DAMPING: f32 = 0.85;
const ITERS: usize = 20;

/// The transition matrix as CSR by destination (row `v` lists the sources `u` of the edges into
/// `v` with `1 / outdeg(u)`), for the pull form `r' = d P r + (1 - d) / n`.
fn pull_matrix(g: &Graph) -> Csr {
    let outdeg = data::degrees(g.n, &g.src);
    let (ptr, col) = data::adjacency(&Graph { n: g.n, src: g.dst.clone(), dst: g.src.clone() });
    let val = col.iter().map(|&u| 1.0 / outdeg[u as usize] as f32).collect();
    Csr { rows: g.n, cols: g.n, ptr, col, val }
}

fn pagerank(h: &mut Harness, g: &Graph) {
    let n = g.n;
    let want: Vec<f32> = to_f32(&data::pagerank_host(g, f64::from(DAMPING), ITERS));
    let sorted = data::sort_edges(g, &g.dst);
    let ctx = h.ctx.clone();
    let tele = (1.0 - DAMPING) / n as f32;
    let uniform = vec![1.0 / n as f32; n];

    // Push form: gather the scaled rank `c = r / outdeg` of the source, add it at the destination.
    for (tag, graph) in [("push_random", g), ("push_dst_sorted", &sorted)] {
        let name = format!("pagerank/{tag}");
        if !h.want(&name) {
            continue;
        }
        let dev = upload(h, graph);
        let outdeg = bincount(&dev.src, n);
        let inv = (1.0f32 / outdeg.to_f32()).cache();
        inv.eval().unwrap();
        let c0 = &up_f32(h, &uniform, n) * &inv;
        c0.eval().unwrap();
        let go = || {
            let [c] = iterate_all([&c0], ITERS, |[c]| {
                let mut acc = Array::<f32>::zeros_on(&ctx, n).unwrap();
                acc.put_add(&dev.dst, c.take(&dev.src)).unwrap();
                [(acc * DAMPING + tele) * &inv]
            });
            c
        };
        check(&name, &(go() / &inv).to_vec().unwrap(), &want, 5e-4, 1e-12);
        let size = format!("{} nodes {} edges x{ITERS}", label(n), label(g.m()));
        case_trim(h, &name, &size, Work::Items((g.m() * ITERS) as f64, "Medge/s"), &mut || go().eval().unwrap());
    }

    // Pull form: the sparse product with the transition matrix by destination, rows split in pieces.
    let name = "pagerank/pull_split";
    if h.want(name) {
        let p = pull_matrix(g);
        let pieces = spmv::upload_pieces(h, &p, 16);
        let r0 = up_f32(h, &uniform, n);
        let go = || {
            let [r] = iterate_all([&r0], ITERS, |[r]| [spmv::split(&pieces, r) * DAMPING + tele]);
            r
        };
        check(name, &go().to_vec().unwrap(), &want, 5e-4, 1e-12);
        let size = format!("{} nodes {} edges x{ITERS}", label(n), label(g.m()));
        case_trim(h, name, &size, Work::Items((g.m() * ITERS) as f64, "Medge/s"), &mut || go().eval().unwrap());
    }
}

// -------------------------------------------------------------------------------------------------
// BFS
// -------------------------------------------------------------------------------------------------

const INF: u32 = u32::MAX;

/// One dense level: every edge whose source is at distance `lvl` offers `lvl + 1` to its target.
fn bfs_level(dist: &U, lvl: &U, dev: &Dev) -> U {
    let mut next = dist.copy();
    let offer = select(&dist.take(&dev.src).eq(lvl), lvl + 1u32, INF);
    next.put_min(&dev.dst, offer).unwrap();
    next
}

fn bfs(h: &mut Harness, g: &Graph) {
    let n = g.n;
    let (ptr, col) = data::adjacency(g);
    let outdeg = data::degrees(n, &g.src);
    let source = (0..n).max_by_key(|&v| outdeg[v]).unwrap();
    let want = data::bfs_host(&ptr, &col, source, INF);
    let depth = want.iter().copied().filter(|&d| d != INF).max().unwrap();
    // Edges an ideal search looks at: the out-edges of every reached node.
    let reached_edges: usize = (0..n).filter(|&v| want[v] != INF).map(|v| outdeg[v] as usize).sum();
    let size = format!("{} nodes {} edges depth {depth}", label(n), label(g.m()));
    let work = Work::Items(reached_edges as f64, "Medge/s");
    let ctx = h.ctx.clone();
    let mut start = vec![INF; n];
    start[source] = 0;
    let init = up_u32(h, &start, n);
    let dev = upload(h, g);

    let name = "bfs/dense_sync";
    if h.want(name) {
        // One level per round trip: the host reads how many nodes the level reached.
        let go = || {
            let mut dist = init.copy();
            let mut lvl = Array::<u32>::zeros_on(&ctx, 1).unwrap();
            loop {
                let next = bfs_level(&dist, &lvl, &dev);
                lvl = &lvl + 1u32;
                lvl.eval().unwrap();
                let fresh = next.eq(&lvl).sum().item().unwrap();
                dist = next;
                if fresh == 0 {
                    return dist;
                }
            }
        };
        exact(name, &go().to_vec().unwrap(), &want);
        case_trim(h, name, &size, work, &mut || go().eval().unwrap());
    }

    let name = "bfs/dense_fixed";
    if h.want(name) {
        // As many levels as the search has, in one device loop with no readback.
        let lvl0 = Array::<u32>::zeros_on(&ctx, 1).unwrap();
        let go = || iterate_all([&init, &lvl0], depth as usize, |[d, l]| [bfs_level(d, l, &dev), l + 1u32]);
        exact(name, &go()[0].to_vec().unwrap(), &want);
        case_trim(h, name, &size, work, &mut || go()[0].eval().unwrap());
    }

    let name = "bfs/frontier";
    if h.want(name) {
        // Frontier compaction: expand only the out-edges of the nodes reached by the last level.
        // `cumsum` of their degrees numbers the edges, `searchsorted` finds each edge's frontier
        // node, a gather through the row pointer reads the neighbour; `nonzero` makes the next
        // frontier. Two readbacks per level (edge count, new frontier).
        let (dptr, dcol) = (up_u32(h, &ptr, n + 1), up_u32(h, &col, g.m()));
        let deg = up_u32(h, &outdeg, n);
        let go = || {
            let mut dist = init.copy();
            let mut frontier = Array::<u32>::from_slice_on(&ctx, &[source as u32], 1).unwrap();
            let mut level = 0u32;
            loop {
                let f = frontier.len();
                let deg_f = deg.take(&frontier);
                let off = deg_f.cumsum(0);
                let total = off.slice(0, f - 1..f).item().unwrap() as usize;
                if total == 0 {
                    return dist;
                }
                let j = Array::<u32>::range_on(&ctx, total, 0).unwrap();
                let owner = off.searchsorted(&j, Side::Right);
                let first = (&off - &deg_f).take(&owner);
                let edge = dptr.take(&frontier.take(&owner)) + (&j - first);
                dist.put_min(&dcol.take(&edge), level + 1).unwrap();
                level += 1;
                match dist.eq(level).nonzero().unwrap() {
                    Some(next) => frontier = next,
                    None => return dist,
                }
            }
        };
        exact(name, &go().to_vec().unwrap(), &want);
        case_trim(h, name, &size, work, &mut || go().eval().unwrap());
    }
}

// -------------------------------------------------------------------------------------------------
// Connected components
// -------------------------------------------------------------------------------------------------

/// Min-label propagation over the edges in both directions (state: labels and the labels before
/// the step; the loop stops on the GPU when a step changes none), optionally with a pointer-jump
/// `label[v] = label[label[v]]` after every step. Returns the components and the steps run.
fn components(dev: &Dev, n: usize, ctx: &Context, jump: bool, max_steps: usize) -> (U, usize) {
    let ids = Array::<u32>::range_on(ctx, n, 0).unwrap();
    ids.eval().unwrap();
    let run = iterate_until_all(
        [ids.clone(), ids],
        Until::new(max_steps).record("changed", |s: &[U; 2]| s[0].ne(&s[1]).sum().to_f32()).stop_when(|h| h["changed"].last().le(0.0)),
        |s, _| {
            let mut l = s[0].copy();
            l.put_min(&dev.dst, s[0].take(&dev.src)).unwrap();
            l.put_min(&dev.src, s[0].take(&dev.dst)).unwrap();
            let l = if jump { l.take(&l) } else { l };
            [l, s[0].copy()]
        },
    )
    .unwrap();
    assert_eq!(run.reason, StopReason::Predicate(0), "label propagation did not converge in {max_steps} steps");
    let [labels, _] = run.state;
    (labels, run.steps)
}

fn cc_cases(h: &mut Harness, tag: &str, g: &Graph) {
    if !wants(h, &format!("cc/{tag}")) {
        return;
    }
    let want = data::components_host(g);
    let comps = want.iter().enumerate().filter(|&(v, &l)| l as usize == v).count();
    let dev = upload(h, g);
    let ctx = h.ctx.clone();
    let max_steps = 8192;
    for (kind, jump) in [("plain", false), ("jump", true)] {
        let name = format!("cc/{tag}_{kind}");
        if !h.want(&name) {
            continue;
        }
        let (labels, steps) = components(&dev, g.n, &ctx, jump, max_steps);
        exact(&name, &labels.to_vec().unwrap(), &want);
        drop(labels);
        let size = format!("{} nodes {} edges {comps} comps x{steps}", label(g.n), label(g.m()));
        // Per step: a gather and a scatter-min for each direction.
        let work = Work::Items((2 * g.m() * steps) as f64, "Medge/s");
        case_trim(h, &name, &size, work, &mut || components(&dev, g.n, &ctx, jump, max_steps).0.eval().unwrap());
    }
}

// -------------------------------------------------------------------------------------------------
// Single-source shortest paths
// -------------------------------------------------------------------------------------------------

/// Bellman-Ford: every round relaxes all edges (`put_min` of `dist[src] + w` at the target, an
/// `f32` atomic minimum) until no distance changes. Weights are multiples of 1/8 in `1/8..=1`, so
/// path lengths are exact in `f32` and the result equals Dijkstra's.
fn bellman_ford(dev: &Dev, weight: &A, n: usize, source: usize, ctx: &Context, max_steps: usize) -> (A, usize) {
    let mut start = vec![f32::INFINITY; n];
    start[source] = 0.0;
    let dist = Array::<f32>::from_slice_on(ctx, &start, n).unwrap();
    let run = iterate_until_all(
        [dist.clone(), dist],
        Until::new(max_steps).record("changed", |s: &[A; 2]| s[0].ne(&s[1]).sum().to_f32()).stop_when(|h| h["changed"].last().le(0.0)),
        |s, _| {
            let mut next = s[0].copy();
            next.put_min(&dev.dst, s[0].take(&dev.src) + weight).unwrap();
            [next, s[0].copy()]
        },
    )
    .unwrap();
    assert_eq!(run.reason, StopReason::Predicate(0), "Bellman-Ford did not converge in {max_steps} rounds");
    let [dist, _] = run.state;
    (dist, run.steps)
}

fn sssp_cases(h: &mut Harness, tag: &str, g: &Graph) {
    let name = format!("sssp/{tag}_bellman_ford");
    if !h.want(&name) {
        return;
    }
    let mut rng = data::Rng::new(64);
    let w: Vec<f32> = (0..g.m()).map(|_| (1 + rng.below(8)) as f32 / 8.0).collect();
    let outdeg = data::degrees(g.n, &g.src);
    let source = (0..g.n).max_by_key(|&v| outdeg[v]).unwrap();
    let want = data::dijkstra_host(g, &w, source);
    let dev = upload(h, g);
    let weight = up_f32(h, &w, g.m());
    let ctx = h.ctx.clone();
    let (dist, steps) = bellman_ford(&dev, &weight, g.n, source, &ctx, 8192);
    exact(&name, &dist.to_vec().unwrap(), &want);
    drop(dist);
    let size = format!("{} nodes {} edges x{steps}", label(g.n), label(g.m()));
    case_trim(h, &name, &size, Work::Items((g.m() * steps) as f64, "Medge/s"), &mut || bellman_ford(&dev, &weight, g.n, source, &ctx, 8192).0.eval().unwrap());
}

// -------------------------------------------------------------------------------------------------
// Degree statistics of a bipartite graph
// -------------------------------------------------------------------------------------------------

const CAP: u32 = 1024;

struct Stats {
    hist_users: U,
    hist_items: U,
    wedges_users: A,
    wedges_items: A,
    nbr_deg: A,
    max_item: A,
}

/// Degrees, capped degree histograms, wedge counts (`sum d (d - 1) / 2` per side: the 2-paths of a
/// triangle-free graph), the mean item degree among each user's items and the largest item degree.
fn stats(users: &U, items: &U, nu: usize, ni: usize) -> Stats {
    let (du, di) = (bincount(users, nu), bincount(items, ni));
    let wedges = |d: &U| {
        let d = d.to_f32();
        (&d * (&d - 1.0f32) * 0.5f32).sum()
    };
    Stats {
        hist_users: bincount(&du.minimum(CAP), CAP as usize + 1),
        hist_items: bincount(&di.minimum(CAP), CAP as usize + 1),
        wedges_users: wedges(&du),
        wedges_items: wedges(&di),
        nbr_deg: di.take(items).to_f32().group_sum(users, nu) / du.to_f32().maximum(1.0f32),
        max_item: di.max().to_f32(),
    }
}

fn degree_cases(h: &mut Harness) {
    if !wants(h, "degree") {
        return;
    }
    let nu = scale(h, 65_536, 262_144, 1_000_000);
    let (ni, m) = (nu / 4, 8 * nu);
    let g = data::bipartite(nu, ni, m, 71);
    let (users, items) = (up_u32(h, &g.src, m), up_u32(h, &g.dst, m));
    let size = format!("{} users {} items {} edges", label(nu), label(ni), label(m));

    let (du, di) = (data::degrees(nu, &g.src), data::degrees(ni, &g.dst));
    let name = "degree/count";
    if h.want(name) {
        let count = || (bincount(&users, nu), bincount(&items, ni));
        let (a, b) = count();
        exact(name, &a.to_vec().unwrap(), &du);
        exact(name, &b.to_vec().unwrap(), &di);
        drop((a, b));
        case_trim(h, name, &size, Work::Items(2.0 * m as f64, "Medge/s"), &mut || {
            let (a, b) = count();
            a.eval().unwrap();
            b.eval().unwrap();
        });
    }

    let name = "degree/stats";
    if h.want(name) {
        let hist = |d: &[u32]| {
            let mut hist = vec![0u32; CAP as usize + 1];
            d.iter().for_each(|&x| hist[x.min(CAP) as usize] += 1);
            hist
        };
        let wedges = |d: &[u32]| d.iter().map(|&x| f64::from(x) * (f64::from(x) - 1.0) / 2.0).sum::<f64>();
        let mut nbr_sum = vec![0.0f64; nu];
        for (&u, &i) in g.src.iter().zip(&g.dst) {
            nbr_sum[u as usize] += f64::from(di[i as usize]);
        }
        let nbr: Vec<f32> = (0..nu).map(|u| (nbr_sum[u] / f64::from(du[u].max(1))) as f32).collect();
        let s = stats(&users, &items, nu, ni);
        exact(name, &s.hist_users.to_vec().unwrap(), &hist(&du));
        exact(name, &s.hist_items.to_vec().unwrap(), &hist(&di));
        check_scalar(name, f64::from(s.wedges_users.item().unwrap()), wedges(&du), 1e-3);
        check_scalar(name, f64::from(s.wedges_items.item().unwrap()), wedges(&di), 1e-3);
        check(name, &s.nbr_deg.to_vec().unwrap(), &nbr, 1e-4, 1e-4);
        assert_eq!(s.max_item.item().unwrap(), *di.iter().max().unwrap() as f32, "{name}: largest item degree");
        drop(s);
        case_trim(h, name, &size, Work::Items(m as f64, "Medge/s"), &mut || {
            let s = stats(&users, &items, nu, ni);
            s.hist_users.eval().unwrap();
            s.hist_items.eval().unwrap();
            s.wedges_users.eval().unwrap();
            s.wedges_items.eval().unwrap();
            s.nbr_deg.eval().unwrap();
            s.max_item.eval().unwrap();
        });
    }
}

// -------------------------------------------------------------------------------------------------
// k-hop neighbourhood sizes
// -------------------------------------------------------------------------------------------------

/// For `B` seeds at once, the number of nodes within `K` hops: the reach of every seed is a row of
/// a `[B, n]` `f32` array (seed fastest), and a hop adds the reach of every edge's source to its
/// target with one `put_add` of `B m` values, then clamps to 0/1.
fn khop_cases(h: &mut Harness) {
    if !wants(h, "khop") {
        return;
    }
    const B: usize = 16;
    const K: usize = 3;
    let n = scale(h, 16_384, 65_536, 262_144);
    let g = data::power_law_graph(n, 8 * n, 81);
    let m = g.m();
    let (ptr, col) = data::adjacency(&g);
    let mut rng = data::Rng::new(82);
    let seeds: Vec<usize> = (0..B).map(|_| rng.below(n)).collect();
    let want: Vec<f32> = seeds.iter().map(|&s| data::bfs_host(&ptr, &col, s, K as u32).iter().filter(|&&d| d != INF).count() as f32).collect();

    let ctx = h.ctx.clone();
    let dev = upload(h, &g);
    let seed_axis = Array::<u32>::range_on(&ctx, [B, m], 0).unwrap();
    let at = |ids: &U| ids.reshape([1, m]).broadcast_to([B, m]) * B as u32 + &seed_axis;
    let (from, to) = (at(&dev.src), at(&dev.dst));
    let mut start = vec![0.0f32; B * n];
    for (b, &s) in seeds.iter().enumerate() {
        start[b + B * s] = 1.0;
    }
    let reach0 = up_f32(h, &start, [B, n]);
    let go = || {
        let [r] = iterate_all([&reach0], K, |[r]| {
            let mut next = r.copy();
            next.put_add(&to, r.take(&from)).unwrap();
            [select(&next.gt(0.0f32), 1.0f32, 0.0f32)]
        });
        r
    };
    let name = "khop/multi_seed";
    exact(name, &go().sum_axis(1).to_vec().unwrap(), &want);
    let size = format!("{} nodes {} edges {B} seeds x{K} hops", label(n), label(m));
    case_trim(h, name, &size, Work::Items((m * B * K) as f64, "Medge/s"), &mut || go().eval().unwrap());
}
