//! Host side of the `sparse` suite: deterministic matrices and graphs, their reference algorithms
//! (in `f64` where sums matter) and the packed layouts (ELL, split rows) the device cases upload.

/// A small deterministic generator (xorshift64*).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, n: usize) -> usize {
        ((self.next() >> 11) % n.max(1) as u64) as usize
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A multiple of 1/16 in `[-1, 1]`: sums and products of such numbers are exact in `f32` while
    /// they stay below 2^16, so a result is the same in any order of summation and can be checked
    /// for equality.
    pub fn quant(&mut self) -> f32 {
        (self.below(33) as f32 - 16.0) / 16.0
    }
}

/// A compressed-sparse-row matrix.
pub struct Csr {
    pub rows: usize,
    pub cols: usize,
    pub ptr: Vec<u32>,
    pub col: Vec<u32>,
    pub val: Vec<f32>,
}

impl Csr {
    pub fn nnz(&self) -> usize {
        self.col.len()
    }

    /// The row of every stored entry (CSR expanded to COO rows).
    pub fn row_of(&self) -> Vec<u32> {
        let mut row = Vec::with_capacity(self.nnz());
        for r in 0..self.rows {
            row.extend(std::iter::repeat_n(r as u32, (self.ptr[r + 1] - self.ptr[r]) as usize));
        }
        row
    }

    pub fn max_row(&self) -> usize {
        self.ptr.windows(2).map(|w| (w[1] - w[0]) as usize).max().unwrap_or(0)
    }

    /// Rows `r` get the columns `cols(r, rng)` (any order, duplicates allowed) with values that are multiples of 1/16 (see [`Rng::quant`]).
    fn build(rows: usize, cols: usize, seed: u64, mut fill: impl FnMut(usize, &mut Rng, &mut Vec<u32>)) -> Csr {
        let mut rng = Rng::new(seed);
        let mut a = Csr { rows, cols, ptr: Vec::with_capacity(rows + 1), col: Vec::new(), val: Vec::new() };
        a.ptr.push(0);
        for r in 0..rows {
            fill(r, &mut rng, &mut a.col);
            a.ptr.push(a.col.len() as u32);
        }
        a.val = (0..a.col.len()).map(|_| rng.quant()).collect();
        a
    }
}

/// `2 * half + 1` diagonals (clipped at the corners): the band of a 1-D stencil.
pub fn banded(rows: usize, half: usize, seed: u64) -> Csr {
    Csr::build(rows, rows, seed, |r, _, out| {
        let lo = r.saturating_sub(half);
        let hi = (r + half).min(rows - 1);
        out.extend(lo as u32..=hi as u32);
    })
}

/// The 5-point Laplacian of a `g x g` grid with Dirichlet boundaries: 4 on the diagonal, -1 for
/// each neighbour inside the grid. Symmetric positive definite.
pub fn poisson2d(g: usize) -> Csr {
    let n = g * g;
    let mut a = Csr { rows: n, cols: n, ptr: Vec::with_capacity(n + 1), col: Vec::with_capacity(5 * n), val: Vec::with_capacity(5 * n) };
    a.ptr.push(0);
    for i in 0..n {
        let (x, y) = (i % g, i / g);
        let mut push = |c: usize, v: f32| {
            a.col.push(c as u32);
            a.val.push(v);
        };
        if y > 0 {
            push(i - g, -1.0);
        }
        if x > 0 {
            push(i - 1, -1.0);
        }
        push(i, 4.0);
        if x + 1 < g {
            push(i + 1, -1.0);
        }
        if y + 1 < g {
            push(i + g, -1.0);
        }
        a.ptr.push(a.col.len() as u32);
    }
    a
}

/// Rows of `lo..=hi` entries at random columns.
pub fn uniform(rows: usize, lo: usize, hi: usize, seed: u64) -> Csr {
    Csr::build(rows, rows, seed, |_, rng, out| {
        let d = lo + rng.below(hi - lo + 1);
        out.extend((0..d).map(|_| rng.below(rows) as u32));
    })
}

/// Rows of 1 to 16 entries at random columns, except that the first 100 rows and a quarter of the
/// others are empty.
pub fn with_empty_rows(rows: usize, seed: u64) -> Csr {
    Csr::build(rows, rows, seed, |r, rng, out| {
        if r >= 100 && rng.below(4) != 0 {
            let d = 1 + rng.below(16);
            out.extend((0..d).map(|_| rng.below(rows) as u32));
        }
    })
}

/// Row lengths follow a Pareto law (`P(len > x) ~ (dmin / x)^(alpha - 1)`, at most `cap`) at
/// random columns: a few hub rows hold a large share of the entries.
pub fn power_law(rows: usize, dmin: usize, alpha: f64, cap: usize, seed: u64) -> Csr {
    Csr::build(rows, rows, seed, |_, rng, out| {
        let u = 1.0 - rng.unit();
        let d = ((dmin as f64 * u.powf(-1.0 / (alpha - 1.0))) as usize).clamp(dmin, cap);
        out.extend((0..d).map(|_| rng.below(rows) as u32));
    })
}

/// `y = A x` in `f64`.
pub fn spmv_host(a: &Csr, x: &[f32]) -> Vec<f64> {
    (0..a.rows).map(|r| (a.ptr[r] as usize..a.ptr[r + 1] as usize).map(|k| f64::from(a.val[k]) * f64::from(x[a.col[k] as usize])).sum()).collect()
}

/// `n` multiples of 1/16 in `[-1, 1]`.
pub fn quantized(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.quant()).collect()
}

/// ELLPACK: every row padded to `w` entries; entry `j` of row `i` is at `j + w * i`, so a sum
/// along axis 0 of the `[w, rows]` array is the row sum. Padding reads column `i` with value 0.
pub struct Ell {
    pub w: usize,
    pub col: Vec<u32>,
    pub val: Vec<f32>,
}

pub fn ell(a: &Csr) -> Ell {
    let w = a.max_row();
    let mut e = Ell { w, col: Vec::with_capacity(w * a.rows), val: Vec::with_capacity(w * a.rows) };
    for r in 0..a.rows {
        let (lo, hi) = (a.ptr[r] as usize, a.ptr[r + 1] as usize);
        e.col.extend_from_slice(&a.col[lo..hi]);
        e.val.extend_from_slice(&a.val[lo..hi]);
        e.col.extend(std::iter::repeat_n(r as u32, w - (hi - lo)));
        e.val.extend(std::iter::repeat_n(0.0, w - (hi - lo)));
    }
    e
}

/// Rows cut into pieces of at most `l` entries (the last piece of a row is padded with zeros):
/// the `[l, pieces]` array has equal-length lines that `sum_axis(0)` reduces, and `row` says which
/// row each piece belongs to.
pub struct Pieces {
    pub row: Vec<u32>,
    pub col: Vec<u32>,
    pub val: Vec<f32>,
}

pub fn pieces(a: &Csr, l: usize) -> Pieces {
    let mut p = Pieces { row: Vec::new(), col: Vec::new(), val: Vec::new() };
    for r in 0..a.rows {
        let (lo, hi) = (a.ptr[r] as usize, a.ptr[r + 1] as usize);
        for s in (lo..hi).step_by(l) {
            let e = (s + l).min(hi);
            p.row.push(r as u32);
            p.col.extend_from_slice(&a.col[s..e]);
            p.val.extend_from_slice(&a.val[s..e]);
            p.col.extend(std::iter::repeat_n(r as u32, l - (e - s)));
            p.val.extend(std::iter::repeat_n(0.0, l - (e - s)));
        }
    }
    p
}

// -------------------------------------------------------------------------------------------------
// Graphs
// -------------------------------------------------------------------------------------------------

/// A directed multigraph as an edge list.
pub struct Graph {
    pub n: usize,
    pub src: Vec<u32>,
    pub dst: Vec<u32>,
}

impl Graph {
    pub fn m(&self) -> usize {
        self.src.len()
    }
}

/// A random permutation of `0..n`.
fn permutation(n: usize, rng: &mut Rng) -> Vec<u32> {
    let mut p: Vec<u32> = (0..n as u32).collect();
    for i in (1..n).rev() {
        p.swap(i, rng.below(i + 1));
    }
    p
}

/// A power-law graph: every node has at least one out-edge (`i -> skewed`), the other `m - n`
/// edges go from a skewed to a more skewed node (`id = n * u^p` for uniform `u`, which gives a
/// degree law with a heavy tail; the hub ids are scattered by a random permutation). Edges are in
/// random order. Self-loops and repeated edges occur.
pub fn power_law_graph(n: usize, m: usize, seed: u64) -> Graph {
    assert!(m >= n);
    let mut rng = Rng::new(seed);
    let perm = permutation(n, &mut rng);
    let skewed = |rng: &mut Rng, p: f64| perm[((rng.unit().powf(p) * n as f64) as usize).min(n - 1)];
    let mut src = Vec::with_capacity(m);
    let mut dst = Vec::with_capacity(m);
    for i in 0..n {
        src.push(i as u32);
        dst.push(skewed(&mut rng, 3.0));
    }
    for _ in n..m {
        src.push(skewed(&mut rng, 2.0));
        dst.push(skewed(&mut rng, 3.0));
    }
    // Shuffle the edges.
    for e in (1..m).rev() {
        let o = rng.below(e + 1);
        src.swap(e, o);
        dst.swap(e, o);
    }
    Graph { n, src, dst }
}

/// A bipartite (hence triangle-free) graph of `nu` users and `ni` items: edge `e` joins user
/// `src[e]` (uniformly random) and item `dst[e]`, whose popularity follows a power law (`id = ni *
/// u^3`, ids scattered by a permutation). `n` of the result is `nu`.
pub fn bipartite(nu: usize, ni: usize, m: usize, seed: u64) -> Graph {
    let mut rng = Rng::new(seed);
    let perm = permutation(ni, &mut rng);
    let src = (0..m).map(|_| rng.below(nu) as u32).collect();
    let dst = (0..m).map(|_| perm[((rng.unit().powi(3) * ni as f64) as usize).min(ni - 1)]).collect();
    Graph { n: nu, src, dst }
}

/// `m` uniformly random edges.
pub fn random_graph(n: usize, m: usize, seed: u64) -> Graph {
    let mut rng = Rng::new(seed);
    let src = (0..m).map(|_| rng.below(n) as u32).collect();
    let dst = (0..m).map(|_| rng.below(n) as u32).collect();
    Graph { n, src, dst }
}

/// The 4-neighbour `g x g` grid, one edge per adjacent pair, node ids scattered at random.
pub fn grid_graph(g: usize, seed: u64) -> Graph {
    let mut rng = Rng::new(seed);
    let perm = permutation(g * g, &mut rng);
    let (mut src, mut dst) = (Vec::new(), Vec::new());
    for y in 0..g {
        for x in 0..g {
            let i = y * g + x;
            if x + 1 < g {
                src.push(perm[i]);
                dst.push(perm[i + 1]);
            }
            if y + 1 < g {
                src.push(perm[i]);
                dst.push(perm[i + g]);
            }
        }
    }
    Graph { n: g * g, src, dst }
}

/// Every edge in both directions (the edges `e` first, then their reverses).
pub fn symmetrize(g: &Graph) -> Graph {
    Graph { n: g.n, src: g.src.iter().chain(&g.dst).copied().collect(), dst: g.dst.iter().chain(&g.src).copied().collect() }
}

/// The edges in stable order of `key`.
pub fn sort_edges(g: &Graph, key: &[u32]) -> Graph {
    let mut order: Vec<u32> = (0..g.m() as u32).collect();
    order.sort_by_key(|&e| key[e as usize]);
    Graph { n: g.n, src: order.iter().map(|&e| g.src[e as usize]).collect(), dst: order.iter().map(|&e| g.dst[e as usize]).collect() }
}

/// Out-adjacency in CSR form (`ptr`, `col`), neighbours in edge order.
pub fn adjacency(g: &Graph) -> (Vec<u32>, Vec<u32>) {
    let mut ptr = vec![0u32; g.n + 1];
    for &s in &g.src {
        ptr[s as usize + 1] += 1;
    }
    for i in 0..g.n {
        ptr[i + 1] += ptr[i];
    }
    let mut next = ptr.clone();
    let mut col = vec![0u32; g.m()];
    for (&s, &d) in g.src.iter().zip(&g.dst) {
        col[next[s as usize] as usize] = d;
        next[s as usize] += 1;
    }
    (ptr, col)
}

pub fn degrees(n: usize, ends: &[u32]) -> Vec<u32> {
    let mut d = vec![0u32; n];
    ends.iter().for_each(|&e| d[e as usize] += 1);
    d
}

/// Breadth-first distances from `source` (`u32::MAX` where unreachable), stopping after `limit` levels.
pub fn bfs_host(ptr: &[u32], col: &[u32], source: usize, limit: u32) -> Vec<u32> {
    let mut dist = vec![u32::MAX; ptr.len() - 1];
    dist[source] = 0;
    let mut frontier = vec![source as u32];
    let mut level = 0;
    while !frontier.is_empty() && level < limit {
        level += 1;
        let mut next = Vec::new();
        for &u in &frontier {
            for &v in &col[ptr[u as usize] as usize..ptr[u as usize + 1] as usize] {
                if dist[v as usize] == u32::MAX {
                    dist[v as usize] = level;
                    next.push(v);
                }
            }
        }
        frontier = next;
    }
    dist
}

/// The smallest node id of every node's (weakly) connected component.
pub fn components_host(g: &Graph) -> Vec<u32> {
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    let mut parent: Vec<u32> = (0..g.n as u32).collect();
    for (&s, &d) in g.src.iter().zip(&g.dst) {
        let (a, b) = (find(&mut parent, s), find(&mut parent, d));
        // Keep the smaller id as the root.
        parent[a.max(b) as usize] = a.min(b);
    }
    (0..g.n as u32).map(|v| find(&mut parent, v)).collect()
}

/// PageRank power iterations in `f64` with damping `d` on a graph where every node has an
/// out-edge; returns the ranks after `iters` steps from the uniform start.
pub fn pagerank_host(g: &Graph, d: f64, iters: usize) -> Vec<f64> {
    let outdeg = degrees(g.n, &g.src);
    let mut r = vec![1.0 / g.n as f64; g.n];
    for _ in 0..iters {
        let mut next = vec![0.0f64; g.n];
        for (&s, &t) in g.src.iter().zip(&g.dst) {
            next[t as usize] += r[s as usize] / f64::from(outdeg[s as usize]);
        }
        for v in &mut next {
            *v = *v * d + (1.0 - d) / g.n as f64;
        }
        r = next;
    }
    r
}

/// Shortest distances from `source` along the edges with weights `w` (`INFINITY` where unreachable).
pub fn dijkstra_host(g: &Graph, w: &[f32], source: usize) -> Vec<f32> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let (ptr, _) = adjacency(g);
    // Edge ids by source, in edge order.
    let mut by_src = vec![0u32; g.m()];
    let mut next = ptr.clone();
    for (e, &s) in g.src.iter().enumerate() {
        by_src[next[s as usize] as usize] = e as u32;
        next[s as usize] += 1;
    }
    // Weights are multiples of 1/8: integer eighths keep the heap exact.
    let mut dist = vec![u64::MAX; g.n];
    let mut heap = BinaryHeap::new();
    dist[source] = 0;
    heap.push(Reverse((0u64, source as u32)));
    while let Some(Reverse((d, u))) = heap.pop() {
        if d > dist[u as usize] {
            continue;
        }
        for &e in &by_src[ptr[u as usize] as usize..ptr[u as usize + 1] as usize] {
            let nd = d + (w[e as usize] * 8.0) as u64;
            let v = g.dst[e as usize] as usize;
            if nd < dist[v] {
                dist[v] = nd;
                heap.push(Reverse((nd, v as u32)));
            }
        }
    }
    dist.iter().map(|&d| if d == u64::MAX { f32::INFINITY } else { d as f32 / 8.0 }).collect()
}
