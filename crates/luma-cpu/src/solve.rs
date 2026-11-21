//! Native line solves: the block Thomas algorithm of `luma`'s tridiagonal builtins, run as Rust
//! instead of compiled shader code.
//!
//! The shader solves one grid line per invocation with the whole line's serial chain, which the
//! packet-per-invocation code generator turns into four dependent chains per core and fissioned
//! loops over per-line scratch; here a workgroup is [`LineSolveSpec::lines_per_group`] lines, each
//! solved in registers by a monomorphized elimination, with the swept upper blocks kept in a small
//! per-thread buffer that stays in cache instead of a scratch array of the whole grid. The launch
//! is the shader's (see [`LineSolveSpec`] for the bindings and the uniform), so recorded steps,
//! graphs and profiles treat it like any kernel.
//!
//! The arithmetic is the shader's, in the same order (Gaussian elimination with partial pivoting,
//! the first row of largest magnitude wins ties, the reciprocal of the pivot), so the results
//! agree with the compiled kernel up to the rounding of fused operations.

// The elimination indexes blocks by row and column on purpose.
#![allow(clippy::needless_range_loop)]

use std::cell::RefCell;
use std::ops::{Add, Mul, Sub};
use std::slice;

use luma_core::{BackendError, LineSolveSpec, NativeKernel};

use crate::abi::{WorkgroupCtx, WorkgroupFn};

/// A line solve compiled to a monomorphized function.
#[derive(Debug)]
pub struct NativeLineSolve {
    pub(crate) spec: LineSolveSpec,
    pub(crate) func: WorkgroupFn,
    /// What a workgroup costs, learned by the pool (see [`crate::exec::CostHint`]).
    pub(crate) cost: crate::exec::CostHint,
}

impl NativeKernel for NativeLineSolve {}

impl NativeLineSolve {
    /// The solve of `spec`, `None` if its block size is not supported.
    pub fn new(spec: &LineSolveSpec) -> Option<Self> {
        if spec.systems == 0 || spec.lines_per_group == 0 || (!spec.parts && spec.systems != 1) {
            return None;
        }
        let func: WorkgroupFn = match spec.block {
            1 => group::<1>,
            2 => group::<2>,
            3 => group::<3>,
            4 => group::<4>,
            5 => group::<5>,
            6 => group::<6>,
            7 => group::<7>,
            8 => group::<8>,
            _ => return None,
        };
        Some(NativeLineSolve { spec: *spec, func, cost: crate::exec::CostHint::default() })
    }

    fn block(&self) -> usize {
        usize::from(self.spec.block)
    }

    fn systems(&self) -> usize {
        usize::from(self.spec.systems)
    }

    /// Buffers bound before the inputs: the outputs and the scratch.
    fn outputs(&self) -> usize {
        if self.spec.parts { self.systems() * self.block() } else { 1 }
    }

    /// Input buffers of the whole launch, `starts` excluded.
    fn inputs(&self) -> usize {
        if self.spec.parts { self.systems() * (3 * self.block() * self.block() + self.block()) } else { 4 }
    }

    /// Bindings of a launch: outputs, scratch, inputs, `starts` if any, the uniform.
    fn bindings(&self) -> usize {
        self.outputs() + 1 + self.inputs() + usize::from(self.spec.starts) + 1
    }

    /// Checks that the buffers of a launch hold every cell the solve touches (the solve itself
    /// trusts them): `bytes` are the sizes of the bindings, `uniform` the words `{n, stride,
    /// lines}`.
    pub(crate) fn validate(&self, bytes: &[u64], uniform: [u32; 3]) -> Result<(), BackendError> {
        let bad = |what: String| Err(BackendError::Unsupported(format!("native line solve: {what}")));
        if bytes.len() != self.bindings() {
            return bad(format!("{} bindings, expected {}", bytes.len(), self.bindings()));
        }
        let [n, stride, lines] = uniform.map(u64::from);
        if n == 0 || lines == 0 {
            return Ok(());
        }
        if stride == 0 {
            return bad("stride 0".into());
        }
        // One past the last cell a line touches.
        let cells = (lines - 1) % stride + (lines - 1) / stride * stride * n + (n - 1) * stride + 1;
        let (b, bb) = (self.block() as u64, (self.block() * self.block()) as u64);
        let (outs, ins) = (self.outputs(), self.inputs());
        let need = |at: usize, floats: u64, name: &str| -> Result<(), BackendError> {
            if bytes[at] < floats * 4 {
                return bad(format!("{name} (binding {at}) holds {} bytes, {} needed", bytes[at], floats * 4));
            }
            Ok(())
        };
        if self.spec.parts {
            for at in (0..outs).chain(outs + 1..outs + 1 + ins) {
                need(at, cells, "part")?;
            }
        } else {
            need(0, cells * b, "out")?;
            for (at, per) in [(2, bb), (3, bb), (4, bb), (5, b)] {
                need(at, cells * per, "input")?;
            }
        }
        if self.spec.starts {
            need(outs + 1 + ins, lines, "starts")?;
        }
        Ok(())
    }
}

/// Lines solved together by the lane path: lines of a strided axis are neighbouring cells, so
/// their coefficients are neighbouring floats (the parts) or neighbouring blocks (the stacked
/// layout), and the lanes of one vector; the lines of the contiguous axis are `n` cells apart
/// and gathered.
const LANES: usize = 4;

/// The arithmetic of the elimination, on one line (`f32`) or [`LANES`] lines at once ([`Lane`]).
/// The operations round like `f32` ones whatever the width, so a line solves to the same bits
/// alone or in a lane.
trait Real: Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + 'static {
    const ZERO: Self;
    fn recip(self) -> Self;
    /// Partial pivoting of column `col`: swaps the row of largest magnitude at or below the
    /// diagonal (the first of equals; none if it is the diagonal) with the diagonal row of `m`, `r`
    /// and `d`, in every lane on its own.
    fn pivot<const B: usize>(m: &mut Block<Self, B>, r: &mut Block<Self, B>, d: &mut [Self; B], col: usize);
    /// The value at `at`, or for lanes the values at `at + k * gap` (`k < LANES`).
    ///
    /// # Safety
    /// Every float read must be in bounds.
    unsafe fn load(at: *const f32, gap: usize) -> Self;
    /// # Safety
    /// Every float written must be in bounds.
    unsafe fn store(self, at: *mut f32, gap: usize);
}

/// `B x B` row-major: the layout of a block in the stacked buffers.
type Block<T, const B: usize> = [[T; B]; B];

impl Real for f32 {
    const ZERO: f32 = 0.0;

    #[inline(always)]
    fn recip(self) -> f32 {
        1.0 / self
    }

    #[inline(always)]
    fn pivot<const B: usize>(m: &mut Block<f32, B>, r: &mut Block<f32, B>, d: &mut [f32; B], col: usize) {
        let (mut best, mut pivot) = (m[col][col].abs(), col);
        for i in col + 1..B {
            let a = m[i][col].abs();
            if a > best {
                best = a;
                pivot = i;
            }
        }
        if pivot != col {
            m.swap(col, pivot);
            r.swap(col, pivot);
            d.swap(col, pivot);
        }
    }

    #[inline(always)]
    unsafe fn load(at: *const f32, _gap: usize) -> f32 {
        unsafe { *at }
    }

    #[inline(always)]
    unsafe fn store(self, at: *mut f32, _gap: usize) {
        unsafe { *at = self }
    }
}

/// [`LANES`] floats, one per line solved together. `NEIGHBOURS`: the lanes are neighbouring
/// floats in memory (a vector load), else `gap` floats apart.
#[derive(Clone, Copy)]
#[repr(transparent)]
struct Lane<const NEIGHBOURS: bool>([f32; LANES]);

macro_rules! lanewise {
    ($($trait:ident $method:ident $assign:tt),*) => {$(
        impl<const N: bool> $trait for Lane<N> {
            type Output = Lane<N>;
            #[inline(always)]
            fn $method(self, other: Lane<N>) -> Lane<N> {
                let mut out = self.0;
                for k in 0..LANES {
                    out[k] $assign other.0[k];
                }
                Lane(out)
            }
        }
    )*};
}

lanewise!(Add add +=, Sub sub -=, Mul mul *=);

impl<const N: bool> Real for Lane<N> {
    const ZERO: Lane<N> = Lane([0.0; LANES]);

    #[inline(always)]
    fn recip(self) -> Lane<N> {
        let mut out = self.0;
        for value in &mut out {
            *value = 1.0 / *value;
        }
        Lane(out)
    }

    #[inline(always)]
    fn pivot<const B: usize>(m: &mut Block<Self, B>, r: &mut Block<Self, B>, d: &mut [Self; B], col: usize) {
        let mut best = m[col][col].0.map(f32::abs);
        let mut pivot = [col; LANES];
        for i in col + 1..B {
            for k in 0..LANES {
                let a = m[i][col].0[k].abs();
                if a > best[k] {
                    best[k] = a;
                    pivot[k] = i;
                }
            }
        }
        // Diagonally dominant systems never exchange: one test for all lanes.
        if pivot.iter().all(|&p| p == col) {
            return;
        }
        for k in 0..LANES {
            let i = pivot[k];
            if i != col {
                for j in 0..B {
                    let t = m[col][j].0[k];
                    m[col][j].0[k] = m[i][j].0[k];
                    m[i][j].0[k] = t;
                    let t = r[col][j].0[k];
                    r[col][j].0[k] = r[i][j].0[k];
                    r[i][j].0[k] = t;
                }
                let t = d[col].0[k];
                d[col].0[k] = d[i].0[k];
                d[i].0[k] = t;
            }
        }
    }

    #[inline(always)]
    unsafe fn load(at: *const f32, gap: usize) -> Lane<N> {
        if N {
            return unsafe { Lane(at.cast::<[f32; LANES]>().read_unaligned()) };
        }
        let mut out = [0f32; LANES];
        for (k, value) in out.iter_mut().enumerate() {
            *value = unsafe { *at.add(k * gap) };
        }
        Lane(out)
    }

    #[inline(always)]
    unsafe fn store(self, at: *mut f32, gap: usize) {
        if N {
            return unsafe { at.cast::<[f32; LANES]>().write_unaligned(self.0) };
        }
        for (k, &value) in self.0.iter().enumerate() {
            unsafe { *at.add(k * gap) = value };
        }
    }
}

/// A cell's blocks, read from and written to the buffers of a solve: cell `q` of one line, or of
/// [`LANES`] lines (cells `q + k * gap`) for a [`Lane`].
trait Cells<T: Real, const B: usize> {
    /// # Safety
    /// The cell(s) at `q` of every buffer must be in bounds ([`NativeLineSolve::validate`]).
    unsafe fn diag(&self, q: usize) -> Block<T, B>;
    unsafe fn lower(&self, q: usize) -> Block<T, B>;
    unsafe fn upper(&self, q: usize) -> Block<T, B>;
    unsafe fn rhs(&self, q: usize) -> [T; B];
    unsafe fn store(&self, q: usize, x: &[T; B]);
}

/// Stacked blocks: `lo`, `diag`, `up` hold `B * B` floats per cell, `rhs` and `out` `B`. `gap`
/// is the cells between the lines of a [`Lane`].
struct Stacked {
    out: *mut f32,
    lo: *const f32,
    diag: *const f32,
    up: *const f32,
    rhs: *const f32,
    gap: usize,
}

impl Stacked {
    #[inline(always)]
    unsafe fn block<T: Real, const B: usize>(&self, at: *const f32, q: usize) -> Block<T, B> {
        let mut m = [[T::ZERO; B]; B];
        for (i, row) in m.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                *value = unsafe { T::load(at.add(q * B * B + i * B + j), self.gap * B * B) };
            }
        }
        m
    }
}

impl<T: Real, const B: usize> Cells<T, B> for Stacked {
    #[inline(always)]
    unsafe fn diag(&self, q: usize) -> Block<T, B> {
        unsafe { self.block::<T, B>(self.diag, q) }
    }
    #[inline(always)]
    unsafe fn lower(&self, q: usize) -> Block<T, B> {
        unsafe { self.block::<T, B>(self.lo, q) }
    }
    #[inline(always)]
    unsafe fn upper(&self, q: usize) -> Block<T, B> {
        unsafe { self.block::<T, B>(self.up, q) }
    }
    #[inline(always)]
    unsafe fn rhs(&self, q: usize) -> [T; B] {
        let mut r = [T::ZERO; B];
        for (i, value) in r.iter_mut().enumerate() {
            *value = unsafe { T::load(self.rhs.add(q * B + i), self.gap * B) };
        }
        r
    }
    #[inline(always)]
    unsafe fn store(&self, q: usize, x: &[T; B]) {
        for (i, &value) in x.iter().enumerate() {
            unsafe { value.store(self.out.add(q * B + i), self.gap * B) };
        }
    }
}

/// The parts of one system: `ins` are its `3 B^2 + B` input buffers (`lo`, `diag`, `up`, `rhs`),
/// `outs` its `B` outputs, a float per cell each; `gap` as for [`Stacked`].
struct Parts<'a> {
    outs: &'a [*mut u8],
    ins: &'a [*mut u8],
    gap: usize,
}

impl Parts<'_> {
    #[inline(always)]
    unsafe fn block<T: Real, const B: usize>(&self, first: usize, q: usize) -> Block<T, B> {
        let mut m = [[T::ZERO; B]; B];
        for (i, row) in m.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                *value = unsafe { T::load(self.ins[first + i * B + j].cast::<f32>().add(q), self.gap) };
            }
        }
        m
    }
}

impl<T: Real, const B: usize> Cells<T, B> for Parts<'_> {
    #[inline(always)]
    unsafe fn lower(&self, q: usize) -> Block<T, B> {
        unsafe { self.block::<T, B>(0, q) }
    }
    #[inline(always)]
    unsafe fn diag(&self, q: usize) -> Block<T, B> {
        unsafe { self.block::<T, B>(B * B, q) }
    }
    #[inline(always)]
    unsafe fn upper(&self, q: usize) -> Block<T, B> {
        unsafe { self.block::<T, B>(2 * B * B, q) }
    }
    #[inline(always)]
    unsafe fn rhs(&self, q: usize) -> [T; B] {
        let mut r = [T::ZERO; B];
        for (i, value) in r.iter_mut().enumerate() {
            *value = unsafe { T::load(self.ins[3 * B * B + i].cast::<f32>().add(q), self.gap) };
        }
        r
    }
    #[inline(always)]
    unsafe fn store(&self, q: usize, x: &[T; B]) {
        for (c, &value) in x.iter().enumerate() {
            unsafe { value.store(self.outs[c].cast::<f32>().add(q), self.gap) };
        }
    }
}

/// Solves `m X = [r | d]` by Gaussian elimination with partial pivoting and back substitution:
/// `r` and `d` become the solution. A zero pivot gives Inf/NaN.
#[inline(always)]
fn eliminate<T: Real, const B: usize>(m: &mut Block<T, B>, r: &mut Block<T, B>, d: &mut [T; B]) {
    let mut v = [T::ZERO; B];
    for col in 0..B {
        if col + 1 < B {
            T::pivot(m, r, d, col);
        }
        let inv = m[col][col].recip();
        v[col] = inv;
        let (top, below, top_d) = (m[col], r[col], d[col]);
        for i in col + 1..B {
            let f = m[i][col] * inv;
            for j in col + 1..B {
                m[i][j] = m[i][j] - f * top[j];
            }
            for j in 0..B {
                r[i][j] = r[i][j] - f * below[j];
            }
            d[i] = d[i] - f * top_d;
        }
    }
    for i in (0..B).rev() {
        let mut row = r[i];
        let mut x = d[i];
        for k in i + 1..B {
            let f = m[i][k];
            for j in 0..B {
                row[j] = row[j] - f * r[k][j];
            }
            x = x - f * d[k];
        }
        for j in 0..B {
            r[i][j] = row[j] * v[i];
        }
        d[i] = x * v[i];
    }
}

thread_local! {
    /// The swept upper blocks and right-hand sides of the line(s) being solved, per thread.
    static SWEPT: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Solves the line of `n` cells `base`, `base + stride`, .. whose system begins at row `s0` (the
/// rows before it are 0), or [`LANES`] lines `base + k` at once for [`Lane`] (`s0` the same).
///
/// # Safety
/// Every cell of the line(s) must be in bounds of the buffers behind `cells`.
#[inline(always)]
unsafe fn solve_line<T: Real, const B: usize, C: Cells<T, B>>(cells: &C, base: usize, stride: usize, n: usize, s0: usize, swept: &mut [T]) {
    let zero = [T::ZERO; B];
    for s in 0..s0 {
        unsafe { cells.store(base + s * stride, &zero) };
    }
    if s0 >= n {
        return;
    }
    let per = B * B + B;
    let mut pc = [[T::ZERO; B]; B];
    let mut pd = zero;
    for s in s0..n {
        let q = base + s * stride;
        let (mut m, mut r, mut d) = unsafe { (cells.diag(q), cells.upper(q), cells.rhs(q)) };
        if s > s0 {
            let l = unsafe { cells.lower(q) };
            for i in 0..B {
                for j in 0..B {
                    let mut sum = l[i][0] * pc[0][j];
                    for k in 1..B {
                        sum = sum + l[i][k] * pc[k][j];
                    }
                    m[i][j] = m[i][j] - sum;
                }
                let mut sum = l[i][0] * pd[0];
                for k in 1..B {
                    sum = sum + l[i][k] * pd[k];
                }
                d[i] = d[i] - sum;
            }
        }
        eliminate::<T, B>(&mut m, &mut r, &mut d);
        let slot = &mut swept[(s - s0) * per..(s - s0 + 1) * per];
        for i in 0..B {
            for j in 0..B {
                slot[i * B + j] = r[i][j];
            }
            slot[B * B + i] = d[i];
        }
        pc = r;
        pd = d;
    }
    // Back substitution: the last row's solution is its swept right-hand side.
    let mut x = pd;
    unsafe { cells.store(base + (n - 1) * stride, &x) };
    for s in (s0..n - 1).rev() {
        let slot = &swept[(s - s0) * per..(s - s0 + 1) * per];
        let mut y = zero;
        for i in 0..B {
            let mut sum = slot[i * B] * x[0];
            for k in 1..B {
                sum = sum + slot[i * B + k] * x[k];
            }
            y[i] = slot[B * B + i] - sum;
        }
        unsafe { cells.store(base + s * stride, &y) };
        x = y;
    }
}

/// One workgroup: [`LineSolveSpec::lines_per_group`] consecutive lines. Runs of [`LANES`] lines
/// that are neighbouring cells of one row of the grid (lines of a strided axis) or `n` cells apart
/// (lines of the contiguous axis) are solved together, the others one by one; so are all lines
/// when every one begins at a cell of its own (`starts`).
unsafe extern "C" fn group<const B: usize>(ctx: *const WorkgroupCtx) {
    // SAFETY: the dispatcher passes a valid context whose `native` is the kernel it dispatched
    // and whose bindings the launch validated ([`NativeLineSolve::validate`]).
    unsafe {
        let ctx = &*ctx;
        let kernel = &*ctx.native.cast::<NativeLineSolve>();
        let spec = &kernel.spec;
        let bindings = slice::from_raw_parts(ctx.bindings, kernel.bindings());
        let [n, stride, lines] = bindings[kernel.bindings() - 1].cast::<[u32; 3]>().read();
        let (n, stride, lines) = (n as usize, stride as usize, lines as usize);
        let total = lines * kernel.systems();
        let per_group = spec.lines_per_group as usize;
        let [x, y, _, _] = ctx.workgroup_id;
        let first = (y as usize * ctx.num_workgroups[0] as usize + x as usize) * per_group;
        if n == 0 || first >= total {
            return;
        }
        let last = (first + per_group).min(total);
        let (outs, inputs) = (kernel.outputs(), kernel.inputs());
        let starts = spec.starts.then(|| bindings[outs + 1 + inputs].cast::<u32>().cast_const());
        let per = 3 * B * B + B;
        SWEPT.with(|cell| {
            let mut swept = cell.borrow_mut();
            let need = n * (B * B + B) * LANES;
            if swept.len() < need {
                swept.resize(need, 0.0);
            }
            // The scratch as `f32`s and as `Lane`s, which are arrays of `f32`s.
            let (floats, len) = (swept.as_mut_ptr(), swept.len());
            let mut lg = first;
            while lg < last {
                let (sys, l) = (lg / lines, lg % lines);
                let base = l % stride + l / stride * stride * n;
                // The cells between the lines solved together, if they can be.
                let gap = if starts.is_some() || lg + LANES > last || l + LANES > lines {
                    None
                } else if stride >= LANES && l % stride + LANES <= stride {
                    Some(1)
                } else if stride == 1 {
                    Some(n)
                } else {
                    None
                };
                let s0 = starts.map_or(0, |starts| (*starts.add(l) as usize).min(n));
                let gap_or_one = gap.unwrap_or(1);
                let scalar = || slice::from_raw_parts_mut(floats, len);
                let wide = || slice::from_raw_parts_mut(floats.cast::<[f32; LANES]>(), len / LANES);
                macro_rules! solve {
                    ($cells:expr) => {{
                        let cells = $cells;
                        match gap {
                            // Parts of neighbouring lines are neighbouring floats: a vector load.
                            Some(1) if spec.parts => solve_line::<Lane<true>, B, _>(&cells, base, stride, n, s0, lanes(wide())),
                            Some(_) => solve_line::<Lane<false>, B, _>(&cells, base, stride, n, s0, lanes(wide())),
                            None => solve_line::<f32, B, _>(&cells, base, stride, n, s0, scalar()),
                        }
                    }};
                }
                if spec.parts {
                    solve!(Parts { outs: &bindings[sys * B..(sys + 1) * B], ins: &bindings[outs + 1 + sys * per..outs + 1 + (sys + 1) * per], gap: gap_or_one });
                } else {
                    solve!(Stacked {
                        out: bindings[0].cast::<f32>(),
                        lo: bindings[2].cast::<f32>().cast_const(),
                        diag: bindings[3].cast::<f32>().cast_const(),
                        up: bindings[4].cast::<f32>().cast_const(),
                        rhs: bindings[5].cast::<f32>().cast_const(),
                        gap: gap_or_one,
                    });
                }
                lg += if gap.is_some() { LANES } else { 1 };
            }
        });
    }
}

/// `chunks` as the scratch of a lane solve: [`Lane`]s are arrays of `f32`s.
///
/// # Safety
/// `T` is a [`Lane`].
unsafe fn lanes<T>(chunks: &mut [[f32; LANES]]) -> &mut [T] {
    unsafe { slice::from_raw_parts_mut(chunks.as_mut_ptr().cast::<T>(), chunks.len()) }
}
