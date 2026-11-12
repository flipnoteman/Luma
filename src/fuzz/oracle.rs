//! The CPU oracle of the fuzzer: every operation computes eagerly on a `Vec<f64>` with plain
//! loops, in the arithmetic of the array's type (`f32` operations are done in `f32`, `u32` ones
//! wrap, [`Ty::Df32`] values are exact `f64`s). It is the reference, so it is kept simple.
//!
//! Sharing follows luma's documented rules (`tests/clone.rs`, `tests/assign.rs`):
//!
//! * A clone, and the result of `cache`, is another handle on the same data (the `Rc` is shared).
//! * Every other operation creates a new array holding the value as of now: a lazy array is the
//!   value as of when it was written, so later writes never reach it.
//! * A slice, flip or transpose (of a view, composed) remembers the array it is a view of and
//!   the index map: `assign` writes through the map into that array's data, and shows the
//!   assigned data itself; views taken earlier keep their data.
//! * `write`, `+=` and `put` change the data the handle shares with its clones, not other arrays.

use std::cell::RefCell;
use std::rc::Rc;

/// Extents of an array, the first varies fastest.
pub type Dims = [usize; 4];

/// Element types of the fuzzer's arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty {
    F32,
    U32,
    Df32,
}

/// What kind of error an operation fails with (the discriminant of luma's root error).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrKind {
    ShapeMismatch,
    InvalidShape,
    LengthMismatch,
    Chunked,
    Unsupported,
    Other,
}

/// Elementwise unary operations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnOp {
    Neg,
    Abs,
    Floor,
    Ceil,
    Sign,
    Sqrt,
    Exp,
    Ln,
    Sin,
    Cos,
    Tan,
    Tanh,
    Clamp { lo: f32, hi: f32 },
    Powf(f32),
    ToDf32,
    ToF32,
}

/// Elementwise binary operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Minimum,
    Maximum,
    Atan2,
    Hypot,
    Copysign,
}

/// Comparisons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Equal,
    NotEqual,
}

/// Reductions (over everything, or along an axis).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RedOp {
    Sum,
    Min,
    Max,
}

/// Inclusive scans along an axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanOp {
    CumSum,
    ScanMax,
    ScanMin,
}

/// What a coordinate outside the source reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Edge {
    /// Nothing is outside.
    None,
    Fill(f64),
    Clamp,
    Mirror,
    Wrap,
}

/// The right-hand side of an operation: an array or one value.
#[derive(Clone, Copy)]
pub enum Rhs<'a> {
    Array(&'a HostArray),
    Val(f64),
}

/// How a view reads its source: source axis `a` is read at `start + step * out[from]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Axis {
    pub from: usize,
    pub start: i64,
    pub step: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewMap {
    pub src: Dims,
    pub out: Dims,
    pub axes: [Axis; 4],
}

impl ViewMap {
    pub fn identity(src: Dims) -> ViewMap {
        ViewMap { src, out: src, axes: std::array::from_fn(|a| Axis { from: a, start: 0, step: 1 }) }
    }

    /// The map that reads the source of `self` through `self`, then `outer` through that:
    /// `self` is the map from `mid` coordinates to base ones, `outer` from out ones to `mid`.
    pub fn then(&self, outer: &ViewMap) -> ViewMap {
        let axes = std::array::from_fn(|b| {
            let inner = self.axes[b];
            let o = outer.axes[inner.from];
            Axis { from: o.from, start: inner.start + inner.step * o.start, step: inner.step * o.step }
        });
        ViewMap { src: self.src, out: outer.out, axes }
    }

    /// The unclipped source coordinates of the output cell `at`.
    fn source(&self, at: [usize; 4]) -> [i64; 4] {
        std::array::from_fn(|a| self.axes[a].start + self.axes[a].step * at[self.axes[a].from] as i64)
    }
}

/// The shared data of an array handle.
struct Cell {
    data: Vec<f64>,
}

/// An array: clones share the data (`cell`).
#[derive(Clone)]
pub struct HostArray {
    pub dims: Dims,
    pub ty: Ty,
    cell: Rc<RefCell<Cell>>,
    /// The array this one is a slice, flip or transpose of, and the map into it.
    view: Option<(Rc<RefCell<Cell>>, ViewMap)>,
}

pub type Res<T> = Result<T, ErrKind>;

pub fn len_of(dims: Dims) -> usize {
    dims.iter().product()
}

fn strides(dims: Dims) -> [usize; 4] {
    [1, dims[0], dims[0] * dims[1], dims[0] * dims[1] * dims[2]]
}

fn coords(i: usize, dims: Dims) -> [usize; 4] {
    let s = strides(dims);
    std::array::from_fn(|a| (i / s[a]) % dims[a])
}

fn flat(c: [usize; 4], dims: Dims) -> usize {
    let s = strides(dims);
    (0..4).map(|a| c[a] * s[a]).sum()
}

/// `v` as a value of `ty` (what a store of that type does to it).
pub fn store(ty: Ty, v: f64) -> f64 {
    match ty {
        Ty::F32 => f64::from(v as f32),
        Ty::U32 => f64::from(v as i64 as u32),
        Ty::Df32 => v,
    }
}

fn mirror(c: i64, n: i64) -> i64 {
    let a = if c < 0 { -c - 1 } else { c };
    let b = if a >= n { 2 * n - 1 - a } else { a };
    b.clamp(0, n - 1)
}

fn wrap(c: i64, n: i64) -> i64 {
    c.rem_euclid(n)
}

fn sign32(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

pub fn apply_bin(ty: Ty, op: BinOp, a: f64, b: f64) -> f64 {
    match ty {
        Ty::F32 => {
            let (a, b) = (a as f32, b as f32);
            f64::from(match op {
                BinOp::Add => a + b,
                BinOp::Sub => a - b,
                BinOp::Mul => a * b,
                BinOp::Div => a / b,
                BinOp::Minimum => a.min(b),
                BinOp::Maximum => a.max(b),
                BinOp::Atan2 => a.atan2(b),
                BinOp::Hypot => a.hypot(b),
                BinOp::Copysign => a.copysign(b),
                BinOp::Rem => unreachable!("rem of floats"),
            })
        }
        Ty::U32 => {
            let (a, b) = (a as u32, b as u32);
            f64::from(match op {
                BinOp::Add => a.wrapping_add(b),
                BinOp::Sub => a.wrapping_sub(b),
                BinOp::Mul => a.wrapping_mul(b),
                BinOp::Div => a.checked_div(b).unwrap_or(0),
                BinOp::Rem => a.checked_rem(b).unwrap_or(0),
                BinOp::Minimum => a.min(b),
                BinOp::Maximum => a.max(b),
                _ => unreachable!("math on integers"),
            })
        }
        Ty::Df32 => match op {
            BinOp::Add => a + b,
            BinOp::Sub => a - b,
            BinOp::Mul => a * b,
            BinOp::Div => a / b,
            BinOp::Minimum => a.min(b),
            BinOp::Maximum => a.max(b),
            _ => unreachable!("math on double-floats"),
        },
    }
}

pub fn apply_un(ty: Ty, op: UnOp, a: f64) -> f64 {
    match (ty, op) {
        (_, UnOp::ToDf32) => a,
        (_, UnOp::ToF32) => f64::from(a as f32),
        (Ty::F32, op) => {
            let x = a as f32;
            f64::from(match op {
                UnOp::Neg => -x,
                UnOp::Abs => x.abs(),
                UnOp::Floor => x.floor(),
                UnOp::Ceil => x.ceil(),
                UnOp::Sign => sign32(x),
                UnOp::Sqrt => x.sqrt(),
                UnOp::Exp => x.exp(),
                UnOp::Ln => x.ln(),
                UnOp::Sin => x.sin(),
                UnOp::Cos => x.cos(),
                UnOp::Tan => x.tan(),
                UnOp::Tanh => x.tanh(),
                UnOp::Clamp { lo, hi } => x.max(lo).min(hi),
                UnOp::Powf(e) => x.powf(e),
                UnOp::ToDf32 | UnOp::ToF32 => unreachable!(),
            })
        }
        (Ty::Df32, UnOp::Neg) => -a,
        (Ty::Df32, UnOp::Abs) => a.abs(),
        (Ty::Df32, UnOp::Sqrt) => a.sqrt(),
        (ty, op) => unreachable!("{op:?} on {ty:?}"),
    }
}

fn apply_cmp(op: CmpOp, a: f64, b: f64) -> f64 {
    let r = match op {
        CmpOp::Lt => a < b,
        CmpOp::Le => a <= b,
        CmpOp::Gt => a > b,
        CmpOp::Ge => a >= b,
        CmpOp::Equal => a == b,
        CmpOp::NotEqual => a != b,
    };
    f64::from(u8::from(r))
}

impl HostArray {
    pub fn new(dims: Dims, ty: Ty, data: Vec<f64>) -> HostArray {
        assert_eq!(data.len(), len_of(dims));
        let data = data.into_iter().map(|v| store(ty, v)).collect();
        HostArray { dims, ty, cell: Rc::new(RefCell::new(Cell { data })), view: None }
    }

    pub fn len(&self) -> usize {
        len_of(self.dims)
    }

    /// The elements now.
    pub fn data(&self) -> Vec<f64> {
        self.cell.borrow().data.clone()
    }

    /// The array that `cache` returns: the same data, no view relation.
    pub fn cached(&self) -> HostArray {
        HostArray { view: None, ..self.clone() }
    }

    /// A new array holding the elements now (`self * 1`).
    pub fn copied(&self) -> HostArray {
        HostArray::new(self.dims, self.ty, self.data())
    }

    // -- Creation --

    pub fn full(dims: Dims, ty: Ty, value: f64) -> HostArray {
        HostArray::new(dims, ty, vec![value; len_of(dims)])
    }

    /// Each element's coordinate along `axis`.
    pub fn range(dims: Dims, ty: Ty, axis: usize) -> Res<HostArray> {
        if axis >= 4 {
            return Err(ErrKind::InvalidShape);
        }
        let data = (0..len_of(dims)).map(|i| coords(i, dims)[axis] as f64).collect();
        Ok(HostArray::new(dims, ty, data))
    }

    /// `start + i * step` in `f32` with `step = (end - start) / (n - 1)`.
    pub fn linspace(start: f32, end: f32, n: usize) -> HostArray {
        let step = if n > 1 { (end - start) / (n - 1) as f32 } else { 0.0 };
        let data = (0..n).map(|i| f64::from(start + i as f32 * step)).collect();
        HostArray::new([n, 1, 1, 1], Ty::F32, data)
    }

    // -- Elementwise --

    pub fn unary(&self, op: UnOp) -> HostArray {
        let ty = match op {
            UnOp::ToDf32 => Ty::Df32,
            UnOp::ToF32 => Ty::F32,
            _ => self.ty,
        };
        let data = self.data().into_iter().map(|v| apply_un(self.ty, op, v)).collect();
        HostArray::new(self.dims, ty, data)
    }

    /// The two operands of `self op rhs` as vectors of the result's shape, broadcasting a
    /// one-element operand.
    fn operands(&self, rhs: Rhs<'_>) -> Res<(Dims, Vec<f64>, Vec<f64>)> {
        match rhs {
            Rhs::Val(v) => Ok((self.dims, self.data(), vec![v; self.len()])),
            Rhs::Array(o) => {
                if self.dims == o.dims {
                    Ok((self.dims, self.data(), o.data()))
                } else if o.len() == 1 {
                    Ok((self.dims, self.data(), vec![o.data()[0]; self.len()]))
                } else if self.len() == 1 {
                    Ok((o.dims, vec![self.data()[0]; o.len()], o.data()))
                } else {
                    Err(ErrKind::ShapeMismatch)
                }
            }
        }
    }

    /// `self op rhs`, or `rhs op self` if `scalar_left`.
    pub fn binary(&self, op: BinOp, rhs: Rhs<'_>, scalar_left: bool) -> Res<HostArray> {
        let (dims, a, b) = self.operands(rhs)?;
        let data = a
            .into_iter()
            .zip(b)
            .map(|(a, b)| if scalar_left { apply_bin(self.ty, op, b, a) } else { apply_bin(self.ty, op, a, b) })
            .collect();
        Ok(HostArray::new(dims, self.ty, data))
    }

    pub fn compare(&self, op: CmpOp, rhs: Rhs<'_>) -> Res<HostArray> {
        let (dims, a, b) = self.operands(rhs)?;
        let data = a.into_iter().zip(b).map(|(a, b)| apply_cmp(op, a, b)).collect();
        Ok(HostArray::new(dims, Ty::U32, data))
    }

    /// `mask != 0 ? a : b` with the shape of the mask; a branch is an array of that shape, one
    /// element or a value.
    pub fn select(&self, a: Rhs<'_>, b: Rhs<'_>, ty: Ty) -> Res<HostArray> {
        let n = self.len();
        let branch = |rhs: Rhs<'_>| -> Res<Vec<f64>> {
            match rhs {
                Rhs::Val(v) => Ok(vec![v; n]),
                Rhs::Array(o) if o.dims == self.dims => Ok(o.data()),
                Rhs::Array(o) if o.len() == 1 => Ok(vec![o.data()[0]; n]),
                Rhs::Array(_) => Err(ErrKind::ShapeMismatch),
            }
        };
        let (a, b) = (branch(a)?, branch(b)?);
        let data = self.data().into_iter().zip(a.into_iter().zip(b)).map(|(m, (a, b))| if m != 0.0 { a } else { b }).collect();
        Ok(HostArray::new(self.dims, ty, data))
    }

    // -- Views --

    /// The values of `self` read through `map`; `edge` says what coordinates outside read.
    fn gather(&self, map: &ViewMap, edge: Edge) -> Vec<f64> {
        let src = self.data();
        (0..len_of(map.out))
            .map(|o| {
                let raw = map.source(coords(o, map.out));
                let mut at = [0usize; 4];
                for a in 0..4 {
                    let (c, n) = (raw[a], map.src[a] as i64);
                    let inside = (0..n).contains(&c);
                    let c = match edge {
                        Edge::None => {
                            assert!(inside, "view reads outside of its source");
                            c
                        }
                        Edge::Fill(v) => {
                            if !(0..4).all(|b| (0..map.src[b] as i64).contains(&raw[b])) {
                                return v;
                            }
                            c
                        }
                        Edge::Clamp => c.clamp(0, n - 1),
                        Edge::Mirror => mirror(c, n),
                        Edge::Wrap => wrap(c, n),
                    };
                    at[a] = c as usize;
                }
                src[flat(at, map.src)]
            })
            .collect()
    }

    /// The view of `self` through `map`. Without an edge rule it is injective and writes through.
    fn read_through(&self, map: ViewMap, edge: Edge) -> HostArray {
        let data = self.gather(&map, edge);
        let mut out = HostArray::new(map.out, self.ty, data);
        if edge == Edge::None {
            out.view = Some(match &self.view {
                Some((base, inner)) => (Rc::clone(base), inner.then(&map)),
                None => (Rc::clone(&self.cell), map),
            });
        }
        out
    }

    pub fn slice(&self, axis: usize, lo: usize, hi: usize) -> Res<HostArray> {
        if axis >= 4 || lo >= hi || hi > self.dims[axis] {
            return Err(ErrKind::InvalidShape);
        }
        let mut map = ViewMap::identity(self.dims);
        map.out[axis] = hi - lo;
        map.axes[axis].start = lo as i64;
        Ok(self.read_through(map, Edge::None))
    }

    pub fn flip(&self, axis: usize) -> Res<HostArray> {
        if axis >= 4 {
            return Err(ErrKind::InvalidShape);
        }
        let mut map = ViewMap::identity(self.dims);
        map.axes[axis] = Axis { from: axis, start: self.dims[axis] as i64 - 1, step: -1 };
        Ok(self.read_through(map, Edge::None))
    }

    pub fn transpose(&self) -> HostArray {
        let d = self.dims;
        let mut map = ViewMap::identity(d);
        map.out = [d[1], d[0], d[2], d[3]];
        map.axes[0].from = 1;
        map.axes[1].from = 0;
        self.read_through(map, Edge::None)
    }

    /// `widths[k]` cells before and after along axis `k`.
    pub fn pad(&self, widths: &[(usize, usize)], mode: crate::PadMode) -> HostArray {
        let mut map = ViewMap::identity(self.dims);
        for (k, &(before, after)) in widths.iter().enumerate() {
            map.out[k] += before + after;
            map.axes[k].start = -(before as i64);
        }
        let edge = match mode {
            crate::PadMode::Edge => Edge::Clamp,
            crate::PadMode::Mirror => Edge::Mirror,
            crate::PadMode::Wrap => Edge::Wrap,
            crate::PadMode::Zero => Edge::Fill(0.0),
        };
        self.read_through(map, edge)
    }

    /// Element `i` moves to `i + offsets` on every axis; cells nothing moves into follow `edge`.
    pub fn shift(&self, offsets: &[isize], edge: Edge) -> HostArray {
        let mut map = ViewMap::identity(self.dims);
        for (a, &off) in offsets.iter().enumerate() {
            map.axes[a].start = -(off as i64);
        }
        self.read_through(map, edge)
    }

    /// Gathers `self[min(idx[i], len - 1)]` with the shape of `idx`.
    pub fn take(&self, idx: &HostArray) -> HostArray {
        let data = self.data();
        let out = idx.data().into_iter().map(|i| data[(i as usize).min(data.len() - 1)]).collect();
        HostArray::new(idx.dims, self.ty, out)
    }

    pub fn concat(axis: usize, parts: &[&HostArray]) -> Res<HostArray> {
        let first = parts[0];
        if axis >= 4 {
            return Err(ErrKind::InvalidShape);
        }
        let mut dims = first.dims;
        for part in &parts[1..] {
            if (0..4).any(|a| a != axis && part.dims[a] != dims[a]) {
                return Err(ErrKind::ShapeMismatch);
            }
            dims[axis] += part.dims[axis];
        }
        let mut data = vec![0.0; len_of(dims)];
        let mut offset = 0;
        for part in parts {
            let values = part.data();
            for (i, v) in values.into_iter().enumerate() {
                let mut c = coords(i, part.dims);
                c[axis] += offset;
                data[flat(c, dims)] = v;
            }
            offset += part.dims[axis];
        }
        Ok(HostArray::new(dims, first.ty, data))
    }

    // -- Reductions --

    fn fold(&self, op: RedOp, values: &[f64]) -> f64 {
        // Sums of `f32` are accumulated in `f64` and rounded once: the tolerances cover the order.
        if op == RedOp::Sum && self.ty == Ty::F32 {
            return store(self.ty, values.iter().sum::<f64>());
        }
        let bin = match op {
            RedOp::Sum => BinOp::Add,
            RedOp::Min => BinOp::Minimum,
            RedOp::Max => BinOp::Maximum,
        };
        values[1..].iter().fold(values[0], |acc, &v| apply_bin(self.ty, bin, acc, v))
    }

    pub fn reduce(&self, op: RedOp) -> HostArray {
        HostArray::new([1, 1, 1, 1], self.ty, vec![self.fold(op, &self.data())])
    }

    pub fn sum_where(&self, mask: &HostArray) -> Res<HostArray> {
        let selected = mask.select(Rhs::Array(self), Rhs::Val(0.0), self.ty)?;
        Ok(selected.reduce(RedOp::Sum))
    }

    /// The elements of each line along `axis`, in order, with the position of their first.
    fn lines(&self, axis: usize) -> Vec<(usize, Vec<usize>)> {
        let mut out = Vec::new();
        let s = strides(self.dims);
        for i in 0..self.len() {
            if coords(i, self.dims)[axis] == 0 {
                out.push((i, (0..self.dims[axis]).map(|k| i + k * s[axis]).collect()));
            }
        }
        out
    }

    pub fn scan(&self, axis: usize, op: ScanOp) -> HostArray {
        let data = self.data();
        let mut out = vec![0.0; data.len()];
        let bin = match op {
            ScanOp::CumSum => BinOp::Add,
            ScanOp::ScanMax => BinOp::Maximum,
            ScanOp::ScanMin => BinOp::Minimum,
        };
        for (_, line) in self.lines(axis) {
            let mut acc = data[line[0]];
            out[line[0]] = acc;
            for &i in &line[1..] {
                acc = apply_bin(self.ty, bin, acc, data[i]);
                out[i] = acc;
            }
        }
        HostArray::new(self.dims, self.ty, out)
    }

    pub fn reduce_axis(&self, axis: usize, op: RedOp) -> HostArray {
        let data = self.data();
        let mut dims = self.dims;
        dims[axis] = 1;
        let mut out = vec![0.0; len_of(dims)];
        for (first, line) in self.lines(axis) {
            let values: Vec<f64> = line.iter().map(|&i| data[i]).collect();
            out[flat(coords(first, self.dims), dims)] = self.fold(op, &values);
        }
        HostArray::new(dims, self.ty, out)
    }

    // -- Writes --

    fn values_of(&self, rhs: Rhs<'_>) -> Res<Vec<f64>> {
        match rhs {
            Rhs::Val(v) => Ok(vec![store(self.ty, v); self.len()]),
            Rhs::Array(o) if o.dims == self.dims => Ok(o.data()),
            Rhs::Array(_) => Err(ErrKind::ShapeMismatch),
        }
    }

    /// Writes `rhs` into the array this one is a view of (this one itself if it is none), and
    /// makes this one show `rhs`.
    pub fn assign(&self, rhs: Rhs<'_>) -> Res<()> {
        let values = self.values_of(rhs)?;
        if let Some((base, map)) = &self.view {
            let mut base = base.borrow_mut();
            for (o, &v) in values.iter().enumerate() {
                let at: [usize; 4] = map.source(coords(o, self.dims)).map(|c| c as usize);
                base.data[flat(at, map.src)] = v;
            }
        }
        self.cell.borrow_mut().data = values;
        Ok(())
    }

    /// `self[idx[i]] = values[i]` for the `idx[i] < len`; `values` has the shape of `idx`.
    pub fn put(&self, idx: &HostArray, values: Rhs<'_>) -> Res<()> {
        if self.view.is_some() {
            return Err(ErrKind::InvalidShape);
        }
        let values = match values {
            Rhs::Val(v) => vec![store(self.ty, v); idx.len()],
            Rhs::Array(o) if o.dims == idx.dims => o.data(),
            Rhs::Array(_) => return Err(ErrKind::ShapeMismatch),
        };
        let mut cell = self.cell.borrow_mut();
        for (i, v) in idx.data().into_iter().zip(values) {
            if (i as usize) < cell.data.len() {
                cell.data[i as usize] = v;
            }
        }
        Ok(())
    }

    /// Replaces the elements (for every handle on the data).
    pub fn write(&self, values: &[f64]) -> Res<()> {
        if values.len() != self.len() {
            return Err(ErrKind::LengthMismatch);
        }
        self.cell.borrow_mut().data = values.iter().map(|&v| store(self.ty, v)).collect();
        Ok(())
    }

    /// `self = self op rhs` on the shared data.
    pub fn compound(&self, op: BinOp, rhs: Rhs<'_>) -> Res<()> {
        let b = match rhs {
            Rhs::Val(v) => vec![v; self.len()],
            Rhs::Array(o) if o.dims == self.dims => o.data(),
            Rhs::Array(o) if o.len() == 1 => vec![o.data()[0]; self.len()],
            Rhs::Array(_) => return Err(ErrKind::ShapeMismatch),
        };
        let data = self.data().into_iter().zip(b).map(|(a, b)| apply_bin(self.ty, op, a, store(self.ty, b))).collect();
        self.cell.borrow_mut().data = data;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PadMode;

    fn line(values: &[f64]) -> HostArray {
        HostArray::new([values.len(), 1, 1, 1], Ty::F32, values.to_vec())
    }

    fn grid(w: usize, h: usize) -> HostArray {
        HostArray::new([w, h, 1, 1], Ty::F32, (0..w * h).map(|i| i as f64).collect())
    }

    #[test]
    fn pad_modes_follow_the_documented_boundary_rules() {
        let a = line(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(a.pad(&[(2, 2)], PadMode::Edge).data(), [1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0]);
        // `-1` reads 0, `-2` reads 1, `n` reads `n - 1`, `n + 1` reads `n - 2`.
        assert_eq!(a.pad(&[(2, 2)], PadMode::Mirror).data(), [2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 3.0]);
        assert_eq!(a.pad(&[(2, 1)], PadMode::Wrap).data(), [3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0]);
        assert_eq!(a.pad(&[(1, 2)], PadMode::Zero).data(), [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0]);
        // The second axis of a grid.
        let g = grid(2, 2);
        assert_eq!(g.pad(&[(0, 0), (1, 0)], PadMode::Edge).data(), [0.0, 1.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn shifts() {
        let a = line(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(a.shift(&[1], Edge::Wrap).data(), [4.0, 1.0, 2.0, 3.0]);
        assert_eq!(a.shift(&[-1], Edge::Wrap).data(), [2.0, 3.0, 4.0, 1.0]);
        assert_eq!(a.shift(&[1], Edge::Fill(9.0)).data(), [9.0, 1.0, 2.0, 3.0]);
        assert_eq!(a.shift(&[-1], Edge::Clamp).data(), [2.0, 3.0, 4.0, 4.0]);
        assert_eq!(a.shift(&[2], Edge::Mirror).data(), [2.0, 1.0, 1.0, 2.0]);
        // A 2-D shift moves along both axes.
        let g = grid(3, 2);
        assert_eq!(g.shift(&[1, 1], Edge::Wrap).data(), [5.0, 3.0, 4.0, 2.0, 0.0, 1.0]);
    }

    #[test]
    fn views_compose_and_transpose_swaps_the_first_axes() {
        let g = grid(3, 2);
        assert_eq!(g.transpose().dims, [2, 3, 1, 1]);
        assert_eq!(g.transpose().data(), [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        assert_eq!(g.flip(0).unwrap().data(), [2.0, 1.0, 0.0, 5.0, 4.0, 3.0]);
        let s = g.slice(0, 1, 3).unwrap().flip(0).unwrap();
        assert_eq!(s.data(), [2.0, 1.0, 5.0, 4.0]);
        assert!(g.slice(0, 2, 4).is_err());
    }

    #[test]
    fn assign_through_a_slice_of_a_clone_reaches_every_handle_but_not_older_arrays() {
        let a = line(&[1.0, 2.0, 3.0, 4.0]);
        let b = a.clone();
        let old = a.binary(BinOp::Mul, Rhs::Val(1.0), false).unwrap();
        let earlier = a.slice(0, 0, 2).unwrap();
        let v = b.slice(0, 1, 3).unwrap().flip(0).unwrap();
        v.assign(Rhs::Array(&line(&[10.0, 20.0]))).unwrap();
        assert_eq!(a.data(), [1.0, 20.0, 10.0, 4.0]);
        assert_eq!(b.data(), [1.0, 20.0, 10.0, 4.0]);
        assert_eq!(v.data(), [10.0, 20.0]);
        assert_eq!(old.data(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(earlier.data(), [1.0, 2.0]);
        // A second write through the same view lands on the same cells.
        v.assign(Rhs::Val(7.0)).unwrap();
        assert_eq!(a.data(), [1.0, 7.0, 7.0, 4.0]);
        // Not through an expression.
        let t = a.binary(BinOp::Add, Rhs::Val(1.0), false).unwrap();
        t.assign(Rhs::Val(0.0)).unwrap();
        assert_eq!(a.data(), [1.0, 7.0, 7.0, 4.0]);
        assert_eq!(t.data(), [0.0; 4]);
    }

    #[test]
    fn put_skips_indices_past_the_end() {
        let x = line(&[0.0; 5]);
        let idx = HostArray::new([3, 1, 1, 1], Ty::U32, vec![4.0, 1.0, 9.0]);
        x.put(&idx, Rhs::Array(&line(&[1.0, 2.0, 3.0]))).unwrap();
        assert_eq!(x.data(), [0.0, 2.0, 0.0, 0.0, 1.0]);
        assert_eq!(x.slice(0, 0, 2).unwrap().put(&idx, Rhs::Val(1.0)), Err(ErrKind::InvalidShape));
    }

    #[test]
    fn scans_and_axis_reductions() {
        let g = grid(3, 2);
        assert_eq!(g.scan(0, ScanOp::CumSum).data(), [0.0, 1.0, 3.0, 3.0, 7.0, 12.0]);
        assert_eq!(g.scan(1, ScanOp::CumSum).data(), [0.0, 1.0, 2.0, 3.0, 5.0, 7.0]);
        assert_eq!(g.scan(0, ScanOp::ScanMax).data(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(g.reduce_axis(1, RedOp::Sum).data(), [3.0, 5.0, 7.0]);
        assert_eq!(g.reduce_axis(0, RedOp::Max).dims, [1, 2, 1, 1]);
        assert_eq!(g.reduce_axis(0, RedOp::Max).data(), [2.0, 5.0]);
        assert_eq!(g.reduce(RedOp::Min).data(), [0.0]);
    }

    #[test]
    fn broadcast_select_take_concat_and_integer_wraparound() {
        let a = line(&[1.0, 2.0, 3.0]);
        let one = line(&[10.0]);
        assert_eq!(a.binary(BinOp::Add, Rhs::Array(&one), false).unwrap().data(), [11.0, 12.0, 13.0]);
        assert_eq!(one.binary(BinOp::Sub, Rhs::Array(&a), false).unwrap().data(), [9.0, 8.0, 7.0]);
        assert_eq!(a.binary(BinOp::Sub, Rhs::Val(10.0), true).unwrap().data(), [9.0, 8.0, 7.0]);
        assert_eq!(a.binary(BinOp::Add, Rhs::Array(&line(&[1.0, 2.0])), false).err(), Some(ErrKind::ShapeMismatch));
        let m = a.compare(CmpOp::Gt, Rhs::Val(1.5)).unwrap();
        assert_eq!(m.data(), [0.0, 1.0, 1.0]);
        assert_eq!(m.select(Rhs::Array(&a), Rhs::Val(-1.0), Ty::F32).unwrap().data(), [-1.0, 2.0, 3.0]);
        let idx = HostArray::new([4, 1, 1, 1], Ty::U32, vec![2.0, 0.0, 7.0, 1.0]);
        assert_eq!(a.take(&idx).data(), [3.0, 1.0, 3.0, 2.0]);
        assert_eq!(HostArray::concat(0, &[&a, &one]).unwrap().data(), [1.0, 2.0, 3.0, 10.0]);
        let u = HostArray::new([1, 1, 1, 1], Ty::U32, vec![0.0]);
        assert_eq!(u.binary(BinOp::Sub, Rhs::Val(1.0), false).unwrap().data(), [f64::from(u32::MAX)]);
    }
}
