//! Host execution of small jobs: when the context can run on the host (see
//! `Context::can_run_on_host`) and the data is host resident, elementwise expressions,
//! reductions, matrix products, sorts and scatters that are small enough run on the CPU: no
//! launch, no shader, no transfer. Every `try_*` function returns `None` (or `false`) without
//! side effects when the job does not qualify, and the caller takes the GPU path.
//!
//! Values travel as 32-bit words: the bits of an `f32`, `u32` or `i32`, and 0 or 1 for a `bool`
//! (packed into bits only when a buffer is written).

use std::sync::Arc;

use crate::array::{Array, Storage};
use crate::builtin::{BinOp, Class, CmpOp, ReduceList, ReduceOp, UnOp};
use crate::context::Context;
use crate::gpu::hash::FxMap;
use crate::gpu_type::{ElemLayout, GpuType, ScalarType};
use crate::jit::{AxisMap, Edge, ElemTy, Gen, Node, OpKind, Ref, Scalar, ShiftKind, Src, ViewMap};
use crate::ops::Rhs;
use crate::jit::structured::Structured;
use crate::shape::Shape;
use crate::sort::{KeyTransform, SortKey};

/// Most elements of an array an operation runs on the host for.
pub(crate) const MAX_HOST_ELEMS: usize = 65_536;
/// Most elements a sort, factorization or scatter runs on the host for.
pub(crate) const MAX_SORT_ELEMS: usize = 32_768;
/// Most multiply-adds of a matrix product that runs on the host.
pub(crate) const MAX_MATMUL_FMAS: u64 = 1_000_000;
/// Most element evaluations of one expression: beyond it a fused kernel is cheaper.
const MAX_WORK: usize = 16 * MAX_HOST_ELEMS;

type Words = Arc<Vec<u32>>;

fn supported(ty: ElemTy) -> bool {
    matches!(ty, ElemTy::F32 | ElemTy::U32 | ElemTy::I32 | ElemTy::Bool)
}

/// The elements of a host-resident buffer as words (a packed `bool` unpacked).
fn load(storage: &Storage) -> Option<Vec<u32>> {
    let n = storage.len();
    let layout = storage.layout();
    storage
        .with_host_bytes(|bytes| {
            if layout.is_bits() {
                let words = bytes.get(..n.div_ceil(32) * 4)?;
                Some((0..n).map(|i| (word_at(words, i / 32) >> (i % 32)) & 1).collect())
            } else if layout.stride() == 4 && !layout.is_packed() {
                let bytes = bytes.get(..n * 4)?;
                if let Ok(slice) = bytemuck::try_cast_slice::<u8, u32>(bytes) {
                    Some(slice.to_vec())
                } else {
                    Some(bytes.as_chunks::<4>().0.iter().copied().map(u32::from_ne_bytes).collect())
                }
            } else {
                None
            }
        })
        .flatten()
}

fn word_at(bytes: &[u8], j: usize) -> u32 {
    u32::from_ne_bytes([bytes[4 * j], bytes[4 * j + 1], bytes[4 * j + 2], bytes[4 * j + 3]])
}

fn write_words(dst: &mut [u8], words: &[u32]) {
    let len = words.len() * 4;
    if let Some(target) = dst.get_mut(..len)
        && let Ok(slice) = bytemuck::try_cast_slice_mut::<u8, u32>(target)
    {
        slice.copy_from_slice(words);
        return;
    }
    for (chunk, word) in dst.as_chunks_mut::<4>().0.iter_mut().zip(words) {
        *chunk = word.to_ne_bytes();
    }
}

/// A host buffer of `len` elements of type `ty` holding `words`.
fn store_words(ctx: &Context, len: usize, ty: ElemTy, words: &[u32]) -> Option<Arc<Storage>> {
    if words.len() != len {
        return None;
    }
    let layout = ty.layout();
    Storage::alloc_host(ctx, len, layout, |dst| {
        if layout.is_bits() {
            for (j, chunk) in dst.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let mut word = 0u32;
                for (b, &v) in words[j * 32..(j * 32 + 32).min(len)].iter().enumerate() {
                    word |= u32::from(v != 0) << b;
                }
                *chunk = word.to_ne_bytes();
            }
        } else {
            write_words(dst, words);
        }
    })
    .ok()
}

/// An operand of an operation: one value for every element, or one per element.
#[derive(Clone)]
enum Opnd {
    Scalar(u32),
    Arr(Words),
}

impl Opnd {
    #[inline(always)]
    fn at(&self, i: usize) -> u32 {
        match self {
            Opnd::Scalar(v) => *v,
            Opnd::Arr(words) => words[i],
        }
    }
}

fn zip(len: usize, a: &Opnd, b: &Opnd, c: &Opnd, f: impl Fn(u32, u32, u32) -> u32) -> Vec<u32> {
    match (a, b, c) {
        (Opnd::Arr(a), Opnd::Arr(b), Opnd::Scalar(c)) => a[..len].iter().zip(&b[..len]).map(|(&x, &y)| f(x, y, *c)).collect(),
        (Opnd::Arr(a), Opnd::Scalar(b), Opnd::Scalar(c)) => a[..len].iter().map(|&x| f(x, *b, *c)).collect(),
        (Opnd::Scalar(a), Opnd::Arr(b), Opnd::Scalar(c)) => b[..len].iter().map(|&y| f(*a, y, *c)).collect(),
        (Opnd::Arr(a), Opnd::Arr(b), Opnd::Arr(c)) => (0..len).map(|i| f(a[i], b[i], c[i])).collect(),
        _ => (0..len).map(|i| f(a.at(i), b.at(i), c.at(i))).collect(),
    }
}

/// Evaluates expressions on the host, sharing the work of nodes and buffers read more than once.
struct Eval<'a> {
    ctx: &'a Context,
    nodes: FxMap<u64, Words>,
    bufs: FxMap<u64, Words>,
    work: usize,
}

impl<'a> Eval<'a> {
    fn new(ctx: &'a Context) -> Self {
        Eval { ctx, nodes: FxMap::default(), bufs: FxMap::default(), work: 0 }
    }

    fn charge(&mut self, n: usize) -> Option<()> {
        self.work += n;
        (self.work <= MAX_WORK).then_some(())
    }

    fn buffer(&mut self, storage: &Storage) -> Option<Words> {
        if let Some(words) = self.bufs.get(&storage.id()) {
            return Some(Arc::clone(words));
        }
        if storage.len() > MAX_HOST_ELEMS {
            return None;
        }
        self.charge(storage.len())?;
        let words = Arc::new(load(storage)?);
        self.bufs.insert(storage.id(), Arc::clone(&words));
        Some(words)
    }

    /// The value of an operand that must have `len` elements if it is an array.
    fn src(&mut self, src: &Src, len: usize) -> Option<Opnd> {
        let words = match src {
            Src::Scalar(s) => return supported(s.ty).then_some(Opnd::Scalar(s.bits as u32)),
            Src::Buffer(storage) => self.buffer(storage)?,
            Src::Node(node) => self.node(node)?,
        };
        (words.len() == len).then_some(Opnd::Arr(words))
    }

    /// The value of a scalar operand.
    fn scalar(src: &Src) -> Option<u32> {
        match src {
            Src::Scalar(s) if supported(s.ty) => Some(s.bits as u32),
            _ => None,
        }
    }

    fn node(&mut self, node: &Arc<Node>) -> Option<Words> {
        if let Some(words) = self.nodes.get(&node.id()) {
            return Some(Arc::clone(words));
        }
        if node.len() > MAX_HOST_ELEMS || node.failure().is_some() || node.is_looped() || !supported(node.op().out_ty()) {
            return None;
        }
        let words = if let Some(storage) = node.evaluated() {
            self.buffer(storage)?
        } else if let Some(reduced) = node.deferred_value() {
            let storage = reduced.host_result(self.ctx)?;
            self.buffer(&storage)?
        } else {
            let args = node.operands()?;
            self.charge(node.len())?;
            let computed = Arc::new(self.compute(node, &args)?);
            if node.has_consumer_outside(|_| false)
                && let Some(storage) = store_words(self.ctx, node.len(), node.op().out_ty(), &computed)
            {
                self.bufs.insert(storage.id(), Arc::clone(&computed));
                node.finish_shared(storage);
            }
            computed
        };
        if words.len() != node.len() {
            return None;
        }
        self.nodes.insert(node.id(), Arc::clone(&words));
        Some(words)
    }

    /// The words of any array that is host resident or computable on the host.
    fn array<T: GpuType>(&mut self, array: &Array<T>) -> Option<Words> {
        let words = if let Some(node) = array.pending() {
            let words = self.node(node)?;
            if node.is_pending()
                && let Some(storage) = store_words(self.ctx, node.len(), node.op().out_ty(), &words)
            {
                self.bufs.insert(storage.id(), Arc::clone(&words));
                node.finish_shared(storage);
            }
            words
        } else if let Some(reduced) = array.structured() {
            match reduced.result_if_recorded() {
                Some(Ok(storage)) => self.buffer(storage)?,
                Some(Err(_)) => return None,
                None => { let storage = reduced.host_result(self.ctx)?; self.buffer(&storage)? }
            }
        } else if let Some(storage) = array.storage_if_valid() {
            self.buffer(storage)?
        } else {
            self.buffer(array.held()?.latest().evaluated()?)?
        };
        (words.len() == array.len()).then_some(words)
    }

    fn compute(&mut self, node: &Arc<Node>, args: &[Src; 3]) -> Option<Vec<u32>> {
        let op = node.op();
        let (ty, len) = (op.ty, node.len());
        if !supported(ty) {
            return None;
        }
        let none = Opnd::Scalar(0);
        let operand = |this: &mut Self, j: usize| if j < op.kind.arity() { this.src(&args[j], len) } else { Some(none.clone()) };
        match op.kind {
            OpKind::Generate(kind) => self.generate(kind, ty, len, args),
            OpKind::Shift(kind) => self.shift(kind, op.ty, len, args),
            kind => {
                let (a, b, c) = (operand(self, 0)?, operand(self, 1)?, operand(self, 2)?);
                Some(match kind {
                    OpKind::Unary(u) => {
                        u.expr_for(ty)?;
                        if u.needs_helper() || ty == ElemTy::F64 {
                            return None;
                        }
                        zip(len, &a, &b, &c, |x, _, _| unary(u, ty, x))
                    }
                    OpKind::Binary(bin) => {
                        bin.expr_for(ty)?;
                        if bin.needs_helper() || bin == BinOp::Mod || ty == ElemTy::F64 {
                            return None;
                        }
                        zip(len, &a, &b, &c, |x, y, _| binary(bin, ty, x, y))
                    }
                    OpKind::Compare(cmp) => zip(len, &a, &b, &c, |x, y, _| compare(cmp, ty, x, y)),
                    OpKind::Classify(class) => {
                        if ty != ElemTy::F32 {
                            return None;
                        }
                        zip(len, &a, &b, &c, |x, _, _| classify(class, x))
                    }
                    OpKind::Powf => {
                        if ty != ElemTy::F32 {
                            return None;
                        }
                        zip(len, &a, &b, &c, |x, y, _| f32::from_bits(x).powf(f32::from_bits(y)).to_bits())
                    }
                    OpKind::Clamp => match ty {
                        ElemTy::F32 => zip(len, &a, &b, &c, |x, lo, hi| f32::from_bits(x).max(f32::from_bits(lo)).min(f32::from_bits(hi)).to_bits()),
                        ElemTy::I32 => zip(len, &a, &b, &c, |x, lo, hi| (x as i32).max(lo as i32).min(hi as i32) as u32),
                        _ => zip(len, &a, &b, &c, |x, lo, hi| x.max(lo).min(hi)),
                    },
                    OpKind::Select => zip(len, &a, &b, &c, |cond, x, y| if cond != 0 { x } else { y }),
                    OpKind::Cast(to) => {
                        let cast = cast_fn(ty, to)?;
                        zip(len, &a, &b, &c, |x, _, _| cast(x))
                    }
                    OpKind::Generate(_) | OpKind::Shift(_) | OpKind::Part(_) | OpKind::Make | OpKind::Call { .. } => return None,
                })
            }
        }
    }

    fn generate(&mut self, kind: Gen, ty: ElemTy, len: usize, args: &[Src; 3]) -> Option<Vec<u32>> {
        let shaped = |coord: u32| match ty {
            ElemTy::F32 => Some((coord as f32).to_bits()),
            ElemTy::U32 | ElemTy::I32 => Some(coord),
            _ => None,
        };
        // Bools have no generators.
        shaped(0)?;
        let at = |i: usize| i as u32;
        match kind {
            Gen::Index => Some((0..len).map(|i| shaped(at(i)).unwrap_or(0)).collect()),
            Gen::RangePow2 => {
                let (shift, mask) = (Self::scalar(&args[0])?, Self::scalar(&args[1])?);
                Some((0..len).map(|i| shaped(at(i).wrapping_shr(shift) & mask).unwrap_or(0)).collect())
            }
            Gen::RangeConst { stride, size, .. } => {
                if stride == 0 || size == 0 {
                    return None;
                }
                Some((0..len).map(|i| shaped((at(i) / stride) % size).unwrap_or(0)).collect())
            }
            Gen::Linspace => {
                if ty != ElemTy::F32 {
                    return None;
                }
                let (start, step) = (f32::from_bits(Self::scalar(&args[0])?), f32::from_bits(Self::scalar(&args[1])?));
                Some((0..len).map(|i| (start + (i as f32) * step).to_bits()).collect())
            }
            Gen::Bits | Gen::Uniform | Gen::Normal | Gen::ParamRange { .. } => None,
        }
    }

    fn shift(&mut self, kind: ShiftKind, ty: ElemTy, len: usize, args: &[Src; 3]) -> Option<Vec<u32>> {
        if !supported(ty) {
            return None;
        }
        match kind {
            ShiftKind::Linear => {
                let (s, total) = (Self::scalar(&args[1])? as usize, Self::scalar(&args[2])? as usize);
                let src = self.src(&args[0], len)?;
                if total != len || s > len {
                    return None;
                }
                Some((0..len).map(|i| src.at(if i < s { i + (len - s) } else { i - s })).collect())
            }
            ShiftKind::Dims { dims, shifts } => {
                let d = dims.map(|d| d as usize);
                if d.contains(&0) || d.iter().product::<usize>() != len {
                    return None;
                }
                let src = self.src(&args[0], len)?;
                let strides = [1, d[0], d[0] * d[1], d[0] * d[1] * d[2]];
                Some(
                    (0..len)
                        .map(|i| {
                            let mut at = 0;
                            for a in 0..4 {
                                let c = (i / strides[a]) % d[a];
                                at += ((c + d[a] - (shifts[a] as usize % d[a])) % d[a]) * strides[a];
                            }
                            src.at(at)
                        })
                        .collect(),
                )
            }
            ShiftKind::View { map, edge } => {
                let src_len: usize = map.src.iter().map(|&d| d as usize).product();
                if map.out_len() != len {
                    return None;
                }
                let src = self.src(&args[0], src_len)?;
                let fill = if edge == Edge::Fill { Self::scalar(&args[1])? } else { 0 };
                view_words(&src, src_len, &map, edge, fill)
            }
            ShiftKind::Take { len: src_len } => {
                let src_len = src_len as usize;
                let src = self.src(&args[0], src_len)?;
                let indices = self.src(&args[1], len)?;
                let top = src_len.checked_sub(1)?;
                Some((0..len).map(|i| src.at((indices.at(i) as usize).min(top))).collect())
            }
            ShiftKind::ParamView { .. } | ShiftKind::ParamTake { .. } | ShiftKind::ParamDims { .. } => None,
        }
    }
}

fn mirror(c: i64, n: i64) -> i64 {
    let a = if c < 0 { -c - 1 } else { c };
    let b = if a >= n { 2 * n - 1 - a } else { a };
    b.clamp(0, n - 1)
}

/// The elements of `src` (`src_len` of them, laid out as `map.src`) that `map` reads, `edge`
/// deciding what coordinates outside read. `None` for a view that reads outside without an edge.
fn view_words(src: &Opnd, src_len: usize, map: &ViewMap, edge: Edge, fill: u32) -> Option<Vec<u32>> {
    let out = map.out.map(|d| d as usize);
    let sd = map.src.map(|d| d as usize);
    if sd.contains(&0) || out.contains(&0) || sd.iter().product::<usize>() != src_len {
        return None;
    }
    let sstride = [1, sd[0], sd[0] * sd[1], sd[0] * sd[1] * sd[2]];
    let total: usize = out.iter().product();
    let mut result = Vec::with_capacity(total);
    if edge == Edge::None {
        for (a, &extent) in sd.iter().enumerate() {
            let (lo, hi) = map.ends(a);
            if lo < 0 || hi < 0 || lo >= extent as i64 || hi >= extent as i64 {
                return None;
            }
        }
        let base0: i64 = (0..4).map(|a| i64::from(map.axes[a].start) * sstride[a] as i64).sum();
        let step_for = |dim: u8| -> i64 {
            (0..4).filter(|&a| map.axes[a].from == dim).map(|a| i64::from(map.axes[a].step) * sstride[a] as i64).sum()
        };
        let [st0, st1, st2, st3] = [0, 1, 2, 3].map(step_for);
        for c3 in 0..out[3] as i64 {
            let b3 = base0 + c3 * st3;
            for c2 in 0..out[2] as i64 {
                let b2 = b3 + c2 * st2;
                for c1 in 0..out[1] as i64 {
                    let mut at = b2 + c1 * st1;
                    if st0 == 0 {
                        result.extend(std::iter::repeat_n(src.at(at as usize), out[0]));
                    } else if st0 == 1
                        && let Opnd::Arr(w) = src
                    {
                        let u = at as usize;
                        result.extend_from_slice(&w[u..u + out[0]]);
                    } else {
                        for _ in 0..out[0] {
                            result.push(src.at(at as usize));
                            at += st0;
                        }
                    }
                }
            }
        }
        return Some(result);
    }
    for c3 in 0..out[3] {
        for c2 in 0..out[2] {
            for c1 in 0..out[1] {
                for c0 in 0..out[0] {
                    let c = [c0, c1, c2, c3];
                    let (mut at, mut filled) = (0usize, false);
                    for a in 0..4 {
                        let axis = map.axes[a];
                        let raw = i64::from(axis.start) + i64::from(axis.step) * c[axis.from as usize] as i64;
                        let n = sd[a] as i64;
                        let inside = (0..n).contains(&raw);
                        let coord = match edge {
                            Edge::None if !inside => return None,
                            Edge::None => raw,
                            Edge::Fill if !inside => {
                                filled = true;
                                0
                            }
                            Edge::Fill => raw,
                            Edge::Clamp => raw.clamp(0, n - 1),
                            Edge::Mirror => mirror(raw, n),
                            Edge::Wrap => raw.rem_euclid(n),
                        };
                        at += coord as usize * sstride[a];
                    }
                    result.push(if filled { fill } else { src.at(at) });
                }
            }
        }
    }
    Some(result)
}

fn f(x: u32) -> f32 {
    f32::from_bits(x)
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

/// `op` of the element `x` of type `ty` (one `expr_for` accepts).
fn unary(op: UnOp, ty: ElemTy, x: u32) -> u32 {
    match ty {
        ElemTy::F32 => {
            let v = f(x);
            let r = match op {
                UnOp::Neg => -v,
                UnOp::Sqrt => v.sqrt(),
                UnOp::Exp => v.exp(),
                UnOp::Ln => v.ln(),
                UnOp::Abs => v.abs(),
                UnOp::Sin => v.sin(),
                UnOp::Cos => v.cos(),
                UnOp::Tan => v.tan(),
                UnOp::Tanh => v.tanh(),
                UnOp::Sign => sign32(v),
                UnOp::Floor => v.floor(),
                UnOp::Ceil => v.ceil(),
                UnOp::Trunc => v.trunc(),
                UnOp::Round => v.round_ties_even(),
                UnOp::Log1p => v.ln_1p(),
                UnOp::Expm1 => v.exp_m1(),
                UnOp::Asin => v.asin(),
                UnOp::Acos => v.acos(),
                UnOp::Atan => v.atan(),
                UnOp::Sinh => v.sinh(),
                UnOp::Cosh => v.cosh(),
                UnOp::Asinh => v.asinh(),
                UnOp::Acosh => v.acosh(),
                UnOp::Atanh => v.atanh(),
                UnOp::Exp2 => v.exp2(),
                UnOp::Log2 => v.log2(),
                UnOp::Log10 => v.log2() * std::f32::consts::LOG10_2,
                UnOp::Rsqrt => 1.0 / v.sqrt(),
                UnOp::Sigmoid => 1.0 / (1.0 + (-v).exp()),
                _ => v,
            };
            r.to_bits()
        }
        ElemTy::U32 => match op {
            UnOp::BitNot => !x,
            UnOp::Shl(k) => x << (u32::from(k) & 31),
            UnOp::Shr(k) => x >> (u32::from(k) & 31),
            UnOp::Rotl(k) => x.rotate_left(u32::from(k) & 31),
            _ => x,
        },
        ElemTy::I32 => {
            let v = x as i32;
            (match op {
                UnOp::Neg => v.wrapping_neg(),
                UnOp::Abs => v.wrapping_abs(),
                UnOp::Sign => v.signum(),
                UnOp::BitNot => !v,
                UnOp::Shl(k) => v << (u32::from(k) & 31),
                UnOp::Shr(k) => v >> (u32::from(k) & 31),
                _ => v,
            }) as u32
        }
        _ => match op {
            UnOp::BitNot => x ^ 1,
            _ => x,
        },
    }
}

/// `op` of `x` and `y` of type `ty` (one `expr_for` accepts); integer division by zero gives the
/// dividend and the remainder 0, as the kernels do.
fn binary(op: BinOp, ty: ElemTy, x: u32, y: u32) -> u32 {
    match ty {
        ElemTy::F32 => {
            let (a, b) = (f(x), f(y));
            let r = match op {
                BinOp::Add => a + b,
                BinOp::Sub => a - b,
                BinOp::Mul => a * b,
                BinOp::Div => a / b,
                BinOp::Rem => a % b,
                BinOp::Min => a.min(b),
                BinOp::Max => a.max(b),
                BinOp::Atan2 => a.atan2(b),
                BinOp::Hypot => a.hypot(b),
                BinOp::Copysign => f32::from_bits((x & 0x7fff_ffff) | (y & 0x8000_0000)),
                _ => a,
            };
            r.to_bits()
        }
        ElemTy::U32 => match op {
            BinOp::Add => x.wrapping_add(y),
            BinOp::Sub => x.wrapping_sub(y),
            BinOp::Mul => x.wrapping_mul(y),
            BinOp::Div => x.checked_div(y).unwrap_or(x),
            BinOp::Rem => x.checked_rem(y).unwrap_or(0),
            BinOp::Min => x.min(y),
            BinOp::Max => x.max(y),
            BinOp::BitAnd => x & y,
            BinOp::BitOr => x | y,
            BinOp::BitXor => x ^ y,
            BinOp::Shl => x << (y & 31),
            BinOp::Shr => x >> (y & 31),
            BinOp::Rotl => x.rotate_left(y & 31),
            _ => x,
        },
        ElemTy::I32 => {
            let (a, b) = (x as i32, y as i32);
            (match op {
                BinOp::Add => a.wrapping_add(b),
                BinOp::Sub => a.wrapping_sub(b),
                BinOp::Mul => a.wrapping_mul(b),
                BinOp::Div => if b == 0 { a } else { a.wrapping_div(b) },
                BinOp::Rem => if b == 0 { 0 } else { a.wrapping_rem(b) },
                BinOp::Min => a.min(b),
                BinOp::Max => a.max(b),
                BinOp::BitAnd => a & b,
                BinOp::BitOr => a | b,
                BinOp::BitXor => a ^ b,
                BinOp::Shl => a << (y & 31),
                BinOp::Shr => a >> (y & 31),
                _ => a,
            }) as u32
        }
        _ => match op {
            BinOp::BitAnd | BinOp::Min => x & y,
            BinOp::BitOr | BinOp::Max => x | y,
            BinOp::BitXor => x ^ y,
            _ => x,
        },
    }
}

fn compare(op: CmpOp, ty: ElemTy, x: u32, y: u32) -> u32 {
    use std::cmp::Ordering;
    let order = match ty {
        ElemTy::F32 => f(x).partial_cmp(&f(y)),
        ElemTy::I32 => Some((x as i32).cmp(&(y as i32))),
        _ => Some(x.cmp(&y)),
    };
    u32::from(match (op, order) {
        (CmpOp::Ne, None) => true,
        (_, None) => false,
        (CmpOp::Lt, Some(o)) => o == Ordering::Less,
        (CmpOp::Le, Some(o)) => o != Ordering::Greater,
        (CmpOp::Gt, Some(o)) => o == Ordering::Greater,
        (CmpOp::Ge, Some(o)) => o != Ordering::Less,
        (CmpOp::Eq, Some(o)) => o == Ordering::Equal,
        (CmpOp::Ne, Some(o)) => o != Ordering::Equal,
    })
}

fn classify(class: Class, x: u32) -> u32 {
    let magnitude = x & 0x7fff_ffff;
    u32::from(match class {
        Class::Finite => magnitude < 0x7f80_0000,
        Class::Nan => magnitude > 0x7f80_0000,
        Class::Inf => magnitude == 0x7f80_0000,
    })
}

/// The conversion from `from` to `to`, `None` for one no kernel performs.
fn cast_fn(from: ElemTy, to: ElemTy) -> Option<fn(u32) -> u32> {
    let convert: fn(u32) -> u32 = match (from, to) {
        (a, b) if a == b => |x| x,
        (ElemTy::U32, ElemTy::F32) => |x| (x as f32).to_bits(),
        (ElemTy::I32, ElemTy::F32) => |x| (x as i32 as f32).to_bits(),
        (ElemTy::Bool, ElemTy::F32) => |x| (x as f32).to_bits(),
        (ElemTy::F32, ElemTy::U32) => |x| {
            let v = f(x);
            if v.is_nan() { 0 } else { v.clamp(0.0, 4_294_967_040.0) as u32 }
        },
        (ElemTy::F32, ElemTy::I32) => |x| {
            let v = f(x);
            if v.is_nan() { 0 } else { v.clamp(-2_147_483_648.0, 2_147_483_520.0) as i32 as u32 }
        },
        (ElemTy::U32 | ElemTy::Bool, ElemTy::I32) | (ElemTy::I32 | ElemTy::Bool, ElemTy::U32) => |x| x,
        _ => return None,
    };
    Some(convert)
}

/// Whether the host may run anything for `ctx` without changing what the GPU would report.
fn host_ok(ctx: &Context) -> bool {
    ctx.can_run_on_host() && !ctx.check_finite()
}

/// Evaluates a pending node on the host and finishes it with the buffer, or `None` (nothing
/// done) if it, or anything below it, cannot run there.
pub(crate) fn try_eval_node(node: &Arc<Node>) -> Option<Arc<Storage>> {
    let ctx = node.context();
    if !ctx.can_run_on_host() || node.len() > MAX_HOST_ELEMS || node.failure().is_some() || node.is_looped() {
        return None;
    }
    if let Some(storage) = node.evaluated() {
        return storage.with_host_bytes(|_| ()).map(|_| Arc::clone(storage));
    }
    if !host_ok(ctx) || !supported(node.op().out_ty()) {
        return None;
    }
    if let Some(reduced) = node.deferred_value() {
        return reduced.host_result(ctx).map(|storage| node.finish_host(storage));
    }
    let words = Eval::new(ctx).node(node)?;
    let storage = store_words(ctx, node.len(), node.op().out_ty(), &words)?;
    Some(node.finish_host(storage))
}

/// Evaluates a group of pending nodes on the host in one shared [`Eval`], or `false` if any
/// cannot run there.
pub(crate) fn try_eval_group(nodes: &[Arc<Node>]) -> bool {
    let Some(first) = nodes.first() else { return true };
    let ctx = first.context();
    if !host_ok(ctx) {
        return false;
    }
    let mut ev = Eval::new(ctx);
    let mut out = Vec::with_capacity(nodes.len());
    for node in nodes {
        if !node.is_pending() {
            continue;
        }
        let Some(words) = ev.node(node) else { return false };
        out.push((node, words));
    }
    for (node, words) in out {
        if let Some(storage) = store_words(ctx, node.len(), node.op().out_ty(), &words) {
            node.finish_shared(storage);
        } else {
            return false;
        }
    }
    true
}

/// Whether the host may try to evaluate the reduction `reduced`: what `reduced_words` checks before it evaluates anything (the
/// `fits` of the host reduce strategy).
pub(crate) fn reduced_fits(reduced: &Structured, ctx: &Context) -> bool {
    let operand = reduced.operand();
    if reduced.is_axis() || !host_ok(ctx) || operand.len() > MAX_HOST_ELEMS || operand.failure().is_some() {
        return false;
    }
    let ty = reduced.ty();
    supported(ty) && (ty == ElemTy::F32 || reduced.epi().is_none())
}

/// The words a reduction leaves (one per op; the bit of a `bool`'s word), epilogue applied.
pub(crate) fn reduced_words(reduced: &Structured, ctx: &Context) -> Option<Vec<u32>> {
    if !reduced_fits(reduced, ctx) {
        return None;
    }
    let operand = reduced.operand();
    let ty = reduced.ty();
    let epi = reduced.epi();
    let words = Eval::new(ctx).node(operand)?;
    let mut out = Vec::with_capacity(reduced.ops().as_slice().len());
    for &op in reduced.ops().as_slice() {
        let mut value = fold(ty, op, words.iter().copied())?;
        if let Some(epi) = epi {
            // A program over the value and scalars (see `Array::absorb_epilogue`): each instruction is a unary or binary operation.
            let mut temps: Vec<u32> = Vec::with_capacity(epi.instrs.len());
            for instr in epi.instrs.iter() {
                let arg = |r: Ref| match r {
                    Ref::Acc => value,
                    Ref::Tmp(t) => temps[t as usize],
                    Ref::Scalar(k) => epi.scalars[k as usize].words()[0],
                    other => unreachable!("a whole reduction's epilogue reads the value, scalars and temporaries, not {other:?}"),
                };
                temps.push(match instr.op.kind {
                    OpKind::Unary(u) => unary(u, ElemTy::F32, arg(instr.args[0])),
                    OpKind::Binary(op) => binary(op, ElemTy::F32, arg(instr.args[0]), arg(instr.args[1])),
                    _ => unreachable!("a whole reduction's epilogue has unary and binary operations"),
                });
            }
            value = temps[epi.outputs.first().map_or(temps.len() - 1, |&at| at as usize)];
        }
        out.push(value);
    }
    Some(out)
}

/// The buffer of a reduction computed on the host.
pub(crate) fn try_eval_reduced(reduced: &Structured, ctx: &Context) -> Option<Arc<Storage>> {
    let words = reduced_words(reduced, ctx)?;
    let layout = reduced.layout();
    Storage::alloc_host(ctx, words.len(), layout, |dst| {
        if layout.is_bits() {
            write_words(dst, &[words[0] & 1]);
        } else {
            write_words(dst, &words);
        }
    })
    .ok()
}

/// The reduction `op` of the elements `it` of type `ty`: floats sum in `f64`.
fn fold(ty: ElemTy, op: ReduceOp, mut it: impl Iterator<Item = u32>) -> Option<u32> {
    let first = it.next()?;
    Some(match ty {
        ElemTy::F32 => match op {
            ReduceOp::Sum => (it.fold(f64::from(f(first)), |acc, w| acc + f64::from(f(w))) as f32).to_bits(),
            ReduceOp::SumSquares => {
                let sq = |w: u32| f64::from(f(w)) * f64::from(f(w));
                (it.fold(sq(first), |acc, w| acc + sq(w)) as f32).to_bits()
            }
            ReduceOp::Min => it.fold(f(first), |acc, w| acc.min(f(w))).to_bits(),
            ReduceOp::Max => it.fold(f(first), |acc, w| acc.max(f(w))).to_bits(),
            ReduceOp::Prod => it.fold(f(first), |acc, w| acc * f(w)).to_bits(),
            ReduceOp::ArgMax | ReduceOp::ArgMin => return None,
        },
        ElemTy::U32 => match op {
            ReduceOp::Sum => it.fold(first, u32::wrapping_add),
            ReduceOp::SumSquares => it.fold(first.wrapping_mul(first), |acc, w| acc.wrapping_add(w.wrapping_mul(w))),
            ReduceOp::Min => it.fold(first, u32::min),
            ReduceOp::Max => it.fold(first, u32::max),
            ReduceOp::Prod => it.fold(first, u32::wrapping_mul),
            ReduceOp::ArgMax | ReduceOp::ArgMin => return None,
        },
        ElemTy::I32 => match op {
            ReduceOp::Sum => it.fold(first, |acc, w| (acc as i32).wrapping_add(w as i32) as u32),
            ReduceOp::SumSquares => {
                let sq = |w: u32| (w as i32).wrapping_mul(w as i32);
                it.fold(sq(first), |acc, w| acc.wrapping_add(sq(w))) as u32
            }
            ReduceOp::Min => it.fold(first as i32, |acc, w| acc.min(w as i32)) as u32,
            ReduceOp::Max => it.fold(first as i32, |acc, w| acc.max(w as i32)) as u32,
            ReduceOp::Prod => it.fold(first, |acc, w| (acc as i32).wrapping_mul(w as i32) as u32),
            ReduceOp::ArgMax | ReduceOp::ArgMin => return None,
        },
        ElemTy::Bool => match op {
            ReduceOp::Min => it.fold(first, |acc, w| acc & w),
            ReduceOp::Max => it.fold(first, |acc, w| acc | w),
            _ => return None,
        },
        _ => return None,
    })
}

/// The words of `array` if it is host resident or computable on the host within the limits.
fn array_words<T: GpuType>(array: &Array<T>, limit: usize) -> Option<(Words, ElemTy)> {
    let ty = ElemTy::try_of::<T>()?;
    let ctx = array.context();
    if !supported(ty) || !host_ok(ctx) || array.len() > limit || array.error().is_some() {
        return None;
    }
    Some((Eval::new(ctx).array(array)?, ty))
}

/// The reductions `ops` of every line of `array` along an axis (`geometry`: cells per line,
/// distance of the cells, lines), one array of `out_dims` per op.
pub(crate) fn try_axis_reduce<T: GpuType>(array: &Array<T>, geometry: [u32; 3], out_dims: [usize; 4], ops: ReduceList) -> Option<Vec<Array<T>>> {
    let (words, ty) = array_words(array, MAX_HOST_ELEMS)?;
    if ty == ElemTy::Bool {
        return None;
    }
    let [n, stride, lines] = geometry.map(|g| g as usize);
    if n * lines != words.len() {
        return None;
    }
    let ctx = array.context();
    let mut outs = Vec::with_capacity(ops.as_slice().len());
    for &op in ops.as_slice() {
        let out: Vec<u32> = if ty == ElemTy::F32 && op == ReduceOp::Sum && stride > 1 && lines == stride {
            let mut acc = vec![0.0f64; lines];
            for row in words.chunks_exact(stride) {
                for (a, &w) in acc.iter_mut().zip(row) {
                    *a += f64::from(f32::from_bits(w));
                }
            }
            acc.into_iter().map(|s| (s as f32).to_bits()).collect()
        } else {
            let mut out = Vec::with_capacity(lines);
            for l in 0..lines {
                let base = (l % stride) + (l / stride) * stride * n;
                out.push(fold(ty, op, (0..n).map(|s| words[base + s * stride]))?);
            }
            out
        };
        let storage = store_words(ctx, lines, ty, &out)?;
        outs.push(Array::from_storage(ctx, Shape::from(out_dims), storage));
    }
    Some(outs)
}

/// Whether `candidate` replaces `best` in an arg reduction: a NaN beats every number, the first of
/// equals stays.
fn better(ty: ElemTy, op: ReduceOp, candidate: u32, best: u32) -> bool {
    let largest = op == ReduceOp::ArgMax;
    match ty {
        ElemTy::F32 => {
            let (c, b) = (f(candidate), f(best));
            if c.is_nan() {
                !b.is_nan()
            } else if b.is_nan() {
                false
            } else if largest {
                c > b
            } else {
                c < b
            }
        }
        ElemTy::I32 => {
            let (c, b) = (candidate as i32, best as i32);
            if largest { c > b } else { c < b }
        }
        _ => if largest { candidate > best } else { candidate < best },
    }
}

/// The arg reduction `op` over the lines `geometry` of `array`, as the `u32` array of `out_dims`.
pub(crate) fn try_arg_reduce<T: GpuType>(array: &Array<T>, geometry: [u32; 3], out_dims: [usize; 4], op: ReduceOp) -> Option<Array<u32>> {
    let (words, ty) = array_words(array, MAX_HOST_ELEMS)?;
    if ty == ElemTy::Bool || !op.is_arg() {
        return None;
    }
    let [n, stride, lines] = geometry.map(|g| g as usize);
    if n == 0 || n * lines != words.len() {
        return None;
    }
    let ctx = array.context();
    let mut out = Vec::with_capacity(lines);
    for l in 0..lines {
        let base = (l % stride) + (l / stride) * stride * n;
        let (mut at, mut best) = (0usize, words[base]);
        for s in 1..n {
            let value = words[base + s * stride];
            if better(ty, op, value, best) {
                (at, best) = (s, value);
            }
        }
        out.push(at as u32);
    }
    let storage = store_words(ctx, lines, ElemTy::U32, &out)?;
    Some(Array::from_storage(ctx, Shape::from(out_dims), storage))
}

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

#[cfg(target_os = "macos")]
const ROW_MAJOR: i32 = 101;
#[cfg(target_os = "macos")]
const NO_TRANS: i32 = 111;
#[cfg(target_os = "macos")]
const TRANS: i32 = 112;

/// `c = op(a) op(b)` for row-major `m x k` and `k x n` operands (stored transposed if `ta` / `tb`).
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn sgemm(a: &[f32], ta: bool, b: &[f32], tb: bool, c: &mut [f32], m: usize, n: usize, k: usize) {
    let (lda, ldb) = (if ta { m } else { k }, if tb { k } else { n });
    // SAFETY: the operands hold `m * k`, `k * n` and `m * n` floats (checked by the caller) and
    // the leading dimensions are those of their storage orders.
    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            if ta { TRANS } else { NO_TRANS },
            if tb { TRANS } else { NO_TRANS },
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            lda as i32,
            b.as_ptr(),
            ldb as i32,
            0.0,
            c.as_mut_ptr(),
            n as i32,
        );
    }
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
fn sgemm(a: &[f32], ta: bool, b: &[f32], tb: bool, c: &mut [f32], m: usize, n: usize, k: usize) {
    let transposed = |x: &[f32], rows: usize, cols: usize| -> Vec<f32> {
        // `x` is `cols x rows` row-major; the result is `rows x cols` row-major.
        let mut out = vec![0.0; x.len()];
        for r in 0..rows {
            for q in 0..cols {
                out[r * cols + q] = x[q * rows + r];
            }
        }
        out
    };
    let (a_rm, b_rm);
    let a = if ta { a_rm = transposed(a, m, k); &a_rm[..] } else { a };
    let b = if tb { b_rm = transposed(b, k, n); &b_rm[..] } else { b };
    c.fill(0.0);
    for row in 0..m {
        let out = &mut c[row * n..(row + 1) * n];
        for kk in 0..k {
            let av = a[row * k + kk];
            for (o, &bv) in out.iter_mut().zip(&b[kk * n..(kk + 1) * n]) {
                *o += av * bv;
            }
        }
    }
}

/// The words of the array a pending 2-D transpose reads, if `array` is one.
fn transposed_source(ev: &mut Eval<'_>, array: &Array<f32>) -> Option<Words> {
    let node = array.pending()?;
    let op = node.op();
    let OpKind::Shift(ShiftKind::View { map, edge: Edge::None }) = op.kind else { return None };
    let unit = |from: u8| AxisMap { from, start: 0, step: 1 };
    // A reshape of a pending view shares its node, which keeps the shape it was made with.
    let swaps = op.ty == ElemTy::F32
        && array.shape().dims().map(|extent| u32::try_from(extent).ok()) == map.out.map(Some)
        && map.axes[0] == unit(1)
        && map.axes[1] == unit(0)
        && map.axes[2] == unit(2)
        && map.axes[3] == unit(3)
        && map.src[0] == map.out[1]
        && map.src[1] == map.out[0]
        && map.src[2..] == [1, 1]
        && map.out[2..] == [1, 1];
    if !swaps || node.len() > MAX_HOST_ELEMS || node.failure().is_some() || node.is_looped() {
        return None;
    }
    let args = node.operands()?;
    match ev.src(&args[0], node.len())? {
        Opnd::Arr(words) => Some(words),
        Opnd::Scalar(_) => None,
    }
}

/// `lhs` (`m x k`) times `rhs` (`k x n`), row-major, on the host. A transposed operand is read
/// from the array it transposes and handed to the product as transposed.
pub(crate) fn try_matmul(ctx: &Context, lhs: &Array<f32>, rhs: &Array<f32>, m: usize, n: usize, k: usize) -> Option<Array<f32>> {
    if !host_ok(ctx) || lhs.error().is_some() || rhs.error().is_some() {
        return None;
    }
    let mut ev = Eval::new(ctx);
    let mut operand = |array: &Array<f32>| match transposed_source(&mut ev, array) {
        Some(words) => Some((words, true)),
        None => ev.array(array).map(|words| (words, false)),
    };
    let (a, ta) = operand(lhs)?;
    let (b, tb) = operand(rhs)?;
    if a.len() != m * k || b.len() != k * n {
        return None;
    }
    let (a, b): (&[f32], &[f32]) = (bytemuck::cast_slice(&a[..]), bytemuck::cast_slice(&b[..]));
    let storage = Storage::alloc_host(ctx, m * n, ElemLayout::plain(4), |dst| match bytemuck::try_cast_slice_mut::<u8, f32>(dst) {
        Ok(c) => sgemm(a, ta, b, tb, c, m, n, k),
        Err(_) => {
            let mut c = vec![0.0f32; m * n];
            sgemm(a, ta, b, tb, &mut c, m, n, k);
            dst.copy_from_slice(bytemuck::cast_slice(&c));
        }
    })
    .ok()?;
    Some(Array::from_storage(ctx, Shape::from([n, m, 1, 1]), storage))
}

/// The word that orders like the key, as the radix sort transforms it.
fn key_in(transform: KeyTransform, b: u32) -> u32 {
    match transform {
        KeyTransform::Identity => b,
        KeyTransform::Signed => b ^ 0x8000_0000,
        KeyTransform::Float => {
            let c = if (b & 0x7fff_ffff) > 0x7f80_0000 { 0x7fc0_0000 } else { b };
            if c & 0x8000_0000 != 0 { !c } else { c | 0x8000_0000 }
        }
    }
}

/// The inverse of [`key_in`] (a NaN comes back as the positive quiet NaN).
fn key_out(transform: KeyTransform, k: u32) -> u32 {
    match transform {
        KeyTransform::Identity => k,
        KeyTransform::Signed => k ^ 0x8000_0000,
        KeyTransform::Float => if k & 0x8000_0000 != 0 { k & 0x7fff_ffff } else { !k },
    }
}

/// The stable sort of the elements of `array`: (key, flat index) in order.
fn sorted_keys<T: SortKey>(array: &Array<T>, descending: bool) -> Option<(Vec<(u32, u32)>, ElemTy)> {
    let (words, ty) = array_words(array, MAX_SORT_ELEMS)?;
    if ty == ElemTy::Bool {
        return None;
    }
    let mut pairs: Vec<(u32, u32)> = words.iter().enumerate().map(|(i, &w)| (key_in(T::TRANSFORM, w), i as u32)).collect();
    if descending {
        pairs.sort_by_key(|a| std::cmp::Reverse(a.0));
    } else {
        pairs.sort_by_key(|p| p.0);
    }
    Some((pairs, ty))
}

pub(crate) fn try_sort<T: SortKey>(array: &Array<T>, descending: bool) -> Option<Array<T>> {
    let (pairs, ty) = sorted_keys(array, descending)?;
    let ctx = array.context();
    let out: Vec<u32> = pairs.iter().map(|p| key_out(T::TRANSFORM, p.0)).collect();
    Some(Array::from_storage(ctx, array.shape(), store_words(ctx, out.len(), ty, &out)?))
}

pub(crate) fn try_argsort<T: SortKey>(array: &Array<T>, descending: bool) -> Option<Arc<Storage>> {
    let (pairs, _) = sorted_keys(array, descending)?;
    let out: Vec<u32> = pairs.iter().map(|p| p.1).collect();
    store_words(array.context(), out.len(), ElemTy::U32, &out)
}

pub(crate) fn try_sort_with_indices<T: SortKey>(array: &Array<T>) -> Option<(Array<T>, Array<u32>)> {
    let (pairs, ty) = sorted_keys(array, false)?;
    let ctx = array.context();
    let keys: Vec<u32> = pairs.iter().map(|p| key_out(T::TRANSFORM, p.0)).collect();
    let perm: Vec<u32> = pairs.iter().map(|p| p.1).collect();
    let keys = Array::from_storage(ctx, array.shape(), store_words(ctx, keys.len(), ty, &keys)?);
    let perm = Array::from_storage(ctx, array.shape(), store_words(ctx, perm.len(), ElemTy::U32, &perm)?);
    Some((keys, perm))
}

/// The sorted distinct values of `array` and the index into them of every element. Floats treat
/// all NaNs as one value and `-0.0` and `+0.0` as one (the first in sorted order, `-0.0`).
pub(crate) fn try_factorize<T: SortKey>(array: &Array<T>) -> Option<(Array<T>, Array<u32>)> {
    let (words, ty) = array_words(array, MAX_SORT_ELEMS)?;
    if ty == ElemTy::Bool {
        return None;
    }
    let ctx = array.context();
    let n = words.len();
    if ty != ElemTy::F32 && n > 0 {
        let mut min_k = u32::MAX;
        let mut max_k = 0u32;
        for &w in words.iter() {
            let k = key_in(T::TRANSFORM, w);
            min_k = min_k.min(k);
            max_k = max_k.max(k);
        }
        let span = (max_k - min_k) as usize;
        if span < 4096 {
            let mut bits = [0u64; 64];
            for &w in words.iter() {
                let d = (key_in(T::TRANSFORM, w) - min_k) as usize;
                bits[d >> 6] |= 1u64 << (d & 63);
            }
            let mut uniques = Vec::new();
            let mut rank = vec![0u32; span + 1];
            for (wi, &word) in bits.iter().enumerate() {
                let mut b = word;
                while b != 0 {
                    let bit = b.trailing_zeros() as usize;
                    let d = wi * 64 + bit;
                    rank[d] = uniques.len() as u32;
                    uniques.push(key_out(T::TRANSFORM, min_k + d as u32));
                    b &= b - 1;
                }
            }
            let codes: Vec<u32> = words.iter().map(|&w| rank[(key_in(T::TRANSFORM, w) - min_k) as usize]).collect();
            let count = uniques.len();
            let uniques = Array::from_storage(ctx, Shape::from(count), store_words(ctx, count, ty, &uniques)?);
            let codes = Array::from_storage(ctx, array.shape(), store_words(ctx, n, ElemTy::U32, &codes)?);
            return Some((uniques, codes));
        }
    }
    let (pairs, _) = sorted_keys(array, false)?;
    let same = |a: u32, b: u32| if ty == ElemTy::F32 { (f(a).is_nan() && f(b).is_nan()) || f(a) == f(b) } else { a == b };
    let mut uniques: Vec<u32> = Vec::new();
    let mut codes = vec![0u32; n];
    for &(key, at) in &pairs {
        let word = key_out(T::TRANSFORM, key);
        if uniques.last().is_none_or(|&last| !same(last, word)) {
            uniques.push(word);
        }
        codes[at as usize] = uniques.len() as u32 - 1;
    }
    let count = uniques.len();
    let uniques = Array::from_storage(ctx, Shape::from(count), store_words(ctx, count, ty, &uniques)?);
    let codes = Array::from_storage(ctx, array.shape(), store_words(ctx, n, ElemTy::U32, &codes)?);
    Some((uniques, codes))
}

/// `target[indices[i]] = values[i]` in place on a host-resident target; `false` (nothing written)
/// if any part is not on the host.
pub(crate) fn try_put<T: ScalarType>(target: &Array<T>, indices: &Array<u32>, values: &Rhs<'_, T>) -> bool {
    put(target, indices, values).is_some()
}

fn put<T: ScalarType>(target: &Array<T>, indices: &Array<u32>, values: &Rhs<'_, T>) -> Option<()> {
    let ctx = target.context();
    let ty = ElemTy::try_of::<T>()?;
    if !host_ok(ctx) || !matches!(ty, ElemTy::F32 | ElemTy::U32 | ElemTy::I32) || !ctx.same_as(indices.context()) {
        return None;
    }
    let storage = target.storage_if_valid()?;
    if storage.segments() != 1 || storage.is_alias() || storage.layout() != ElemLayout::plain(4) || storage.with_host_bytes(|_| ()).is_none() {
        return None;
    }
    let (index_words, _) = array_words(indices, MAX_SORT_ELEMS)?;
    let value = match values {
        Rhs::Scalar(v) => Opnd::Scalar(Scalar::of(*v).ok()?.bits as u32),
        Rhs::Array(other) => {
            if !ctx.same_as(other.context()) || other.shape() != indices.shape() {
                return None;
            }
            Opnd::Arr(array_words(other, MAX_SORT_ELEMS)?.0)
        }
    };
    // What was recorded against the old contents runs before the write.
    storage.readers.flush();
    ctx.jit().note_write();
    let len = storage.len();
    storage.with_host_bytes_mut(|bytes| {
        for (i, &at) in index_words.iter().enumerate() {
            let at = at as usize;
            if at < len {
                bytes[4 * at..4 * at + 4].copy_from_slice(&value.at(i).to_ne_bytes());
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A program with a call is never evaluated on the host: the device runs it (and the host would run `* 2.0`).
    #[test]
    fn a_call_is_not_evaluated_on_the_host() {
        let ctx = Context::new().unwrap();
        let f = ctx.elementwise_fn::<f32>("ho_f", "fn ho_f(x: f32) -> f32 { return x * 2.0; }").unwrap();
        let x = Array::from_slice_on(&ctx, &[1.0f32, 2.0, 3.0], 3).unwrap();
        let plain = &x * 2.0;
        let call = f.call1(&x);
        let plain_on_host = plain.pending().is_some_and(|node| Eval::new(&ctx).node(node).is_some());
        let call_on_host = call.pending().is_some_and(|node| Eval::new(&ctx).node(node).is_some());
        // Where the host evaluates anything here, it evaluates the product and declines the call.
        assert!(!call_on_host);
        assert!(plain_on_host || !ctx.zero_copy() || ctx.is_cpu(), "the control did not run on the host either");
        assert_eq!(call.to_vec().unwrap(), vec![2.0, 4.0, 6.0]);
    }
}
