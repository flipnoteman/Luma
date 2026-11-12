//! Records of arrays: a struct of arrays of one element type (the conserved state of a flow
//! solver, a gradient) that is sliced, padded, assigned and combined with arithmetic as one
//! value. Implement it with `#[derive(luma::Fields)]` on a struct whose fields are `Array<T>`
//! of one element type `T` or other records of that type.

use std::borrow::Borrow;
use std::ops::RangeBounds;
use std::panic::Location;
use std::sync::Arc;

use crate::array::Array;
use crate::builtin::BinOp;
use crate::error::Result;
use crate::gpu_type::ScalarType;
use crate::jit::assign::{self, Region};
use crate::jit;
use crate::ops::{eval_all, select_at, Rhs};
use crate::shape::Offsets;
use crate::view::{finish, PadMode};

/// The operator of a record operation (an implementation detail of `#[derive(Fields)]`).
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub enum FieldOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl FieldOp {
    fn parts(self) -> (BinOp, &'static str) {
        match self {
            FieldOp::Add => (BinOp::Add, "add"),
            FieldOp::Sub => (BinOp::Sub, "sub"),
            FieldOp::Mul => (BinOp::Mul, "mul"),
            FieldOp::Div => (BinOp::Div, "div"),
        }
    }
}

/// A record of arrays of one element type and one shape (a struct of arrays): the conserved state
/// of a flow solver, a gradient, ...
///
/// Implemented by `Array<T>` (one field), `[F; N]` and tuples of up to six records, and by
/// every struct with `#[derive(Fields)]` (`luma::Fields` also names the derive macro). All
/// the arrays of a record have the same element type [`Fields::Elem`]: `f32` for the usual
/// state, [`Df32`](crate::Df32) or `f64` for one that needs more precision.
///
/// ```ignore
/// #[derive(luma::Fields)]
/// struct Cons { rho: Array<f32>, mu: Array<f32>, mv: Array<f32>, e: Array<f32> }
/// ```
///
/// The derive infers the element type from the fields (which must agree) and also provides
/// `+ - * /` between records, with an element value (`f32` here) and with an `Array<Elem>`
/// (applied to every field), `+= -= *= /=` and unary `-`. Every method applies the array
/// operation of the same name to each field and keeps its laziness; failures poison the fields
/// with the caller's source location. A record has at least one field.
pub trait Fields: Sized {
    /// The element type of every array of the record (`f32`, [`Df32`](crate::Df32), `f64`, ...).
    type Elem: ScalarType;

    /// Number of arrays in the record.
    const COUNT: usize;

    /// Calls `f` with every array, in declaration order.
    fn each<'a>(&'a self, f: &mut dyn FnMut(&'a Array<Self::Elem>));

    /// Builds a record from `COUNT` arrays taken from `next`, in declaration order.
    fn build(next: &mut dyn FnMut() -> Array<Self::Elem>) -> Self;

    /// The arrays of the record, in declaration order.
    fn arrays(&self) -> Vec<&Array<Self::Elem>> {
        let mut out = Vec::with_capacity(Self::COUNT);
        self.each(&mut |a| out.push(a));
        out
    }

    /// A record of `f(field)` for every field.
    fn map(&self, mut f: impl FnMut(&Array<Self::Elem>) -> Array<Self::Elem>) -> Self {
        let mut out = Vec::with_capacity(Self::COUNT);
        self.each(&mut |a| out.push(f(a)));
        from_vec(out)
    }

    /// A record of `f(field, other field)` for every field.
    fn zip(&self, other: &Self, f: impl FnMut(&Array<Self::Elem>, &Array<Self::Elem>) -> Array<Self::Elem>) -> Self {
        from_vec(zip_vec(self, other, f))
    }

    /// The sum of the products of the fields: `sum_k self.k * other.k`, a lazy elementwise array.
    #[track_caller]
    fn dot(&self, other: &Self) -> Array<Self::Elem> {
        let location = Location::caller();
        let mut acc: Option<Array<Self::Elem>> = None;
        let products = zip_vec(self, other, |a, b| a.binary_at(Rhs::Array(b), BinOp::Mul, false, "mul", location));
        for p in products {
            acc = Some(match acc {
                None => p,
                Some(sum) => sum.binary_at(Rhs::Array(&p), BinOp::Add, false, "add", location),
            });
        }
        acc.expect("a record has at least one field")
    }

    /// [`Array::slice`] of every field.
    #[track_caller]
    fn slice(&self, axis: usize, range: impl RangeBounds<usize> + Clone) -> Self {
        let location = Location::caller();
        self.map(|a| a.slice_at(axis, range.clone(), location))
    }

    /// [`Array::pad`] of every field.
    #[track_caller]
    fn pad<const N: usize>(&self, widths: [(usize, usize); N], mode: PadMode) -> Self {
        let location = Location::caller();
        self.map(|a| a.pad_at(widths, mode, location))
    }

    /// [`Array::flip`] of every field.
    #[track_caller]
    fn flip(&self, axis: usize) -> Self {
        let location = Location::caller();
        self.map(|a| a.flip_at(axis, location))
    }

    /// [`Array::transpose`] of every field.
    #[track_caller]
    fn transpose(&self) -> Self {
        let location = Location::caller();
        self.map(|a| a.transpose_at(location))
    }

    /// [`Array::shift`] of every field.
    #[track_caller]
    fn shift(&self, offsets: impl Into<Offsets>) -> Self {
        let (location, offsets) = (Location::caller(), offsets.into());
        self.map(|a| a.shift_at(offsets, location))
    }

    /// [`Array::shift_fill`] of every field.
    #[track_caller]
    fn shift_fill(&self, offsets: impl Into<Offsets>, value: Self::Elem) -> Self {
        let (location, offsets) = (Location::caller(), offsets.into());
        self.map(|a| a.shift_fill_at(offsets, value, location))
    }

    /// [`Array::shift_clamp`] of every field.
    #[track_caller]
    fn shift_clamp(&self, offsets: impl Into<Offsets>) -> Self {
        let (location, offsets) = (Location::caller(), offsets.into());
        self.map(|a| a.shift_clamp_at(offsets, location))
    }

    /// [`Array::shift_mirror`] of every field.
    #[track_caller]
    fn shift_mirror(&self, offsets: impl Into<Offsets>) -> Self {
        let (location, offsets) = (Location::caller(), offsets.into());
        self.map(|a| a.shift_mirror_at(offsets, location))
    }

    /// [`Array::take`] of every field.
    #[track_caller]
    fn take(&self, indices: &Array<u32>) -> Self {
        let location = Location::caller();
        self.map(|a| a.take_at(indices, location))
    }

    /// Per field, `a` where `mask` is non-zero and `b` elsewhere (see [`select`](crate::select)).
    #[track_caller]
    fn select(mask: &Array<u32>, a: &Self, b: &Self) -> Self {
        let location = Location::caller();
        let theirs = b.arrays();
        let mut out = Vec::with_capacity(Self::COUNT);
        a.each(&mut |x| out.push(select_at(mask, x, theirs[out.len()], location)));
        from_vec(out)
    }

    /// Writes `rhs` into the arrays this record's fields are slices, flips or transposes of
    /// (see [`Array::assign`]), field by field, in one kernel (several outputs). `rhs` is a
    /// record (`&rec`, or `rec` by value, which lets its expressions fuse into the write).
    /// `w.slice(0, ..2).assign(mirror(w.slice(0, 2..4).flip(0)))` fills the ghost cells of a
    /// whole state. Fields whose base is still a pending expression are not written at all:
    /// the record denotes the new values from then on and nothing is launched, so a padded
    /// record with ghost cells assigned stays one expression.
    #[track_caller]
    fn assign(&self, rhs: impl Borrow<Self>) -> Result<()> {
        let location = Location::caller();
        let theirs = rhs.borrow().arrays();
        // Everything that can fail is checked before the first field is written.
        let mut checked = Ok(());
        let mut k = 0;
        self.each(&mut |a| {
            if checked.is_ok() {
                checked = a.src().and_then(|_| a.check_write(Rhs::Array(theirs[k])));
            }
            k += 1;
        });
        let ctx = self.arrays()[0].context();
        // Fields of pending bases are written lazily; the others are collected for the kernel.
        let mut eager = vec![true; Self::COUNT];
        if checked.is_ok() {
            let mut k = 0;
            self.each(&mut |a| {
                if checked.is_ok() {
                    match a.assign_lazy(Rhs::Array(theirs[k]), location) {
                        Ok(done) => eager[k] = !done,
                        Err(error) => checked = Err(error),
                    }
                }
                k += 1;
            });
        }
        let regions: crate::ops::Outcome<Vec<Region>> = match checked {
            Err(error) => Err(error),
            Ok(()) => {
                // The pending bases the remaining fields are views of are evaluated together,
                // one kernel per length, instead of one kernel per field when each region forces
                // its own. A failure surfaces when the field's region forces its base.
                let mut bases = Vec::new();
                let mut k = 0;
                self.each(&mut |a| {
                    if eager[k]
                        && let Some(view) = a.view()
                        && let Some(held) = &view.held
                        && held.latest().is_pending()
                    {
                        bases.push(Arc::clone(held.latest()));
                    }
                    k += 1;
                });
                let _ = jit::force_group(&bases);
                let mut regions = Vec::with_capacity(Self::COUNT);
                let mut failure = None;
                let mut k = 0;
                self.each(&mut |a| {
                    if eager[k] && failure.is_none() {
                        match a.region(Rhs::Array(theirs[k])) {
                            Ok(region) => regions.push(region),
                            Err(error) => failure = Some(error),
                        }
                    }
                    k += 1;
                });
                match failure {
                    Some(error) => Err(error),
                    None => Ok(regions),
                }
            }
        };
        // Only the regions hold the right-hand side now: an owned record's temporaries are
        // private to the kernel that stores them.
        drop(rhs);
        let result = match regions {
            Ok(regions) if regions.is_empty() => Ok(()),
            Ok(regions) => assign::assign(ctx, regions, "assign", location),
            Err(error) => Err(error),
        };
        finish(result, "assign", location)
    }

    /// [`Array::put`] on every field: `self.k[indices[i]] = values.k[i]`, one kernel per field.
    #[track_caller]
    fn put(&self, indices: &Array<u32>, values: &Self) -> Result<()> {
        let theirs = values.arrays();
        let mut result = Ok(());
        let mut k = 0;
        self.each(&mut |a| {
            if result.is_ok() {
                result = a.put(indices, theirs[k]);
            }
            k += 1;
        });
        result
    }

    /// [`Array::cache`] of every field: the pending expressions of the fields are recorded as
    /// kernels now, together (fields that share work compute it once), and the record reads
    /// evaluated buffers afterwards.
    #[track_caller]
    fn cache(&self) -> Self {
        let location = Location::caller();
        let pending: Vec<_> = self.arrays().into_iter().filter_map(|a| a.pending().cloned()).collect();
        jit::evaluate_group(&pending);
        self.map(|a| a.cache_at(location))
    }

    /// Evaluates every field without waiting, together ([`eval_all`]).
    fn eval(&self) -> Result<()> {
        eval_all(&self.arrays())
    }

    #[doc(hidden)]
    #[track_caller]
    fn binary(&self, other: &Self, op: FieldOp) -> Self {
        let (location, (op, name)) = (Location::caller(), op.parts());
        from_vec(zip_vec(self, other, |a, b| a.binary_at(Rhs::Array(b), op, false, name, location)))
    }

    #[doc(hidden)]
    #[track_caller]
    fn binary_array(&self, other: &Array<Self::Elem>, op: FieldOp, array_left: bool) -> Self {
        let (location, (op, name)) = (Location::caller(), op.parts());
        self.map(|a| {
            if array_left {
                other.binary_at(Rhs::Array(a), op, false, name, location)
            } else {
                a.binary_at(Rhs::Array(other), op, false, name, location)
            }
        })
    }

    #[doc(hidden)]
    #[track_caller]
    fn binary_scalar(&self, scalar: Self::Elem, op: FieldOp, scalar_left: bool) -> Self {
        let (location, (op, name)) = (Location::caller(), op.parts());
        self.map(|a| a.binary_at(Rhs::Scalar(scalar), op, scalar_left, name, location))
    }

    #[doc(hidden)]
    #[track_caller]
    fn negate(&self) -> Self {
        let location = Location::caller();
        self.map(|a| a.neg_at(location))
    }
}

/// `f(field, other field)` for every field.
fn zip_vec<F: Fields>(
    lhs: &F,
    rhs: &F,
    mut f: impl FnMut(&Array<F::Elem>, &Array<F::Elem>) -> Array<F::Elem>,
) -> Vec<Array<F::Elem>> {
    let theirs = rhs.arrays();
    let mut out = Vec::with_capacity(F::COUNT);
    lhs.each(&mut |a| out.push(f(a, theirs[out.len()])));
    out
}

/// A record from `COUNT` arrays in a vector.
fn from_vec<F: Fields>(arrays: Vec<Array<F::Elem>>) -> F {
    let mut arrays = arrays.into_iter();
    F::build(&mut || arrays.next().expect("a record is built from COUNT arrays"))
}

impl<T: ScalarType> Fields for Array<T> {
    type Elem = T;
    const COUNT: usize = 1;

    fn each<'a>(&'a self, f: &mut dyn FnMut(&'a Array<Self::Elem>)) {
        f(self);
    }

    fn build(next: &mut dyn FnMut() -> Array<Self::Elem>) -> Self {
        next()
    }
}

impl<F: Fields, const N: usize> Fields for [F; N] {
    type Elem = F::Elem;
    const COUNT: usize = N * F::COUNT;

    fn each<'a>(&'a self, f: &mut dyn FnMut(&'a Array<Self::Elem>)) {
        for field in self {
            field.each(f);
        }
    }

    fn build(next: &mut dyn FnMut() -> Array<Self::Elem>) -> Self {
        std::array::from_fn(|_| F::build(next))
    }
}

macro_rules! tuple_fields {
    ($F:ident $f:ident $(, $T:ident $n:ident)*) => {
        impl<$F: Fields $(, $T: Fields<Elem = $F::Elem>)*> Fields for ($F, $($T,)*) {
            type Elem = $F::Elem;
            const COUNT: usize = $F::COUNT $(+ $T::COUNT)*;

            fn each<'a>(&'a self, f: &mut dyn FnMut(&'a Array<Self::Elem>)) {
                let ($f, $($n,)*) = self;
                $f.each(f);
                $($n.each(f);)*
            }

            fn build(next: &mut dyn FnMut() -> Array<Self::Elem>) -> Self {
                ($F::build(next), $($T::build(next),)*)
            }
        }
    };
}

tuple_fields!(A a);
tuple_fields!(A a, B b);
tuple_fields!(A a, B b, C c);
tuple_fields!(A a, B b, C c, D d);
tuple_fields!(A a, B b, C c, D d, E e);
tuple_fields!(A a, B b, C c, D d, E e, G g);
