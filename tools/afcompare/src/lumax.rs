//! luma helpers.

use luma::{Array, Context, ContextOptions};

pub fn open() -> Context {
    let ctx = Context::with_options(ContextOptions::from_env()).expect("no luma device");
    assert!(!ctx.is_cpu() && ctx.backend() == luma::BackendKind::Cuda, "afcompare needs the CUDA backend (LUMA_BACKEND=cuda), got {:?}", ctx.backend());
    ctx
}

/// An evaluated, synchronised array.
pub fn arr<T: luma::ScalarType>(ctx: &Context, data: &[T], shape: impl Into<luma::Shape>) -> Array<T> {
    let a = Array::from_slice_on(ctx, data, shape).unwrap();
    a.eval().unwrap();
    ctx.sync().unwrap();
    a
}

pub fn host<T: luma::ScalarType>(a: &Array<T>) -> Vec<T> {
    a.to_vec().unwrap()
}
