//! ArrayFire helpers: backend selection, host transfers and the cuBLAS math mode.

use arrayfire as af;
use std::ffi::c_int;

pub fn init() {
    af::set_backend(af::Backend::CUDA);
    af::set_device(0);
}

pub fn sync() {
    af::sync(0);
}

pub fn dims(d: &[u64]) -> af::Dim4 {
    let mut x = [1u64; 4];
    x[..d.len()].copy_from_slice(d);
    af::Dim4::new(&x)
}

/// An evaluated, synchronised device array.
pub fn arr<T: af::HasAfEnum>(data: &[T], d: &[u64]) -> af::Array<T> {
    let a = af::Array::new(data, dims(d));
    a.eval();
    sync();
    a
}

pub fn host<T: af::HasAfEnum + Default + Clone>(a: &af::Array<T>) -> Vec<T> {
    let mut v = vec![T::default(); a.elements()];
    a.host(&mut v);
    v
}

pub fn info() -> String {
    let (major, minor, patch) = af::get_version();
    format!("ArrayFire {major}.{minor}.{patch} (CUDA backend): {}", af::device_info().0.trim())
}

/// cuBLAS math modes of `cublasMath_t`.
pub const CUBLAS_DEFAULT_MATH: c_int = 0;
pub const CUBLAS_TENSOR_OP_MATH: c_int = 1;
pub const CUBLAS_TF32_TENSOR_OP_MATH: c_int = 3;

/// `afcu_cublasSetMathMode` of `libafcuda`, which is not part of the unified `libaf`.
pub fn set_math_mode(mode: c_int) -> Result<(), String> {
    type SetMode = unsafe extern "C" fn(c_int) -> c_int;
    unsafe {
        let lib = libc::dlopen(c"libafcuda.so.3".as_ptr(), libc::RTLD_NOW);
        if lib.is_null() {
            return Err("cannot dlopen libafcuda.so.3".into());
        }
        let sym = libc::dlsym(lib, c"afcu_cublasSetMathMode".as_ptr());
        if sym.is_null() {
            return Err("afcu_cublasSetMathMode not found in libafcuda".into());
        }
        let f: SetMode = std::mem::transmute(sym);
        match f(mode) {
            0 => Ok(()),
            code => Err(format!("afcu_cublasSetMathMode({mode}) returned af_err {code}")),
        }
    }
}
