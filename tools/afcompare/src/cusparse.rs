//! cuSPARSE called directly (`dlopen` of the library ArrayFire ships): what ArrayFire's sparse
//! `matmul` and `sparseConvertTo` call underneath. ArrayFire 3.10.1's own sparse entry points
//! segfault in `cusparseCreateSpMatDescr` on the machines measured here (see `sparse.rs`), so the
//! comparison rows of the sparse group use cuSPARSE itself as the reference: the generic
//! `cusparseSpMV` / `cusparseSpMM` (default algorithms, the preprocessing buffer allocated once, as
//! ArrayFire's wrapper does per call at best) and `cusparseCsr2cscEx2`.
//!
//! Device memory comes from ArrayFire arrays (`device_ptr`), on ArrayFire's context.

use std::ffi::{c_int, c_void};

type A = arrayfire::Array<f32>;
pub type AI = arrayfire::Array<i32>;

type Handle = *mut c_void;
type Descr = *mut c_void;

const INDEX_32I: c_int = 2;
const BASE_ZERO: c_int = 0;
const R_32F: c_int = 0;
const OP_N: c_int = 0;
const OP_T: c_int = 1;
const ORDER_COL: c_int = 1;
const ALG_DEFAULT: c_int = 0;
const CSR2CSC_ALG1: c_int = 1;

struct Lib {
    create: unsafe extern "C" fn(*mut Handle) -> c_int,
    create_csr: unsafe extern "C" fn(*mut Descr, i64, i64, i64, *mut c_void, *mut c_void, *mut c_void, c_int, c_int, c_int, c_int) -> c_int,
    create_dnvec: unsafe extern "C" fn(*mut Descr, i64, *mut c_void, c_int) -> c_int,
    create_dnmat: unsafe extern "C" fn(*mut Descr, i64, i64, i64, *mut c_void, c_int, c_int) -> c_int,
    spmv_size: unsafe extern "C" fn(Handle, c_int, *const f32, Descr, Descr, *const f32, Descr, c_int, c_int, *mut usize) -> c_int,
    spmv: unsafe extern "C" fn(Handle, c_int, *const f32, Descr, Descr, *const f32, Descr, c_int, c_int, *mut c_void) -> c_int,
    spmm_size: unsafe extern "C" fn(Handle, c_int, c_int, *const f32, Descr, Descr, *const f32, Descr, c_int, c_int, *mut usize) -> c_int,
    spmm: unsafe extern "C" fn(Handle, c_int, c_int, *const f32, Descr, Descr, *const f32, Descr, c_int, c_int, *mut c_void) -> c_int,
    csr2csc_size: unsafe extern "C" fn(Handle, c_int, c_int, c_int, *const c_void, *const c_void, *const c_void, *mut c_void, *mut c_void, *mut c_void, c_int, c_int, c_int, c_int, *mut usize) -> c_int,
    csr2csc: unsafe extern "C" fn(Handle, c_int, c_int, c_int, *const c_void, *const c_void, *const c_void, *mut c_void, *mut c_void, *mut c_void, c_int, c_int, c_int, c_int, *mut c_void) -> c_int,
    sync: unsafe extern "C" fn() -> c_int,
}

fn sym<T>(lib: *mut c_void, name: &std::ffi::CStr) -> T {
    unsafe {
        let p = libc::dlsym(lib, name.as_ptr());
        assert!(!p.is_null(), "missing symbol {name:?}");
        std::mem::transmute_copy(&p)
    }
}

fn check(what: &str, status: c_int) {
    assert_eq!(status, 0, "{what} returned status {status}");
}

/// A cuSPARSE handle with the entry points of the generic API.
pub struct Cusparse {
    lib: Lib,
    handle: Handle,
}

/// A CSR matrix descriptor over device arrays (kept alive by the caller).
pub struct Csr {
    descr: Descr,
    rows: i64,
    cols: i64,
    nnz: i64,
}

/// A prepared product: descriptors and the workspace.
pub struct Product {
    a: Csr,
    x: Descr,
    y: Descr,
    buffer: Option<A>,
    transposed: bool,
    spmm: bool,
}

impl Cusparse {
    pub fn open() -> Result<Cusparse, String> {
        unsafe {
            let sp = libc::dlopen(c"libcusparse.so.12".as_ptr(), libc::RTLD_NOW);
            if sp.is_null() {
                return Err("cannot dlopen libcusparse.so.12".into());
            }
            let cu = libc::dlopen(c"libcuda.so.1".as_ptr(), libc::RTLD_NOW);
            if cu.is_null() {
                return Err("cannot dlopen libcuda.so.1".into());
            }
            let lib = Lib {
                create: sym(sp, c"cusparseCreate"),
                create_csr: sym(sp, c"cusparseCreateCsr"),
                create_dnvec: sym(sp, c"cusparseCreateDnVec"),
                create_dnmat: sym(sp, c"cusparseCreateDnMat"),
                spmv_size: sym(sp, c"cusparseSpMV_bufferSize"),
                spmv: sym(sp, c"cusparseSpMV"),
                spmm_size: sym(sp, c"cusparseSpMM_bufferSize"),
                spmm: sym(sp, c"cusparseSpMM"),
                csr2csc_size: sym(sp, c"cusparseCsr2cscEx2_bufferSize"),
                csr2csc: sym(sp, c"cusparseCsr2cscEx2"),
                sync: sym(cu, c"cuCtxSynchronize"),
            };
            let mut handle = std::ptr::null_mut();
            check("cusparseCreate", (lib.create)(&mut handle));
            Ok(Cusparse { lib, handle })
        }
    }

    pub fn sync(&self) {
        check("cuCtxSynchronize", unsafe { (self.lib.sync)() });
    }

    /// A CSR descriptor over ArrayFire arrays `ptr`, `col` (`i32`) and `val` (`f32`).
    pub fn csr(&self, rows: usize, cols: usize, ptr: &AI, col: &AI, val: &A) -> Csr {
        unsafe {
            let mut descr = std::ptr::null_mut();
            let nnz = val.elements() as i64;
            check(
                "cusparseCreateCsr",
                (self.lib.create_csr)(&mut descr, rows as i64, cols as i64, nnz, ptr.device_ptr(), col.device_ptr(), val.device_ptr(), INDEX_32I, INDEX_32I, BASE_ZERO, R_32F),
            );
            Csr { descr, rows: rows as i64, cols: cols as i64, nnz }
        }
    }

    /// `y = A x` (or `A^T x`), `x` of `c` columns in column-major `(n, c)`, `y` `(m, c)`; `c == 1`
    /// is the SpMV of the vector `x`.
    pub fn product(&self, a: Csr, x: &A, y: &A, c: usize, transposed: bool) -> Product {
        unsafe {
            let (n_in, n_out) = if transposed { (a.rows, a.cols) } else { (a.cols, a.rows) };
            let (mut dx, mut dy) = (std::ptr::null_mut(), std::ptr::null_mut());
            let (alpha, beta) = (1.0f32, 0.0f32);
            let op = if transposed { OP_T } else { OP_N };
            let mut size = 0usize;
            let spmm = c > 1;
            if spmm {
                check("DnMat x", (self.lib.create_dnmat)(&mut dx, n_in, c as i64, n_in, x.device_ptr(), R_32F, ORDER_COL));
                check("DnMat y", (self.lib.create_dnmat)(&mut dy, n_out, c as i64, n_out, y.device_ptr(), R_32F, ORDER_COL));
                check("SpMM_bufferSize", (self.lib.spmm_size)(self.handle, op, OP_N, &alpha, a.descr, dx, &beta, dy, R_32F, ALG_DEFAULT, &mut size));
            } else {
                check("DnVec x", (self.lib.create_dnvec)(&mut dx, n_in, x.device_ptr(), R_32F));
                check("DnVec y", (self.lib.create_dnvec)(&mut dy, n_out, y.device_ptr(), R_32F));
                check("SpMV_bufferSize", (self.lib.spmv_size)(self.handle, op, &alpha, a.descr, dx, &beta, dy, R_32F, ALG_DEFAULT, &mut size));
            }
            let buffer = (size > 0).then(|| crate::afx::arr(&vec![0.0f32; size.div_ceil(4)], &[size.div_ceil(4) as u64]));
            Product { a, x: dx, y: dy, buffer, transposed, spmm }
        }
    }

    /// Runs the product (asynchronously on the default stream; call [`Cusparse::sync`]).
    pub fn run(&self, p: &Product) {
        unsafe {
            let (alpha, beta) = (1.0f32, 0.0f32);
            let op = if p.transposed { OP_T } else { OP_N };
            let buf = p.buffer.as_ref().map_or(std::ptr::null_mut(), |b| b.device_ptr());
            if p.spmm {
                check("cusparseSpMM", (self.lib.spmm)(self.handle, op, OP_N, &alpha, p.a.descr, p.x, &beta, p.y, R_32F, ALG_DEFAULT, buf));
            } else {
                check("cusparseSpMV", (self.lib.spmv)(self.handle, op, &alpha, p.a.descr, p.x, &beta, p.y, R_32F, ALG_DEFAULT, buf));
            }
        }
    }

    /// CSR to CSC of the matrix: `(val_out, col_ptr_out, row_idx_out)` are device arrays of the
    /// right sizes; the workspace is allocated here once.
    pub fn csr_to_csc_setup(&self, a: &Csr, ptr: &AI, col: &AI, val: &A, out: &(A, AI, AI)) -> Option<A> {
        unsafe {
            let mut size = 0usize;
            check(
                "Csr2cscEx2_bufferSize",
                (self.lib.csr2csc_size)(self.handle, a.rows as c_int, a.cols as c_int, a.nnz as c_int, val.device_ptr(), ptr.device_ptr(), col.device_ptr(), out.0.device_ptr(), out.1.device_ptr(), out.2.device_ptr(), R_32F, 1, BASE_ZERO, CSR2CSC_ALG1, &mut size),
            );
            (size > 0).then(|| crate::afx::arr(&vec![0.0f32; size.div_ceil(4)], &[size.div_ceil(4) as u64]))
        }
    }

    pub fn csr_to_csc(&self, a: &Csr, ptr: &AI, col: &AI, val: &A, out: &(A, AI, AI), buffer: &Option<A>) {
        unsafe {
            let buf = buffer.as_ref().map_or(std::ptr::null_mut(), |b| b.device_ptr());
            check(
                "Csr2cscEx2",
                (self.lib.csr2csc)(self.handle, a.rows as c_int, a.cols as c_int, a.nnz as c_int, val.device_ptr(), ptr.device_ptr(), col.device_ptr(), out.0.device_ptr(), out.1.device_ptr(), out.2.device_ptr(), R_32F, 1, BASE_ZERO, CSR2CSC_ALG1, buf),
            );
        }
    }
}
