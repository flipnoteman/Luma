//! Loader for the NVVM compiler that ships inside the NVIDIA driver.
//!
//! NVVM IR text (LLVM 7 dialect) plus NVIDIA's `libdevice` is compiled to PTX
//! without the CUDA toolkit being installed.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::sync::Mutex;

use libloading::Library;

const LIBDEVICE: &[u8] = include_bytes!("../vendor/libdevice.10.bc");

#[cfg(windows)]
const CANDIDATES: &[&str] = &["nvvm64.dll", "nvvm70.dll", "nvvm64_40_0.dll"];
#[cfg(not(windows))]
const CANDIDATES: &[&str] = &["libnvidia-nvvm.so.4", "libnvidia-nvvm70.so.4", "libnvvm.so.4"];

const SELF_TEST: &str = r#"target triple = "nvptx64-nvidia-cuda"
target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
define void @luma_main(float* %x) nounwind {
entry:
  store float 1.0, float* %x, align 4
  ret void
}
!nvvm.annotations = !{!0}
!0 = !{void (float*)* @luma_main, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0}
"#;

type Prog = *mut c_void;
type Res = c_int;

pub(crate) struct Nvvm {
    pub version: (i32, i32),
    lock: Mutex<()>,
    create: unsafe extern "C" fn(*mut Prog) -> Res,
    destroy: unsafe extern "C" fn(*mut Prog) -> Res,
    add: unsafe extern "C" fn(Prog, *const c_char, usize, *const c_char) -> Res,
    lazy_add: unsafe extern "C" fn(Prog, *const c_char, usize, *const c_char) -> Res,
    compile: unsafe extern "C" fn(Prog, c_int, *const *const c_char) -> Res,
    result_size: unsafe extern "C" fn(Prog, *mut usize) -> Res,
    result: unsafe extern "C" fn(Prog, *mut c_char) -> Res,
    log_size: unsafe extern "C" fn(Prog, *mut usize) -> Res,
    log: unsafe extern "C" fn(Prog, *mut c_char) -> Res,
    error_string: unsafe extern "C" fn(c_int) -> *const c_char,
    // Keeps the function pointers above valid; must outlive them.
    _lib: Library,
}

impl Nvvm {
    /// Tries each candidate library; the first that loads, resolves every symbol, reports IR
    /// major 2 and compiles the self-test for the device wins.
    pub(crate) fn open(major: i32, minor: i32) -> Result<Nvvm, String> {
        let mut reasons = Vec::new();
        for name in CANDIDATES {
            match Self::try_open(name, major, minor) {
                Ok(n) => return Ok(n),
                Err(e) => reasons.push(format!("{name}: {e}")),
            }
        }
        Err(reasons.join("; "))
    }

    fn try_open(name: &str, major: i32, minor: i32) -> Result<Nvvm, String> {
        // SAFETY: loading the driver's NVVM library runs its initialisers, as any driver load does.
        let lib = unsafe { Library::new(name) }.map_err(|e| e.to_string())?;
        macro_rules! sym {
            ($n:literal) => {
                // SAFETY: the symbol is declared with the documented NVVM C signature.
                *unsafe { lib.get($n) }
                    .map_err(|e| format!("missing symbol: {e}"))?
            };
        }
        let version_fn: unsafe extern "C" fn(*mut c_int, *mut c_int) -> Res = sym!(b"nvvmVersion\0");
        let ir_version_fn: unsafe extern "C" fn(*mut c_int, *mut c_int, *mut c_int, *mut c_int) -> Res =
            sym!(b"nvvmIRVersion\0");
        let mut n = Nvvm {
            version: (0, 0),
            lock: Mutex::new(()),
            create: sym!(b"nvvmCreateProgram\0"),
            destroy: sym!(b"nvvmDestroyProgram\0"),
            add: sym!(b"nvvmAddModuleToProgram\0"),
            lazy_add: sym!(b"nvvmLazyAddModuleToProgram\0"),
            compile: sym!(b"nvvmCompileProgram\0"),
            result_size: sym!(b"nvvmGetCompiledResultSize\0"),
            result: sym!(b"nvvmGetCompiledResult\0"),
            log_size: sym!(b"nvvmGetProgramLogSize\0"),
            log: sym!(b"nvvmGetProgramLog\0"),
            error_string: sym!(b"nvvmGetErrorString\0"),
            _lib: lib,
        };
        let (mut vmaj, mut vmin) = (0, 0);
        let r = unsafe { version_fn(&mut vmaj, &mut vmin) };
        if r != 0 {
            return Err(format!("nvvmVersion failed: {}", n.err_str(r)));
        }
        let (mut imaj, mut imin, mut dmaj, mut dmin) = (0, 0, 0, 0);
        let r = unsafe { ir_version_fn(&mut imaj, &mut imin, &mut dmaj, &mut dmin) };
        if r != 0 {
            return Err(format!("nvvmIRVersion failed: {}", n.err_str(r)));
        }
        if imaj != 2 {
            return Err(format!("unsupported NVVM IR version {imaj}.{imin}"));
        }
        n.version = (vmaj, vmin);
        n.compile(SELF_TEST, major, minor)
            .map_err(|e| format!("self-test failed: {e}"))?;
        Ok(n)
    }

    fn err_str(&self, r: Res) -> String {
        // SAFETY: returns a static NUL-terminated string.
        let p = unsafe { (self.error_string)(r) };
        if p.is_null() {
            format!("error {r}")
        } else {
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        }
    }

    fn program_log(&self, prog: Prog) -> String {
        let mut size = 0usize;
        // SAFETY: `prog` is a live program.
        if unsafe { (self.log_size)(prog, &mut size) } != 0 || size == 0 {
            return String::new();
        }
        let mut buf = vec![0u8; size];
        if unsafe { (self.log)(prog, buf.as_mut_ptr().cast()) } != 0 {
            return String::new();
        }
        while buf.last() == Some(&0) {
            buf.pop();
        }
        String::from_utf8_lossy(&buf).into_owned()
    }

    fn run(&self, prog: Prog, ir: &str, major: i32, minor: i32) -> Result<String, String> {
        let fail = |r: Res| Err(format!("{}\n{}", self.err_str(r), self.program_log(prog)));
        // SAFETY: pointers and lengths come from live slices; `prog` is live.
        unsafe {
            let r = (self.add)(prog, ir.as_ptr().cast(), ir.len(), c"luma".as_ptr());
            if r != 0 {
                return fail(r);
            }
            let r = (self.lazy_add)(
                prog,
                LIBDEVICE.as_ptr().cast(),
                LIBDEVICE.len(),
                c"libdevice".as_ptr(),
            );
            if r != 0 {
                return fail(r);
            }
            let arch = std::ffi::CString::new(format!("-arch=compute_{major}{minor}"))
                .map_err(|e| e.to_string())?;
            // f32 `/` and `sqrt` as WGSL specifies them (2.5 ulp and the device's square root), as
            // the other GPU backends compile them, instead of the slower IEEE-rounded sequences;
            // f32 subnormals flushed to zero, which WGSL allows and the Vulkan driver does, so the
            // approximations need no range fix-ups. add/sub/mul stay correctly rounded; f64 is
            // unaffected.
            let opts = [arch.as_ptr(), c"-prec-div=0".as_ptr(), c"-prec-sqrt=0".as_ptr(), c"-ftz=1".as_ptr()];
            let r = (self.compile)(prog, opts.len() as c_int, opts.as_ptr());
            if r != 0 {
                return fail(r);
            }
            let mut size = 0usize;
            let r = (self.result_size)(prog, &mut size);
            if r != 0 {
                return fail(r);
            }
            let mut buf = vec![0u8; size];
            let r = (self.result)(prog, buf.as_mut_ptr().cast());
            if r != 0 {
                return fail(r);
            }
            while buf.last() == Some(&0) {
                buf.pop();
            }
            String::from_utf8(buf).map_err(|e| e.to_string())
        }
    }

    /// IR text -> PTX text. Err = "<nvvmGetErrorString>\n<program log>".
    pub(crate) fn compile(&self, ir: &str, major: i32, minor: i32) -> Result<String, String> {
        let _guard = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let mut prog: Prog = std::ptr::null_mut();
        // SAFETY: `prog` is a valid out-pointer.
        let r = unsafe { (self.create)(&mut prog) };
        if r != 0 {
            return Err(format!("{}\n", self.err_str(r)));
        }
        let out = self.run(prog, ir, major, minor);
        // SAFETY: `prog` was created above and is destroyed exactly once.
        unsafe { (self.destroy)(&mut prog) };
        out
    }
}
