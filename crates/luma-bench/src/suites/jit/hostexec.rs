//! The measurement behind W3-HOSTCPU: what running the host strategies on the CPU backend (an
//! auxiliary CPU context, the "CPU-JIT") would buy against today's paths. Rows time work on the
//! context under test and on a CPU context opened next to it, in one process.
//!
//! * `hostexec/small_*`: one family over `2^6 .. 2^16` elements. On a zero-copy device with
//!   `LUMA_HOST_FAST=on` the context-under-test rows are the `host.rs` interpreter (and the GPU path
//!   with `LUMA_HOST_FAST=off`); the `_cpujit` rows are the CPU backend's JIT on the same program.
//!   The CPU rows keep the data resident in the CPU context, which is what a zero-copy import would
//!   give (`_cpujit`), and time a call that evaluates and waits.
//! * `hostexec/{map,chain16,sum,minmax}_*`: big arrays that start in host memory. `_dev` uploads
//!   them and runs on the device (the upload-and-run of a spilled array; `_dev_rb` also reads the
//!   result back), `_cpujit` runs on the CPU context over resident data (the zero-copy bound) and
//!   `_cpucopy` includes the copy into a CPU buffer. `hostexec/upload` is the bare upload.

use super::common::*;
use crate::harness::*;

/// `y = x * 1.0001 + 0.5` and a 16-operation chain.
fn map(x: &A) -> A {
    &(x * 1.0001) + 0.5
}

fn chain16(x: &A) -> A {
    let mut y = x.clone();
    for i in 0..4 {
        let c = 1.0 + i as f32 * 1e-3;
        y = &(&(&y * c) + 1e-3).abs() - 1e-4;
        y = &(&y * 0.999) + (c * 1e-3);
    }
    y
}

fn minmax(x: &A) -> A {
    let m = &(x * 1.0001) + 0.5;
    &m.min() + &m.max()
}

fn sum(x: &A) -> A {
    (&(x * 1.0001) + 0.5).sum()
}

trait CaseOn {
    fn case_on(&mut self, on_cpu: bool, case: &str, size: &str, work: Work, f: &mut dyn FnMut());
}

impl CaseOn for Harness {
    /// A case run on the CPU context launches nothing on the context under test: `transfer_case` registers it as such.
    fn case_on(&mut self, on_cpu: bool, case: &str, size: &str, work: Work, f: &mut dyn FnMut()) {
        if on_cpu {
            self.transfer_case(case, size, work, f);
        } else {
            self.case(case, size, work, f);
        }
    }
}

fn small(h: &mut Harness, cpu: &luma::Context) {
    let ctx = h.ctx.clone();
    for e in [6u32, 8, 10, 12, 14, 16] {
        let n = 1usize << e;
        let xh = values(n, 21, 0.5, 1.5);
        let tag = format!("n=2^{e}");
        let cols = 64usize.min(n);
        let rows = n / cols;
        for (on_cpu, suffix) in [(false, ""), (true, "_cpujit")] {
            if on_cpu && ctx.is_cpu() {
                continue;
            }
            let c = if on_cpu { cpu } else { &ctx };
            let x = A::from_slice_on(c, &xh, n).unwrap();
            let x2 = x.reshape([rows, cols]);
            let mut s = 1.0f32;
            let rep = 20usize;
            h.case_on(on_cpu, &format!("hostexec/small_map{suffix}"), &tag, per_sec(rep, "call/s"), &mut || {
                for _ in 0..rep {
                    s += 1e-4;
                    force(&(&(&x * s) + 0.5));
                }
            });
            h.case_on(on_cpu, &format!("hostexec/small_sum{suffix}"), &tag, per_sec(rep, "call/s"), &mut || {
                let mut t = 0.0;
                for _ in 0..rep {
                    s += 1e-4;
                    t += (&x * s).sum().item().unwrap();
                }
                std::hint::black_box(t);
            });
            h.case_on(on_cpu, &format!("hostexec/small_axis_sum{suffix}"), &tag, per_sec(rep, "call/s"), &mut || {
                for _ in 0..rep {
                    s += 1e-4;
                    force(&(&x2 * s).sum_axis(1));
                }
            });
            h.case_on(on_cpu, &format!("hostexec/small_argmax{suffix}"), &tag, per_sec(rep, "call/s"), &mut || {
                let mut t = 0u32;
                for _ in 0..rep {
                    s += 1e-4;
                    t = t.wrapping_add((&x * s).argmax().item().unwrap());
                }
                std::hint::black_box(t);
            });
        }
    }
    // Small matmul (the interpreter's cap is 1M FMAs).
    for m in [16usize, 32, 64, 100] {
        let ah = values(m * m, 22, 0.5, 1.5);
        let tag = format!("{m}x{m}");
        for (on_cpu, suffix) in [(false, ""), (true, "_cpujit")] {
            if on_cpu && ctx.is_cpu() {
                continue;
            }
            let c = if on_cpu { cpu } else { &ctx };
            let a = A::from_slice_on(c, &ah, [m, m]).unwrap();
            let rep = 10usize;
            let mut s = 1.0f32;
            h.case_on(on_cpu, &format!("hostexec/small_matmul{suffix}"), &tag, per_sec(rep, "call/s"), &mut || {
                for _ in 0..rep {
                    s += 1e-4;
                    force(&(&a * s).matmul(&a));
                }
            });
        }
    }
}

fn big(h: &mut Harness, cpu: &luma::Context) {
    let ctx = h.ctx.clone();
    let mibs: &[usize] = if h.quick { &[16] } else { &[16, 64, 256] };
    type Fam = (&'static str, fn(&A) -> A, bool);
    let fams: [Fam; 4] = [("map", map, true), ("chain16", chain16, true), ("sum", sum, false), ("minmax", minmax, false)];
    for &mib in mibs {
        let n = mib * (1 << 20) / 4;
        let xh = values(n, 23, 0.5, 1.5);
        let tag = format!("{mib}MiB");
        let bytes = Work::Bytes(4.0 * n as f64);
        h.transfer_case("hostexec/upload", &tag, bytes, &mut || {
            force(&A::from_slice_on(&ctx, &xh, n).unwrap());
        });
        let xc = A::from_slice_on(cpu, &xh, n).unwrap();
        force(&xc);
        for (name, f, vector) in fams {
            let rb = |a: &A| {
                if vector {
                    std::hint::black_box(a.to_vec().unwrap());
                } else {
                    std::hint::black_box(a.item().unwrap());
                }
            };
            let touched = |k: usize| Work::Bytes(4.0 * n as f64 * k as f64);
            let k = if vector { 2 } else { 1 };
            if !ctx.is_cpu() {
                h.case(&format!("hostexec/{name}_dev"), &tag, touched(k), &mut || {
                    let x = A::from_slice_on(&ctx, &xh, n).unwrap();
                    let y = f(&x);
                    force(&y);
                });
                if vector {
                    h.case(&format!("hostexec/{name}_dev_rb"), &tag, touched(k), &mut || {
                        let x = A::from_slice_on(&ctx, &xh, n).unwrap();
                        rb(&f(&x));
                    });
                }
                h.transfer_case(&format!("hostexec/{name}_cpucopy"), &tag, touched(k), &mut || {
                    let x = A::from_slice_on(cpu, &xh, n).unwrap();
                    force(&f(&x));
                });
            }
            h.transfer_case(&format!("hostexec/{name}_cpujit"), &tag, touched(k), &mut || {
                force(&f(&xc));
            });
        }
    }
}

pub fn run(h: &mut Harness) {
    h.guarded("hostexec", |h| {
        let cpu = match luma::Context::cpu() {
            Ok(c) => c,
            Err(e) => {
                println!("hostexec: no CPU context ({e})");
                return;
            }
        };
        small(h, &cpu);
        big(h, &cpu);
    });
}
