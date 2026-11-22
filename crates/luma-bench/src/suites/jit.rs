//! The `jit` suite: the costs of Luma's lazy-fusion JIT and its recording machinery rather than
//! the throughput of one kernel: compiling kernels (cold and cached), launch latency, fusion of
//! long and wide expressions, time loops (eager, traced and replayed, compiled), the memory of a
//! recorded step, host readback and threads sharing a context.

mod common;
mod compile;
mod fusion;
mod launch;
mod launchcost;
mod loops;
mod memory;
mod readback;
mod threads;
mod trace;

use std::time::Instant;

use crate::harness::Harness;

type Group = fn(&mut Harness);

pub fn run(h: &mut Harness) {
    let groups: [(&str, Group); 9] = [
        ("compile", compile::run),
        ("launch", launch::run),
        ("launchcost", launchcost::run),
        ("loop", loops::run),
        ("fusion", fusion::run),
        ("trace", trace::run),
        ("memory", memory::run),
        ("readback", readback::run),
        ("threads", threads::run),
    ];
    let start = Instant::now();
    for (name, group) in groups {
        let t = Instant::now();
        group(h);
        println!("-- jit/{name}: {:.1} s", t.elapsed().as_secs_f64());
    }
    println!("-- jit: {:.1} s", start.elapsed().as_secs_f64());
}
