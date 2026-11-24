//! Prints the device luma opened and its cost model, including the memory hierarchy numbers that
//! calibration measures (the first run on a device calibrates; later runs read the disk cache).

use luma::*;

fn main() -> Result<()> {
    let start = std::time::Instant::now();
    let ctx = Context::new()?;
    println!("opened in {:.0} ms", start.elapsed().as_secs_f64() * 1e3);
    println!("device: {} ({})", ctx.device_name(), ctx.backend());
    let cost = ctx.cost_model();
    println!("launch {:.2} us, bandwidth {:.0} GB/s, f32 {:.0} ops/ns, df32 x{:.1}", cost.launch_us, cost.bandwidth_gbs, cost.f32_ops_per_ns, cost.df32_factor);
    match cost.f64_factor {
        Some(factor) => println!("f64 x{factor:.1}"),
        None => println!("no f64"),
    }
    match ctx.memory_model() {
        Some(memory) => println!("memory: {memory}"),
        None => println!("memory: not calibrated (the profile's fixed numbers are in use)"),
    }
    Ok(())
}
