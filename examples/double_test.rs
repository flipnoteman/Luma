use luma::*;

fn main() -> Result<()> {
    let t = std::time::Instant::now();

    let ctx = Context::new()?;
    let a = ctx.from_slice(&[3, 1, 1, 1], &[1u32, 2u32, 3u32])?;
    let b = ctx.from_slice(&[3, 1, 1, 1], &[10u32, 20u32, 30u32])?;

    // Chain of operations — all GPU-resident, no CPU copies in between
    //
    //   a = [1, 2, 3]
    //   b = [10, 20, 30]
    //
    //   step 1: add(a, b)           -> [11, 22, 33]
    //   step 2: multiply(step1, a)  -> [11, 44, 99]
    //   step 3: add_scalar(step2,1) -> [12, 45, 100]
    //   step 4: double(step3)       -> [24, 90, 200]
    //   step 5: subtract(step4, b)  -> [14, 70, 170]
    //   step 6: divide_scalar(step5, 2) -> [7, 35, 85]
    //
    // Every step is submitted to the GPU without waiting; read() is the only sync point.
    let result = a.add(&b)?                            // [11, 22, 33]
        .multiply(&a)?                                 // [11, 44, 99]
        .add_scalar(1u32)?                             // [12, 45, 100]
        .double()?                                     // [24, 90, 200]
        .subtract(&b)?                                 // [14, 70, 170]
        .divide_scalar(2u32)?                          // [7, 35, 85]
        .read()?;

    println!("Chain result: {:?}", result);
    println!("Program Time: {:?}", t.elapsed());
    Ok(())
}
