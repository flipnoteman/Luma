use luma::*;

fn main() -> Result<()> {
    let t = std::time::Instant::now();

    let a = Array::from_slice(&[1u32, 2, 3], 3)?;
    let b = Array::from_slice(&[10u32, 20, 30], 3)?;

    // One expression, all GPU-resident, no CPU copies in between:
    //
    //   a = [1, 2, 3]
    //   b = [10, 20, 30]
    //
    //   (a + b)          -> [11, 22, 33]
    //   * a              -> [11, 44, 99]
    //   + 1              -> [12, 45, 100]
    //   * 2              -> [24, 90, 200]
    //   - b              -> [14, 70, 170]
    //   / 2              -> [7, 35, 85]
    //
    // Every step is recorded into the context's batch; to_vec() submits it and waits.
    let result = ((((&a + &b) * &a + 1) * 2 - &b) / 2).to_vec()?;

    println!("Chain result: {result:?}");

    // Float math, comparisons, selection and reductions work the same way.
    let x = Array::from_slice(&[-2.0f32, -0.5, 0.0, 1.5, 4.0], 5)?;
    let relu = select(&x.gt(0.0), &x, &Array::zeros(5)?);
    println!("relu(x)   = {:?}", relu.to_vec()?);
    println!("sqrt(|x|) = {:?}", x.abs().sqrt().to_vec()?);
    println!(
        "sum, min, max, mean, norm = {}, {}, {}, {}, {}",
        x.sum().item()?,
        x.min().item()?,
        x.max().item()?,
        x.mean().item()?,
        x.norm().item()?
    );

    println!("Program Time: {:?}", t.elapsed());
    Ok(())
}
