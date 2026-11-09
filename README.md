
# Luma
A cross-platform GPU-accelerated linear algebra library WIP

Current Usage Example:

```rs
use luma::*;

fn main() -> Result<()> {
    let t = std::time::Instant::now();

    // A [Context] is an explicit handle to the GPU. It is cheap to clone and creates [Array]s.
    let ctx = Context::new()?;

    let array1 = ctx.from_slice(&[3, 1, 1, 1], &[1u32, 6, 5])?;
    let array2 = ctx.from_slice(&[3, 1, 1, 1], &[4u32, 12, 10])?;

    // Operations are submitted to the GPU without waiting and return new arrays.
    let sum = array1.add(&array2)?;     // [5, 18, 15]
    let doubled = sum.double()?;        // [10, 36, 30]

    // `read()` is the only call that waits for the GPU and copies data back to the host.
    println!("Result = {:?}; shape = {:?}", doubled.read()?, doubled.shape());

    println!("Program Time: {:?}", t.elapsed());
    Ok(())
}
```
