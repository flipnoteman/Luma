
# Luma
A cross-platform GPU-accelerated linear algebra library WIP

Current Usage Example:

```rs
use luma::*;

#[tokio::main]
async fn main() {
    let t = std::time::Instant::now();

    // Can now instantiate an [Array] with macros.
    let array1 = array!(&[1u64, 1, 1, 1], &[1u32, 6u32, 5u32]);

    // [double_test] calls [test_fn()] in Executor which Doubles all the values in the array.
    let _ = array1.double_test().await;

    println!("Time: {:?}", t.elapsed())
}
```
