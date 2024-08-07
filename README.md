
# Luma
### A cross-platform GPU-accelerated linear algebra library 
## WIP

Current Usage Example:

```rs
use luma::Array;

#[tokio::main]
async fn main() {
    let t = std::time::Instant::now();
    let array1 = Array::new(&[1u64, 1, 1, 1], &[1u32, 6u32, 5u32])
        .await
        .unwrap();
    let array2 = Array::new(&[3u64, 1, 1, 1], &[4u32, 3u32, 1u32])
        .await
        .unwrap();

    // [double_test] calls [test_fn()] in Executor which Doubles all the values in the array.
    let _ = array1.double_test().await;

    // [double_test] calls [test_fn()] in Executor which Doubles all the values in the array.
    let _ = array2.double_test().await;

    println!("Time: {:?}", t.elapsed())
}
```
