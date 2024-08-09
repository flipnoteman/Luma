
# Luma
A cross-platform GPU-accelerated linear algebra library WIP

Current Usage Example:

```rs
use luma::*;

#[tokio::main]
async fn main() {
    let t = std::time::Instant::now();

    // Can now instantiate an [Array] with macros.
    let array1 = array!(&[3u64, 1, 1, 1], &[1u32, 6u32, 5u32]);
    let array2 = array!(&[3u64, 1, 1, 1], &[4u32, 12u32, 10u32]);
    // [double_test] calls [test_fn()] in Executor which Doubles all the values in the array.
    let res1 = array1.double_test().await.unwrap();
    let res2 = array2.double_test().await.unwrap();

    println!("Result for {} = {:?}", array1.id(), res1);
    println!("Result for {} = {:?}", array2.id(), res2);

    println!("Time: {:?}", t.elapsed())
}
```
