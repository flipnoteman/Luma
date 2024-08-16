
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
    let test_1 = std::time::Instant::now();
    let res1 = array1.double_test().await.unwrap();
    println!("Result for {} = {:?}; time = {:?}", array1.id(), res1, test_1.elapsed());
    let test_2 = std::time::Instant::now();
    let res2 = array2.double_test().await.unwrap();
    println!("Result for {} = {:?}; time = {:?}", array2.id(), res2, test_2.elapsed());

    println!("Program Time: {:?}", t.elapsed())
}
```
