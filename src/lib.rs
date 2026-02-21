#![allow(dead_code)]
extern crate core;

mod array;
mod executor;
mod gpu;
mod gpu_type;
mod operations;

pub use array::Array;
pub use gpu_type::GpuType;

use crate::executor::Executor;
use tokio::sync::OnceCell;

/// Instantiates a new [Array]
/// The first argument is the dimensions of the array, while the second is the data to initialize it
/// with.
///
/// # Example
/// ```no_run
/// use luma::Array;
/// async {
///     let array1 = luma::array!(&[3, 1, 1, 1], &[1u32, 6u32, 5u32]);
/// };
/// ```
#[macro_export]
macro_rules! array {
    ($dims:expr, $data:expr) => {
        Array::new($dims, $data)
        .await.expect("Could not create Array.");
    };
}

const PROJECT_DIR: &str = env!("CARGO_MANIFEST_DIR");
const SHADERS_PATH: &str = "./operations";

/// Static thread-safe executor with interior mutability.
static EXECUTOR: OnceCell<Executor> = OnceCell::const_new();

/// Ensure the executor is initialized. Only the first caller creates it;
/// concurrent callers wait for that initialization to finish.
async fn ensure_executor() -> &'static Executor {
    EXECUTOR.get_or_init(|| async {
        Executor::new(&format!("{}/{}", PROJECT_DIR, SHADERS_PATH)).await.unwrap()
    }).await
}
