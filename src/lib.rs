#![allow(dead_code)]
use std::sync::OnceLock;
use bytemuck::Pod;
use uuid::Uuid;
mod execution;
use crate::execution::Executor;

/// Instantiates a new [Array]
/// The first argument is the dimensions of the array, while the second is the data to initialize it
/// with.
///
/// # Example
/// ```
///  let array1 = luma::array!(&[3u64, 1, 1, 1], &[1u32, 6u32, 5u32]);
/// ```
#[macro_export]
macro_rules! array {
    ($dims:expr, $data:expr) => {
        Array::new($dims, $data)
        .await.expect("Could not create Array.");
    };
}


const SHADERS_PATH: &str = "./Luma/operations";

/// Static thread-safe executor with interior mutability.
static EXECUTOR: OnceLock<Executor> = OnceLock::new();

/// Instantiates a new [Array]
/// The first argument is the dimensions of the array, while the second is the data to initialize it
/// with.
///
/// # Example
/// ```
///  let array1 = luma::Array::new(&[3u64, 1, 1, 1], &[1u32, 6u32, 5u32]).await.expect("Could not create Array.");;
/// ```
#[derive(Debug)]
pub struct Array {
    dimensions: [u64; 4],
    id: String,
}

impl Drop for Array {
    /// We need to handle when it goes out of scope by deleting it from our [Executor]
    fn drop(&mut self) {
        EXECUTOR.get().expect("Could not drop value").drop(&self.id);
    }
}

impl Array {
    pub async fn new<T>(dimensions: &[u64; 4], data: &[T]) -> Result<Self, String>
    where
        T: Pod,
    {
        // Set up the executor only if not already initialized.
        std::thread::spawn(|| {
            Box::pin(
                async {
                    if EXECUTOR.get().is_none() {
                        let ex = Executor::new(SHADERS_PATH).await.unwrap();
                        EXECUTOR.set(ex).unwrap();
                    }
                }
            )
        }).join().unwrap().await;

        let id = Uuid::new_v4();
        // Setup input output buffers with our data
        // TODO: Incorporate the dimensions array
        EXECUTOR.get().unwrap().setup_buffers(data, id.into()).await?;

        Ok(Array {
            dimensions: *dimensions,
            id: id.into()
        })
    }

    pub fn id(&self) -> String {
        self.id.clone()
    }

    pub async fn double_test(&self) -> Result<Vec<u32>, String> {
        EXECUTOR.get().unwrap().execute_op(&self.id).await
    }
}

