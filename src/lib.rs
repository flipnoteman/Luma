#![allow(dead_code)]
extern crate core;
use std::sync::OnceLock;
use bytemuck::Pod;
use uuid::Uuid;
mod execution;
mod utils;

use crate::execution::{Executor, Operation};

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

const PROJECT_DIR: &str = env!("CARGO_MANIFEST_DIR");
const SHADERS_PATH: &str = "./operations";

/// Static thread-safe executor with interior mutability.
static EXECUTOR: OnceLock<Executor> = OnceLock::new();

/// Instantiates a new [Array]
/// The first argument is the dimensions of the array, while the second is the data to initialize it
/// with.
///
/// # Example
/// ```
/// async {
///     let array1 = luma::Array::new(&[3, 1, 1, 1], &[1u32, 6u32, 5u32]).await.expect("Could not create Array.");
/// }
/// ```
#[derive(Debug)]
pub struct Array {
    dimensions: [usize; 4],
    id: String,
}

impl Drop for Array {
    /// We need to handle when it goes out of scope by deleting it from our [Executor]
    fn drop(&mut self) {
        EXECUTOR.get().expect("Could not drop value").drop(&self.id);
    }
}

impl Array {
    pub async fn new<T>(dimensions: &[usize; 4], data: &[T]) -> Result<Self, String>
    where
        T: Pod + std::fmt::Debug,
    {
        // Set up the executor only if not already initialized.
        std::thread::spawn(|| {
            Box::pin(
                async {
                    if EXECUTOR.get().is_none() {
                        let ex = Executor::new(&format!("{}/{}", PROJECT_DIR, SHADERS_PATH)).await.unwrap();
                        EXECUTOR.set(ex).unwrap();
                    }
                }
            )
        }).join().unwrap().await;

        // let test = vec![vec![3, 5, 6], vec![1, 2, 3], vec![2, 3, 6]];
        // println!("Dimensions: {:?}", utils::extrapolate_dimensions(&test));

        let id = Uuid::new_v4();
        // Setup input output buffers with our data
        // TODO: Incorporate the dimensions array
        EXECUTOR.get().unwrap().setup_buffers(dimensions, data, id.into()).await?;

        Ok(Array {
            dimensions: *dimensions,
            id: id.into()
        })
    }

    pub fn id(&self) -> String {
        self.id.clone()
    }

    pub async fn double_test(&self) -> Result<Vec<u32>, String> {
        EXECUTOR.get().unwrap().execute_op(&self.id, Operation::DOUBLE).await
    }
}

