
use bytemuck::Pod;
use execution::Executor;

mod execution;

const SHADERS_PATH: &str = "./Luma/operations";

#[derive(Debug)]
pub struct Array {
    dimensions: [u64; 4],
    executor: Executor,
}

impl Array {
    pub async fn new<T>(dimensions: &[u64; 4], data: &[T]) -> Result<Self, String>
    where
        T: Pod,
    {
        // Instantiate our Executor
        let mut ex = Box::new(Executor::new(SHADERS_PATH).await?);

        // Setup input output buffers with our data
        // TODO: Incorporate the dimensions array
        ex.setup_buffers(data).await?;

        Ok(Array {
            dimensions: *dimensions,
            executor: *ex,
        })
    }

    pub async fn double_test(&self) -> Result<(), String> {
        self.executor.test_fn().await
    }
}
