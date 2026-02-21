use std::marker::PhantomData;
use uuid::Uuid;

use crate::gpu_type::GpuType;
use crate::operations::Operation;
use crate::{ensure_executor, EXECUTOR};

/// GPU-resident array. Data stays on the GPU until `to_vec()` is called.
///
/// # Example
/// ```
/// async {
///     let a = luma::Array::<u32>::new(&[3, 1, 1, 1], &[1u32, 2u32, 3u32]).await.unwrap();
///     let b = a.double().await.unwrap();
///     let result = b.to_vec().await.unwrap();
/// };
/// ```
#[derive(Debug)]
pub struct Array<T: GpuType> {
    dimensions: [usize; 4],
    id: String,
    _marker: PhantomData<T>,
}

impl<T: GpuType> Drop for Array<T> {
    fn drop(&mut self) {
        EXECUTOR.get().expect("Could not drop value").drop(&self.id);
    }
}

impl<T: GpuType> Array<T> {
    pub async fn new(dimensions: &[usize; 4], data: &[T]) -> Result<Self, String> {
        ensure_executor().await;

        let id = Uuid::new_v4();
        let type_suffix = T::type_suffix().to_string();
        EXECUTOR.get().unwrap().setup_buffers(dimensions, data, id.to_string(), type_suffix).await?;

        Ok(Array {
            dimensions: *dimensions,
            id: id.to_string(),
            _marker: PhantomData,
        })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn dimensions(&self) -> &[usize; 4] {
        &self.dimensions
    }

    /// Read data back from the GPU to CPU. This is the only way data leaves the GPU.
    pub async fn to_vec(&self) -> Result<Vec<T>, String> {
        let bytes = EXECUTOR.get().unwrap().readback(&self.id).await?;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    // -- Unary operations --

    pub async fn double(&self) -> Result<Array<T>, String> {
        let new_id = EXECUTOR.get().unwrap().execute_unary_gpu(&self.id, &Operation::DOUBLE).await?;
        Ok(Array {
            dimensions: self.dimensions,
            id: new_id,
            _marker: PhantomData,
        })
    }

    // -- Binary operations (array-array) --

    pub async fn add(&self, rhs: &Array<T>) -> Result<Array<T>, String> {
        self.binary_op(rhs, &Operation::ADD).await
    }

    pub async fn subtract(&self, rhs: &Array<T>) -> Result<Array<T>, String> {
        self.binary_op(rhs, &Operation::SUBTRACT).await
    }

    pub async fn multiply(&self, rhs: &Array<T>) -> Result<Array<T>, String> {
        self.binary_op(rhs, &Operation::MULTIPLY).await
    }

    pub async fn divide(&self, rhs: &Array<T>) -> Result<Array<T>, String> {
        self.binary_op(rhs, &Operation::DIVIDE).await
    }

    // -- Scalar operations --

    pub async fn add_scalar(&self, scalar: T) -> Result<Array<T>, String> {
        let tmp = Array::<T>::new(&[1, 1, 1, 1], &[scalar]).await?;
        self.binary_op_unchecked(&tmp, &Operation::ADD).await
    }

    pub async fn subtract_scalar(&self, scalar: T) -> Result<Array<T>, String> {
        let tmp = Array::<T>::new(&[1, 1, 1, 1], &[scalar]).await?;
        self.binary_op_unchecked(&tmp, &Operation::SUBTRACT).await
    }

    pub async fn multiply_scalar(&self, scalar: T) -> Result<Array<T>, String> {
        let tmp = Array::<T>::new(&[1, 1, 1, 1], &[scalar]).await?;
        self.binary_op_unchecked(&tmp, &Operation::MULTIPLY).await
    }

    pub async fn divide_scalar(&self, scalar: T) -> Result<Array<T>, String> {
        let tmp = Array::<T>::new(&[1, 1, 1, 1], &[scalar]).await?;
        self.binary_op_unchecked(&tmp, &Operation::DIVIDE).await
    }

    // -- Internal helpers --

    async fn binary_op(&self, rhs: &Array<T>, op: &Operation) -> Result<Array<T>, String> {
        if self.dimensions != rhs.dimensions {
            return Err(format!(
                "Dimension mismatch: {:?} vs {:?}",
                self.dimensions, rhs.dimensions
            ));
        }
        self.binary_op_unchecked(rhs, op).await
    }

    async fn binary_op_unchecked(&self, rhs: &Array<T>, op: &Operation) -> Result<Array<T>, String> {
        let new_id = EXECUTOR.get().unwrap().execute_binary_gpu(&self.id, &rhs.id, op).await?;
        Ok(Array {
            dimensions: self.dimensions,
            id: new_id,
            _marker: PhantomData,
        })
    }
}
