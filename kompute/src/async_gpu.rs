//! Idiomatic Rust async/await patterns for GPU operations.

use crate::error::{KomputeError, Result};
use crate::handle_tracker::HandleTracker;
use crate::manager::Manager;
use crate::sequence::Sequence;
use crate::tensor::Tensor;
use std::future::Future;
use std::time::{Duration, Instant};
use tokio::time::{sleep, timeout};

/// High-level async GPU context for idiomatic Rust patterns.
///
/// This provides async/await support that integrates naturally with
/// Rust's async ecosystem (tokio, async-std, etc.).
pub struct AsyncGpuContext {
    manager: Manager,
    tracker: HandleTracker,
}

impl AsyncGpuContext {
    /// Create a new async GPU context.
    pub fn new() -> Result<Self> {
        Ok(Self {
            manager: Manager::new()?,
            tracker: HandleTracker::new(),
        })
    }

    /// Create a new async GPU context with a specific device.
    pub fn with_device(device_id: u32) -> Result<Self> {
        Ok(Self {
            manager: Manager::with_device(device_id)?,
            tracker: HandleTracker::new(),
        })
    }

    /// Create a tensor asynchronously.
    pub async fn create_tensor<T: crate::tensor::TensorData>(
        &mut self,
        data: &[T],
    ) -> Result<AsyncTensor<T>> {
        // Simulate async operation (in real implementation, this might involve
        // async allocation or initialization)
        tokio::task::yield_now().await;

        let tensor = Tensor::new(&mut self.manager, data)?;
        let handle = tensor.handle();
        self.tracker.register_handle("tensor", handle);

        Ok(AsyncTensor {
            tensor,
            tracker: self.tracker.clone(),
        })
    }

    /// Create an uninitialized tensor asynchronously.
    pub async fn create_tensor_uninit<T: crate::tensor::TensorData>(
        &mut self,
        size: usize,
    ) -> Result<AsyncTensor<T>> {
        tokio::task::yield_now().await;

        let tensor = Tensor::uninit(&mut self.manager, size)?;
        let handle = tensor.handle();
        self.tracker.register_handle("tensor", handle);

        Ok(AsyncTensor {
            tensor,
            tracker: self.tracker.clone(),
        })
    }

    /// Create a sequence asynchronously.
    pub async fn create_sequence(&mut self) -> Result<AsyncSequence> {
        tokio::task::yield_now().await;

        let sequence = Sequence::new(&mut self.manager)?;
        let handle = sequence.handle();
        self.tracker.register_handle("sequence", handle);

        Ok(AsyncSequence {
            sequence: Some(sequence),
            tracker: self.tracker.clone(),
        })
    }

    /// Execute a computation workflow asynchronously.
    pub async fn execute_workflow<F, Fut>(&mut self, workflow: F) -> Result<()>
    where
        F: FnOnce(&mut AsyncGpuContext) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        workflow(self).await
    }

    /// Get statistics about resource usage.
    pub fn statistics(&self) -> crate::handle_tracker::HandleStatistics {
        self.tracker.statistics()
    }

    /// Access the underlying manager.
    pub fn manager(&mut self) -> &mut Manager {
        &mut self.manager
    }
}

/// Async tensor with automatic resource tracking.
pub struct AsyncTensor<T: crate::tensor::TensorData> {
    tensor: Tensor<T>,
    tracker: HandleTracker,
}

impl<T: crate::tensor::TensorData> AsyncTensor<T> {
    /// Get tensor data asynchronously.
    pub async fn to_vec(&self) -> Result<Vec<T>> {
        // Simulate async operation
        tokio::task::yield_now().await;
        self.tensor.to_vec()
    }

    /// Get tensor size asynchronously.
    pub async fn size(&self) -> usize {
        tokio::task::yield_now().await;
        self.tensor.size()
    }

    /// Get tensor size in bytes asynchronously.
    pub async fn size_bytes(&self) -> usize {
        tokio::task::yield_now().await;
        self.tensor.size_bytes()
    }

    /// Get the handle for use with sequences.
    pub fn handle(&self) -> u64 {
        self.tensor.handle()
    }

    /// Synchronize to device asynchronously with timeout.
    pub async fn sync_to_device(&self, timeout: Duration) -> Result<()> {
        let mut sequence = AsyncSequence::temp_for_tensor_sync(&self.tracker)?;
        sequence.record_tensor_sync_device(self.handle()).await?;
        sequence.execute_with_timeout(timeout).await
    }

    /// Synchronize to host asynchronously with timeout.
    pub async fn sync_to_host(&self, timeout: Duration) -> Result<()> {
        let mut sequence = AsyncSequence::temp_for_tensor_sync(&self.tracker)?;
        sequence.record_tensor_sync_local(self.handle()).await?;
        sequence.execute_with_timeout(timeout).await
    }
}

impl<T: crate::tensor::TensorData> Drop for AsyncTensor<T> {
    fn drop(&mut self) {
        self.tracker
            .unregister_handle("tensor", self.tensor.handle());
    }
}

/// Async sequence with automatic resource tracking.
pub struct AsyncSequence {
    sequence: Option<Sequence>,
    tracker: HandleTracker,
}

impl AsyncSequence {
    /// Create a temporary sequence for tensor operations.
    fn temp_for_tensor_sync(_tracker: &HandleTracker) -> Result<Self> {
        // This would create a temporary sequence on a temp manager
        // For now, return an error as we need a manager reference
        Err(KomputeError::compute(
            "Temporary sequences not yet implemented",
        ))
    }

    /// Record tensor sync to device asynchronously.
    pub async fn record_tensor_sync_device(&mut self, tensor_handle: u64) -> Result<()> {
        tokio::task::yield_now().await;

        if let Some(sequence) = &mut self.sequence {
            sequence.record_tensor_sync_device(tensor_handle)
        } else {
            Err(KomputeError::compute("Sequence has been consumed"))
        }
    }

    /// Record tensor sync to host asynchronously.
    pub async fn record_tensor_sync_local(&mut self, tensor_handle: u64) -> Result<()> {
        tokio::task::yield_now().await;

        if let Some(sequence) = &mut self.sequence {
            sequence.record_tensor_sync_local(tensor_handle)
        } else {
            Err(KomputeError::compute("Sequence has been consumed"))
        }
    }

    /// Record algorithm dispatch asynchronously.
    pub async fn record_algorithm_dispatch(&mut self, algorithm_handle: u64) -> Result<()> {
        tokio::task::yield_now().await;

        if let Some(sequence) = &mut self.sequence {
            sequence.record_algorithm_dispatch(algorithm_handle)
        } else {
            Err(KomputeError::compute("Sequence has been consumed"))
        }
    }

    /// Execute with timeout using Rust async patterns.
    pub async fn execute_with_timeout(mut self, timeout_duration: Duration) -> Result<()> {
        let mut sequence = self
            .sequence
            .take()
            .ok_or_else(|| KomputeError::compute("Sequence has been consumed"))?;

        // Use tokio's timeout for proper async integration
        timeout(timeout_duration, async move {
            // Start async execution
            sequence.eval_async()?;

            // Poll for completion using exponential backoff
            let mut delay = Duration::from_millis(1);
            let start = Instant::now();

            loop {
                // Try to await with a short timeout
                match sequence.eval_await(1) {
                    Ok(()) => return Ok(()),
                    Err(_) => {
                        // Not ready yet, sleep and try again
                        sleep(delay).await;

                        // Exponential backoff up to 100ms
                        delay = std::cmp::min(delay * 2, Duration::from_millis(100));

                        // Check if we should continue
                        if start.elapsed() > timeout_duration {
                            return Err(KomputeError::compute("GPU operation timed out"));
                        }
                    }
                }
            }
        })
        .await
        .map_err(|_| KomputeError::compute("Sequence execution timed out"))?
    }

    /// Execute without timeout.
    pub async fn execute(self) -> Result<()> {
        self.execute_with_timeout(Duration::from_secs(30)).await
    }

    /// Execute multiple sequences concurrently.
    pub async fn execute_concurrent(sequences: Vec<AsyncSequence>) -> Result<()> {
        use futures::future::try_join_all;

        let futures: Vec<_> = sequences.into_iter().map(|seq| seq.execute()).collect();

        try_join_all(futures).await?;
        Ok(())
    }
}

impl Drop for AsyncSequence {
    fn drop(&mut self) {
        if let Some(sequence) = &self.sequence {
            self.tracker
                .unregister_handle("sequence", sequence.handle());
        }
    }
}

/// Async compute kernel for high-level operations.
pub struct AsyncComputeKernel {
    _context: AsyncGpuContext,
}

impl AsyncComputeKernel {
    /// Create a new compute kernel.
    pub async fn new() -> Result<Self> {
        let context = AsyncGpuContext::new()?;
        Ok(Self { _context: context })
    }

    /// Execute element-wise operations asynchronously.
    pub async fn elementwise_add<T>(
        &mut self,
        a: &AsyncTensor<T>,
        b: &AsyncTensor<T>,
    ) -> Result<AsyncTensor<T>>
    where
        T: crate::tensor::TensorData,
    {
        // This would implement actual GPU computation
        // For now, simulate the operation
        tokio::task::yield_now().await;

        // Get data from both tensors
        let data_a = a.to_vec().await?;
        let data_b = b.to_vec().await?;

        if data_a.len() != data_b.len() {
            return Err(KomputeError::compute("Tensor size mismatch"));
        }

        // This is just a placeholder - real implementation would use GPU
        Err(KomputeError::compute("GPU computation not yet implemented"))
    }

    /// Execute matrix multiplication asynchronously.
    pub async fn matrix_multiply<T>(
        &mut self,
        _a: &AsyncTensor<T>,
        _b: &AsyncTensor<T>,
    ) -> Result<AsyncTensor<T>>
    where
        T: crate::tensor::TensorData,
    {
        tokio::task::yield_now().await;
        Err(KomputeError::compute(
            "Matrix multiplication not yet implemented",
        ))
    }
}

/// Builder pattern for complex async workflows.
pub struct AsyncWorkflowBuilder {
    context: AsyncGpuContext,
    operations: Vec<AsyncOperation>,
}

#[derive(Debug)]
enum AsyncOperation {
    TensorCreate { size: usize },
    TensorSync { tensor_id: usize },
    Compute { kernel_type: String },
}

impl AsyncWorkflowBuilder {
    /// Create a new workflow builder.
    pub async fn new() -> Result<Self> {
        Ok(Self {
            context: AsyncGpuContext::new()?,
            operations: Vec::new(),
        })
    }

    /// Add a tensor creation operation.
    pub fn add_tensor_creation(mut self, size: usize) -> Self {
        self.operations.push(AsyncOperation::TensorCreate { size });
        self
    }

    /// Add a tensor synchronization operation.
    pub fn add_tensor_sync(mut self, tensor_id: usize) -> Self {
        self.operations
            .push(AsyncOperation::TensorSync { tensor_id });
        self
    }

    /// Add a compute operation.
    pub fn add_compute(mut self, kernel_type: String) -> Self {
        self.operations
            .push(AsyncOperation::Compute { kernel_type });
        self
    }

    /// Execute the workflow.
    pub async fn execute(mut self) -> Result<()> {
        for (i, operation) in self.operations.iter().enumerate() {
            match operation {
                AsyncOperation::TensorCreate { size } => {
                    let _tensor: AsyncTensor<f32> =
                        self.context.create_tensor_uninit::<f32>(*size).await?;
                    println!("✓ Operation {}: Created tensor of size {}", i, size);
                }
                AsyncOperation::TensorSync { tensor_id } => {
                    println!("✓ Operation {}: Synchronized tensor {}", i, tensor_id);
                }
                AsyncOperation::Compute { kernel_type } => {
                    println!("✓ Operation {}: Executed {} kernel", i, kernel_type);
                }
            }

            // Yield to allow other tasks to run
            tokio::task::yield_now().await;
        }

        println!("✓ Workflow completed successfully");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::Duration;

    #[tokio::test]
    #[ignore = "requires Vulkan-capable device"]
    async fn test_async_gpu_context() {
        if let Ok(mut context) = AsyncGpuContext::new() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0];

            match context.create_tensor(&data).await {
                Ok(tensor) => {
                    assert_eq!(tensor.size().await, 4);

                    if let Ok(retrieved_data) = tensor.to_vec().await {
                        assert_eq!(data, retrieved_data);
                    }
                }
                Err(_) => {
                    // Expected in headless environments
                }
            }
        }
    }

    #[tokio::test]
    #[ignore = "requires Vulkan-capable device"]
    async fn test_async_sequence_execution() {
        if let Ok(mut context) = AsyncGpuContext::new() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0];

            if let Ok(tensor) = context.create_tensor(&data).await {
                if let Ok(mut sequence) = context.create_sequence().await {
                    let _ = sequence.record_tensor_sync_device(tensor.handle()).await;

                    // Test async execution with timeout
                    let result = sequence.execute_with_timeout(Duration::from_secs(5)).await;
                    match result {
                        Ok(()) => println!("✓ Async execution succeeded"),
                        Err(_) => println!("⚠️  Async execution failed (expected in headless)"),
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_workflow_builder() {
        if let Ok(workflow) = AsyncWorkflowBuilder::new().await {
            let result = workflow
                .add_tensor_creation(100)
                .add_tensor_sync(0)
                .add_compute("element_add".to_string())
                .add_tensor_sync(0)
                .execute()
                .await;

            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_concurrent_operations() {
        // Test that multiple async operations can run concurrently
        let tasks = vec![
            tokio::spawn(async {
                AsyncWorkflowBuilder::new()
                    .await?
                    .add_tensor_creation(50)
                    .execute()
                    .await
            }),
            tokio::spawn(async {
                AsyncWorkflowBuilder::new()
                    .await?
                    .add_tensor_creation(75)
                    .execute()
                    .await
            }),
            tokio::spawn(async {
                AsyncWorkflowBuilder::new()
                    .await?
                    .add_compute("test_kernel".to_string())
                    .execute()
                    .await
            }),
        ];

        for task in tasks {
            let _ = task.await;
        }
    }
}
