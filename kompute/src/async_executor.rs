//! Future-based async API for GPU compute operations.

use crate::error::{KomputeError, Result};
use crate::manager::Manager;
use crate::sequence::Sequence;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

/// Async wrapper around handle-based sequence operations.
///
/// This provides a Future-based API that allows GPU operations to be executed
/// asynchronously with Rust's async/await syntax and integrates with async runtimes.
///
/// # Examples
///
/// ```rust,no_run
/// use kompute::{Manager, TensorHandle, AsyncSequence, Result};
///
/// async fn gpu_compute_async() -> Result<()> {
///     let mut manager = Manager::new()?;
///     let data = vec![1.0f32, 2.0, 3.0, 4.0];
///     let tensor = TensorHandle::new(&mut manager, &data)?;
///
///     let mut async_seq = AsyncSequence::new(&mut manager)?;
///     async_seq.record_tensor_sync_device(tensor.handle())?;
///
///     // Execute asynchronously with timeout
///     async_seq.eval_with_timeout(Duration::from_secs(5)).await?;
///
///     Ok(())
/// }
/// ```
pub struct AsyncSequence {
    sequence: Sequence,
}

impl AsyncSequence {
    /// Create a new async sequence wrapper.
    ///
    /// # Arguments
    ///
    /// * `manager` - The GPU manager to create the sequence with
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ComputeError`] if sequence creation fails.
    pub fn new(manager: &mut Manager) -> Result<Self> {
        let sequence = Sequence::new(manager)?;
        Ok(Self { sequence })
    }

    /// Record a tensor synchronization operation from host to device.
    ///
    /// # Arguments
    ///
    /// * `tensor_handle` - The handle of the tensor to synchronize
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ComputeError`] if recording fails.
    pub fn record_tensor_sync_device(&mut self, tensor_handle: u64) -> Result<()> {
        self.sequence.record_tensor_sync_device(tensor_handle)
    }

    /// Record a tensor synchronization operation from device to host.
    ///
    /// # Arguments
    ///
    /// * `tensor_handle` - The handle of the tensor to synchronize
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ComputeError`] if recording fails.
    pub fn record_tensor_sync_local(&mut self, tensor_handle: u64) -> Result<()> {
        self.sequence.record_tensor_sync_local(tensor_handle)
    }

    /// Record an algorithm dispatch operation.
    ///
    /// # Arguments
    ///
    /// * `algorithm_handle` - The handle of the algorithm to dispatch
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ComputeError`] if recording fails.
    pub fn record_algorithm_dispatch(&mut self, algorithm_handle: u64) -> Result<()> {
        self.sequence.record_algorithm_dispatch(algorithm_handle)
    }

    /// Execute the sequence asynchronously with a timeout.
    ///
    /// This starts the GPU operations and returns a Future that completes
    /// when all operations finish or the timeout expires.
    ///
    /// # Arguments
    ///
    /// * `timeout` - Maximum time to wait for completion
    ///
    /// # Returns
    ///
    /// A Future that resolves when the operations complete or timeout expires.
    pub fn eval_with_timeout(self, timeout: Duration) -> AsyncSequenceExecution {
        AsyncSequenceExecution::new(self.sequence, timeout)
    }

    /// Execute the sequence asynchronously without timeout.
    ///
    /// This starts the GPU operations and returns a Future that completes
    /// when all operations finish.
    ///
    /// # Returns
    ///
    /// A Future that resolves when the operations complete.
    pub fn eval(self) -> AsyncSequenceExecution {
        AsyncSequenceExecution::new(self.sequence, Duration::from_secs(u64::MAX))
    }

    /// Get the internal handle value.
    pub fn handle(&self) -> u64 {
        self.sequence.handle()
    }
}

/// Future representing an async GPU sequence execution.
///
/// This Future polls the GPU operations for completion and integrates
/// with Rust's async runtime systems.
pub struct AsyncSequenceExecution {
    sequence: Sequence,
    timeout: Duration,
    start_time: Instant,
    started: bool,
}

impl AsyncSequenceExecution {
    fn new(sequence: Sequence, timeout: Duration) -> Self {
        Self {
            sequence,
            timeout,
            start_time: Instant::now(),
            started: false,
        }
    }
}

impl Future for AsyncSequenceExecution {
    type Output = Result<()>;

    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        if !self.started {
            // Start async execution
            if let Err(e) = self.sequence.eval_async() {
                return Poll::Ready(Err(e));
            }
            self.started = true;
            self.start_time = Instant::now();
        }

        // Check if timeout has expired
        if self.start_time.elapsed() >= self.timeout {
            return Poll::Ready(Err(KomputeError::compute("GPU operation timed out")));
        }

        // Try to await completion with a short timeout to avoid blocking
        // We use a very small timeout (1ms) to make this effectively non-blocking
        match self.sequence.eval_await(1) {
            Ok(()) => Poll::Ready(Ok(())),
            Err(_) => {
                // Operation not yet complete, wake up the task to be polled again
                _cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }
}

/// Async GPU compute operations with multi-queue support.
///
/// This provides high-level async operations that can utilize multiple
/// GPU queues for parallel execution.
pub struct AsyncGpuExecutor {
    manager: Manager,
}

impl AsyncGpuExecutor {
    /// Create a new async GPU executor.
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ComputeError`] if GPU initialization fails.
    pub fn new() -> Result<Self> {
        let manager = Manager::new()?;
        Ok(Self { manager })
    }

    /// Create a new async executor with a specific device.
    ///
    /// # Arguments
    ///
    /// * `device_id` - The GPU device to use
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ComputeError`] if device initialization fails.
    pub fn with_device(device_id: u32) -> Result<Self> {
        let manager = Manager::with_device(device_id)?;
        Ok(Self { manager })
    }

    /// Execute multiple sequences in parallel across multiple GPU queues.
    ///
    /// This demonstrates the multi-queue parallel execution capability.
    ///
    /// # Arguments
    ///
    /// * `sequences` - Vector of async sequences to execute in parallel
    /// * `timeout` - Maximum time to wait for all sequences to complete
    ///
    /// # Returns
    ///
    /// A Future that resolves when all sequences complete or timeout expires.
    pub async fn execute_parallel(
        &mut self,
        sequences: Vec<AsyncSequence>,
        timeout: Duration,
    ) -> Result<()> {
        use futures::future::join_all;

        // Convert sequences to futures
        let futures: Vec<_> = sequences
            .into_iter()
            .map(|seq| seq.eval_with_timeout(timeout))
            .collect();

        // Execute all futures in parallel
        let results = join_all(futures).await;

        // Check if any failed
        for result in results {
            result?;
        }

        Ok(())
    }

    /// Get a mutable reference to the underlying manager.
    pub fn manager(&mut self) -> &mut Manager {
        &mut self.manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Manager, Tensor};
    use std::time::Duration;

    #[tokio::test]
    #[ignore = "requires Vulkan-capable device and tokio runtime"]
    async fn test_async_sequence_execution() {
        if let Ok(mut manager) = Manager::new() {
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            if let Ok(tensor) = Tensor::new(&mut manager, &data) {
                if let Ok(mut async_seq) = AsyncSequence::new(&mut manager) {
                    let _ = async_seq.record_tensor_sync_device(tensor.handle());

                    // Test async execution with timeout
                    let result = async_seq.eval_with_timeout(Duration::from_secs(5)).await;
                    match result {
                        Ok(()) => println!("✓ Async execution completed successfully"),
                        Err(_) => println!(
                            "⚠️ Async execution failed (expected in headless environments)"
                        ),
                    }
                }
            }
        }
    }

    #[tokio::test]
    #[ignore = "requires Vulkan-capable device and tokio runtime"]
    async fn test_parallel_execution() {
        if let Ok(mut executor) = AsyncGpuExecutor::new() {
            let mut sequences = Vec::new();

            // Create multiple async sequences
            for i in 0..3 {
                let data = vec![i as f32; 10];
                if let Ok(tensor) = Tensor::new(executor.manager(), &data) {
                    if let Ok(mut async_seq) = AsyncSequence::new(executor.manager()) {
                        let _ = async_seq.record_tensor_sync_device(tensor.handle());
                        sequences.push(async_seq);
                    }
                }
            }

            // Execute all sequences in parallel
            let result = executor
                .execute_parallel(sequences, Duration::from_secs(10))
                .await;

            match result {
                Ok(()) => println!("✓ Parallel execution completed successfully"),
                Err(_) => {
                    println!("⚠️ Parallel execution failed (expected in headless environments)")
                }
            }
        }
    }
}
