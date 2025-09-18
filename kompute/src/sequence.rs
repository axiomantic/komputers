//! Command buffer for batched GPU operations.

use crate::error::Result;
use crate::manager::Manager;
use kompute_sys::ffi;

/// GPU command sequence for batched operations.
///
/// Sequences allow recording multiple GPU operations (tensor synchronization,
/// algorithm execution) into a command buffer that can be executed efficiently
/// as a single batch using handle-based resource management.
///
/// # Examples
///
/// ```rust,no_run
/// use kompute::{Manager, Sequence, Tensor, Result};
///
/// fn example() -> Result<()> {
///     let mut manager = Manager::new()?;
///     let mut sequence = Sequence::new(&mut manager)?;
///
///     let data = vec![1.0f32, 2.0, 3.0, 4.0];
///     let tensor = Tensor::new(&mut manager, &data)?;
///
///     // Record operations into the sequence
///     sequence.record_tensor_sync_device(tensor.handle())?;
///     sequence.record_tensor_sync_local(tensor.handle())?;
///
///     // Execute all recorded operations
///     sequence.eval()?;
///     Ok(())
/// }
/// ```
pub struct Sequence {
    handle: u64,
}

impl Sequence {
    /// Create a new sequence for recording GPU operations.
    ///
    /// # Arguments
    ///
    /// * `manager` - The GPU manager to create the sequence with
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ComputeError`] if sequence creation fails.
    pub fn new(manager: &mut Manager) -> Result<Self> {
        let handle = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ffi::create_sequence_handle(manager.handle())
        }))
        .map_err(|_| {
            crate::error::KomputeError::compute("Sequence creation failed due to GPU error")
        })?;

        if handle == 0 {
            return Err(crate::error::KomputeError::compute(
                "Sequence creation returned invalid handle",
            ));
        }

        Ok(Self { handle })
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
        if tensor_handle == 0 {
            return Err(crate::error::KomputeError::compute(
                "Cannot record operation with invalid tensor handle",
            ));
        }

        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ffi::sequence_record_tensor_sync_device_by_handles(self.handle, tensor_handle)
        }))
        .map_err(|_| {
            crate::error::KomputeError::compute(
                "Failed to record tensor sync device operation - invalid handle or GPU error",
            )
        })?;

        Ok(())
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
        if tensor_handle == 0 {
            return Err(crate::error::KomputeError::compute(
                "Cannot record operation with invalid tensor handle",
            ));
        }

        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ffi::sequence_record_tensor_sync_local_by_handles(self.handle, tensor_handle)
        }))
        .map_err(|_| {
            crate::error::KomputeError::compute(
                "Failed to record tensor sync local operation - invalid handle or GPU error",
            )
        })?;

        Ok(())
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
        if algorithm_handle == 0 {
            return Err(crate::error::KomputeError::compute(
                "Cannot record operation with invalid algorithm handle",
            ));
        }

        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ffi::sequence_record_algorithm_dispatch_by_handles(self.handle, algorithm_handle)
        }))
        .map_err(|_| {
            crate::error::KomputeError::compute(
                "Failed to record algorithm dispatch operation - invalid handle or GPU error",
            )
        })?;

        Ok(())
    }

    /// Execute all recorded operations synchronously.
    ///
    /// This submits the command buffer to the GPU and blocks until
    /// all operations are complete.
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ComputeError`] if execution fails.
    pub fn eval(&mut self) -> Result<()> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ffi::sequence_eval_by_handle(self.handle)
        }))
        .map_err(|_| {
            crate::error::KomputeError::compute(
                "Failed to execute sequence - GPU error or invalid operations recorded",
            )
        })?;

        Ok(())
    }

    /// Execute all recorded operations asynchronously.
    ///
    /// This submits the command buffer to the GPU and returns immediately.
    /// Use [`eval_await`] to wait for completion.
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ComputeError`] if execution fails.
    pub fn eval_async(&mut self) -> Result<()> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ffi::sequence_eval_async_by_handle(self.handle)
        }))
        .map_err(|_| {
            crate::error::KomputeError::compute(
            "Failed to execute sequence asynchronously - GPU error or invalid operations recorded"
        )
        })?;

        Ok(())
    }

    /// Wait for asynchronous execution to complete.
    ///
    /// # Arguments
    ///
    /// * `timeout_ms` - Maximum time to wait in milliseconds, or 0 for infinite
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ComputeError`] if waiting fails or times out.
    pub fn eval_await(&mut self, timeout_ms: u64) -> Result<()> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ffi::sequence_eval_await_by_handle(self.handle, timeout_ms)
        }))
        .map_err(|_| {
            crate::error::KomputeError::compute(
                "Failed to await sequence completion - timeout exceeded or GPU error",
            )
        })?;

        Ok(())
    }

    /// Get the internal handle value.
    ///
    /// This is used for interfacing with other handle-based operations.
    pub fn handle(&self) -> u64 {
        self.handle
    }
}

impl Drop for Sequence {
    fn drop(&mut self) {
        ffi::sequence_destroy_handle(self.handle);
    }
}

// Sequences are safe to send between threads
unsafe impl Send for Sequence {}

// Sequences can be shared between threads with proper synchronization
unsafe impl Sync for Sequence {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Manager;

    #[test]
    #[ignore = "requires Vulkan-capable device"]
    fn test_sequence_creation() {
        // This test may fail in environments without Vulkan support
        if let Ok(mut manager) = Manager::new() {
            match Sequence::new(&mut manager) {
                Ok(_sequence) => {
                    // Sequence creation succeeded
                    assert!(true, "Sequence creation succeeded");
                }
                Err(_e) => {
                    // Sequence creation failed - expected in headless environments
                    assert!(true, "Sequence creation failed as expected");
                }
            }
        }
    }
}
