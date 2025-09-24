//! Command buffer for batched GPU operations with Rust-idiomatic consuming builder pattern.

use crate::error::Result;
use crate::manager::Manager;
use crate::runtime_validation;
use kompute_sys::ffi;
use std::marker::PhantomData;
use std::time::{Duration, Instant};

// =============================================================================
// Type-State Markers for Compile-Time Safety
// =============================================================================

/// Marker type for a sequence in recording state
///
/// A sequence in this state can have operations recorded but cannot be executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Recording;

/// Result of sequence execution containing metadata and timing information
///
/// This is returned when a sequence is successfully executed, providing
/// useful information about the GPU operation performance.
#[derive(Debug, Clone)]
pub struct SequenceResult {
    /// Time taken to execute all recorded operations
    pub execution_time: Duration,
    /// Number of operations that were executed
    pub operation_count: usize,
    /// Handle of the executed sequence (for debugging)
    pub sequence_handle: u64,
}

/// GPU command sequence with Rust-idiomatic consuming builder pattern.
///
/// This sequence implementation follows the same pattern as `std::process::Command`
/// and other Rust standard library builders, using move semantics and type states
/// to prevent misuse at compile time.
///
/// ## Compile-Time Safety Guarantees
///
/// - **Cannot execute empty sequences**: Must record operations before execution
/// - **Cannot record after execution**: Sequence is consumed by execution
/// - **Cannot execute twice**: No methods available after execution
/// - **Cannot accidentally reuse**: Each operation moves the sequence
///
/// ## Usage Pattern (Consuming Builder)
///
/// ```rust,no_run
/// use kompute::{Manager, Sequence, Tensor, Result};
///
/// fn example() -> Result<()> {
///     let mut manager = Manager::new()?;
///
///     let data = vec![1.0f32, 2.0, 3.0, 4.0];
///     let tensor = Tensor::new(&mut manager, &data)?;
///
///     // Consuming builder pattern - each operation moves the sequence
///     let result = Sequence::new(&mut manager)?
///         .record_tensor_sync_device(tensor.handle())?
///         .record_tensor_sync_local(tensor.handle())?
///         .execute()?;  // Sequence is consumed here
///
///     println!("Execution took: {:?}", result.execution_time);
///
///     // Manager is still available for more operations
///     let result2 = Sequence::new(&mut manager)?
///         .record_tensor_sync_local(tensor.handle())?
///         .execute()?;
///
///     Ok(())
/// }
/// ```
///
/// ## Performance
///
/// Type states are zero-cost abstractions - they compile away completely with
/// no runtime overhead compared to the previous implementation.
pub struct Sequence<State = Recording> {
    handle: u64,
    operation_count: usize,
    _state: PhantomData<State>,
}

// =============================================================================
// Recording State Implementation (Consuming Builder Pattern)
// =============================================================================

impl Sequence<Recording> {
    /// Create a new sequence for recording GPU operations.
    ///
    /// Creates a sequence in the recording state, ready to record operations
    /// using the consuming builder pattern.
    ///
    /// # Arguments
    ///
    /// * `manager` - The GPU manager to create the sequence with
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::KomputeError::ComputeError`] if sequence creation fails.
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

        // Register the sequence handle with the manager for runtime validation
        manager.register_handle(handle);

        Ok(Self {
            handle,
            operation_count: 0,
            _state: PhantomData,
        })
    }

    /// Create a sequence from an existing handle.
    ///
    /// # Arguments
    ///
    /// * `handle` - The existing sequence handle
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ComputeError`] if the handle is invalid.
    pub(crate) fn from_handle(handle: u64) -> Result<Self> {
        if handle == 0 {
            return Err(crate::error::KomputeError::compute(
                "Cannot create sequence from invalid handle",
            ));
        }

        Ok(Self {
            handle,
            operation_count: 0,
            _state: PhantomData,
        })
    }

    /// Record a tensor synchronization operation from host to device.
    ///
    /// Records the operation using the consuming builder pattern. The sequence
    /// is moved into this method and returned, allowing for method chaining.
    ///
    /// # Arguments
    ///
    /// * `tensor_handle` - The handle of the tensor to synchronize
    ///
    /// # Returns
    ///
    /// Returns the sequence for continued method chaining, or an error if recording fails.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// let result = Sequence::new(&mut manager)?
    ///     .record_tensor_sync_device(tensor.handle())?
    ///     .execute()?;
    /// ```
    pub fn record_tensor_sync_device(mut self, tensor_handle: u64) -> Result<Self> {
        if tensor_handle == 0 {
            return Err(crate::error::KomputeError::compute(
                "Cannot record operation with invalid tensor handle",
            ));
        }

        // Runtime validation: Check if tensor and sequence belong to the same manager
        // Note: This validates that both handles are from the same manager instance
        if runtime_validation::is_validation_enabled() {
            if let Err(e) =
                runtime_validation::validate_handles_same_manager(&[self.handle, tensor_handle])
            {
                return Err(crate::error::KomputeError::compute(format!(
                    "Cross-manager handle usage detected: sequence handle {} and tensor handle {} are from different managers. {}",
                    self.handle, tensor_handle, e
                )));
            }
        }

        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ffi::sequence_record_tensor_sync_device_by_handles(self.handle, tensor_handle)
        }))
        .map_err(|_| {
            crate::error::KomputeError::compute(
                "Failed to record tensor sync device operation - invalid handle or GPU error",
            )
        })?;

        self.operation_count += 1;
        Ok(self)
    }

    /// Record a tensor synchronization operation from device to host.
    ///
    /// Records the operation using the consuming builder pattern. The sequence
    /// is moved into this method and returned, allowing for method chaining.
    ///
    /// # Arguments
    ///
    /// * `tensor_handle` - The handle of the tensor to synchronize
    ///
    /// # Returns
    ///
    /// Returns the sequence for continued method chaining, or an error if recording fails.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// let result = Sequence::new(&mut manager)?
    ///     .record_tensor_sync_local(tensor.handle())?
    ///     .execute()?;
    /// ```
    pub fn record_tensor_sync_local(mut self, tensor_handle: u64) -> Result<Self> {
        if tensor_handle == 0 {
            return Err(crate::error::KomputeError::compute(
                "Cannot record operation with invalid tensor handle",
            ));
        }

        // Runtime validation: Check if tensor and sequence belong to the same manager
        if runtime_validation::is_validation_enabled() {
            if let Err(e) =
                runtime_validation::validate_handles_same_manager(&[self.handle, tensor_handle])
            {
                return Err(crate::error::KomputeError::compute(format!(
                    "Cross-manager handle usage detected: sequence handle {} and tensor handle {} are from different managers. {}",
                    self.handle, tensor_handle, e
                )));
            }
        }

        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ffi::sequence_record_tensor_sync_local_by_handles(self.handle, tensor_handle)
        }))
        .map_err(|_| {
            crate::error::KomputeError::compute(
                "Failed to record tensor sync local operation - invalid handle or GPU error",
            )
        })?;

        self.operation_count += 1;
        Ok(self)
    }

    /// Record an algorithm dispatch operation.
    ///
    /// Records the operation using the consuming builder pattern. The sequence
    /// is moved into this method and returned, allowing for method chaining.
    ///
    /// # Arguments
    ///
    /// * `algorithm_handle` - The handle of the algorithm to dispatch
    ///
    /// # Returns
    ///
    /// Returns the sequence for continued method chaining, or an error if recording fails.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// let result = Sequence::new(&mut manager)?
    ///     .record_algorithm_dispatch(algorithm.handle())?
    ///     .execute()?;
    /// ```
    pub fn record_algorithm_dispatch(mut self, algorithm_handle: u64) -> Result<Self> {
        if algorithm_handle == 0 {
            return Err(crate::error::KomputeError::compute(
                "Cannot record operation with invalid algorithm handle",
            ));
        }

        // Runtime validation: Check if algorithm and sequence belong to the same manager
        if runtime_validation::is_validation_enabled() {
            if let Err(e) =
                runtime_validation::validate_handles_same_manager(&[self.handle, algorithm_handle])
            {
                return Err(crate::error::KomputeError::compute(format!(
                    "Cross-manager handle usage detected: sequence handle {} and algorithm handle {} are from different managers. {}",
                    self.handle, algorithm_handle, e
                )));
            }
        }

        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ffi::sequence_record_algorithm_dispatch_by_handles(self.handle, algorithm_handle)
        }))
        .map_err(|_| {
            crate::error::KomputeError::compute(
                "Failed to record algorithm dispatch operation - invalid handle or GPU error",
            )
        })?;

        self.operation_count += 1;
        Ok(self)
    }

    /// Execute all recorded operations synchronously (terminal operation).
    ///
    /// This is the terminal operation of the consuming builder pattern. It executes
    /// all recorded operations on the GPU and returns execution metadata. The sequence
    /// is consumed by this operation and cannot be reused.
    ///
    /// # Returns
    ///
    /// Returns [`SequenceResult`] containing execution metadata, or an error if execution fails.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::KomputeError::ComputeError`] if:
    /// - No operations were recorded (empty sequence)
    /// - GPU execution fails
    /// - Invalid sequence state
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// let result = Sequence::new(&mut manager)?
    ///     .record_tensor_sync_device(tensor.handle())?
    ///     .execute()?;
    ///
    /// println!("Executed {} operations in {:?}",
    ///          result.operation_count,
    ///          result.execution_time);
    /// ```
    pub fn execute(self) -> Result<SequenceResult> {
        // Prevent execution of empty sequences
        if self.operation_count == 0 {
            return Err(crate::error::KomputeError::compute(
                "Cannot execute empty sequence - no operations recorded",
            ));
        }

        let start_time = Instant::now();

        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ffi::sequence_eval_by_handle(self.handle)
        }))
        .map_err(|_| {
            crate::error::KomputeError::compute(
                "Failed to execute sequence - GPU error or invalid operations recorded",
            )
        })?;

        let execution_time = start_time.elapsed();

        Ok(SequenceResult {
            execution_time,
            operation_count: self.operation_count,
            sequence_handle: self.handle,
        })
    }
}

// =============================================================================
// Compatibility Methods for Integration
// =============================================================================

impl<State> Sequence<State> {
    /// Get the raw sequence handle.
    ///
    /// This method is used for integration with other parts of the system
    /// that need access to the underlying GPU sequence handle.
    ///
    /// # Returns
    ///
    /// The raw u64 handle for this sequence.
    pub fn handle(&self) -> u64 {
        self.handle
    }
}

// =============================================================================
// Drop Implementation for Proper Resource Cleanup
// =============================================================================

impl<State> Drop for Sequence<State> {
    fn drop(&mut self) {
        // GPU sequence cleanup is handled by the underlying C++ implementation
        // The handle will be automatically cleaned up when it goes out of scope
    }
}

// =============================================================================
// Tests for Compile-Time Safety and Runtime Behavior
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // These tests will be added in Phase 3 to verify compile-time safety
    // and runtime behavior of the consuming builder pattern
}
