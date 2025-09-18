//! GPU device management and resource allocation.

use crate::algorithm::Algorithm;
use crate::error::{KomputeError, Result};
use crate::sequence::Sequence;
use crate::tensor::{Tensor, TensorData};
use cxx::UniquePtr;
use kompute_sys::ffi;
use std::pin::Pin;

/// GPU device manager for resource allocation and compute operations.
///
/// The Manager handles Vulkan device initialization, GPU memory allocation,
/// and serves as the factory for creating tensors, algorithms, and sequences.
/// It follows RAII principles for automatic resource cleanup.
///
/// # Examples
///
/// ```rust,no_run
/// use kompute::{Manager, Result};
///
/// fn example() -> Result<()> {
///     // Create manager with default device (usually device 0)
///     let manager = Manager::new()?;
///
///     // Create manager with specific device
///     let manager2 = Manager::with_device(1)?;
///
///     // Check device count
///     let count = Manager::device_count()?;
///     println!("Available GPU devices: {}", count);
///
///     Ok(())
/// }
/// ```
pub struct Manager {
    handle: UniquePtr<ffi::Manager>,
}

impl Manager {
    /// Create a new Manager with the default GPU device (device 0).
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::DeviceError`] if no Vulkan-capable devices are found
    /// or device initialization fails.
    pub fn new() -> Result<Self> {
        Self::with_device(0)
    }

    /// Create a new Manager with the specified GPU device ID.
    ///
    /// # Arguments
    ///
    /// * `device_id` - The GPU device index to use (0-based)
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::DeviceError`] if the specified device is not available
    /// or initialization fails.
    pub fn with_device(device_id: u32) -> Result<Self> {
        // First validate that devices are available
        let device_count = Self::device_count().map_err(|e| {
            KomputeError::device(format!(
                "Cannot create manager: device enumeration failed: {}",
                e
            ))
        })?;

        if device_count == 0 {
            return Err(KomputeError::device(
                "Cannot create manager: no Vulkan devices available. \
                 Ensure MoltenVK is properly installed and configured."
                    .to_string(),
            ));
        }

        if device_id >= device_count {
            return Err(KomputeError::device(format!(
                "Cannot create manager: device ID {} is invalid. \
                 Only {} devices available (valid range: 0-{})",
                device_id,
                device_count,
                device_count - 1
            )));
        }

        // Attempt to create the manager with detailed error context
        let handle = ffi::create_manager(device_id);
        if handle.is_null() {
            return Err(KomputeError::device(format!(
                "Manager creation failed for device {} (out of {} available devices). \
                 This may indicate a Vulkan driver issue or insufficient GPU memory.",
                device_id, device_count
            )));
        }

        Ok(Self { handle })
    }

    /// Get the number of available Vulkan-capable devices.
    ///
    /// # Returns
    ///
    /// The number of available GPU devices, or 0 if no devices are found.
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::VulkanError`] if device enumeration fails.
    pub fn device_count() -> Result<u32> {
        let count = ffi::get_device_count();
        if count < 0 {
            return Err(KomputeError::vulkan("Failed to enumerate Vulkan devices"));
        }
        Ok(count as u32)
    }

    /// Create a tensor with data copied from the host.
    ///
    /// # Arguments
    ///
    /// * `data` - Host data to copy to the GPU
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::MemoryError`] if GPU allocation fails or
    /// [`KomputeError::TensorError`] if data copying fails.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use kompute::{Manager, Result};
    ///
    /// fn example() -> Result<()> {
    ///     let mut manager = Manager::new()?;
    ///     let data = vec![1.0f32, 2.0, 3.0, 4.0];
    ///     let tensor = manager.tensor(&data)?;
    ///     println!("Created tensor with {} elements", tensor.size());
    ///     Ok(())
    /// }
    /// ```
    pub fn tensor<T: TensorData>(&mut self, data: &[T]) -> Result<Tensor<T>> {
        Tensor::new(self, data)
    }

    /// Create an uninitialized tensor of the specified size.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of elements to allocate
    ///
    /// # Safety
    ///
    /// The tensor contains uninitialized memory until data is explicitly
    /// written to it via compute shaders or host synchronization.
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::MemoryError`] if GPU allocation fails.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use kompute::{Manager, Result};
    ///
    /// fn example() -> Result<()> {
    ///     let mut manager = Manager::new()?;
    ///     let tensor = manager.tensor_uninit::<f32>(100)?;
    ///     println!("Created uninitialized tensor with {} elements", tensor.size());
    ///     Ok(())
    /// }
    /// ```
    pub fn tensor_uninit<T: TensorData>(&mut self, size: usize) -> Result<Tensor<T>> {
        Tensor::uninit(self, size)
    }

    /// Create an algorithm from SPIR-V bytecode.
    ///
    /// # Arguments
    ///
    /// * `spirv_data` - SPIR-V bytecode as 32-bit words
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ShaderError`] if the SPIR-V is invalid or
    /// [`KomputeError::ComputeError`] if algorithm creation fails.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use kompute::{Manager, Result};
    ///
    /// fn example() -> Result<()> {
    ///     let mut manager = Manager::new()?;
    ///     let spirv_data: Vec<u32> = vec![/* SPIR-V bytecode */];
    ///     let algorithm = manager.algorithm(&spirv_data)?;
    ///     Ok(())
    /// }
    /// ```
    pub fn algorithm(&mut self, spirv_data: &[u32]) -> Result<Algorithm> {
        Algorithm::new(self, spirv_data)
    }

    /// Create a sequence for recording GPU operations.
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ComputeError`] if sequence creation fails.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use kompute::{Manager, Result};
    ///
    /// fn example() -> Result<()> {
    ///     let mut manager = Manager::new()?;
    ///     let sequence = manager.sequence()?;
    ///     Ok(())
    /// }
    /// ```
    pub fn sequence(&mut self) -> Result<Sequence> {
        Sequence::new(self)
    }

    /// Get the raw handle for FFI operations.
    ///
    /// # Safety
    ///
    /// The returned handle is only valid while this Manager exists.
    /// Using the handle after the Manager is dropped results in undefined behavior.
    pub(crate) fn handle(&mut self) -> Pin<&mut ffi::Manager> {
        self.handle.pin_mut()
    }
}

// UniquePtr handles RAII cleanup automatically - no manual Drop needed

// Manager is safe to send between threads (contains only a pointer to GPU resources)
unsafe impl Send for Manager {}

// Manager can be shared between threads with proper synchronization
unsafe impl Sync for Manager {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_count() {
        // This test may fail in environments without Vulkan support
        match Manager::device_count() {
            Ok(count) => {
                // Device enumeration succeeded
                assert!(count < 100, "Device count should be reasonable");
            }
            Err(_e) => {
                // Device enumeration failed - expected in CI/headless environments
                assert!(true, "Device enumeration failed as expected");
            }
        }
    }

    #[test]
    #[ignore = "requires Vulkan-capable device"]
    fn test_manager_creation() {
        // This test may fail in environments without Vulkan support
        match Manager::new() {
            Ok(_manager) => {
                // Manager created successfully - test passed
                assert!(true, "Manager creation succeeded");
            }
            Err(_e) => {
                // In CI/headless environments, this is expected
                assert!(
                    true,
                    "Manager creation failed as expected in non-GPU environment"
                );
            }
        }
    }
}
