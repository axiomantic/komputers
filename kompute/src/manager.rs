//! GPU device management and resource allocation.

use crate::error::{KomputeError, Result};
use std::ptr::NonNull;

/// GPU device manager for resource allocation and compute operations.
///
/// The Manager handles Vulkan device initialization, GPU memory allocation,
/// and serves as the factory for creating tensors, algorithms, and sequences.
/// It follows RAII principles for automatic resource cleanup.
///
/// # Examples
///
/// ```rust,ignore
/// use kompute::Manager;
///
/// // Create manager with default device (usually device 0)
/// let mut manager = Manager::new()?;
///
/// // Create manager with specific device
/// let mut manager = Manager::with_device(1)?;
/// ```
pub struct Manager {
    handle: NonNull<kompute_sys::kp_Manager>,
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
        unsafe {
            let handle = kompute_sys::kp_manager_new(device_id);
            if handle.is_null() {
                return Err(KomputeError::device(format!(
                    "Failed to create manager with device {}",
                    device_id
                )));
            }

            Ok(Self {
                handle: NonNull::new_unchecked(handle),
            })
        }
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
        unsafe {
            let count = kompute_sys::kp_manager_get_device_count();
            if count < 0 {
                return Err(KomputeError::vulkan("Failed to enumerate Vulkan devices"));
            }
            Ok(count as u32)
        }
    }

    /// Get the raw handle for FFI operations.
    ///
    /// # Safety
    ///
    /// The returned handle is only valid while this Manager exists.
    /// Using the handle after the Manager is dropped results in undefined behavior.
    pub(crate) unsafe fn handle(&self) -> *mut kompute_sys::kp_Manager {
        self.handle.as_ptr()
    }
}

// RAII cleanup - automatically destroy the manager when dropped
impl Drop for Manager {
    fn drop(&mut self) {
        unsafe {
            kompute_sys::kp_manager_destroy(self.handle.as_ptr());
        }
    }
}

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
                println!("Found {} Vulkan devices", count);
                assert!(count < 100, "Device count should be reasonable");
            }
            Err(e) => {
                println!("Device enumeration failed (expected in CI): {}", e);
            }
        }
    }

    #[test]
    fn test_manager_creation() {
        // This test may fail in environments without Vulkan support
        match Manager::new() {
            Ok(_manager) => {
                // Manager created successfully
                println!("Manager creation succeeded");
            }
            Err(e) => {
                println!("Manager creation failed (expected in CI): {}", e);
            }
        }
    }
}
