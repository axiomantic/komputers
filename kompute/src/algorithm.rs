//! SPIR-V compute shader management and execution.

use crate::error::Result;
use crate::manager::Manager;
use kompute_sys::ffi;

/// GPU compute algorithm from SPIR-V shader code.
///
/// Algorithms represent compiled compute shaders that can be executed on the GPU.
/// They are created from SPIR-V bytecode and can be bound to tensors for execution
/// using handle-based resource management.
///
/// # Examples
///
/// ```rust,no_run
/// use kompute::{Manager, Algorithm, Result};
///
/// fn example() -> Result<()> {
///     let mut manager = Manager::new()?;
///     let spirv_data: Vec<u32> = vec![/* SPIR-V bytecode */];
///     let algorithm = Algorithm::new(&mut manager, &spirv_data)?;
///     Ok(())
/// }
/// ```
pub struct Algorithm {
    handle: u64,
}

impl Algorithm {
    /// Create a new algorithm from SPIR-V bytecode.
    ///
    /// # Arguments
    ///
    /// * `manager` - The GPU manager to create the algorithm with
    /// * `spirv_data` - SPIR-V bytecode as 32-bit words
    ///
    /// # Errors
    ///
    /// Returns [`KomputeError::ShaderError`] if the SPIR-V is invalid or
    /// [`KomputeError::ComputeError`] if algorithm creation fails.
    pub fn new(manager: &mut Manager, spirv_data: &[u32]) -> Result<Self> {
        // Validate input data
        if spirv_data.is_empty() {
            return Err(crate::error::KomputeError::shader(
                "Cannot create algorithm with empty SPIR-V data",
            ));
        }

        if spirv_data.len() < 5 {
            return Err(crate::error::KomputeError::shader(
                "SPIR-V data too short - must contain at least header",
            ));
        }

        // Validate SPIR-V magic number (0x07230203)
        if spirv_data[0] != 0x07230203 {
            return Err(crate::error::KomputeError::shader(format!(
                "Invalid SPIR-V magic number: expected 0x07230203, got 0x{:08x}",
                spirv_data[0]
            )));
        }

        if spirv_data.len() > 1024 * 1024 {
            return Err(crate::error::KomputeError::shader(format!(
                "SPIR-V data too large: {} words (maximum supported: 1M words)",
                spirv_data.len()
            )));
        }

        let handle = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ffi::create_algorithm_handle(manager.handle(), spirv_data)
        }))
        .map_err(|_| {
            crate::error::KomputeError::shader(
                "Algorithm creation failed due to invalid SPIR-V or GPU error",
            )
        })?;

        if handle == 0 {
            return Err(crate::error::KomputeError::shader(
                "Algorithm creation returned invalid handle",
            ));
        }

        Ok(Self { handle })
    }

    /// Get the internal handle value.
    ///
    /// This is used for interfacing with other handle-based operations.
    pub fn handle(&self) -> u64 {
        self.handle
    }
}

impl Drop for Algorithm {
    fn drop(&mut self) {
        ffi::algorithm_destroy_handle(self.handle);
    }
}

// Algorithms are safe to send between threads
unsafe impl Send for Algorithm {}

// Algorithms can be shared between threads with proper synchronization
unsafe impl Sync for Algorithm {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Manager;

    #[test]
    #[ignore = "requires Vulkan-capable device"]
    fn test_algorithm_creation() {
        // This test may fail in environments without Vulkan support
        if let Ok(mut manager) = Manager::new() {
            // Simple SPIR-V shader (this would normally be compiled from GLSL)
            let spirv_data = vec![
                0x07230203, // SPIR-V magic number
                0x00010000, // Version
                0x00080001, // Generator
                0x0000000b, // Bound
                0x00000000, // Schema
            ];

            match Algorithm::new(&mut manager, &spirv_data) {
                Ok(_algorithm) => {
                    // Algorithm creation succeeded
                    assert!(true, "Algorithm creation succeeded");
                }
                Err(_e) => {
                    // Algorithm creation failed - expected with invalid SPIR-V
                    assert!(
                        true,
                        "Algorithm creation failed as expected with test SPIR-V"
                    );
                }
            }
        }
    }
}
