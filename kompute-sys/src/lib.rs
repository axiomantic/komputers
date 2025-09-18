//! Low-level Rust bindings for the Kompute GPU compute framework.
//!
//! This crate provides safe Rust bindings to the Kompute C++ library using cxx,
//! enabling GPU compute operations through Vulkan with zero-cost abstractions.
//!
//! # Examples
//!
//! ```rust,no_run
//! use kompute_sys::ffi;
//!
//! // Check if devices are available
//! let device_count = ffi::get_device_count();
//! if device_count > 0 {
//!     let mut manager = ffi::create_manager(0); // Use device 0
//!     let data = vec![1.0f32, 2.0, 3.0, 4.0];
//!     let tensor_handle = ffi::create_tensor_f32_handle(manager.pin_mut(), &data);
//!     println!("Created tensor with {} elements", ffi::tensor_size_by_handle(tensor_handle));
//! }
//! ```

pub mod bridge;
pub use bridge::ffi;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires Vulkan-capable device"]
    fn test_device_detection() {
        // This test requires a Vulkan-capable device
        let device_count = ffi::get_device_count();

        // In environments without Vulkan, device_count may be -1
        // This is acceptable - we're testing that the function doesn't crash
        if device_count >= 0 {
            // Found Vulkan devices
            if device_count > 0 {
                // Test creating a manager
                let manager = ffi::create_manager(0);
                assert!(!manager.is_null(), "Manager creation should succeed");
                // UniquePtr handles destruction automatically
            }
        }
        // else: No Vulkan devices available (expected in CI)

        // The key success is that we didn't segfault
        // Function completed without crashing
    }

    #[test]
    #[ignore = "requires Vulkan-capable device"]
    fn test_complete_workflow() {
        // Test a complete GPU compute workflow if Vulkan is available
        let device_count = ffi::get_device_count();
        if device_count > 0 {
            let mut manager = ffi::create_manager(0);
            if !manager.is_null() {
                // Create tensor
                let data = [1.0f32, 2.0, 3.0, 4.0];
                let tensor = ffi::create_tensor_f32(manager.pin_mut(), &data);

                if !tensor.is_null() {
                    // Note: Sequence operations are not fully functional due to SharedPtr limitations
                    // This is documented in the current implementation

                    // Create sequence
                    let sequence = ffi::create_sequence(manager.pin_mut());
                    assert!(!sequence.is_null(), "Sequence creation should succeed");

                    // SharedPtr handles destruction automatically
                }
                // UniquePtr handles manager destruction automatically
            }
        }
    }
}
