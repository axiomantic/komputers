//! Low-level unsafe Rust bindings for the Kompute GPU compute framework.
//!
//! This crate provides direct FFI bindings to the Kompute C++ library, enabling
//! GPU compute operations through Vulkan. These bindings are intentionally minimal
//! and unsafe - for safe, idiomatic Rust APIs, use the `kompute` crate instead.
//!
//! # Safety
//!
//! All functions in this crate are marked as `unsafe` because they directly
//! interface with C++ code and GPU memory. Proper resource management and
//! lifetime handling are the responsibility of the caller.
//!
//! # Examples
//!
//! ```rust,ignore
//! use kompute_sys::*;
//!
//! unsafe {
//!     let manager = kp_manager_new(0); // Use device 0
//!     // ... perform GPU operations
//!     kp_manager_destroy(manager);
//! }
//! ```

#![no_std]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

// Include the generated bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_detection() {
        unsafe {
            // This test requires a Vulkan-capable device
            let device_count = kp_manager_get_device_count();
            assert!(device_count >= 0, "Device count should be non-negative");

            if device_count > 0 {
                // Test creating and destroying a manager
                let manager = kp_manager_new(0);
                assert!(!manager.is_null(), "Manager creation should succeed");
                kp_manager_destroy(manager);
            }
        }
    }
}
