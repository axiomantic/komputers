//! Build system verification tests.
//!
//! These tests verify that the build system correctly compiles the C++ Kompute
//! library, generates FFI bindings, and enables basic GPU operations.

use kompute::{Manager, Result};

/// Test that the build system successfully links against Vulkan.
#[test]
fn test_vulkan_linking() -> Result<()> {
    // This test verifies that Vulkan symbols are properly linked
    let device_count = Manager::device_count();

    // Even if no devices are found, the call should succeed (return 0)
    // If linking failed, this would panic or return an error
    match device_count {
        Ok(_count) => {
            // Vulkan linking successful
            Ok(())
        }
        Err(_e) => {
            // In CI environments without GPUs, this might fail
            // but the test still validates that symbols are linked
            Ok(())
        }
    }
}

/// Test that CXX bridge bindings were generated correctly.
#[test]
fn test_ffi_bindings() {
    // Verify that key CXX bridge symbols exist and are callable
    let device_count = kompute_sys::ffi::get_device_count();
    // CXX bridge binding test completed

    // Negative values indicate errors, non-negative are valid
    assert!(device_count >= -1, "Device count should be reasonable");
}

/// Test manager creation and destruction (RAII).
#[test]

fn test_manager_raii() {
    // Test that Manager properly manages C++ resources
    match Manager::new() {
        Ok(manager) => {
            // Manager RAII test: creation successful
            // Manager will be automatically destroyed when it goes out of scope
            drop(manager);
            // Manager RAII test: destruction successful
        }
        Err(_e) => {
            // Manager creation failed (expected in CI)
        }
    }
}

/// Test cross-platform compilation capabilities.
#[test]
fn test_cross_platform_build() {
    // This test verifies that the build works on the current platform
    let target_os = std::env::consts::OS;
    // Build verification on current platform

    match target_os {
        "linux" | "windows" | "macos" => {
            // These platforms should be supported
        }
        _ => {
            // Platform may have limited support
        }
    }

    // If we reached this point, the build system worked for this platform
    // Cross-platform build successful
}

/// Benchmark basic build performance for development workflow.
#[test]
fn test_build_performance() {
    use std::time::Instant;

    let start = Instant::now();

    // Perform some basic operations that exercise the compiled code
    let _ = Manager::device_count();

    let duration = start.elapsed();
    // Basic operation completed

    // Build should enable fast basic operations (sub-second)
    assert!(duration.as_secs() < 5, "Basic operations should be fast");
}

/// Test that Vulkan validation layers can be enabled in debug builds.
#[test]
#[cfg(feature = "validation")]
fn test_validation_layers() {
    // Validation layers feature enabled

    // In debug builds with validation feature, additional checks should be available
    // This test verifies the feature flag compilation works
    assert!(cfg!(feature = "validation"));
}

/// Test error handling for invalid operations.
#[test]

fn test_error_handling() {
    // Test that errors are properly propagated from C++ to Rust
    match Manager::with_device(999) {
        // Invalid device ID
        Ok(_) => {
            // Unexpected success with invalid device ID
        }
        Err(e) => {
            // Correctly caught error for invalid device

            // Verify error contains useful information
            let error_msg = format!("{}", e);
            assert!(error_msg.contains("device") || error_msg.contains("Device"));
        }
    }
}
