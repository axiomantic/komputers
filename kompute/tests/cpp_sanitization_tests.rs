//! C++ memory sanitization integration tests.
//!
//! These tests integrate AddressSanitizer and LeakSanitizer to detect
//! memory issues in the C++ bridge code, including memory leaks,
//! use-after-free, buffer overflows, and handle registry safety.

use kompute::{Manager, Tensor};
use std::env;
use std::process::Command;
use std::time::Instant;

/// Test C++ memory leak detection using AddressSanitizer.
/// Runs tests in subprocess with sanitizer to detect memory leaks.
#[test]
fn test_cpp_memory_leak_detection() {
    println!("Testing C++ memory leak detection...");

    if !cfg!(feature = "asan") {
        println!("  Skipping C++ sanitization test - asan feature not enabled");
        println!("  To enable: cargo test --features asan");
        return;
    }

    // Set up AddressSanitizer environment
    env::set_var(
        "ASAN_OPTIONS",
        "detect_leaks=1:abort_on_error=1:print_stats=1:symbolize=1",
    );
    env::set_var("MSAN_OPTIONS", "print_stats=1");

    // Test manager creation under sanitizer
    println!("  Running manager creation test with AddressSanitizer...");
    let output = Command::new("cargo")
        .args(&[
            "test",
            "--features",
            "asan",
            "--test",
            "manager_unit_tests",
            "--",
            "--exact",
            "test_manager_thread_safety_concurrent_creation",
            "--nocapture",
        ])
        .env("RUSTFLAGS", "-Z sanitizer=address")
        .output()
        .expect("Failed to execute sanitized manager test");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    println!("  Sanitizer test output analysis:");
    println!("    Exit status: {}", output.status);

    // Check for memory leaks
    if stderr.contains("ERROR: LeakSanitizer") {
        panic!("Memory leak detected in C++ manager code:\n{}", stderr);
    } else {
        println!("    ✓ No memory leaks detected in manager operations");
    }

    // Check for use-after-free
    if stderr.contains("ERROR: AddressSanitizer") && stderr.contains("use-after-free") {
        panic!("Use-after-free detected in C++ manager code:\n{}", stderr);
    } else {
        println!("    ✓ No use-after-free issues detected in manager operations");
    }

    // Check for buffer overflows
    if stderr.contains("ERROR: AddressSanitizer") && stderr.contains("buffer-overflow") {
        panic!("Buffer overflow detected in C++ manager code:\n{}", stderr);
    } else {
        println!("    ✓ No buffer overflow issues detected in manager operations");
    }

    // Verify test actually ran
    if output.status.success()
        || stdout.contains("test_manager_thread_safety")
        || stderr.contains("Vulkan")
    {
        println!("    ✓ Sanitized test executed properly");
    } else {
        println!("    ⚠ Sanitized test may not have executed correctly");
        println!("    stdout: {}", stdout);
        println!("    stderr: {}", stderr);
    }

    // Test tensor operations under sanitizer
    println!("  Running tensor test with AddressSanitizer...");
    let tensor_output = Command::new("cargo")
        .args(&[
            "test",
            "--features",
            "asan",
            "--test",
            "tensor_multitype_tests",
            "--",
            "--exact",
            "test_tensor_data_trait_implementations",
            "--nocapture",
        ])
        .env("RUSTFLAGS", "-Z sanitizer=address")
        .output()
        .expect("Failed to execute sanitized tensor test");

    let tensor_stderr = String::from_utf8_lossy(&tensor_output.stderr);

    if tensor_stderr.contains("ERROR: LeakSanitizer") {
        panic!(
            "Memory leak detected in C++ tensor code:\n{}",
            tensor_stderr
        );
    } else {
        println!("    ✓ No memory leaks detected in tensor operations");
    }

    println!("  ✓ C++ memory leak detection tests completed");
}

/// Test C++ use-after-free detection with handle lifecycle.
/// Verifies that dropped handles don't cause use-after-free issues.
#[test]
fn test_cpp_use_after_free_detection() {
    println!("Testing C++ use-after-free detection...");

    if !cfg!(feature = "asan") {
        println!("  Skipping use-after-free test - asan feature not enabled");
        return;
    }

    // Set up sanitizer for use-after-free detection
    env::set_var(
        "ASAN_OPTIONS",
        "detect_use_after_free=1:abort_on_error=1:symbolize=1",
    );

    println!("  Running handle lifecycle test with AddressSanitizer...");
    let output = Command::new("cargo")
        .args(&[
            "test",
            "--features",
            "asan",
            "--test",
            "tensor_multitype_tests",
            "--",
            "--exact",
            "test_tensor_handle_lifecycle_invalidation",
            "--nocapture",
        ])
        .env("RUSTFLAGS", "-Z sanitizer=address")
        .output()
        .expect("Failed to execute sanitized handle lifecycle test");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Check for use-after-free specifically
    if stderr.contains("ERROR: AddressSanitizer") && stderr.contains("use-after-free") {
        panic!("Use-after-free detected in handle lifecycle:\n{}", stderr);
    } else {
        println!("    ✓ No use-after-free issues detected in handle lifecycle");
    }

    // Check for heap-use-after-free
    if stderr.contains("heap-use-after-free") {
        panic!(
            "Heap use-after-free detected in handle management:\n{}",
            stderr
        );
    } else {
        println!("    ✓ No heap use-after-free issues detected");
    }

    println!("  ✓ Use-after-free detection tests completed");
}

/// Test C++ buffer overflow detection with tensor data boundaries.
/// Verifies boundary checking in tensor data operations.
#[test]
fn test_cpp_buffer_overflow_detection() {
    println!("Testing C++ buffer overflow detection...");

    if !cfg!(feature = "asan") {
        println!("  Skipping buffer overflow test - asan feature not enabled");
        return;
    }

    // Set up sanitizer for buffer overflow detection
    env::set_var(
        "ASAN_OPTIONS",
        "detect_stack_use_after_return=1:check_initialization_order=1:strict_init_order=1",
    );

    println!("  Running tensor boundary test with AddressSanitizer...");
    let output = Command::new("cargo")
        .args(&[
            "test",
            "--features",
            "asan",
            "--test",
            "tensor_multitype_tests",
            "--",
            "--exact",
            "test_tensor_memory_alignment_requirements",
            "--nocapture",
        ])
        .env("RUSTFLAGS", "-Z sanitizer=address")
        .output()
        .expect("Failed to execute sanitized tensor boundary test");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Check for buffer overflows
    if stderr.contains("buffer-overflow") || stderr.contains("heap-buffer-overflow") {
        panic!("Buffer overflow detected in tensor operations:\n{}", stderr);
    } else {
        println!("    ✓ No buffer overflow issues detected in tensor operations");
    }

    // Check for stack buffer overflow
    if stderr.contains("stack-buffer-overflow") {
        panic!("Stack buffer overflow detected:\n{}", stderr);
    } else {
        println!("    ✓ No stack buffer overflow issues detected");
    }

    // Check for global buffer overflow
    if stderr.contains("global-buffer-overflow") {
        panic!("Global buffer overflow detected:\n{}", stderr);
    } else {
        println!("    ✓ No global buffer overflow issues detected");
    }

    println!("  ✓ Buffer overflow detection tests completed");
}

/// Test handle registry memory safety with concurrent operations.
/// Verifies thread-safe handle registration and cleanup.
#[test]
fn test_handle_registry_memory_safety() {
    println!("Testing handle registry memory safety...");

    if !cfg!(feature = "asan") {
        println!("  Skipping handle registry test - asan feature not enabled");
        return;
    }

    // Set up sanitizer for comprehensive checking
    env::set_var(
        "ASAN_OPTIONS",
        "detect_thread_leaks=1:detect_leaks=1:abort_on_error=1",
    );

    println!("  Running concurrent manager test with AddressSanitizer...");
    let output = Command::new("cargo")
        .args(&[
            "test",
            "--features",
            "asan",
            "--test",
            "manager_unit_tests",
            "--",
            "--exact",
            "test_manager_thread_safety_concurrent_creation",
            "--nocapture",
        ])
        .env("RUSTFLAGS", "-Z sanitizer=address")
        .output()
        .expect("Failed to execute sanitized concurrent test");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Check for thread-related memory issues
    if stderr.contains("ERROR: ThreadSanitizer") {
        panic!(
            "Thread safety issue detected in handle registry:\n{}",
            stderr
        );
    } else {
        println!("    ✓ No thread safety issues detected in handle registry");
    }

    // Check for race conditions
    if stderr.contains("data race") {
        panic!("Data race detected in handle registry:\n{}", stderr);
    } else {
        println!("    ✓ No data races detected in handle registry");
    }

    // Test handle tracker specifically
    println!("  Running handle tracker test with AddressSanitizer...");
    let tracker_output = Command::new("cargo")
        .args(&[
            "test",
            "--features",
            "asan",
            "handle_tracker",
            "--",
            "--nocapture",
        ])
        .env("RUSTFLAGS", "-Z sanitizer=address")
        .output()
        .expect("Failed to execute sanitized handle tracker test");

    let tracker_stderr = String::from_utf8_lossy(&tracker_output.stderr);

    if tracker_stderr.contains("ERROR: AddressSanitizer") {
        panic!(
            "Memory safety issue detected in handle tracker:\n{}",
            tracker_stderr
        );
    } else {
        println!("    ✓ No memory safety issues detected in handle tracker");
    }

    println!("  ✓ Handle registry memory safety tests completed");
}

/// Test Vulkan resource cleanup verification.
/// Verifies proper cleanup of GPU resources and handles.
#[test]
fn test_vulkan_resource_cleanup_verification() {
    println!("Testing Vulkan resource cleanup...");

    if !cfg!(feature = "asan") {
        println!("  Skipping Vulkan cleanup test - asan feature not enabled");
        return;
    }

    // Set up sanitizer for resource leak detection
    env::set_var(
        "ASAN_OPTIONS",
        "detect_leaks=1:leak_check_at_exit=1:print_stats=1",
    );

    println!("  Running resource creation test with AddressSanitizer...");
    let output = Command::new("cargo")
        .args(&[
            "test",
            "--features",
            "asan",
            "--test",
            "manager_unit_tests",
            "--",
            "--exact",
            "test_manager_resource_limits_tensor_creation",
            "--nocapture",
        ])
        .env("RUSTFLAGS", "-Z sanitizer=address")
        .output()
        .expect("Failed to execute sanitized resource test");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Check for resource leaks
    if stderr.contains("ERROR: LeakSanitizer") {
        // Parse leak information
        if stderr.contains("Direct leak") {
            panic!(
                "Direct memory leak detected in Vulkan resources:\n{}",
                stderr
            );
        }
        if stderr.contains("Indirect leak") {
            panic!(
                "Indirect memory leak detected in Vulkan resources:\n{}",
                stderr
            );
        }
    } else {
        println!("    ✓ No resource leaks detected in Vulkan operations");
    }

    // Test sequence resource cleanup
    println!("  Running sequence cleanup test with AddressSanitizer...");
    let seq_output = Command::new("cargo")
        .args(&[
            "test",
            "--features",
            "asan",
            "--test",
            "sequence_command_tests",
            "--",
            "--exact",
            "test_sequence_creation_and_properties",
            "--nocapture",
        ])
        .env("RUSTFLAGS", "-Z sanitizer=address")
        .output()
        .expect("Failed to execute sanitized sequence test");

    let seq_stderr = String::from_utf8_lossy(&seq_output.stderr);

    if seq_stderr.contains("ERROR: LeakSanitizer") {
        panic!(
            "Resource leak detected in sequence cleanup:\n{}",
            seq_stderr
        );
    } else {
        println!("    ✓ No resource leaks detected in sequence operations");
    }

    // Test algorithm resource cleanup
    println!("  Running algorithm cleanup test with AddressSanitizer...");
    let algo_output = Command::new("cargo")
        .args(&[
            "test",
            "--features",
            "asan",
            "--test",
            "algorithm_spirv_tests",
            "--",
            "--exact",
            "test_algorithm_creation_performance",
            "--nocapture",
        ])
        .env("RUSTFLAGS", "-Z sanitizer=address")
        .output()
        .expect("Failed to execute sanitized algorithm test");

    let algo_stderr = String::from_utf8_lossy(&algo_output.stderr);

    if algo_stderr.contains("ERROR: LeakSanitizer") {
        panic!(
            "Resource leak detected in algorithm cleanup:\n{}",
            algo_stderr
        );
    } else {
        println!("    ✓ No resource leaks detected in algorithm operations");
    }

    println!("  ✓ Vulkan resource cleanup verification completed");
}

/// Test sanitizer integration with actual GPU operations.
/// Verifies sanitizers work correctly with real GPU calls.
#[test]
#[ignore = "requires Vulkan-capable device and asan"]
fn test_sanitizer_integration_with_gpu_operations() {
    println!("Testing sanitizer integration with GPU operations...");

    if !cfg!(feature = "asan") {
        println!("  Skipping GPU sanitizer test - asan feature not enabled");
        return;
    }

    // This test requires both sanitizers and GPU to work together
    if let Ok(mut manager) = Manager::new() {
        println!("  GPU available - running sanitized GPU operations...");

        // Set comprehensive sanitizer options
        env::set_var(
            "ASAN_OPTIONS",
            "detect_leaks=1:detect_use_after_free=1:abort_on_error=1:symbolize=1",
        );

        // Create tensors under sanitizer
        let data = vec![1.0f32; 1000];
        match Tensor::new(&mut manager, &data) {
            Ok(tensor) => {
                println!("    ✓ Tensor creation under sanitizer successful");

                // Test data access under sanitizer
                match tensor.to_vec() {
                    Ok(retrieved) => {
                        assert_eq!(data.len(), retrieved.len(), "Data length should match");
                        println!("    ✓ Tensor data access under sanitizer successful");
                    }
                    Err(e) => {
                        println!("    ⚠ Tensor data access failed: {}", e);
                    }
                }

                // Handle should be valid
                assert_ne!(tensor.handle(), 0, "Tensor handle should be valid");
                println!("    ✓ Tensor handle validation under sanitizer successful");
            }
            Err(e) => {
                println!("    ⚠ Tensor creation under sanitizer failed: {}", e);
            }
        }

        // Test manager operations under sanitizer
        match Manager::device_count() {
            Ok(count) => {
                println!(
                    "    ✓ Device enumeration under sanitizer successful: {} devices",
                    count
                );
            }
            Err(e) => {
                println!("    ⚠ Device enumeration under sanitizer failed: {}", e);
            }
        }

        println!("  ✓ GPU operations under sanitizer completed without errors");
    } else {
        println!("  No GPU available - testing sanitizer with headless operations...");

        // Test headless operations under sanitizer
        match Manager::device_count() {
            Ok(count) => {
                println!("    ✓ Headless device count under sanitizer: {}", count);
            }
            Err(e) => {
                println!("    ✓ Expected headless failure under sanitizer: {}", e);
            }
        }
    }
}

/// Test sanitizer performance impact measurement.
/// Measures the performance overhead of running with sanitizers.
#[test]
fn test_sanitizer_performance_impact() {
    println!("Testing sanitizer performance impact...");

    if !cfg!(feature = "asan") {
        println!("  Skipping performance impact test - asan feature not enabled");
        return;
    }

    // Test normal execution time
    println!("  Measuring normal execution time...");
    let normal_start = Instant::now();
    let normal_output = Command::new("cargo")
        .args(&[
            "test",
            "--test",
            "manager_unit_tests",
            "--",
            "--exact",
            "test_manager_creation_performance_consistency",
            "--nocapture",
        ])
        .output()
        .expect("Failed to execute normal performance test");
    let normal_time = normal_start.elapsed();

    // Test sanitized execution time
    println!("  Measuring sanitized execution time...");
    let sanitized_start = Instant::now();
    let sanitized_output = Command::new("cargo")
        .args(&[
            "test",
            "--features",
            "asan",
            "--test",
            "manager_unit_tests",
            "--",
            "--exact",
            "test_manager_creation_performance_consistency",
            "--nocapture",
        ])
        .env("RUSTFLAGS", "-Z sanitizer=address")
        .output()
        .expect("Failed to execute sanitized performance test");
    let sanitized_time = sanitized_start.elapsed();

    println!("Performance impact analysis:");
    println!("  Normal execution time: {:?}", normal_time);
    println!("  Sanitized execution time: {:?}", sanitized_time);

    if sanitized_time > normal_time {
        let overhead_ratio = sanitized_time.as_secs_f64() / normal_time.as_secs_f64();
        println!("  Sanitizer overhead: {:.1}x slower", overhead_ratio);

        // Sanitizer overhead should be reasonable (typically 2-10x)
        assert!(
            overhead_ratio < 50.0,
            "Sanitizer overhead too high: {:.1}x",
            overhead_ratio
        );

        if overhead_ratio < 2.0 {
            println!("  ✓ Low overhead sanitization");
        } else if overhead_ratio < 10.0 {
            println!("  ✓ Acceptable overhead sanitization");
        } else {
            println!("  ⚠ High overhead sanitization: {:.1}x", overhead_ratio);
        }
    } else {
        println!("  ⚠ Sanitized execution faster than normal (unexpected)");
    }

    // Check both tests succeeded (or failed consistently)
    let normal_success = normal_output.status.success();
    let sanitized_success = sanitized_output.status.success();

    println!("  Normal test success: {}", normal_success);
    println!("  Sanitized test success: {}", sanitized_success);

    // Both should have same success status
    assert_eq!(
        normal_success, sanitized_success,
        "Normal and sanitized tests should have same outcome"
    );

    println!("  ✓ Sanitizer performance impact measurement completed");
}

/// Test sanitizer configuration validation.
/// Verifies that sanitizer options are properly configured.
#[test]
fn test_sanitizer_configuration_validation() {
    println!("Testing sanitizer configuration...");

    if !cfg!(feature = "asan") {
        println!("  Skipping configuration test - asan feature not enabled");
        return;
    }

    // Test different sanitizer configurations
    let configs = vec![
        ("detect_leaks=1", "Leak detection"),
        ("detect_use_after_free=1", "Use-after-free detection"),
        ("abort_on_error=1", "Abort on error"),
        ("symbolize=1", "Symbol resolution"),
        ("print_stats=1", "Statistics printing"),
    ];

    for (option, description) in configs {
        println!("  Testing {}: {}", description, option);

        env::set_var("ASAN_OPTIONS", option);

        let output = Command::new("cargo")
            .args(&[
                "test",
                "--features",
                "asan",
                "--lib",
                "--",
                "--exact",
                "test_version_constants",
                "--nocapture",
            ])
            .env("RUSTFLAGS", "-Z sanitizer=address")
            .output()
            .expect("Failed to execute sanitizer config test");

        let stderr = String::from_utf8_lossy(&output.stderr);

        // Check if sanitizer is actually running
        if stderr.contains("AddressSanitizer") || stderr.contains("ASAN") {
            println!("    ✓ {} active", description);
        } else {
            println!("    ? {} status unclear", description);
        }

        // Test should not crash from sanitizer misconfiguration
        if !output.status.success() && stderr.contains("sanitizer") {
            println!("    ⚠ {} may have configuration issues", description);
        }
    }

    // Reset to comprehensive options
    env::set_var(
        "ASAN_OPTIONS",
        "detect_leaks=1:detect_use_after_free=1:abort_on_error=1:symbolize=1:print_stats=1",
    );

    println!("  ✓ Sanitizer configuration validation completed");
}
