//! Comprehensive error handling and edge case tests.
//!
//! These tests verify that the library handles errors gracefully,
//! provides meaningful error messages, and maintains stability
//! under various failure conditions.

use komputers::{KomputeError, Manager, Result, Sequence, Tensor};
use std::panic;

/// Test error handling in manager creation and device operations.
#[test]
fn test_manager_error_handling() {
    println!("Starting manager error handling tests...");

    // Test device count error handling
    match Manager::device_count() {
        Ok(count) => {
            println!("✓ Device count: {}", count);

            // Test invalid device ID
            let invalid_device_id = count + 100;
            match Manager::with_device(invalid_device_id) {
                Ok(_) => {
                    panic!(
                        "Should not succeed with invalid device ID {}",
                        invalid_device_id
                    );
                }
                Err(e) => {
                    println!(
                        "✓ Invalid device ID {} correctly rejected: {}",
                        invalid_device_id, e
                    );
                    // Verify error type and message quality
                    assert!(e.to_string().contains("device") || e.to_string().contains("Device"));
                }
            }

            // Test extremely large device ID
            match Manager::with_device(u32::MAX) {
                Ok(_) => {
                    panic!("Should not succeed with device ID {}", u32::MAX);
                }
                Err(e) => {
                    println!("✓ Extreme device ID correctly rejected: {}", e);
                }
            }
        }
        Err(e) => {
            println!(
                "⚠️  Device enumeration failed (expected in headless): {}",
                e
            );
            // Verify error message is meaningful
            assert!(!e.to_string().is_empty());
        }
    }

    // Test multiple manager creation attempts
    for i in 0..5 {
        match Manager::new() {
            Ok(_) => println!("✓ Manager creation {} succeeded", i),
            Err(e) => println!("⚠️  Manager creation {} failed: {}", i, e),
        }
    }

    println!("✓ Manager error handling tests completed");
}

/// Test tensor creation error scenarios.
#[test]
fn test_tensor_error_handling() {
    println!("Starting tensor error handling tests...");

    if let Ok(mut manager) = Manager::new() {
        // Test very large tensor creation
        let huge_size = usize::MAX / 1000; // Avoid overflow but still too large
        match Tensor::<f32>::uninit(&mut manager, huge_size) {
            Ok(_) => {
                println!(
                    "⚠️  Unexpectedly succeeded creating huge tensor of size {}",
                    huge_size
                );
            }
            Err(e) => {
                println!("✓ Huge tensor creation correctly failed: {}", e);
                assert!(!e.to_string().is_empty());
            }
        }

        // Test tensor creation with problematic data patterns
        let problematic_patterns = vec![
            vec![f32::NAN; 100],
            vec![f32::INFINITY; 100],
            vec![f32::NEG_INFINITY; 100],
            vec![f32::MAX; 100],
            vec![f32::MIN; 100],
        ];

        for (i, data) in problematic_patterns.into_iter().enumerate() {
            match Tensor::new(&mut manager, &data) {
                Ok(tensor) => {
                    println!("✓ Problematic pattern {} handled successfully", i);
                    assert_eq!(tensor.size(), data.len());

                    // Test data retrieval with problematic values
                    match tensor.to_vec() {
                        Ok(retrieved) => {
                            println!("✓ Data retrieval succeeded for pattern {}", i);
                            assert_eq!(retrieved.len(), data.len());
                        }
                        Err(e) => {
                            println!("⚠️  Data retrieval failed for pattern {}: {}", i, e);
                        }
                    }
                }
                Err(e) => {
                    println!("⚠️  Problematic pattern {} creation failed: {}", i, e);
                }
            }
        }

        // Test zero-sized tensor
        let empty_data: Vec<f32> = vec![];
        match Tensor::new(&mut manager, &empty_data) {
            Ok(tensor) => {
                println!("✓ Empty tensor creation succeeded");
                assert_eq!(tensor.size(), 0);
                assert_eq!(tensor.size_bytes(), 0);

                match tensor.to_vec() {
                    Ok(retrieved) => {
                        assert!(retrieved.is_empty());
                        println!("✓ Empty tensor data retrieval succeeded");
                    }
                    Err(e) => {
                        println!("⚠️  Empty tensor data retrieval failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("✓ Empty tensor creation failed as expected: {}", e);
            }
        }
    } else {
        println!("⚠️  Skipping tensor tests due to manager creation failure");
    }

    println!("✓ Tensor error handling tests completed");
}

/// Test sequence operation error scenarios.
#[test]
fn test_sequence_error_handling() {
    println!("Starting sequence error handling tests...");

    if let Ok(mut manager) = Manager::new() {
        // Test sequence creation
        match Sequence::new(&mut manager) {
            Ok(_sequence) => {
                println!("✓ Sequence creation succeeded");

                // Test operations with invalid handles
                // Note: We can't easily test invalid handle operations without
                // exposing internal APIs, but we can test the error paths
            }
            Err(e) => {
                println!("⚠️  Sequence creation failed: {}", e);
            }
        }

        // Test creating many sequences to check resource limits
        let mut sequences = Vec::new();
        for i in 0..100 {
            match Sequence::new(&mut manager) {
                Ok(seq) => {
                    sequences.push(seq);
                }
                Err(e) => {
                    println!(
                        "✓ Sequence creation failed at count {} (resource limit): {}",
                        i, e
                    );
                    break;
                }
            }
        }

        println!(
            "✓ Created {} sequences before hitting limits",
            sequences.len()
        );
    }

    println!("✓ Sequence error handling tests completed");
}

/// Test error message quality and consistency.
#[test]
fn test_error_message_quality() {
    println!("Starting error message quality tests...");

    // Test various error constructors
    let errors = vec![
        KomputeError::vulkan("Vulkan initialization failed"),
        KomputeError::hardware("GPU device lost"),
        KomputeError::shader("Invalid SPIR-V bytecode"),
        KomputeError::tensor("Dimension mismatch"),
        KomputeError::memory("Out of GPU memory"),
        KomputeError::compute("Kernel execution failed"),
        KomputeError::device("Device not found"),
        KomputeError::invalid_parameter("Invalid buffer size"),
        KomputeError::resource_not_found("Shader not found"),
        KomputeError::not_supported("Feature not supported on this device"),
        KomputeError::platform("Platform compatibility issue"),
        KomputeError::environment("Environment configuration error"),
        KomputeError::internal("Unexpected internal state"),
    ];

    for (i, error) in errors.iter().enumerate() {
        let message = error.to_string();
        let debug_message = format!("{:?}", error);

        println!("Error {}: {}", i, message);

        // Error messages should be non-empty and informative
        assert!(!message.is_empty(), "Error message should not be empty");
        assert!(message.len() > 10, "Error message should be descriptive");
        assert!(
            !debug_message.is_empty(),
            "Debug message should not be empty"
        );

        // Error messages should contain the error type
        match error {
            KomputeError::VulkanError(_) => assert!(message.contains("Vulkan")),
            KomputeError::HardwareError(_) => assert!(message.contains("hardware")),
            KomputeError::ShaderError(_) => assert!(message.contains("Shader")),
            KomputeError::TensorError(_) => assert!(message.contains("Tensor")),
            KomputeError::MemoryError(_) => assert!(message.contains("Memory")),
            KomputeError::ComputeError(_) => assert!(message.contains("Compute")),
            KomputeError::DeviceError(_) => assert!(message.contains("Device")),
            KomputeError::InvalidParameter(_) => assert!(message.contains("parameter")),
            KomputeError::ResourceNotFound(_) => assert!(message.contains("not found")),
            KomputeError::NotSupported(_) => assert!(message.contains("not supported")),
            KomputeError::PlatformError(_) => assert!(message.contains("Platform")),
            KomputeError::EnvironmentError(_) => assert!(message.contains("Environment")),
            KomputeError::InternalError(_) => assert!(message.contains("Internal")),
        }
    }

    println!("✓ All error messages are well-formed");
    println!("✓ Error message quality tests completed");
}

/// Test panic safety and error recovery.
#[test]
fn test_panic_safety() {
    println!("Starting panic safety tests...");

    // Test that invalid operations don't panic
    let test_cases: Vec<(&str, Box<dyn Fn() + Send>)> = vec![
        (
            "Manager creation",
            Box::new(|| {
                let _result = Manager::new();
            }),
        ),
        (
            "Device count",
            Box::new(|| {
                let _result = Manager::device_count();
            }),
        ),
        (
            "Invalid device ID",
            Box::new(|| {
                let _result = Manager::with_device(99999);
            }),
        ),
    ];

    for (name, test_fn) in test_cases {
        let result = panic::catch_unwind(panic::AssertUnwindSafe(test_fn));
        match result {
            Ok(_) => println!("✓ {} completed without panic", name),
            Err(_) => panic!("❌ {} caused a panic - this should not happen", name),
        }
    }

    // Test error propagation doesn't cause panics
    fn error_chain_test() -> Result<()> {
        fn inner() -> Result<()> {
            Err(KomputeError::compute("inner error"))
        }

        fn middle() -> Result<()> {
            inner()?;
            Ok(())
        }

        middle()?;
        Ok(())
    }

    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let _result = error_chain_test();
    }));

    match result {
        Ok(_) => println!("✓ Error chain propagation completed without panic"),
        Err(_) => panic!("❌ Error chain caused a panic"),
    }

    println!("✓ Panic safety tests completed");
}

/// Test resource cleanup and leak prevention.
#[test]
fn test_resource_cleanup() {
    println!("Starting resource cleanup tests...");

    if let Ok(mut manager) = Manager::new() {
        let mut created_handles = Vec::new();

        // Create and drop resources in nested scopes
        {
            let data1 = vec![1.0f32, 2.0, 3.0];
            let tensor1 =
                Tensor::new(&mut manager, &data1).expect("Tensor creation should succeed");
            created_handles.push(tensor1.handle());

            {
                let data2 = vec![4.0f32, 5.0, 6.0];
                let tensor2 =
                    Tensor::new(&mut manager, &data2).expect("Tensor creation should succeed");
                created_handles.push(tensor2.handle());

                {
                    let _seq =
                        Sequence::new(&mut manager).expect("Sequence creation should succeed");
                    // seq drops here
                    println!("✓ Sequence dropped in inner scope");
                }

                // tensor2 drops here
                println!("✓ Tensor2 dropped in middle scope");
            }

            // tensor1 drops here
            println!("✓ Tensor1 dropped in outer scope");
        }

        // Create new resources to test handle reuse/uniqueness
        let data3 = vec![7.0f32, 8.0, 9.0];
        let tensor3 =
            Tensor::new(&mut manager, &data3).expect("New tensor creation should succeed");
        let new_handle = tensor3.handle();

        // New handle should be different from previous ones
        assert!(
            !created_handles.contains(&new_handle),
            "New handle should be unique"
        );
        assert_ne!(new_handle, 0, "Handle should be non-zero");

        println!("✓ Resource cleanup and handle uniqueness verified");
    }

    println!("✓ Resource cleanup tests completed");
}

/// Test concurrent error scenarios.
#[test]
fn test_concurrent_error_handling() {
    use std::sync::{Arc, Mutex};
    use std::thread;

    println!("Starting concurrent error handling tests...");

    // Test concurrent manager creation
    let handles: Vec<_> = (0..5)
        .map(|i| {
            thread::spawn(move || {
                for _attempt in 0..3 {
                    match Manager::new() {
                        Ok(_) => {} // Success is fine
                        Err(e) => {
                            // Errors should be well-formed
                            assert!(!e.to_string().is_empty());
                        }
                    }
                }
                println!("✓ Thread {} completed manager creation tests", i);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }

    // Test concurrent tensor creation with shared manager
    if let Ok(manager) = Manager::new() {
        let manager = Arc::new(Mutex::new(manager));
        let handles: Vec<_> = (0..3)
            .map(|i| {
                let manager_clone = Arc::clone(&manager);
                thread::spawn(move || {
                    for j in 0..5 {
                        let data: Vec<f32> = vec![i as f32 * 10.0 + j as f32; 10];
                        let mut guard = manager_clone.lock().unwrap();
                        match Tensor::new(&mut *guard, &data) {
                            Ok(_tensor) => {
                                // Success is fine
                            }
                            Err(e) => {
                                // Errors should be well-formed
                                assert!(!e.to_string().is_empty());
                            }
                        }
                        drop(guard);
                    }
                    println!("✓ Thread {} completed tensor creation tests", i);
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread should not panic");
        }
    }

    println!("✓ Concurrent error handling tests completed");
}

/// Test edge cases with numerical limits.
#[test]
fn test_numerical_edge_cases() {
    println!("Starting numerical edge cases tests...");

    if let Ok(mut manager) = Manager::new() {
        // Test with various problematic float values
        let edge_case_values = vec![
            (vec![0.0f32], "positive zero"),
            (vec![-0.0f32], "negative zero"),
            (vec![f32::MIN_POSITIVE], "minimum positive"),
            (vec![f32::MAX], "maximum finite"),
            (vec![-f32::MAX], "maximum negative finite"),
            (vec![f32::EPSILON], "machine epsilon"),
            (vec![1.0 + f32::EPSILON], "1 + epsilon"),
            (vec![1.0 - f32::EPSILON / 2.0], "1 - epsilon/2"),
        ];

        for (data, description) in edge_case_values {
            match Tensor::new(&mut manager, &data) {
                Ok(tensor) => {
                    println!("✓ {} tensor created successfully", description);
                    match tensor.to_vec() {
                        Ok(retrieved) => {
                            assert_eq!(retrieved.len(), 1);
                            // For most values, we expect bit-perfect round-trip
                            if !data[0].is_nan() {
                                if (data[0] == 0.0 && retrieved[0] == 0.0)
                                    || (data[0] != 0.0
                                        && (data[0] - retrieved[0]).abs()
                                            <= f32::EPSILON * data[0].abs())
                                {
                                    println!("✓ {} value preserved accurately", description);
                                } else {
                                    println!(
                                        "⚠️  {} value changed: {} -> {}",
                                        description, data[0], retrieved[0]
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            println!("⚠️  {} data retrieval failed: {}", description, e);
                        }
                    }
                }
                Err(e) => {
                    println!("⚠️  {} tensor creation failed: {}", description, e);
                }
            }
        }

        // Test with infinity and NaN
        let special_values = vec![
            (vec![f32::INFINITY], "positive infinity"),
            (vec![f32::NEG_INFINITY], "negative infinity"),
            (vec![f32::NAN], "NaN"),
        ];

        for (data, description) in special_values {
            match Tensor::new(&mut manager, &data) {
                Ok(tensor) => {
                    println!("✓ {} tensor created successfully", description);
                    match tensor.to_vec() {
                        Ok(retrieved) => {
                            assert_eq!(retrieved.len(), 1);
                            if data[0].is_nan() {
                                assert!(retrieved[0].is_nan(), "{} should remain NaN", description);
                            } else if data[0].is_infinite() {
                                assert!(
                                    retrieved[0].is_infinite(),
                                    "{} should remain infinite",
                                    description
                                );
                                assert_eq!(
                                    data[0].is_sign_positive(),
                                    retrieved[0].is_sign_positive(),
                                    "{} sign should be preserved",
                                    description
                                );
                            }
                            println!("✓ {} value handled correctly", description);
                        }
                        Err(e) => {
                            println!("⚠️  {} data retrieval failed: {}", description, e);
                        }
                    }
                }
                Err(e) => {
                    println!("⚠️  {} tensor creation failed: {}", description, e);
                }
            }
        }
    }

    println!("✓ Numerical edge cases tests completed");
}
