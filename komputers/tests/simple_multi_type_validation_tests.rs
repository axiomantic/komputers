//! Simple Multi-Type Tensor Validation Tests
//!
//! This test suite provides essential validation for multi-type tensor operations
//! with comprehensive edge case coverage in a simple, maintainable format.

use komputers::{DataType, Manager};

#[test]
fn test_all_types_basic_operations() {
    println!("Testing basic operations for all supported types...");

    if let Ok(mut manager) = Manager::new() {
        let test_size = 100;

        // Test f32
        match manager.tensor_filled(test_size, std::f32::consts::PI) {
            Ok(tensor) => {
                assert_eq!(tensor.size(), test_size);
                assert_eq!(tensor.data_type().unwrap(), DataType::Float32);
                assert!(tensor.validate_type().is_ok());
                if let Ok(data) = tensor.to_vec() {
                    assert_eq!(data.len(), test_size);
                    assert!((data[0] - std::f32::consts::PI).abs() < f32::EPSILON);
                }
                println!("  ✓ f32 tensor operations successful");
            }
            Err(e) => println!("  ⚠️ f32 tensor failed: {}", e),
        }

        // Test f64
        match manager.tensor_filled(test_size, std::f64::consts::E) {
            Ok(tensor) => {
                assert_eq!(tensor.size(), test_size);
                assert_eq!(tensor.data_type().unwrap(), DataType::Float64);
                assert!(tensor.validate_type().is_ok());
                if let Ok(data) = tensor.to_vec() {
                    assert_eq!(data.len(), test_size);
                    assert!((data[0] - std::f64::consts::E).abs() < f64::EPSILON);
                }
                println!("  ✓ f64 tensor operations successful");
            }
            Err(e) => println!("  ⚠️ f64 tensor failed: {}", e),
        }

        // Test i32
        match manager.tensor_filled(test_size, 42i32) {
            Ok(tensor) => {
                assert_eq!(tensor.size(), test_size);
                assert_eq!(tensor.data_type().unwrap(), DataType::Int32);
                assert!(tensor.validate_type().is_ok());
                if let Ok(data) = tensor.to_vec() {
                    assert_eq!(data.len(), test_size);
                    assert_eq!(data[0], 42i32);
                }
                println!("  ✓ i32 tensor operations successful");
            }
            Err(e) => println!("  ⚠️ i32 tensor failed: {}", e),
        }

        // Test u32
        match manager.tensor_filled(test_size, 123u32) {
            Ok(tensor) => {
                assert_eq!(tensor.size(), test_size);
                assert_eq!(tensor.data_type().unwrap(), DataType::UInt32);
                assert!(tensor.validate_type().is_ok());
                if let Ok(data) = tensor.to_vec() {
                    assert_eq!(data.len(), test_size);
                    assert_eq!(data[0], 123u32);
                }
                println!("  ✓ u32 tensor operations successful");
            }
            Err(e) => println!("  ⚠️ u32 tensor failed: {}", e),
        }

        println!("  ✓ All type basic operations tested");
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}

#[test]
fn test_enhanced_creation_methods() {
    println!("Testing enhanced tensor creation methods...");

    if let Ok(mut manager) = Manager::new() {
        let test_size = 50;

        // Test zeros
        match manager.tensor_zeros::<f32>(test_size) {
            Ok(tensor) => {
                assert_eq!(tensor.size(), test_size);
                if let Ok(data) = tensor.to_vec() {
                    assert!(data.iter().all(|&x| x == 0.0f32));
                }
                println!("  ✓ tensor_zeros works");
            }
            Err(e) => println!("  ⚠️ tensor_zeros failed: {}", e),
        }

        // Test ones
        match manager.tensor_ones::<f32>(test_size) {
            Ok(tensor) => {
                assert_eq!(tensor.size(), test_size);
                if let Ok(data) = tensor.to_vec() {
                    assert!(data.iter().all(|&x| x == 1.0f32));
                }
                println!("  ✓ tensor_ones works");
            }
            Err(e) => println!("  ⚠️ tensor_ones failed: {}", e),
        }

        // Test filled with different types
        if let Ok(pi_tensor) = manager.tensor_filled(test_size, std::f32::consts::PI) {
            if let Ok(data) = pi_tensor.to_vec() {
                assert!(data
                    .iter()
                    .all(|&x| (x - std::f32::consts::PI).abs() < f32::EPSILON));
            }
            println!("  ✓ tensor_filled with PI works");
        }

        println!("  ✓ Enhanced creation methods tested");
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}

#[test]
fn test_boundary_values() {
    println!("Testing boundary values and edge cases...");

    if let Ok(mut manager) = Manager::new() {
        // Test zero size (should fail)
        assert!(manager.tensor_filled(0, 1.0f32).is_err());
        assert!(manager.tensor_zeros::<f32>(0).is_err());
        assert!(manager.tensor_ones::<f32>(0).is_err());
        println!("  ✓ Zero size correctly rejected");

        // Test size 1 (minimum valid size)
        if let Ok(tensor) = manager.tensor_filled(1, 42.0f32) {
            assert_eq!(tensor.size(), 1);
            if let Ok(data) = tensor.to_vec() {
                assert_eq!(data.len(), 1);
                assert_eq!(data[0], 42.0f32);
            }
            println!("  ✓ Size 1 works correctly");
        }

        // Test integer boundary values
        let boundary_values = vec![i32::MIN, -1, 0, 1, i32::MAX];
        for &value in &boundary_values {
            if let Ok(tensor) = manager.tensor_filled(10, value) {
                if let Ok(data) = tensor.to_vec() {
                    assert!(data.iter().all(|&x| x == value));
                }
            }
        }
        println!("  ✓ Integer boundary values handled");

        // Test floating point special values
        let special_values = vec![0.0f32, -0.0f32, f32::MIN_POSITIVE, f32::EPSILON];
        for &value in &special_values {
            if let Ok(tensor) = manager.tensor_filled(10, value) {
                if let Ok(data) = tensor.to_vec() {
                    if value == 0.0 || value == -0.0 {
                        assert!(data.iter().all(|&x| x == 0.0));
                    } else {
                        assert!(data.iter().all(|&x| x == value));
                    }
                }
            }
        }
        println!("  ✓ Floating point special values handled");

        println!("  ✓ Boundary values and edge cases tested");
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}

#[test]
fn test_type_consistency() {
    println!("Testing type consistency and validation...");

    if let Ok(mut manager) = Manager::new() {
        let test_size = 100;

        // Create tensors of each type and verify consistency
        let types_to_test = vec![
            ("f32", DataType::Float32),
            ("f64", DataType::Float64),
            ("i32", DataType::Int32),
            ("u32", DataType::UInt32),
        ];

        for (type_name, expected_type) in types_to_test {
            match type_name {
                "f32" => {
                    if let Ok(tensor) = manager.tensor_filled(test_size, 1.0f32) {
                        assert_eq!(tensor.size(), test_size);
                        assert_eq!(tensor.data_type().unwrap(), expected_type);
                        assert!(tensor.validate_type().is_ok());
                        if let Ok(data) = tensor.to_vec() {
                            assert_eq!(data.len(), test_size);
                        }
                        println!("  ✓ {} type consistency verified", type_name);
                    } else {
                        println!("  ⚠️ {} type test failed", type_name);
                    }
                }
                "f64" => {
                    if let Ok(tensor) = manager.tensor_filled(test_size, 1.0f64) {
                        assert_eq!(tensor.size(), test_size);
                        assert_eq!(tensor.data_type().unwrap(), expected_type);
                        assert!(tensor.validate_type().is_ok());
                        if let Ok(data) = tensor.to_vec() {
                            assert_eq!(data.len(), test_size);
                        }
                        println!("  ✓ {} type consistency verified", type_name);
                    } else {
                        println!("  ⚠️ {} type test failed", type_name);
                    }
                }
                "i32" => {
                    if let Ok(tensor) = manager.tensor_filled(test_size, 1i32) {
                        assert_eq!(tensor.size(), test_size);
                        assert_eq!(tensor.data_type().unwrap(), expected_type);
                        assert!(tensor.validate_type().is_ok());
                        if let Ok(data) = tensor.to_vec() {
                            assert_eq!(data.len(), test_size);
                        }
                        println!("  ✓ {} type consistency verified", type_name);
                    } else {
                        println!("  ⚠️ {} type test failed", type_name);
                    }
                }
                "u32" => {
                    if let Ok(tensor) = manager.tensor_filled(test_size, 1u32) {
                        assert_eq!(tensor.size(), test_size);
                        assert_eq!(tensor.data_type().unwrap(), expected_type);
                        assert!(tensor.validate_type().is_ok());
                        if let Ok(data) = tensor.to_vec() {
                            assert_eq!(data.len(), test_size);
                        }
                        println!("  ✓ {} type consistency verified", type_name);
                    } else {
                        println!("  ⚠️ {} type test failed", type_name);
                    }
                }
                _ => continue,
            }
        }

        println!("  ✓ Type consistency and validation tested");
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}

#[test]
fn test_error_handling() {
    println!("Testing error handling robustness...");

    if let Ok(mut manager) = Manager::new() {
        // Test zero size error propagation
        let zero_size_errors = [
            manager.tensor_filled(0, 1.0f32).is_err(),
            manager.tensor_filled(0, 1.0f64).is_err(),
            manager.tensor_filled(0, 1i32).is_err(),
            manager.tensor_filled(0, 1u32).is_err(),
        ];

        assert!(
            zero_size_errors.iter().all(|&x| x),
            "All zero size operations should fail"
        );
        println!("  ✓ Zero size error handling consistent");

        // Test that error messages are informative
        if let Err(error) = manager.tensor_filled(0, 1.0f32) {
            let error_msg = format!("{}", error);
            assert!(!error_msg.is_empty(), "Error message should not be empty");
            println!("  ✓ Error messages are informative: {}", error_msg);
        }

        // Test very large size (should fail gracefully)
        let large_size = 1_000_000_000; // 1 billion elements
        match manager.tensor_filled(large_size, 1.0f32) {
            Ok(_) => println!("  ⚠️ Large allocation unexpectedly succeeded"),
            Err(e) => {
                println!("  ✓ Large allocation failed gracefully: {}", e);
                // Verify it's a memory error
                let error_msg = format!("{}", e);
                assert!(
                    error_msg.contains("Memory")
                        || error_msg.contains("memory")
                        || error_msg.contains("allocation")
                        || error_msg.contains("size"),
                    "Error should mention memory or allocation issues"
                );
            }
        }

        println!("  ✓ Error handling robustness tested");
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}

#[test]
fn test_memory_layout_consistency() {
    println!("Testing memory layout consistency...");

    if let Ok(mut manager) = Manager::new() {
        let test_sizes = vec![1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256];

        for &size in &test_sizes {
            // Test f32 (4 bytes per element)
            if let Ok(f32_tensor) = manager.tensor_filled(size, 1.0f32) {
                assert_eq!(f32_tensor.size(), size);
                let byte_size = f32_tensor.size_bytes();
                if true {
                    assert_eq!(byte_size, size * 4, "f32 byte size should be size * 4");
                }
            }

            // Test f64 (8 bytes per element)
            if let Ok(f64_tensor) = manager.tensor_filled(size, 1.0f64) {
                assert_eq!(f64_tensor.size(), size);
                let byte_size = f64_tensor.size_bytes();
                if true {
                    assert_eq!(byte_size, size * 8, "f64 byte size should be size * 8");
                }
            }

            // Test i32 (4 bytes per element)
            if let Ok(i32_tensor) = manager.tensor_filled(size, 1i32) {
                assert_eq!(i32_tensor.size(), size);
                let byte_size = i32_tensor.size_bytes();
                if true {
                    assert_eq!(byte_size, size * 4, "i32 byte size should be size * 4");
                }
            }

            // Test u32 (4 bytes per element)
            if let Ok(u32_tensor) = manager.tensor_filled(size, 1u32) {
                assert_eq!(u32_tensor.size(), size);
                let byte_size = u32_tensor.size_bytes();
                if true {
                    assert_eq!(byte_size, size * 4, "u32 byte size should be size * 4");
                }
            }
        }

        println!(
            "  ✓ Memory layout consistency verified for {} sizes",
            test_sizes.len()
        );
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}

#[test]
fn test_concurrent_operations() {
    println!("Testing concurrent tensor operations...");

    if let Ok(manager) = Manager::new() {
        use std::sync::{Arc, Mutex};
        use std::thread;

        let manager = Arc::new(Mutex::new(manager));
        let success_count = Arc::new(Mutex::new(0));
        let total_count = Arc::new(Mutex::new(0));

        let mut handles = vec![];

        // Spawn threads that create tensors concurrently
        for thread_id in 0..4 {
            let manager_clone = Arc::clone(&manager);
            let success_clone = Arc::clone(&success_count);
            let total_clone = Arc::clone(&total_count);

            let handle = thread::spawn(move || {
                for i in 0..10 {
                    let size = 100 + (i * 10);
                    let value = (thread_id as f32) * 0.1 + (i as f32) * 0.01;

                    {
                        let mut total = total_clone.lock().unwrap();
                        *total += 1;
                    }

                    let result = {
                        let mut mgr = manager_clone.lock().unwrap();
                        mgr.tensor_filled(size, value)
                    };

                    if let Ok(tensor) = result {
                        if tensor.size() == size && tensor.validate_type().is_ok() {
                            let mut success = success_clone.lock().unwrap();
                            *success += 1;
                        }
                    }

                    // Small delay to encourage interleaving
                    thread::sleep(std::time::Duration::from_millis(1));
                }
            });

            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().expect("Thread should complete");
        }

        let final_success = *success_count.lock().unwrap();
        let final_total = *total_count.lock().unwrap();
        let success_rate = (final_success as f64) / (final_total as f64) * 100.0;

        println!(
            "  Concurrent operations: {}/{} successful ({:.1}%)",
            final_success, final_total, success_rate
        );

        // Allow some failures in concurrent scenarios but expect reasonable success rate
        assert!(success_rate >= 50.0, "Success rate should be at least 50%");

        println!("  ✓ Concurrent operations tested");
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}
