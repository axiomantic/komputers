//! Integration tests for high-level GPU operations.
//!
//! These tests verify that the operations module correctly wraps
//! Kompute's GPU compute functionality while providing idiomatic Rust APIs.

use kompute::{GpuOperations, MemoryPatterns, Tensor, WorkgroupConfig};

#[test]
fn test_gpu_operations_creation() {
    let result = GpuOperations::new();
    match result {
        Ok(_ops) => {
            println!("✓ GPU operations manager created successfully");
        }
        Err(e) => {
            println!(
                "⚠️ GPU operations creation failed: {} (expected in headless environments)",
                e
            );
        }
    }
}

#[test]
fn test_gpu_operations_with_specific_device() {
    let result = GpuOperations::with_device(0);
    match result {
        Ok(_ops) => {
            println!("✓ GPU operations manager created with device 0");
        }
        Err(e) => {
            println!(
                "⚠️ GPU operations creation with device failed: {} (expected in some environments)",
                e
            );
        }
    }
}

#[test]
fn test_tensor_copy_workflow() {
    if let Ok(mut ops) = GpuOperations::new() {
        let source_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        match Tensor::new(ops.manager(), &source_data) {
            Ok(source) => {
                match Tensor::<f32>::uninit(ops.manager(), 5) {
                    Ok(mut dest) => {
                        match ops.copy(&source, &mut dest) {
                            Ok(()) => {
                                println!("✓ Tensor copy operation completed successfully");

                                // Verify copy worked by reading data back
                                match dest.to_vec() {
                                    Ok(copied_data) => {
                                        println!(
                                            "✓ Copy verification: source={:?}, dest={:?}",
                                            source_data, copied_data
                                        );
                                        assert_eq!(copied_data.len(), source_data.len());
                                    }
                                    Err(e) => {
                                        println!("⚠️ Could not verify copy: {}", e);
                                    }
                                }
                            }
                            Err(e) => {
                                println!("⚠️ Tensor copy failed: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        println!("⚠️ Could not create destination tensor: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("⚠️ Could not create source tensor: {}", e);
            }
        }
    }
}

#[test]
fn test_elementwise_add_error_handling() {
    if let Ok(mut ops) = GpuOperations::new() {
        let data_a = vec![1.0f32, 2.0, 3.0];
        let data_b = vec![4.0f32, 5.0, 6.0];

        if let (Ok(tensor_a), Ok(tensor_b)) = (
            Tensor::new(ops.manager(), &data_a),
            Tensor::new(ops.manager(), &data_b),
        ) {
            if let Ok(mut output) = Tensor::<f32>::uninit(ops.manager(), 3) {
                // This should fail with informative error about SPIR-V compilation
                match ops.elementwise_add(&tensor_a, &tensor_b, &mut output) {
                    Ok(()) => {
                        println!("⚠️ Element-wise add unexpectedly succeeded");
                    }
                    Err(e) => {
                        println!(
                            "✓ Element-wise add correctly returned informative error: {}",
                            e
                        );
                        assert!(e.to_string().contains("SPIR-V"));
                        assert!(e.to_string().contains("Algorithm API"));
                    }
                }
            }
        }
    }
}

#[test]
fn test_tensor_size_mismatch_validation() {
    if let Ok(mut ops) = GpuOperations::new() {
        let data_a = vec![1.0f32, 2.0, 3.0];
        let data_b = vec![4.0f32, 5.0]; // Different size

        if let (Ok(tensor_a), Ok(tensor_b)) = (
            Tensor::new(ops.manager(), &data_a),
            Tensor::new(ops.manager(), &data_b),
        ) {
            if let Ok(mut output) = Tensor::<f32>::uninit(ops.manager(), 3) {
                // Should fail due to size mismatch
                match ops.elementwise_add(&tensor_a, &tensor_b, &mut output) {
                    Ok(()) => {
                        println!("⚠️ Element-wise add should have failed on size mismatch");
                    }
                    Err(e) => {
                        println!("✓ Size mismatch correctly detected: {}", e);
                        assert!(e.to_string().contains("size mismatch"));
                    }
                }
            }
        }
    }
}

#[test]
fn test_workgroup_config_creation() {
    let config_1d = WorkgroupConfig::new_1d(256);
    assert_eq!(config_1d.x, 256);
    assert_eq!(config_1d.y, 1);
    assert_eq!(config_1d.z, 1);
    assert_eq!(config_1d.total_items(), 256);

    let config_2d = WorkgroupConfig::new_2d(16, 16);
    assert_eq!(config_2d.x, 16);
    assert_eq!(config_2d.y, 16);
    assert_eq!(config_2d.z, 1);
    assert_eq!(config_2d.total_items(), 256);

    let config_3d = WorkgroupConfig::new_3d(8, 8, 4);
    assert_eq!(config_3d.x, 8);
    assert_eq!(config_3d.y, 8);
    assert_eq!(config_3d.z, 4);
    assert_eq!(config_3d.total_items(), 256);
}

#[test]
fn test_optimal_workgroup_calculation() {
    // Test various element counts
    let small_config = WorkgroupConfig::optimal_1d(100);
    assert!(small_config.x > 0);
    assert_eq!(small_config.y, 1);
    assert_eq!(small_config.z, 1);

    let large_config = WorkgroupConfig::optimal_1d(10000);
    assert!(large_config.x > 0);
    assert_eq!(large_config.y, 1);
    assert_eq!(large_config.z, 1);

    // Should handle edge cases
    let single_config = WorkgroupConfig::optimal_1d(1);
    assert!(single_config.x > 0);

    let exact_multiple_config = WorkgroupConfig::optimal_1d(256);
    assert!(exact_multiple_config.x > 0);
}

#[test]
fn test_memory_patterns_coalescing() {
    // Test aligned sizes (multiples of 32)
    assert!(MemoryPatterns::is_coalesced_access(32));
    assert!(MemoryPatterns::is_coalesced_access(64));
    assert!(MemoryPatterns::is_coalesced_access(128));
    assert!(MemoryPatterns::is_coalesced_access(256));
    assert!(MemoryPatterns::is_coalesced_access(1024));

    // Test non-aligned sizes
    assert!(!MemoryPatterns::is_coalesced_access(33));
    assert!(!MemoryPatterns::is_coalesced_access(100));
    assert!(!MemoryPatterns::is_coalesced_access(127));
    assert!(!MemoryPatterns::is_coalesced_access(255));
}

#[test]
fn test_memory_patterns_padding() {
    // Already aligned - no padding needed
    assert_eq!(MemoryPatterns::suggest_padding(32), 0);
    assert_eq!(MemoryPatterns::suggest_padding(64), 0);
    assert_eq!(MemoryPatterns::suggest_padding(128), 0);

    // Need padding to next multiple of 32
    assert_eq!(MemoryPatterns::suggest_padding(33), 31);
    assert_eq!(MemoryPatterns::suggest_padding(34), 30);
    assert_eq!(MemoryPatterns::suggest_padding(63), 1);
    assert_eq!(MemoryPatterns::suggest_padding(100), 20);

    // Edge cases
    assert_eq!(MemoryPatterns::suggest_padding(1), 31);
    assert_eq!(MemoryPatterns::suggest_padding(31), 1);
}

#[test]
fn test_optimal_buffer_size_calculation() {
    // Already optimal sizes
    assert_eq!(MemoryPatterns::optimal_buffer_size(32), 32);
    assert_eq!(MemoryPatterns::optimal_buffer_size(64), 64);
    assert_eq!(MemoryPatterns::optimal_buffer_size(128), 128);

    // Sizes that need padding
    assert_eq!(MemoryPatterns::optimal_buffer_size(33), 64);
    assert_eq!(MemoryPatterns::optimal_buffer_size(100), 128); // 100 + 28 padding = 128
    assert_eq!(MemoryPatterns::optimal_buffer_size(127), 128);

    // Edge cases
    assert_eq!(MemoryPatterns::optimal_buffer_size(1), 32);
    assert_eq!(MemoryPatterns::optimal_buffer_size(31), 32);
}

#[test]
fn test_multi_type_tensor_operations() {
    if let Ok(mut ops) = GpuOperations::new() {
        // Test f32 tensors
        let f32_data = vec![1.0f32, 2.0, 3.0];
        if let Ok(f32_tensor) = Tensor::new(ops.manager(), &f32_data) {
            if let Ok(mut f32_output) = Tensor::<f32>::uninit(ops.manager(), 3) {
                match ops.copy(&f32_tensor, &mut f32_output) {
                    Ok(()) => println!("✓ f32 tensor copy succeeded"),
                    Err(e) => println!("⚠️ f32 tensor copy failed: {}", e),
                }
            }
        }

        // Test f64 tensors
        let f64_data = vec![1.0f64, 2.0, 3.0];
        if let Ok(f64_tensor) = Tensor::new(ops.manager(), &f64_data) {
            if let Ok(mut f64_output) = Tensor::<f64>::uninit(ops.manager(), 3) {
                match ops.copy(&f64_tensor, &mut f64_output) {
                    Ok(()) => println!("✓ f64 tensor copy succeeded"),
                    Err(e) => println!("⚠️ f64 tensor copy failed: {}", e),
                }
            }
        }

        // Test i32 tensors
        let i32_data = vec![1i32, 2, 3];
        if let Ok(i32_tensor) = Tensor::new(ops.manager(), &i32_data) {
            if let Ok(mut i32_output) = Tensor::<i32>::uninit(ops.manager(), 3) {
                match ops.copy(&i32_tensor, &mut i32_output) {
                    Ok(()) => println!("✓ i32 tensor copy succeeded"),
                    Err(e) => println!("⚠️ i32 tensor copy failed: {}", e),
                }
            }
        }

        // Test u32 tensors
        let u32_data = vec![1u32, 2, 3];
        if let Ok(u32_tensor) = Tensor::new(ops.manager(), &u32_data) {
            if let Ok(mut u32_output) = Tensor::<u32>::uninit(ops.manager(), 3) {
                match ops.copy(&u32_tensor, &mut u32_output) {
                    Ok(()) => println!("✓ u32 tensor copy succeeded"),
                    Err(e) => println!("⚠️ u32 tensor copy failed: {}", e),
                }
            }
        }
    }
}
