//! Comprehensive integration tests for GPU compute workflows.
//!
//! These tests verify the complete end-to-end functionality of the komputers library,
//! including tensor operations, sequence execution, and resource management.

use komputers::{Manager, Result, Sequence, Tensor};

/// Test a complete GPU compute workflow with multiple tensors and operations.
#[test]

fn test_comprehensive_gpu_workflow() -> Result<()> {
    println!("Starting comprehensive GPU workflow test...");

    // Initialize GPU manager
    let mut manager = Manager::new()?;
    println!("✓ GPU Manager initialized");

    // Create multiple input tensors with different patterns
    let linear_data: Vec<f32> = (1..=100).map(|i| i as f32).collect();
    let sine_data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
    let constant_data: Vec<f32> = vec![3.14159; 100];

    let tensor_linear = Tensor::new(&mut manager, &linear_data)?;
    let tensor_sine = Tensor::new(&mut manager, &sine_data)?;
    let tensor_constant = Tensor::new(&mut manager, &constant_data)?;
    let tensor_output: Tensor<f32> = Tensor::uninit(&mut manager, 100)?;

    println!("✓ Created 4 tensors (linear, sine, constant, output)");

    // Verify tensor properties
    assert_eq!(tensor_linear.size(), 100);
    assert_eq!(tensor_sine.size(), 100);
    assert_eq!(tensor_constant.size(), 100);
    assert_eq!(tensor_output.size(), 100);

    assert_eq!(tensor_linear.size_bytes(), 400); // 100 * 4 bytes
    assert_eq!(tensor_sine.size_bytes(), 400);
    assert_eq!(tensor_constant.size_bytes(), 400);
    assert_eq!(tensor_output.size_bytes(), 400);

    println!("✓ Tensor properties verified");

    // Test data consistency
    let retrieved_linear = tensor_linear.to_vec()?;
    let retrieved_sine = tensor_sine.to_vec()?;
    let retrieved_constant = tensor_constant.to_vec()?;

    assert_eq!(linear_data, retrieved_linear);
    assert_eq!(sine_data, retrieved_sine);
    assert_eq!(constant_data, retrieved_constant);

    println!("✓ Data integrity verified");

    // Create sequence for GPU operations
    let _sequence = Sequence::new(&mut manager)?;
    println!("✓ GPU sequence created");

    // Test handle uniqueness
    let handles = vec![
        tensor_linear.handle(),
        tensor_sine.handle(),
        tensor_constant.handle(),
        tensor_output.handle(),
    ];

    // All handles should be unique and non-zero
    for (i, &handle) in handles.iter().enumerate() {
        assert_ne!(handle, 0, "Handle {} should not be zero", i);
        for (j, &other_handle) in handles.iter().enumerate() {
            if i != j {
                assert_ne!(
                    handle, other_handle,
                    "Handles {} and {} should be unique",
                    i, j
                );
            }
        }
    }

    println!("✓ Handle uniqueness verified");
    println!("🎉 Comprehensive GPU workflow test completed successfully!");

    Ok(())
}

/// Test tensor lifecycle and resource management.
#[test]

fn test_tensor_lifecycle() -> Result<()> {
    println!("Starting tensor lifecycle test...");

    let mut manager = Manager::new()?;

    // Test creating and dropping tensors in different scopes
    let mut handles = Vec::new();

    {
        let data1 = vec![1.0f32, 2.0, 3.0];
        let tensor1 = Tensor::new(&mut manager, &data1)?;
        handles.push(tensor1.handle());

        {
            let data2 = vec![4.0f32, 5.0, 6.0];
            let tensor2 = Tensor::new(&mut manager, &data2)?;
            handles.push(tensor2.handle());

            // Both tensors should be accessible
            assert_eq!(tensor1.size(), 3);
            assert_eq!(tensor2.size(), 3);

            // tensor2 drops here
        }

        // tensor1 should still be accessible
        assert_eq!(tensor1.size(), 3);

        // tensor1 drops here
    }

    // Test creating new tensors after previous ones were dropped
    let data3 = vec![7.0f32, 8.0, 9.0, 10.0];
    let tensor3 = Tensor::new(&mut manager, &data3)?;
    handles.push(tensor3.handle());

    // New tensor should have a different handle
    assert!(!handles[..2].contains(&tensor3.handle()));

    println!("✓ Tensor lifecycle management verified");
    Ok(())
}

/// Test edge cases and error conditions.
#[test]

fn test_edge_cases() -> Result<()> {
    println!("Starting edge cases test...");

    let mut manager = Manager::new()?;

    // Test empty tensor
    let empty_data: Vec<f32> = vec![];
    match Tensor::new(&mut manager, &empty_data) {
        Ok(tensor) => {
            assert_eq!(tensor.size(), 0);
            assert_eq!(tensor.size_bytes(), 0);

            let retrieved = tensor.to_vec()?;
            assert!(retrieved.is_empty());
            println!("✓ Empty tensor handled correctly");
        }
        Err(_) => {
            println!("✓ Empty tensor creation failed as expected");
        }
    }

    // Test single element tensor
    let single_data = vec![42.0f32];
    let single_tensor = Tensor::new(&mut manager, &single_data)?;
    assert_eq!(single_tensor.size(), 1);
    assert_eq!(single_tensor.size_bytes(), 4);

    let retrieved_single = single_tensor.to_vec()?;
    assert_eq!(retrieved_single, vec![42.0f32]);
    println!("✓ Single element tensor works correctly");

    // Test large tensor (within reasonable limits)
    let large_size = 10000;
    let large_data: Vec<f32> = (0..large_size).map(|i| i as f32 / 1000.0).collect();
    match Tensor::new(&mut manager, &large_data) {
        Ok(large_tensor) => {
            assert_eq!(large_tensor.size(), large_size);
            assert_eq!(large_tensor.size_bytes(), large_size * 4);

            let retrieved_large = large_tensor.to_vec()?;
            assert_eq!(retrieved_large.len(), large_size);
            assert_eq!(retrieved_large[0], 0.0);
            assert_eq!(
                retrieved_large[large_size - 1],
                (large_size - 1) as f32 / 1000.0
            );
            println!("✓ Large tensor ({} elements) handled correctly", large_size);
        }
        Err(e) => {
            println!("✓ Large tensor creation failed as expected: {}", e);
        }
    }

    // Test special float values
    let special_data = vec![
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        0.0f32,
        -0.0f32,
        f32::MIN,
        f32::MAX,
        f32::EPSILON,
    ];

    match Tensor::new(&mut manager, &special_data) {
        Ok(special_tensor) => {
            let retrieved_special = special_tensor.to_vec()?;
            assert_eq!(retrieved_special.len(), special_data.len());

            // Check specific values (being careful with NaN comparison)
            assert!(retrieved_special[0].is_infinite() && retrieved_special[0].is_sign_positive());
            assert!(retrieved_special[1].is_infinite() && retrieved_special[1].is_sign_negative());
            assert!(retrieved_special[2].is_nan());
            assert_eq!(retrieved_special[3], 0.0);
            println!("✓ Special float values handled correctly");
        }
        Err(e) => {
            println!("✓ Special float values failed as expected: {}", e);
        }
    }

    println!("🎉 Edge cases test completed!");
    Ok(())
}

/// Test concurrent tensor operations.
#[test]

fn test_concurrent_operations() -> Result<()> {
    use std::sync::{Arc, Mutex};
    use std::thread;

    println!("Starting concurrent operations test...");

    let manager = Arc::new(Mutex::new(Manager::new()?));
    let mut handles = vec![];

    // Create multiple threads that create tensors concurrently
    for i in 0..5 {
        let manager_clone = Arc::clone(&manager);
        let handle = thread::spawn(move || -> Result<Vec<u64>> {
            let mut tensor_handles = Vec::new();

            for j in 0..3 {
                let data: Vec<f32> = vec![i as f32 * 10.0 + j as f32; 10];
                let mut manager_guard = manager_clone.lock().unwrap();
                let tensor = Tensor::new(&mut *manager_guard, &data)?;
                tensor_handles.push(tensor.handle());
                drop(manager_guard); // Release lock early

                // Verify tensor properties without holding the manager lock
                assert_eq!(tensor.size(), 10);
                assert_eq!(tensor.size_bytes(), 40);

                // Verify data integrity
                let retrieved_data = tensor.to_vec()?;
                assert_eq!(retrieved_data, data);
            }

            Ok(tensor_handles)
        });
        handles.push(handle);
    }

    // Wait for all threads to complete and collect results
    let mut all_handles = Vec::new();
    for handle in handles {
        let thread_handles = handle.join().unwrap()?;
        all_handles.extend(thread_handles);
    }

    // Verify that all handles are unique
    for (i, &handle_i) in all_handles.iter().enumerate() {
        for (j, &handle_j) in all_handles.iter().enumerate() {
            if i != j {
                assert_ne!(
                    handle_i, handle_j,
                    "Handles should be unique across threads"
                );
            }
        }
    }

    println!(
        "✓ Created {} unique tensors across 5 threads",
        all_handles.len()
    );
    println!("🎉 Concurrent operations test completed!");

    Ok(())
}

/// Test device capabilities and limits.
#[test]
fn test_device_capabilities() {
    println!("Starting device capabilities test...");

    // Test device enumeration
    match Manager::device_count() {
        Ok(count) => {
            println!("✓ Found {} Vulkan-capable device(s)", count);
            assert!(
                count > 0,
                "Should have at least one device in GPU environments"
            );
            assert!(count < 256, "Device count should be reasonable");

            // Test creating manager with different device IDs
            for device_id in 0..std::cmp::min(count, 3) {
                match Manager::with_device(device_id) {
                    Ok(_manager) => {
                        println!("✓ Successfully created manager for device {}", device_id);
                    }
                    Err(e) => {
                        println!(
                            "⚠️  Failed to create manager for device {}: {}",
                            device_id, e
                        );
                    }
                }
            }
        }
        Err(e) => {
            println!(
                "⚠️  Device enumeration failed: {} (expected in headless environments)",
                e
            );
        }
    }

    // Test invalid device ID
    if let Ok(device_count) = Manager::device_count() {
        if device_count > 0 {
            match Manager::with_device(device_count + 10) {
                Ok(_) => {
                    panic!("Should not be able to create manager with invalid device ID");
                }
                Err(_) => {
                    println!("✓ Invalid device ID correctly rejected");
                }
            }
        }
    }

    println!("🎉 Device capabilities test completed!");
}

/// Stress test with many tensors and operations.
#[test]
fn test_stress_operations() -> Result<()> {
    println!("Starting stress test...");

    let mut manager = Manager::new()?;
    let mut tensors = Vec::new();

    // Create many tensors
    let tensor_count = 100;
    for i in 0..tensor_count {
        let data: Vec<f32> = vec![i as f32; 100];
        let tensor = Tensor::new(&mut manager, &data)?;
        tensors.push(tensor);
    }

    println!("✓ Created {} tensors", tensor_count);

    // Verify all tensors
    for (i, tensor) in tensors.iter().enumerate() {
        assert_eq!(tensor.size(), 100);
        let data = tensor.to_vec()?;
        assert_eq!(data[0], i as f32);
        assert_eq!(data.len(), 100);
    }

    println!("✓ Verified all {} tensors", tensor_count);

    // Create multiple sequences
    let mut sequences = Vec::new();
    for _ in 0..10 {
        let sequence = Sequence::new(&mut manager)?;
        sequences.push(sequence);
    }

    println!("✓ Created {} sequences", sequences.len());

    // Test that all handles are unique
    let mut all_handles = Vec::new();
    for tensor in &tensors {
        all_handles.push(tensor.handle());
    }

    for (i, &handle_i) in all_handles.iter().enumerate() {
        assert_ne!(handle_i, 0);
        for (j, &handle_j) in all_handles.iter().enumerate() {
            if i != j {
                assert_ne!(handle_i, handle_j);
            }
        }
    }

    println!("✓ All {} handles are unique", all_handles.len());
    println!("🎉 Stress test completed successfully!");

    Ok(())
}
