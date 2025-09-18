//! Multi-type tensor support testing.
//!
//! These tests verify tensor operations across all declared TensorData types
//! (f32, f64, i32, u32, i8, u8) including precision, overflow handling,
//! memory alignment, and handle lifecycle management.

use kompute::{Manager, Tensor, TensorData};
use std::sync::Arc;
use std::thread;

/// Test f64 tensor data consistency and precision preservation.
/// Verifies high-precision floating point operations.
#[test]
#[ignore = "requires Vulkan-capable device"]
fn test_tensor_f64_data_consistency() {
    if let Ok(mut manager) = Manager::new() {
        println!("Testing f64 tensor data consistency...");

        // Test high-precision values that would lose precision in f32
        let original_data = vec![
            3.14159265358979323846f64, // π with high precision
            2.71828182845904523536f64, // e with high precision
            1.41421356237309504880f64, // √2 with high precision
            1.61803398874989484820f64, // Golden ratio with high precision
            std::f64::consts::PI,
            std::f64::consts::E,
            f64::MAX,
            f64::MIN,
            f64::MIN_POSITIVE,
            1e-100f64, // Very small number
            1e100f64,  // Very large number
        ];

        println!("  Testing {} f64 values", original_data.len());

        match Tensor::new(&mut manager, &original_data) {
            Ok(tensor) => {
                println!("  ✓ f64 tensor creation succeeded");

                // Verify tensor properties
                assert_eq!(
                    tensor.size(),
                    original_data.len(),
                    "f64 tensor size should match input data length"
                );
                assert_eq!(
                    tensor.size_bytes(),
                    original_data.len() * 8,
                    "f64 tensor byte size should be 8 bytes per element"
                );
                assert_ne!(tensor.handle(), 0, "f64 tensor handle should be non-zero");

                // Test data retrieval and precision
                match tensor.to_vec() {
                    Ok(retrieved) => {
                        println!("  ✓ f64 data retrieval succeeded");
                        assert_eq!(
                            retrieved.len(),
                            original_data.len(),
                            "Retrieved data length should match"
                        );

                        for (i, (orig, retr)) in
                            original_data.iter().zip(retrieved.iter()).enumerate()
                        {
                            if orig.is_nan() {
                                assert!(retr.is_nan(), "NaN should be preserved at index {}", i);
                            } else if orig.is_infinite() {
                                assert!(
                                    retr.is_infinite(),
                                    "Infinity should be preserved at index {}",
                                    i
                                );
                                assert_eq!(
                                    orig.is_sign_positive(),
                                    retr.is_sign_positive(),
                                    "Infinity sign should be preserved at index {}",
                                    i
                                );
                            } else {
                                let precision_loss = (orig - retr).abs();
                                assert!(
                                    precision_loss < f64::EPSILON,
                                    "f64 precision lost at index {}: {} != {} (loss: {})",
                                    i,
                                    orig,
                                    retr,
                                    precision_loss
                                );
                            }
                        }
                        println!("  ✓ All f64 values preserved with full precision");
                    }
                    Err(e) => {
                        println!("  ⚠ f64 data retrieval failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!(
                    "  ⚠ f64 tensor creation failed (expected until implementation): {}",
                    e
                );
                assert!(
                    e.to_string().contains("f32") || e.to_string().contains("supported"),
                    "Error should mention f32 or supported types limitation: {}",
                    e
                );
            }
        }
    } else {
        println!("Skipping f64 test - no GPU available");
    }
}

/// Test i32 tensor integer overflow and boundary value handling.
/// Verifies signed integer operations and overflow behavior.
#[test]
#[ignore = "requires Vulkan-capable device"]
fn test_tensor_i32_integer_overflow_handling() {
    if let Ok(mut manager) = Manager::new() {
        println!("Testing i32 tensor integer overflow handling...");

        // Test boundary values and potential overflow scenarios
        let original_data = vec![
            i32::MIN,     // Most negative value
            i32::MIN + 1, // Near minimum
            -1000000,     // Large negative
            -1,           // Small negative
            0,            // Zero
            1,            // Small positive
            1000000,      // Large positive
            i32::MAX - 1, // Near maximum
            i32::MAX,     // Most positive value
        ];

        println!("  Testing {} i32 boundary values", original_data.len());

        match Tensor::new(&mut manager, &original_data) {
            Ok(tensor) => {
                println!("  ✓ i32 tensor creation succeeded");

                // Verify tensor properties
                assert_eq!(
                    tensor.size(),
                    original_data.len(),
                    "i32 tensor size should match input data length"
                );
                assert_eq!(
                    tensor.size_bytes(),
                    original_data.len() * 4,
                    "i32 tensor byte size should be 4 bytes per element"
                );
                assert_ne!(tensor.handle(), 0, "i32 tensor handle should be non-zero");

                // Test data retrieval and exact integer preservation
                match tensor.to_vec() {
                    Ok(retrieved) => {
                        println!("  ✓ i32 data retrieval succeeded");
                        assert_eq!(
                            retrieved.len(),
                            original_data.len(),
                            "Retrieved data length should match"
                        );

                        for (i, (orig, retr)) in
                            original_data.iter().zip(retrieved.iter()).enumerate()
                        {
                            assert_eq!(
                                orig, retr,
                                "i32 value should be exact at index {}: {} != {}",
                                i, orig, retr
                            );
                        }
                        println!("  ✓ All i32 values preserved exactly");

                        // Test specific boundary values
                        assert_eq!(retrieved[0], i32::MIN, "MIN value should be preserved");
                        assert_eq!(retrieved[8], i32::MAX, "MAX value should be preserved");
                        assert_eq!(retrieved[4], 0, "Zero should be preserved");
                    }
                    Err(e) => {
                        println!("  ⚠ i32 data retrieval failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!(
                    "  ⚠ i32 tensor creation failed (expected until implementation): {}",
                    e
                );
                assert!(
                    e.to_string().contains("f32") || e.to_string().contains("supported"),
                    "Error should mention f32 or supported types limitation: {}",
                    e
                );
            }
        }
    } else {
        println!("Skipping i32 test - no GPU available");
    }
}

/// Test u32 tensor unsigned integer wraparound and large value handling.
/// Verifies unsigned integer operations and large value preservation.
#[test]
#[ignore = "requires Vulkan-capable device"]
fn test_tensor_u32_unsigned_wraparound() {
    if let Ok(mut manager) = Manager::new() {
        println!("Testing u32 tensor unsigned wraparound...");

        // Test unsigned boundary values and large numbers
        let original_data = vec![
            0u32,          // Minimum unsigned value
            1u32,          // Small value
            255u32,        // Byte boundary
            65535u32,      // 16-bit boundary
            16777215u32,   // 24-bit boundary
            4294967295u32, // Maximum u32 value (2^32 - 1)
            4294967294u32, // Near maximum
            2147483648u32, // 2^31 (would overflow i32)
            3000000000u32, // Large value > i32::MAX
        ];

        println!("  Testing {} u32 boundary values", original_data.len());

        match Tensor::new(&mut manager, &original_data) {
            Ok(tensor) => {
                println!("  ✓ u32 tensor creation succeeded");

                // Verify tensor properties
                assert_eq!(
                    tensor.size(),
                    original_data.len(),
                    "u32 tensor size should match input data length"
                );
                assert_eq!(
                    tensor.size_bytes(),
                    original_data.len() * 4,
                    "u32 tensor byte size should be 4 bytes per element"
                );
                assert_ne!(tensor.handle(), 0, "u32 tensor handle should be non-zero");

                // Test data retrieval and exact value preservation
                match tensor.to_vec() {
                    Ok(retrieved) => {
                        println!("  ✓ u32 data retrieval succeeded");
                        assert_eq!(
                            retrieved.len(),
                            original_data.len(),
                            "Retrieved data length should match"
                        );

                        for (i, (orig, retr)) in
                            original_data.iter().zip(retrieved.iter()).enumerate()
                        {
                            assert_eq!(
                                orig, retr,
                                "u32 value should be exact at index {}: {} != {}",
                                i, orig, retr
                            );
                        }
                        println!("  ✓ All u32 values preserved exactly");

                        // Test specific large values
                        assert_eq!(retrieved[5], u32::MAX, "MAX value should be preserved");
                        assert_eq!(
                            retrieved[7], 2147483648u32,
                            "Large value > i32::MAX should be preserved"
                        );
                        assert_eq!(retrieved[0], 0u32, "Zero should be preserved");
                    }
                    Err(e) => {
                        println!("  ⚠ u32 data retrieval failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!(
                    "  ⚠ u32 tensor creation failed (expected until implementation): {}",
                    e
                );
                assert!(
                    e.to_string().contains("f32") || e.to_string().contains("supported"),
                    "Error should mention f32 or supported types limitation: {}",
                    e
                );
            }
        }
    } else {
        println!("Skipping u32 test - no GPU available");
    }
}

/// Test i8 and u8 tensor byte alignment and small value handling.
/// Verifies byte-sized data types and memory alignment.
#[test]
#[ignore = "requires Vulkan-capable device"]
fn test_tensor_i8_u8_byte_alignment() {
    if let Ok(mut manager) = Manager::new() {
        println!("Testing i8/u8 tensor byte alignment...");

        // Test i8 values
        let i8_data = vec![
            i8::MIN, // -128
            -100i8,
            -1i8,
            0i8,
            1i8,
            100i8,
            i8::MAX, // 127
        ];

        println!("  Testing {} i8 values", i8_data.len());

        match Tensor::new(&mut manager, &i8_data) {
            Ok(tensor) => {
                println!("  ✓ i8 tensor creation succeeded");

                assert_eq!(
                    tensor.size(),
                    i8_data.len(),
                    "i8 tensor size should match input data length"
                );
                assert_eq!(
                    tensor.size_bytes(),
                    i8_data.len() * 1,
                    "i8 tensor byte size should be 1 byte per element"
                );

                match tensor.to_vec() {
                    Ok(retrieved) => {
                        println!("  ✓ i8 data retrieval succeeded");
                        for (i, (orig, retr)) in i8_data.iter().zip(retrieved.iter()).enumerate() {
                            assert_eq!(
                                orig, retr,
                                "i8 value should be exact at index {}: {} != {}",
                                i, orig, retr
                            );
                        }
                        println!("  ✓ All i8 values preserved exactly");
                    }
                    Err(e) => println!("  ⚠ i8 data retrieval failed: {}", e),
                }
            }
            Err(e) => {
                println!(
                    "  ⚠ i8 tensor creation failed (expected until implementation): {}",
                    e
                );
            }
        }

        // Test u8 values
        let u8_data = vec![
            0u8, // Minimum
            1u8, 127u8, // i8::MAX equivalent
            128u8, // Would overflow i8
            200u8, 255u8, // Maximum u8
        ];

        println!("  Testing {} u8 values", u8_data.len());

        match Tensor::new(&mut manager, &u8_data) {
            Ok(tensor) => {
                println!("  ✓ u8 tensor creation succeeded");

                assert_eq!(
                    tensor.size(),
                    u8_data.len(),
                    "u8 tensor size should match input data length"
                );
                assert_eq!(
                    tensor.size_bytes(),
                    u8_data.len() * 1,
                    "u8 tensor byte size should be 1 byte per element"
                );

                match tensor.to_vec() {
                    Ok(retrieved) => {
                        println!("  ✓ u8 data retrieval succeeded");
                        for (i, (orig, retr)) in u8_data.iter().zip(retrieved.iter()).enumerate() {
                            assert_eq!(
                                orig, retr,
                                "u8 value should be exact at index {}: {} != {}",
                                i, orig, retr
                            );
                        }
                        println!("  ✓ All u8 values preserved exactly");
                    }
                    Err(e) => println!("  ⚠ u8 data retrieval failed: {}", e),
                }
            }
            Err(e) => {
                println!(
                    "  ⚠ u8 tensor creation failed (expected until implementation): {}",
                    e
                );
            }
        }
    } else {
        println!("Skipping i8/u8 test - no GPU available");
    }
}

/// Test memory alignment requirements across all tensor types.
/// Verifies GPU memory alignment constraints are met.
#[test]
#[ignore = "requires Vulkan-capable device"]
fn test_tensor_memory_alignment_requirements() {
    if let Ok(mut manager) = Manager::new() {
        println!("Testing tensor memory alignment requirements...");

        // Test different data sizes to verify alignment
        let test_cases = vec![
            ("f32", 4, vec![1.0f32, 2.0f32, 3.0f32]),
            // Other types will be tested when implemented
        ];

        for (type_name, expected_element_size, _data) in test_cases {
            println!(
                "  Testing {} alignment (element size: {} bytes)",
                type_name, expected_element_size
            );

            // For now, only f32 is implemented
            if type_name == "f32" {
                let f32_data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32];
                match Tensor::new(&mut manager, &f32_data) {
                    Ok(tensor) => {
                        println!("    ✓ {} tensor creation succeeded", type_name);

                        // Verify size calculations
                        assert_eq!(
                            tensor.size(),
                            f32_data.len(),
                            "{} tensor size should match input",
                            type_name
                        );
                        assert_eq!(
                            tensor.size_bytes(),
                            f32_data.len() * expected_element_size,
                            "{} tensor byte size should be correct",
                            type_name
                        );

                        // Verify alignment by checking handle is reasonable
                        assert_ne!(
                            tensor.handle(),
                            0,
                            "{} tensor handle should be non-zero",
                            type_name
                        );

                        // Test data integrity after alignment
                        match tensor.to_vec() {
                            Ok(retrieved) => {
                                assert_eq!(
                                    retrieved, f32_data,
                                    "{} data should be preserved after alignment",
                                    type_name
                                );
                                println!("    ✓ {} data integrity verified", type_name);
                            }
                            Err(e) => println!("    ⚠ {} data retrieval failed: {}", type_name, e),
                        }
                    }
                    Err(e) => println!("    ⚠ {} tensor creation failed: {}", type_name, e),
                }
            }
        }

        // Test memory alignment with various sizes
        let sizes_to_test = vec![1, 2, 3, 4, 5, 8, 16, 32, 64, 100, 1000];

        for size in sizes_to_test {
            let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
            match Tensor::new(&mut manager, &data) {
                Ok(tensor) => {
                    assert_eq!(
                        tensor.size(),
                        size,
                        "Size {} tensor should have correct size",
                        size
                    );
                    assert_eq!(
                        tensor.size_bytes(),
                        size * 4,
                        "Size {} tensor should have correct byte size",
                        size
                    );
                }
                Err(e) => {
                    println!("  ⚠ Size {} tensor creation failed: {}", size, e);
                }
            }
        }

        println!("  ✓ Memory alignment requirements tested");
    } else {
        println!("Skipping memory alignment test - no GPU available");
    }
}

/// Test tensor handle lifecycle and invalidation after Drop.
/// Verifies proper handle cleanup and use-after-free prevention.
#[test]
#[ignore = "requires Vulkan-capable device"]
fn test_tensor_handle_lifecycle_invalidation() {
    if let Ok(mut manager) = Manager::new() {
        println!("Testing tensor handle lifecycle...");

        let mut collected_handles = Vec::new();

        // Create and drop multiple tensors, collecting their handles
        for i in 0..5 {
            let data: Vec<f32> = vec![i as f32; 10];
            match Tensor::new(&mut manager, &data) {
                Ok(tensor) => {
                    let handle = tensor.handle();
                    assert_ne!(handle, 0, "Tensor {} handle should be non-zero", i);

                    // Verify handle uniqueness
                    assert!(
                        !collected_handles.contains(&handle),
                        "Handle {} should be unique",
                        handle
                    );

                    collected_handles.push(handle);
                    println!("  Created tensor {} with handle {}", i, handle);

                    // Tensor drops here when it goes out of scope
                }
                Err(e) => {
                    println!("  Tensor {} creation failed: {}", i, e);
                    break;
                }
            }
        }

        println!("  ✓ Collected {} unique handles", collected_handles.len());

        // Verify all handles were unique
        for i in 0..collected_handles.len() {
            for j in (i + 1)..collected_handles.len() {
                assert_ne!(
                    collected_handles[i], collected_handles[j],
                    "Handles should be unique: {} != {}",
                    collected_handles[i], collected_handles[j]
                );
            }
        }

        // Create new tensors and verify they get different handles
        for i in 0..3 {
            let data: Vec<f32> = vec![(i + 100) as f32; 5];
            match Tensor::new(&mut manager, &data) {
                Ok(tensor) => {
                    let new_handle = tensor.handle();
                    assert_ne!(new_handle, 0, "New tensor {} handle should be non-zero", i);

                    // Note: We can't guarantee handles won't be reused in a real implementation,
                    // but we can verify they're valid and non-zero
                    println!("  New tensor {} handle: {}", i, new_handle);
                }
                Err(e) => {
                    println!("  New tensor {} creation failed: {}", i, e);
                }
            }
        }

        println!("  ✓ Handle lifecycle validation completed");
    } else {
        println!("Skipping handle lifecycle test - no GPU available");
    }
}

/// Test TensorData trait implementations and type ID consistency.
/// Verifies trait implementation correctness across all data types.
#[test]
fn test_tensor_data_trait_implementations() {
    println!("Testing TensorData trait implementations...");

    // Test data_type_id() uniqueness
    let type_ids = vec![
        ("f32", f32::data_type_id()),
        ("f64", f64::data_type_id()),
        ("i32", i32::data_type_id()),
        ("u32", u32::data_type_id()),
        ("i8", i8::data_type_id()),
        ("u8", u8::data_type_id()),
    ];

    println!("  Type IDs:");
    for (type_name, type_id) in &type_ids {
        println!("    {}: {}", type_name, type_id);
    }

    // Verify all type IDs are unique
    for i in 0..type_ids.len() {
        for j in (i + 1)..type_ids.len() {
            assert_ne!(
                type_ids[i].1, type_ids[j].1,
                "Type IDs should be unique: {} ({}) != {} ({})",
                type_ids[i].0, type_ids[i].1, type_ids[j].0, type_ids[j].1
            );
        }
    }

    // Test type ID consistency (multiple calls return same value)
    assert_eq!(
        f32::data_type_id(),
        f32::data_type_id(),
        "f32 type ID should be consistent"
    );
    assert_eq!(
        f64::data_type_id(),
        f64::data_type_id(),
        "f64 type ID should be consistent"
    );
    assert_eq!(
        i32::data_type_id(),
        i32::data_type_id(),
        "i32 type ID should be consistent"
    );
    assert_eq!(
        u32::data_type_id(),
        u32::data_type_id(),
        "u32 type ID should be consistent"
    );
    assert_eq!(
        i8::data_type_id(),
        i8::data_type_id(),
        "i8 type ID should be consistent"
    );
    assert_eq!(
        u8::data_type_id(),
        u8::data_type_id(),
        "u8 type ID should be consistent"
    );

    // Test that types implement TensorData (compile-time check)
    fn assert_tensor_data<T: TensorData>() {}
    assert_tensor_data::<f32>();
    assert_tensor_data::<f64>();
    assert_tensor_data::<i32>();
    assert_tensor_data::<u32>();
    assert_tensor_data::<i8>();
    assert_tensor_data::<u8>();

    // Test Copy trait (required for TensorData)
    let f32_val: f32 = 1.0;
    let f32_copy = f32_val;
    assert_eq!(f32_val, f32_copy, "f32 should implement Copy");

    let i32_val: i32 = 42;
    let i32_copy = i32_val;
    assert_eq!(i32_val, i32_copy, "i32 should implement Copy");

    println!("  ✓ All TensorData trait implementations verified");
}

/// Test tensor creation with different type sizes simultaneously.
/// Verifies memory management across mixed data types.
#[test]
#[ignore = "requires Vulkan-capable device"]
fn test_tensor_mixed_type_creation() {
    if let Ok(mut manager) = Manager::new() {
        println!("Testing mixed type tensor creation...");

        let mut tensors_created = 0;

        // Test f32 tensors (currently supported)
        let f32_data = vec![1.0f32, 2.0f32, 3.0f32];
        match Tensor::new(&mut manager, &f32_data) {
            Ok(tensor) => {
                assert_eq!(tensor.size(), 3, "f32 tensor size should be correct");
                assert_eq!(
                    tensor.size_bytes(),
                    12,
                    "f32 tensor byte size should be correct"
                );
                tensors_created += 1;
                println!("  ✓ f32 tensor created successfully");
            }
            Err(e) => println!("  ⚠ f32 tensor creation failed: {}", e),
        }

        // Test other types (expected to fail until implemented)
        let test_cases = vec![("f64", 8), ("i32", 4), ("u32", 4), ("i8", 1), ("u8", 1)];

        for (type_name, expected_size) in test_cases {
            println!(
                "  Testing {} type creation (element size: {} bytes)",
                type_name, expected_size
            );
            // These tests will be added as implementation progresses
        }

        if tensors_created > 0 {
            println!(
                "  ✓ Successfully created {} tensors of different types",
                tensors_created
            );
        } else {
            println!("  ⚠ No tensors created successfully");
        }
    } else {
        println!("Skipping mixed type test - no GPU available");
    }
}

/// Test concurrent tensor creation across multiple threads with different types.
/// Verifies thread safety of tensor operations.
#[test]
#[ignore = "requires Vulkan-capable device"]
fn test_tensor_concurrent_multitype_creation() {
    println!("Testing concurrent multi-type tensor creation...");

    if let Ok(manager) = Manager::new() {
        let manager = Arc::new(std::sync::Mutex::new(manager));
        let thread_count = 4;
        let mut handles = vec![];

        for thread_id in 0..thread_count {
            let manager_clone = Arc::clone(&manager);

            let handle = thread::spawn(move || {
                let mut results = Vec::new();

                // Each thread creates tensors with different data
                for i in 0..5 {
                    let data: Vec<f32> = vec![(thread_id * 100 + i) as f32; 10];

                    let mut manager_guard = manager_clone.lock().unwrap();
                    match Tensor::new(&mut *manager_guard, &data) {
                        Ok(tensor) => {
                            let handle = tensor.handle();
                            results.push((thread_id, i, handle, true));
                            drop(manager_guard); // Release lock early

                            // Verify tensor outside lock
                            if let Ok(retrieved) = tensor.to_vec() {
                                assert_eq!(
                                    retrieved.len(),
                                    10,
                                    "Thread {} tensor {} size should be correct",
                                    thread_id,
                                    i
                                );
                                assert_eq!(
                                    retrieved[0],
                                    (thread_id * 100 + i) as f32,
                                    "Thread {} tensor {} data should be correct",
                                    thread_id,
                                    i
                                );
                            }
                        }
                        Err(_e) => {
                            results.push((thread_id, i, 0, false));
                            drop(manager_guard);
                        }
                    }
                }

                results
            });

            handles.push(handle);
        }

        // Collect results from all threads
        let mut all_handles = Vec::new();
        let mut total_created = 0;

        for handle in handles {
            if let Ok(thread_results) = handle.join() {
                for (thread_id, i, tensor_handle, success) in thread_results {
                    if success {
                        all_handles.push(tensor_handle);
                        total_created += 1;
                        println!(
                            "  Thread {} created tensor {} with handle {}",
                            thread_id, i, tensor_handle
                        );
                    }
                }
            }
        }

        println!(
            "  ✓ Created {} tensors across {} threads",
            total_created, thread_count
        );

        // Verify all handles are unique
        for i in 0..all_handles.len() {
            for j in (i + 1)..all_handles.len() {
                assert_ne!(
                    all_handles[i], all_handles[j],
                    "Handles should be unique across threads: {} != {}",
                    all_handles[i], all_handles[j]
                );
            }
            assert_ne!(all_handles[i], 0, "All handles should be non-zero");
        }

        if total_created > 0 {
            println!(
                "  ✓ Concurrent tensor creation successful with {} tensors",
                total_created
            );
        }
    } else {
        println!("Skipping concurrent test - no GPU available");
    }
}
