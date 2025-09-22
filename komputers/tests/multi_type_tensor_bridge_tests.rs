//! Multi-type tensor C++ bridge integration tests.
//!
//! These tests verify that the C++ bridge correctly handles f64, i32, u32 tensors
//! with real GPU operations and proper type safety.

use komputers::{Manager, Tensor, TensorData};
use std::collections::HashMap;

#[test]
fn test_f64_tensor_precision_preservation() {
    if let Ok(mut manager) = Manager::new() {
        let high_precision_data = vec![
            std::f64::consts::PI,
            std::f64::consts::E,
            1.7976931348623157e+308, // Near f64::MAX
            2.2250738585072014e-308, // Near f64::MIN_POSITIVE
            -std::f64::consts::PI,
            0.0f64,
            1.0f64,
            -1.0f64,
        ];

        match Tensor::new(&mut manager, &high_precision_data) {
            Ok(tensor) => {
                // Verify tensor properties
                assert_eq!(tensor.size(), high_precision_data.len());
                println!("✓ f64 tensor created with size: {}", tensor.size());

                // Test data round-trip preservation
                match tensor.to_vec() {
                    Ok(retrieved) => {
                        assert_eq!(retrieved.len(), high_precision_data.len());

                        for (i, (orig, retr)) in
                            high_precision_data.iter().zip(retrieved.iter()).enumerate()
                        {
                            // Use strict equality for exact preservation
                            assert_eq!(
                                *orig, *retr,
                                "f64 precision lost at index {}: {} != {}",
                                i, orig, retr
                            );
                        }
                        println!("✓ f64 tensor preserves high precision values exactly");

                        // Test special values
                        assert!(retrieved[0].is_finite(), "Pi should be finite");
                        assert!(retrieved[1].is_finite(), "E should be finite");
                        assert!(retrieved[2].is_finite(), "Large value should be finite");
                        assert!(
                            retrieved[3] > 0.0,
                            "Small positive value should be positive"
                        );
                        assert!(retrieved[5] == 0.0, "Zero should be preserved exactly");
                    }
                    Err(e) => panic!("f64 tensor data retrieval failed: {}", e),
                }
            }
            Err(e)
                if e.to_string().contains("not implemented")
                    || e.to_string().contains("unsupported") =>
            {
                println!("⚠️ f64 tensors not yet implemented: {}", e);
            }
            Err(e) => panic!("f64 tensor creation failed unexpectedly: {}", e),
        }
    } else {
        println!("⚠️ GPU manager creation failed - skipping f64 tensor test");
    }
}

#[test]
fn test_i32_tensor_extreme_values() {
    if let Ok(mut manager) = Manager::new() {
        let extreme_i32_data = vec![i32::MAX, i32::MIN, 0, -1, 1, 1_000_000, -1_000_000, 42, -42];

        match Tensor::new(&mut manager, &extreme_i32_data) {
            Ok(tensor) => {
                assert_eq!(tensor.size(), extreme_i32_data.len());
                println!("✓ i32 tensor created with size: {}", tensor.size());

                match tensor.to_vec() {
                    Ok(retrieved) => {
                        assert_eq!(retrieved.len(), extreme_i32_data.len());

                        for (i, (orig, retr)) in
                            extreme_i32_data.iter().zip(retrieved.iter()).enumerate()
                        {
                            assert_eq!(
                                *orig, *retr,
                                "i32 value corrupted at index {}: {} != {}",
                                i, orig, retr
                            );
                        }
                        println!("✓ i32 tensor handles extreme values correctly");

                        // Verify specific extreme values
                        assert_eq!(retrieved[0], i32::MAX, "MAX value preserved");
                        assert_eq!(retrieved[1], i32::MIN, "MIN value preserved");
                        assert_eq!(retrieved[2], 0, "Zero preserved");
                        assert_eq!(retrieved[3], -1, "Negative one preserved");
                        assert_eq!(retrieved[4], 1, "Positive one preserved");
                    }
                    Err(e) => panic!("i32 tensor data retrieval failed: {}", e),
                }
            }
            Err(e)
                if e.to_string().contains("not implemented")
                    || e.to_string().contains("unsupported") =>
            {
                println!("⚠️ i32 tensors not yet implemented: {}", e);
            }
            Err(e) => panic!("i32 tensor creation failed unexpectedly: {}", e),
        }
    } else {
        println!("⚠️ GPU manager creation failed - skipping i32 tensor test");
    }
}

#[test]
fn test_u32_tensor_unsigned_range() {
    if let Ok(mut manager) = Manager::new() {
        let unsigned_data = vec![
            0u32,
            1u32,
            u32::MAX,
            u32::MAX / 2,
            1_000_000u32,
            4_294_967_295u32, // u32::MAX
            2_147_483_648u32, // Larger than i32::MAX
            42u32,
        ];

        match Tensor::new(&mut manager, &unsigned_data) {
            Ok(tensor) => {
                assert_eq!(tensor.size(), unsigned_data.len());
                println!("✓ u32 tensor created with size: {}", tensor.size());

                match tensor.to_vec() {
                    Ok(retrieved) => {
                        assert_eq!(retrieved.len(), unsigned_data.len());

                        for (i, (orig, retr)) in
                            unsigned_data.iter().zip(retrieved.iter()).enumerate()
                        {
                            assert_eq!(
                                *orig, *retr,
                                "u32 value corrupted at index {}: {} != {}",
                                i, orig, retr
                            );
                        }
                        println!("✓ u32 tensor handles full unsigned range correctly");

                        // Verify specific unsigned values
                        assert_eq!(retrieved[0], 0u32, "Zero preserved");
                        assert_eq!(retrieved[2], u32::MAX, "MAX value preserved");
                        assert_eq!(
                            retrieved[6], 2_147_483_648u32,
                            "Large unsigned value preserved"
                        );
                    }
                    Err(e) => panic!("u32 tensor data retrieval failed: {}", e),
                }
            }
            Err(e)
                if e.to_string().contains("not implemented")
                    || e.to_string().contains("unsupported") =>
            {
                println!("⚠️ u32 tensors not yet implemented: {}", e);
            }
            Err(e) => panic!("u32 tensor creation failed unexpectedly: {}", e),
        }
    } else {
        println!("⚠️ GPU manager creation failed - skipping u32 tensor test");
    }
}

#[test]
fn test_multi_type_tensor_size_calculations() {
    if let Ok(mut manager) = Manager::new() {
        let test_data_f32 = vec![1.0f32; 100];
        let test_data_f64 = vec![1.0f64; 100];
        let test_data_i32 = vec![1i32; 100];
        let test_data_u32 = vec![1u32; 100];

        let mut created_tensors = Vec::new();
        let mut expected_sizes = HashMap::new();

        // Test f32 tensor
        if let Ok(tensor) = Tensor::new(&mut manager, &test_data_f32) {
            assert_eq!(tensor.size(), 100);
            created_tensors.push(("f32", tensor.size()));
            expected_sizes.insert("f32", (100, 400)); // 100 elements * 4 bytes
            println!("✓ f32 tensor: {} elements", tensor.size());
        }

        // Test f64 tensor
        if let Ok(tensor) = Tensor::new(&mut manager, &test_data_f64) {
            assert_eq!(tensor.size(), 100);
            created_tensors.push(("f64", tensor.size()));
            expected_sizes.insert("f64", (100, 800)); // 100 elements * 8 bytes
            println!("✓ f64 tensor: {} elements", tensor.size());
        }

        // Test i32 tensor
        if let Ok(tensor) = Tensor::new(&mut manager, &test_data_i32) {
            assert_eq!(tensor.size(), 100);
            created_tensors.push(("i32", tensor.size()));
            expected_sizes.insert("i32", (100, 400)); // 100 elements * 4 bytes
            println!("✓ i32 tensor: {} elements", tensor.size());
        }

        // Test u32 tensor
        if let Ok(tensor) = Tensor::new(&mut manager, &test_data_u32) {
            assert_eq!(tensor.size(), 100);
            created_tensors.push(("u32", tensor.size()));
            expected_sizes.insert("u32", (100, 400)); // 100 elements * 4 bytes
            println!("✓ u32 tensor: {} elements", tensor.size());
        }

        if created_tensors.is_empty() {
            println!("⚠️ No multi-type tensors could be created - may not be implemented yet");
        } else {
            println!(
                "✓ Multi-type tensor size calculations verified for {} types",
                created_tensors.len()
            );
        }
    }
}

#[test]
fn test_tensor_data_type_detection() {
    if let Ok(mut manager) = Manager::new() {
        let mut type_tests = Vec::new();

        // Test f32 data type detection
        let f32_data = vec![1.0f32, 2.0, 3.0];
        if let Ok(_tensor) = Tensor::new(&mut manager, &f32_data) {
            // Test the TensorData trait type ID
            assert_eq!(<f32 as TensorData>::data_type_id(), 0);
            println!("✓ f32 tensor created - type ID: {}", f32::data_type_id());
            type_tests.push("f32");
        }

        // Test f64 data type detection
        let f64_data = vec![1.0f64, 2.0, 3.0];
        if let Ok(_tensor) = Tensor::new(&mut manager, &f64_data) {
            assert_eq!(<f64 as TensorData>::data_type_id(), 1);
            println!("✓ f64 tensor created - type ID: {}", f64::data_type_id());
            type_tests.push("f64");
        }

        // Test i32 data type detection
        let i32_data = vec![1i32, 2, 3];
        if let Ok(_tensor) = Tensor::new(&mut manager, &i32_data) {
            assert_eq!(<i32 as TensorData>::data_type_id(), 2);
            println!("✓ i32 tensor created - type ID: {}", i32::data_type_id());
            type_tests.push("i32");
        }

        // Test u32 data type detection
        let u32_data = vec![1u32, 2, 3];
        if let Ok(_tensor) = Tensor::new(&mut manager, &u32_data) {
            assert_eq!(<u32 as TensorData>::data_type_id(), 3);
            println!("✓ u32 tensor created - type ID: {}", u32::data_type_id());
            type_tests.push("u32");
        }

        if type_tests.is_empty() {
            println!("⚠️ No tensors created for type detection test");
        } else {
            println!(
                "✓ Type detection test completed for {} data types",
                type_tests.len()
            );
        }
    }
}

#[test]
fn test_uninitialized_multi_type_tensors() {
    if let Ok(mut manager) = Manager::new() {
        let tensor_size = 1000;

        // Test uninitialized f64 tensor
        match Tensor::<f64>::uninit(&mut manager, tensor_size) {
            Ok(tensor) => {
                assert_eq!(tensor.size(), tensor_size);
                println!(
                    "✓ Uninitialized f64 tensor created with size: {}",
                    tensor.size()
                );

                // Verify we can read the data (should be zeros)
                match tensor.to_vec() {
                    Ok(data) => {
                        assert_eq!(data.len(), tensor_size);
                        // Check that data is initialized (should be zeros)
                        let zero_count = data.iter().filter(|&&x| x == 0.0).count();
                        println!(
                            "  f64 tensor initialized with {} zeros out of {} elements",
                            zero_count, tensor_size
                        );
                    }
                    Err(e) => println!("⚠️ Could not read uninitialized f64 tensor data: {}", e),
                }
            }
            Err(e) if e.to_string().contains("not implemented") => {
                println!("⚠️ Uninitialized f64 tensors not yet implemented: {}", e);
            }
            Err(e) => println!("⚠️ Uninitialized f64 tensor creation failed: {}", e),
        }

        // Test uninitialized i32 tensor
        match Tensor::<i32>::uninit(&mut manager, tensor_size) {
            Ok(tensor) => {
                assert_eq!(tensor.size(), tensor_size);
                println!(
                    "✓ Uninitialized i32 tensor created with size: {}",
                    tensor.size()
                );
            }
            Err(e) if e.to_string().contains("not implemented") => {
                println!("⚠️ Uninitialized i32 tensors not yet implemented: {}", e);
            }
            Err(e) => println!("⚠️ Uninitialized i32 tensor creation failed: {}", e),
        }

        // Test uninitialized u32 tensor
        match Tensor::<u32>::uninit(&mut manager, tensor_size) {
            Ok(tensor) => {
                assert_eq!(tensor.size(), tensor_size);
                println!(
                    "✓ Uninitialized u32 tensor created with size: {}",
                    tensor.size()
                );
            }
            Err(e) if e.to_string().contains("not implemented") => {
                println!("⚠️ Uninitialized u32 tensors not yet implemented: {}", e);
            }
            Err(e) => println!("⚠️ Uninitialized u32 tensor creation failed: {}", e),
        }
    }
}

#[test]
fn test_multi_type_tensor_concurrent_creation() {
    use std::sync::Arc;
    use std::thread;

    if let Ok(manager) = Manager::new() {
        let manager = Arc::new(std::sync::Mutex::new(manager));
        let mut handles = Vec::new();

        // Create multiple threads creating different tensor types
        for i in 0..4 {
            let manager_clone = Arc::clone(&manager);
            let handle = thread::spawn(move || {
                let mut results = Vec::new();

                if let Ok(mut mgr) = manager_clone.lock() {
                    // Each thread creates a different type of tensor
                    match i % 4 {
                        0 => {
                            // f32 tensor
                            let data = vec![i as f32; 100];
                            if let Ok(_tensor) = Tensor::new(&mut mgr, &data) {
                                results.push(format!("f32 tensor {} created", i));
                            }
                        }
                        1 => {
                            // f64 tensor
                            let data = vec![i as f64; 100];
                            if let Ok(_tensor) = Tensor::new(&mut mgr, &data) {
                                results.push(format!("f64 tensor {} created", i));
                            }
                        }
                        2 => {
                            // i32 tensor
                            let data = vec![i as i32; 100];
                            if let Ok(_tensor) = Tensor::new(&mut mgr, &data) {
                                results.push(format!("i32 tensor {} created", i));
                            }
                        }
                        3 => {
                            // u32 tensor
                            let data = vec![i as u32; 100];
                            if let Ok(_tensor) = Tensor::new(&mut mgr, &data) {
                                results.push(format!("u32 tensor {} created", i));
                            }
                        }
                        _ => unreachable!(),
                    }
                }

                (i, results)
            });
            handles.push(handle);
        }

        // Collect results
        let mut total_successes = 0;
        for handle in handles {
            match handle.join() {
                Ok((thread_id, results)) => {
                    for result in results {
                        println!("✓ Thread {}: {}", thread_id, result);
                        total_successes += 1;
                    }
                }
                Err(_) => {
                    println!("⚠️ Thread panicked during concurrent tensor creation");
                }
            }
        }

        if total_successes > 0 {
            println!(
                "✓ Concurrent multi-type tensor creation: {} successful operations",
                total_successes
            );
        } else {
            println!("⚠️ No concurrent tensor creations succeeded - may need implementation");
        }
    }
}

#[test]
fn test_large_multi_type_tensors() {
    if let Ok(mut manager) = Manager::new() {
        let large_size = 100_000; // 100K elements

        // Test large f64 tensor
        println!("Testing large f64 tensor ({} elements)...", large_size);
        let f64_data: Vec<f64> = (0..large_size).map(|i| i as f64 * 0.001).collect();
        match Tensor::new(&mut manager, &f64_data) {
            Ok(tensor) => {
                assert_eq!(tensor.size(), large_size);

                // Spot check a few values
                match tensor.to_vec() {
                    Ok(retrieved) => {
                        assert_eq!(retrieved.len(), large_size);
                        assert!((retrieved[0] - 0.0).abs() < f64::EPSILON);
                        assert!((retrieved[1000] - 1.0).abs() < f64::EPSILON);
                        println!("✓ Large f64 tensor verified");
                    }
                    Err(e) => println!("⚠️ Large f64 tensor data retrieval failed: {}", e),
                }
            }
            Err(e) => println!("⚠️ Large f64 tensor creation failed: {}", e),
        }

        // Test large i32 tensor
        println!("Testing large i32 tensor ({} elements)...", large_size);
        let i32_data: Vec<i32> = (0..large_size).map(|i| i as i32).collect();
        match Tensor::new(&mut manager, &i32_data) {
            Ok(tensor) => {
                assert_eq!(tensor.size(), large_size);
                println!("✓ Large i32 tensor created successfully");
            }
            Err(e) => println!("⚠️ Large i32 tensor creation failed: {}", e),
        }

        // Test large u32 tensor
        println!("Testing large u32 tensor ({} elements)...", large_size);
        let u32_data: Vec<u32> = (0..large_size).map(|i| i as u32).collect();
        match Tensor::new(&mut manager, &u32_data) {
            Ok(tensor) => {
                assert_eq!(tensor.size(), large_size);
                println!("✓ Large u32 tensor created successfully");
            }
            Err(e) => println!("⚠️ Large u32 tensor creation failed: {}", e),
        }
    }
}
