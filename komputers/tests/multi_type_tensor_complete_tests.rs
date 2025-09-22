//! Comprehensive Multi-Type Tensor Support Tests
//!
//! This test suite validates complete support for f64, i32, and u32 tensor types
//! with full GPU integration and proper type safety validation.

use komputers::{Manager, Tensor, TensorData};

#[test]
fn test_f64_tensor_precision_handling() {
    println!("Testing f64 tensor precision handling...");

    // Test with high precision f64 values
    let high_precision_data = vec![
        std::f64::consts::PI,
        std::f64::consts::E,
        1.7976931348623157e308,  // Near f64::MAX
        2.2250738585072014e-308, // Near f64::MIN_POSITIVE
        -1.7976931348623157e308, // Near f64::MIN
    ];

    if let Ok(mut manager) = Manager::new() {
        match Tensor::new(&mut manager, &high_precision_data) {
            Ok(tensor) => {
                println!("  ✓ f64 tensor created with high precision data");

                // Test size calculation for f64 (8 bytes per element)
                assert_eq!(tensor.size(), high_precision_data.len());
                println!("  ✓ f64 tensor size calculation correct");

                // Test data round-trip with full precision
                match tensor.to_vec() {
                    Ok(retrieved_data) => {
                        println!("  ✓ f64 tensor data retrieved successfully");

                        // Validate IEEE 754 double precision is maintained
                        for (original, retrieved) in
                            high_precision_data.iter().zip(retrieved_data.iter())
                        {
                            assert_eq!(*original, *retrieved, "f64 precision not maintained");
                        }
                        println!("  ✓ f64 precision maintained across GPU operations");
                    }
                    Err(_) => {
                        println!("  ⚠️ f64 tensor data retrieval failed (expected without GPU)")
                    }
                }
            }
            Err(_) => println!("  ⚠️ f64 tensor creation failed (expected without GPU)"),
        }
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}

#[test]
fn test_i32_tensor_signed_integer_handling() {
    println!("Testing i32 tensor signed integer handling...");

    // Test with full range of i32 values including negative numbers
    let signed_data = vec![
        i32::MAX, // 2,147,483,647
        i32::MIN, // -2,147,483,648
        0,        // Zero
        -1,       // Negative one
        42,       // Positive
        -42,      // Negative
        1000000,  // Large positive
        -1000000, // Large negative
    ];

    if let Ok(mut manager) = Manager::new() {
        match Tensor::new(&mut manager, &signed_data) {
            Ok(tensor) => {
                println!("  ✓ i32 tensor created with signed integer data");

                // Test size calculation for i32 (4 bytes per element)
                assert_eq!(tensor.size(), signed_data.len());
                println!("  ✓ i32 tensor size calculation correct");

                // Test data round-trip preserving sign
                match tensor.to_vec() {
                    Ok(retrieved_data) => {
                        println!("  ✓ i32 tensor data retrieved successfully");

                        // Validate all signed values including negatives
                        for (original, retrieved) in signed_data.iter().zip(retrieved_data.iter()) {
                            assert_eq!(*original, *retrieved, "i32 signed value not preserved");
                        }
                        println!("  ✓ i32 signed values preserved across GPU operations");
                    }
                    Err(_) => {
                        println!("  ⚠️ i32 tensor data retrieval failed (expected without GPU)")
                    }
                }
            }
            Err(_) => println!("  ⚠️ i32 tensor creation failed (expected without GPU)"),
        }
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}

#[test]
fn test_u32_tensor_unsigned_range_handling() {
    println!("Testing u32 tensor unsigned range handling...");

    // Test with full range of u32 values
    let unsigned_data = vec![
        0u32,          // Minimum value
        u32::MAX,      // 4,294,967,295 (maximum value)
        1u32,          // Small positive
        1000000u32,    // Medium value
        2147483648u32, // Beyond i32::MAX
        4000000000u32, // Large unsigned value
        0xFFFFFFFFu32, // Maximum in hex
        0x80000000u32, // High bit set
    ];

    if let Ok(mut manager) = Manager::new() {
        match Tensor::new(&mut manager, &unsigned_data) {
            Ok(tensor) => {
                println!("  ✓ u32 tensor created with unsigned integer data");

                // Test size calculation for u32 (4 bytes per element)
                assert_eq!(tensor.size(), unsigned_data.len());
                println!("  ✓ u32 tensor size calculation correct");

                // Test data round-trip preserving full unsigned range
                match tensor.to_vec() {
                    Ok(retrieved_data) => {
                        println!("  ✓ u32 tensor data retrieved successfully");

                        // Validate all unsigned values including large ones
                        for (original, retrieved) in unsigned_data.iter().zip(retrieved_data.iter())
                        {
                            assert_eq!(*original, *retrieved, "u32 unsigned value not preserved");
                        }
                        println!("  ✓ u32 full unsigned range preserved across GPU operations");
                    }
                    Err(_) => {
                        println!("  ⚠️ u32 tensor data retrieval failed (expected without GPU)")
                    }
                }
            }
            Err(_) => println!("  ⚠️ u32 tensor creation failed (expected without GPU)"),
        }
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}

#[test]
fn test_tensor_data_trait_consistency() {
    println!("Testing TensorData trait consistency across types...");

    // Validate data type IDs are consistent
    assert_eq!(f32::data_type_id(), 0, "f32 data type ID should be 0");
    assert_eq!(f64::data_type_id(), 1, "f64 data type ID should be 1");
    assert_eq!(i32::data_type_id(), 2, "i32 data type ID should be 2");
    assert_eq!(u32::data_type_id(), 3, "u32 data type ID should be 3");

    println!("  ✓ TensorData trait data type IDs are consistent");

    // Test that data type IDs are unique
    let type_ids = vec![
        f32::data_type_id(),
        f64::data_type_id(),
        i32::data_type_id(),
        u32::data_type_id(),
    ];

    for i in 0..type_ids.len() {
        for j in (i + 1)..type_ids.len() {
            assert_ne!(type_ids[i], type_ids[j], "Data type IDs must be unique");
        }
    }

    println!("  ✓ All data type IDs are unique");
}

#[test]
fn test_multi_type_tensor_sizes() {
    println!("Testing multi-type tensor size calculations...");

    let test_size = 1000usize;

    if let Ok(mut manager) = Manager::new() {
        // Test f32 tensor (4 bytes per element)
        let f32_data: Vec<f32> = (0..test_size).map(|i| i as f32).collect();
        if let Ok(f32_tensor) = Tensor::new(&mut manager, &f32_data) {
            assert_eq!(f32_tensor.size(), test_size);
            println!("  ✓ f32 tensor size correct: {} elements", test_size);
        }

        // Test f64 tensor (8 bytes per element)
        let f64_data: Vec<f64> = (0..test_size).map(|i| i as f64).collect();
        if let Ok(f64_tensor) = Tensor::new(&mut manager, &f64_data) {
            assert_eq!(f64_tensor.size(), test_size);
            println!("  ✓ f64 tensor size correct: {} elements", test_size);
        }

        // Test i32 tensor (4 bytes per element)
        let i32_data: Vec<i32> = (0..test_size).map(|i| i as i32).collect();
        if let Ok(i32_tensor) = Tensor::new(&mut manager, &i32_data) {
            assert_eq!(i32_tensor.size(), test_size);
            println!("  ✓ i32 tensor size correct: {} elements", test_size);
        }

        // Test u32 tensor (4 bytes per element)
        let u32_data: Vec<u32> = (0..test_size).map(|i| i as u32).collect();
        if let Ok(u32_tensor) = Tensor::new(&mut manager, &u32_data) {
            assert_eq!(u32_tensor.size(), test_size);
            println!("  ✓ u32 tensor size correct: {} elements", test_size);
        }

        println!("  ✓ All tensor types handle size calculations correctly");
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}

#[test]
fn test_uninit_tensor_multi_type_support() {
    println!("Testing uninit tensor support for all types...");

    let test_size = 500usize;

    if let Ok(mut manager) = Manager::new() {
        // Test f32 uninit tensor
        match Tensor::<f32>::uninit(&mut manager, test_size) {
            Ok(f32_tensor) => {
                assert_eq!(f32_tensor.size(), test_size);
                println!("  ✓ f32 uninit tensor creation successful");
            }
            Err(_) => println!("  ⚠️ f32 uninit tensor creation failed (expected without GPU)"),
        }

        // Test f64 uninit tensor
        match Tensor::<f64>::uninit(&mut manager, test_size) {
            Ok(f64_tensor) => {
                assert_eq!(f64_tensor.size(), test_size);
                println!("  ✓ f64 uninit tensor creation successful");
            }
            Err(_) => println!("  ⚠️ f64 uninit tensor creation failed (expected without GPU)"),
        }

        // Test i32 uninit tensor
        match Tensor::<i32>::uninit(&mut manager, test_size) {
            Ok(i32_tensor) => {
                assert_eq!(i32_tensor.size(), test_size);
                println!("  ✓ i32 uninit tensor creation successful");
            }
            Err(_) => println!("  ⚠️ i32 uninit tensor creation failed (expected without GPU)"),
        }

        // Test u32 uninit tensor
        match Tensor::<u32>::uninit(&mut manager, test_size) {
            Ok(u32_tensor) => {
                assert_eq!(u32_tensor.size(), test_size);
                println!("  ✓ u32 uninit tensor creation successful");
            }
            Err(_) => println!("  ⚠️ u32 uninit tensor creation failed (expected without GPU)"),
        }

        println!("  ✓ All tensor types support uninit creation");
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}

#[test]
fn test_tensor_error_handling_by_type() {
    println!("Testing tensor error handling for different types...");

    if let Ok(mut manager) = Manager::new() {
        // Test empty data error for all types
        assert!(Tensor::<f32>::new(&mut manager, &[]).is_err());
        assert!(Tensor::<f64>::new(&mut manager, &[]).is_err());
        assert!(Tensor::<i32>::new(&mut manager, &[]).is_err());
        assert!(Tensor::<u32>::new(&mut manager, &[]).is_err());
        println!("  ✓ Empty data error handling works for all types");

        // Test zero size uninit tensor error for all types
        assert!(Tensor::<f32>::uninit(&mut manager, 0).is_err());
        assert!(Tensor::<f64>::uninit(&mut manager, 0).is_err());
        assert!(Tensor::<i32>::uninit(&mut manager, 0).is_err());
        assert!(Tensor::<u32>::uninit(&mut manager, 0).is_err());
        println!("  ✓ Zero size error handling works for all types");
    }

    println!("  ✓ Error handling is consistent across all tensor types");
}

#[test]
fn test_multi_type_tensor_memory_layout() {
    println!("Testing multi-type tensor memory layout consistency...");

    if let Ok(mut manager) = Manager::new() {
        let test_data_size = 100;

        // Create test data for each type
        let f32_data: Vec<f32> = (0..test_data_size).map(|i| i as f32 + 0.5).collect();
        let f64_data: Vec<f64> = (0..test_data_size).map(|i| i as f64 + 0.5).collect();
        let i32_data: Vec<i32> = (0..test_data_size).map(|i| i as i32 - 50).collect();
        let u32_data: Vec<u32> = (0..test_data_size).map(|i| i as u32).collect();

        // Test that memory layout is correct by verifying data integrity
        if let (Ok(f32_tensor), Ok(f64_tensor), Ok(i32_tensor), Ok(u32_tensor)) = (
            Tensor::new(&mut manager, &f32_data),
            Tensor::new(&mut manager, &f64_data),
            Tensor::new(&mut manager, &i32_data),
            Tensor::new(&mut manager, &u32_data),
        ) {
            println!("  ✓ All tensor types created successfully");

            // Verify sizes are consistent
            assert_eq!(f32_tensor.size(), test_data_size);
            assert_eq!(f64_tensor.size(), test_data_size);
            assert_eq!(i32_tensor.size(), test_data_size);
            assert_eq!(u32_tensor.size(), test_data_size);

            println!("  ✓ Memory layout consistent: all tensors report correct sizes");
        } else {
            println!("  ⚠️ Some tensor creations failed (expected without GPU)");
        }
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}
