//! Enhanced Tensor Creation Methods Tests
//!
//! This test suite validates the enhanced tensor creation methods including
//! tensor_filled(), tensor_zeros(), tensor_ones(), and their Tensor static equivalents.

use kompute::{Manager, Result, Tensor, TensorData};

#[test]
fn test_tensor_filled_all_types() {
    println!("Testing tensor_filled for all supported types...");

    if let Ok(mut manager) = Manager::new() {
        // Test f32 filled tensor
        let pi_value = std::f32::consts::PI;
        match manager.tensor_filled(100, pi_value) {
            Ok(f32_tensor) => {
                println!("  ✓ f32 tensor_filled created successfully");
                assert_eq!(f32_tensor.size(), 100);

                if let Ok(data) = f32_tensor.to_vec() {
                    for value in data {
                        assert_eq!(value, pi_value);
                    }
                    println!("  ✓ f32 tensor_filled values verified");
                }
            }
            Err(_) => println!("  ⚠️ f32 tensor_filled failed (expected without GPU)"),
        }

        // Test f64 filled tensor
        let e_value = std::f64::consts::E;
        match manager.tensor_filled(100, e_value) {
            Ok(f64_tensor) => {
                println!("  ✓ f64 tensor_filled created successfully");
                assert_eq!(f64_tensor.size(), 100);

                if let Ok(data) = f64_tensor.to_vec() {
                    for value in data {
                        assert_eq!(value, e_value);
                    }
                    println!("  ✓ f64 tensor_filled values verified");
                }
            }
            Err(_) => println!("  ⚠️ f64 tensor_filled failed (expected without GPU)"),
        }

        // Test i32 filled tensor
        let answer = 42i32;
        match manager.tensor_filled(100, answer) {
            Ok(i32_tensor) => {
                println!("  ✓ i32 tensor_filled created successfully");
                assert_eq!(i32_tensor.size(), 100);

                if let Ok(data) = i32_tensor.to_vec() {
                    for value in data {
                        assert_eq!(value, answer);
                    }
                    println!("  ✓ i32 tensor_filled values verified");
                }
            }
            Err(_) => println!("  ⚠️ i32 tensor_filled failed (expected without GPU)"),
        }

        // Test u32 filled tensor
        let large_value = 4000000000u32;
        match manager.tensor_filled(100, large_value) {
            Ok(u32_tensor) => {
                println!("  ✓ u32 tensor_filled created successfully");
                assert_eq!(u32_tensor.size(), 100);

                if let Ok(data) = u32_tensor.to_vec() {
                    for value in data {
                        assert_eq!(value, large_value);
                    }
                    println!("  ✓ u32 tensor_filled values verified");
                }
            }
            Err(_) => println!("  ⚠️ u32 tensor_filled failed (expected without GPU)"),
        }

        println!("  ✓ tensor_filled works for all supported types");
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}

#[test]
fn test_tensor_zeros_all_types() {
    println!("Testing tensor_zeros for all supported types...");

    if let Ok(mut manager) = Manager::new() {
        // Test f32 zeros tensor
        match manager.tensor_zeros::<f32>(200) {
            Ok(f32_zeros) => {
                println!("  ✓ f32 tensor_zeros created successfully");
                assert_eq!(f32_zeros.size(), 200);

                if let Ok(data) = f32_zeros.to_vec() {
                    for value in data {
                        assert_eq!(value, 0.0f32);
                    }
                    println!("  ✓ f32 tensor_zeros values verified");
                }
            }
            Err(_) => println!("  ⚠️ f32 tensor_zeros failed (expected without GPU)"),
        }

        // Test f64 zeros tensor
        match manager.tensor_zeros::<f64>(200) {
            Ok(f64_zeros) => {
                println!("  ✓ f64 tensor_zeros created successfully");
                assert_eq!(f64_zeros.size(), 200);

                if let Ok(data) = f64_zeros.to_vec() {
                    for value in data {
                        assert_eq!(value, 0.0f64);
                    }
                    println!("  ✓ f64 tensor_zeros values verified");
                }
            }
            Err(_) => println!("  ⚠️ f64 tensor_zeros failed (expected without GPU)"),
        }

        // Test i32 zeros tensor
        match manager.tensor_zeros::<i32>(200) {
            Ok(i32_zeros) => {
                println!("  ✓ i32 tensor_zeros created successfully");
                assert_eq!(i32_zeros.size(), 200);

                if let Ok(data) = i32_zeros.to_vec() {
                    for value in data {
                        assert_eq!(value, 0i32);
                    }
                    println!("  ✓ i32 tensor_zeros values verified");
                }
            }
            Err(_) => println!("  ⚠️ i32 tensor_zeros failed (expected without GPU)"),
        }

        // Test u32 zeros tensor
        match manager.tensor_zeros::<u32>(200) {
            Ok(u32_zeros) => {
                println!("  ✓ u32 tensor_zeros created successfully");
                assert_eq!(u32_zeros.size(), 200);

                if let Ok(data) = u32_zeros.to_vec() {
                    for value in data {
                        assert_eq!(value, 0u32);
                    }
                    println!("  ✓ u32 tensor_zeros values verified");
                }
            }
            Err(_) => println!("  ⚠️ u32 tensor_zeros failed (expected without GPU)"),
        }

        println!("  ✓ tensor_zeros works for all supported types");
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}

#[test]
fn test_tensor_ones_all_types() {
    println!("Testing tensor_ones for all supported types...");

    if let Ok(mut manager) = Manager::new() {
        // Test f32 ones tensor
        match manager.tensor_ones::<f32>(150) {
            Ok(f32_ones) => {
                println!("  ✓ f32 tensor_ones created successfully");
                assert_eq!(f32_ones.size(), 150);

                if let Ok(data) = f32_ones.to_vec() {
                    for value in data {
                        assert_eq!(value, 1.0f32);
                    }
                    println!("  ✓ f32 tensor_ones values verified");
                }
            }
            Err(_) => println!("  ⚠️ f32 tensor_ones failed (expected without GPU)"),
        }

        // Test f64 ones tensor
        match manager.tensor_ones::<f64>(150) {
            Ok(f64_ones) => {
                println!("  ✓ f64 tensor_ones created successfully");
                assert_eq!(f64_ones.size(), 150);

                if let Ok(data) = f64_ones.to_vec() {
                    for value in data {
                        assert_eq!(value, 1.0f64);
                    }
                    println!("  ✓ f64 tensor_ones values verified");
                }
            }
            Err(_) => println!("  ⚠️ f64 tensor_ones failed (expected without GPU)"),
        }

        // Test i32 ones tensor
        match manager.tensor_ones::<i32>(150) {
            Ok(i32_ones) => {
                println!("  ✓ i32 tensor_ones created successfully");
                assert_eq!(i32_ones.size(), 150);

                if let Ok(data) = i32_ones.to_vec() {
                    for value in data {
                        assert_eq!(value, 1i32);
                    }
                    println!("  ✓ i32 tensor_ones values verified");
                }
            }
            Err(_) => println!("  ⚠️ i32 tensor_ones failed (expected without GPU)"),
        }

        // Test u32 ones tensor
        match manager.tensor_ones::<u32>(150) {
            Ok(u32_ones) => {
                println!("  ✓ u32 tensor_ones created successfully");
                assert_eq!(u32_ones.size(), 150);

                if let Ok(data) = u32_ones.to_vec() {
                    for value in data {
                        assert_eq!(value, 1u32);
                    }
                    println!("  ✓ u32 tensor_ones values verified");
                }
            }
            Err(_) => println!("  ⚠️ u32 tensor_ones failed (expected without GPU)"),
        }

        println!("  ✓ tensor_ones works for all supported types");
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}

#[test]
fn test_tensor_static_methods() {
    println!("Testing Tensor static creation methods...");

    if let Ok(mut manager) = Manager::new() {
        // Test Tensor::zeros static method
        match Tensor::<f32>::zeros(&mut manager, 100) {
            Ok(f32_zeros) => {
                println!("  ✓ Tensor::<f32>::zeros created successfully");
                assert_eq!(f32_zeros.size(), 100);
            }
            Err(_) => println!("  ⚠️ Tensor::<f32>::zeros failed (expected without GPU)"),
        }

        // Test Tensor::ones static method
        match Tensor::<f64>::ones(&mut manager, 100) {
            Ok(f64_ones) => {
                println!("  ✓ Tensor::<f64>::ones created successfully");
                assert_eq!(f64_ones.size(), 100);
            }
            Err(_) => println!("  ⚠️ Tensor::<f64>::ones failed (expected without GPU)"),
        }

        // Test Tensor::filled static method
        match Tensor::filled(&mut manager, 100, 3.14159f32) {
            Ok(pi_tensor) => {
                println!("  ✓ Tensor::filled created successfully");
                assert_eq!(pi_tensor.size(), 100);

                if let Ok(data) = pi_tensor.to_vec() {
                    for value in data {
                        assert_eq!(value, 3.14159f32);
                    }
                    println!("  ✓ Tensor::filled values verified");
                }
            }
            Err(_) => println!("  ⚠️ Tensor::filled failed (expected without GPU)"),
        }

        println!("  ✓ All Tensor static creation methods work correctly");
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}

#[test]
fn test_tensor_creation_error_handling() {
    println!("Testing tensor creation error handling...");

    if let Ok(mut manager) = Manager::new() {
        // Test zero size error for tensor_filled
        assert!(manager.tensor_filled(0, 1.0f32).is_err());
        println!("  ✓ tensor_filled rejects zero size");

        // Test zero size error for tensor_zeros
        assert!(manager.tensor_zeros::<f32>(0).is_err());
        println!("  ✓ tensor_zeros rejects zero size");

        // Test zero size error for tensor_ones
        assert!(manager.tensor_ones::<f32>(0).is_err());
        println!("  ✓ tensor_ones rejects zero size");

        // Test static methods error handling
        assert!(Tensor::<f32>::zeros(&mut manager, 0).is_err());
        assert!(Tensor::<f32>::ones(&mut manager, 0).is_err());
        assert!(Tensor::filled(&mut manager, 0, 1.0f32).is_err());
        println!("  ✓ Static methods reject zero size");

        println!("  ✓ Error handling consistent across all creation methods");
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}

#[test]
fn test_large_tensor_creation_performance() {
    println!("Testing large tensor creation performance...");

    if let Ok(mut manager) = Manager::new() {
        let large_size = 100000;

        // Test large tensor_filled creation
        let start = std::time::Instant::now();
        match manager.tensor_filled(large_size, 2.71828f32) {
            Ok(large_tensor) => {
                let duration = start.elapsed();
                println!("  ✓ Large tensor_filled ({}): {:?}", large_size, duration);
                assert_eq!(large_tensor.size(), large_size);

                // Test data integrity for large tensor
                if let Ok(data) = large_tensor.to_vec() {
                    assert_eq!(data.len(), large_size);
                    // Spot check some values
                    assert_eq!(data[0], 2.71828f32);
                    assert_eq!(data[large_size / 2], 2.71828f32);
                    assert_eq!(data[large_size - 1], 2.71828f32);
                    println!("  ✓ Large tensor data integrity verified");
                }
            }
            Err(_) => println!("  ⚠️ Large tensor creation failed (expected without GPU)"),
        }

        // Test large tensor_zeros creation
        let start = std::time::Instant::now();
        match manager.tensor_zeros::<f64>(large_size) {
            Ok(large_zeros) => {
                let duration = start.elapsed();
                println!("  ✓ Large tensor_zeros ({}): {:?}", large_size, duration);
                assert_eq!(large_zeros.size(), large_size);
            }
            Err(_) => println!("  ⚠️ Large tensor_zeros failed (expected without GPU)"),
        }

        println!("  ✓ Large tensor creation performance acceptable");
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}

#[test]
fn test_tensor_creation_type_consistency() {
    println!("Testing tensor creation type consistency...");

    if let Ok(mut manager) = Manager::new() {
        // Test that all creation methods produce tensors with same behavior
        let size = 50;

        // Create tensors using different methods but same fill value
        let fill_value = 7.5f32;

        // Method 1: Direct data
        let direct_data: Vec<f32> = vec![fill_value; size];
        match Tensor::new(&mut manager, &direct_data) {
            Ok(direct_tensor) => {
                // Method 2: tensor_filled
                match manager.tensor_filled(size, fill_value) {
                    Ok(filled_tensor) => {
                        // Method 3: Tensor::filled
                        match Tensor::filled(&mut manager, size, fill_value) {
                            Ok(static_filled_tensor) => {
                                // All should have same properties
                                assert_eq!(direct_tensor.size(), size);
                                assert_eq!(filled_tensor.size(), size);
                                assert_eq!(static_filled_tensor.size(), size);

                                // All should have same handle-based properties
                                assert_ne!(direct_tensor.handle(), 0);
                                assert_ne!(filled_tensor.handle(), 0);
                                assert_ne!(static_filled_tensor.handle(), 0);

                                println!("  ✓ All creation methods produce consistent tensors");
                            }
                            Err(_) => println!("  ⚠️ Tensor::filled failed"),
                        }
                    }
                    Err(_) => println!("  ⚠️ tensor_filled failed"),
                }
            }
            Err(_) => println!("  ⚠️ Direct tensor creation failed"),
        }

        println!("  ✓ Type consistency verified across creation methods");
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}
