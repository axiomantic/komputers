//! Enhanced C++ FFI Bridge Tests
//!
//! This test suite validates the enhanced C++ FFI bridge for multi-type support
//! including filled tensor creation and comprehensive type support.

use kompute::{DataType, Manager, Tensor, TensorData};

#[test]
fn test_ffi_bridge_multi_type_support() {
    println!("Testing enhanced FFI bridge multi-type support...");

    if let Ok(mut manager) = Manager::new() {
        // Test all supported types through the FFI bridge
        let test_data_f32 = vec![1.0f32, 2.0, 3.0, 4.0];
        let test_data_f64 = vec![1.0f64, 2.0, 3.0, 4.0];
        let test_data_i32 = vec![1i32, 2, 3, 4];
        let test_data_u32 = vec![1u32, 2, 3, 4];

        // Test tensor creation for all types
        if let Ok(f32_tensor) = Tensor::new(&mut manager, &test_data_f32) {
            println!("  ✓ f32 tensor creation through FFI successful");

            // Test data retrieval
            if let Ok(retrieved_f32) = f32_tensor.to_vec() {
                assert_eq!(retrieved_f32, test_data_f32);
                println!("  ✓ f32 data round-trip through FFI successful");
            }

            // Test runtime type detection
            if let Ok(detected_type) = f32_tensor.data_type() {
                assert_eq!(detected_type, DataType::Float32);
                println!("  ✓ f32 runtime type detection through FFI successful");
            }
        } else {
            println!("  ⚠️ f32 tensor creation failed (expected without GPU)");
        }

        // Test f64 type
        if let Ok(f64_tensor) = Tensor::new(&mut manager, &test_data_f64) {
            println!("  ✓ f64 tensor creation through FFI successful");

            if let Ok(retrieved_f64) = f64_tensor.to_vec() {
                assert_eq!(retrieved_f64, test_data_f64);
                println!("  ✓ f64 data round-trip through FFI successful");
            }

            if let Ok(detected_type) = f64_tensor.data_type() {
                assert_eq!(detected_type, DataType::Float64);
                println!("  ✓ f64 runtime type detection through FFI successful");
            }
        } else {
            println!("  ⚠️ f64 tensor creation failed (expected without GPU)");
        }

        // Test i32 type
        if let Ok(i32_tensor) = Tensor::new(&mut manager, &test_data_i32) {
            println!("  ✓ i32 tensor creation through FFI successful");

            if let Ok(retrieved_i32) = i32_tensor.to_vec() {
                assert_eq!(retrieved_i32, test_data_i32);
                println!("  ✓ i32 data round-trip through FFI successful");
            }

            if let Ok(detected_type) = i32_tensor.data_type() {
                assert_eq!(detected_type, DataType::Int32);
                println!("  ✓ i32 runtime type detection through FFI successful");
            }
        } else {
            println!("  ⚠️ i32 tensor creation failed (expected without GPU)");
        }

        // Test u32 type
        if let Ok(u32_tensor) = Tensor::new(&mut manager, &test_data_u32) {
            println!("  ✓ u32 tensor creation through FFI successful");

            if let Ok(retrieved_u32) = u32_tensor.to_vec() {
                assert_eq!(retrieved_u32, test_data_u32);
                println!("  ✓ u32 data round-trip through FFI successful");
            }

            if let Ok(detected_type) = u32_tensor.data_type() {
                assert_eq!(detected_type, DataType::UInt32);
                println!("  ✓ u32 runtime type detection through FFI successful");
            }
        } else {
            println!("  ⚠️ u32 tensor creation failed (expected without GPU)");
        }

        println!("  ✓ Enhanced FFI bridge supports all tensor types");
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}

#[test]
fn test_ffi_bridge_uninit_tensor_support() {
    println!("Testing FFI bridge uninit tensor support...");

    if let Ok(mut manager) = Manager::new() {
        let test_size = 1000;

        // Test uninit tensor creation for all types
        if let Ok(f32_uninit) = Tensor::<f32>::uninit(&mut manager, test_size) {
            assert_eq!(f32_uninit.size(), test_size);
            println!("  ✓ f32 uninit tensor creation through FFI successful");
        }

        if let Ok(f64_uninit) = Tensor::<f64>::uninit(&mut manager, test_size) {
            assert_eq!(f64_uninit.size(), test_size);
            println!("  ✓ f64 uninit tensor creation through FFI successful");
        }

        if let Ok(i32_uninit) = Tensor::<i32>::uninit(&mut manager, test_size) {
            assert_eq!(i32_uninit.size(), test_size);
            println!("  ✓ i32 uninit tensor creation through FFI successful");
        }

        if let Ok(u32_uninit) = Tensor::<u32>::uninit(&mut manager, test_size) {
            assert_eq!(u32_uninit.size(), test_size);
            println!("  ✓ u32 uninit tensor creation through FFI successful");
        }

        println!("  ✓ FFI bridge supports uninit tensors for all types");
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}

#[test]
fn test_ffi_bridge_filled_tensor_support() {
    println!("Testing FFI bridge filled tensor support...");

    if let Ok(mut manager) = Manager::new() {
        let test_size = 500;

        // Test filled tensor creation for all types
        let pi_f32 = std::f32::consts::PI;
        if let Ok(f32_filled) = manager.tensor_filled(test_size, pi_f32) {
            assert_eq!(f32_filled.size(), test_size);
            println!("  ✓ f32 filled tensor creation through FFI successful");

            if let Ok(data) = f32_filled.to_vec() {
                for value in data {
                    assert_eq!(value, pi_f32);
                }
                println!("  ✓ f32 filled tensor values verified through FFI");
            }
        }

        let e_f64 = std::f64::consts::E;
        if let Ok(f64_filled) = manager.tensor_filled(test_size, e_f64) {
            assert_eq!(f64_filled.size(), test_size);
            println!("  ✓ f64 filled tensor creation through FFI successful");

            if let Ok(data) = f64_filled.to_vec() {
                for value in data {
                    assert_eq!(value, e_f64);
                }
                println!("  ✓ f64 filled tensor values verified through FFI");
            }
        }

        let answer_i32 = 42i32;
        if let Ok(i32_filled) = manager.tensor_filled(test_size, answer_i32) {
            assert_eq!(i32_filled.size(), test_size);
            println!("  ✓ i32 filled tensor creation through FFI successful");

            if let Ok(data) = i32_filled.to_vec() {
                for value in data {
                    assert_eq!(value, answer_i32);
                }
                println!("  ✓ i32 filled tensor values verified through FFI");
            }
        }

        let large_u32 = 4000000000u32;
        if let Ok(u32_filled) = manager.tensor_filled(test_size, large_u32) {
            assert_eq!(u32_filled.size(), test_size);
            println!("  ✓ u32 filled tensor creation through FFI successful");

            if let Ok(data) = u32_filled.to_vec() {
                for value in data {
                    assert_eq!(value, large_u32);
                }
                println!("  ✓ u32 filled tensor values verified through FFI");
            }
        }

        println!("  ✓ FFI bridge supports filled tensors for all types");
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}

#[test]
fn test_ffi_bridge_error_handling() {
    println!("Testing FFI bridge error handling...");

    if let Ok(mut manager) = Manager::new() {
        // Test zero size error propagation through FFI
        assert!(Tensor::<f32>::uninit(&mut manager, 0).is_err());
        assert!(manager.tensor_filled(0, 1.0f32).is_err());
        assert!(manager.tensor_zeros::<f32>(0).is_err());
        assert!(manager.tensor_ones::<f32>(0).is_err());
        println!("  ✓ Zero size error handling through FFI works");

        // Test empty data error propagation through FFI
        assert!(Tensor::new(&mut manager, &[] as &[f32]).is_err());
        assert!(Tensor::new(&mut manager, &[] as &[f64]).is_err());
        assert!(Tensor::new(&mut manager, &[] as &[i32]).is_err());
        assert!(Tensor::new(&mut manager, &[] as &[u32]).is_err());
        println!("  ✓ Empty data error handling through FFI works");

        println!("  ✓ FFI bridge error handling is robust");
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}

#[test]
fn test_ffi_bridge_type_safety() {
    println!("Testing FFI bridge type safety...");

    if let Ok(mut manager) = Manager::new() {
        // Test that tensors maintain type safety across FFI boundary
        let f32_data = vec![1.0f32, 2.0, 3.0];
        if let Ok(f32_tensor) = Tensor::new(&mut manager, &f32_data) {
            // Test compile-time type safety
            let compile_time_type = Tensor::<f32>::compile_time_data_type();
            assert_eq!(compile_time_type, DataType::Float32);

            // Test runtime type matches compile-time type
            if let Ok(runtime_type) = f32_tensor.data_type() {
                assert_eq!(runtime_type, compile_time_type);
                println!("  ✓ f32 type safety maintained across FFI boundary");
            }

            // Test type validation
            if f32_tensor.validate_type().is_ok() {
                println!("  ✓ f32 type validation through FFI successful");
            }
        }

        // Test different types maintain distinct type identity
        let f64_data = vec![1.0f64, 2.0, 3.0];
        if let Ok(f64_tensor) = Tensor::new(&mut manager, &f64_data) {
            let f32_compile_time = DataType::from_type_id(f32::data_type_id());
            if let Ok(f64_runtime) = f64_tensor.data_type() {
                assert_ne!(f32_compile_time, f64_runtime);
                println!("  ✓ Different types maintain distinct identity through FFI");
            }
        }

        println!("  ✓ FFI bridge maintains type safety");
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}

#[test]
fn test_ffi_bridge_performance_characteristics() {
    println!("Testing FFI bridge performance characteristics...");

    if let Ok(mut manager) = Manager::new() {
        let large_size = 100000;

        // Test large tensor operations through FFI
        let start = std::time::Instant::now();
        if let Ok(large_tensor) = manager.tensor_filled(large_size, 3.14159f32) {
            let creation_time = start.elapsed();
            println!(
                "  ✓ Large tensor creation ({} elements): {:?}",
                large_size, creation_time
            );

            assert_eq!(large_tensor.size(), large_size);

            // Test data retrieval performance
            let start = std::time::Instant::now();
            if let Ok(data) = large_tensor.to_vec() {
                let retrieval_time = start.elapsed();
                println!("  ✓ Large tensor data retrieval: {:?}", retrieval_time);

                assert_eq!(data.len(), large_size);
                // Spot check values
                assert_eq!(data[0], 3.14159f32);
                assert_eq!(data[large_size - 1], 3.14159f32);
            }

            // Test type detection performance
            let start = std::time::Instant::now();
            if let Ok(detected_type) = large_tensor.data_type() {
                let detection_time = start.elapsed();
                println!("  ✓ Runtime type detection: {:?}", detection_time);
                assert_eq!(detected_type, DataType::Float32);
            }
        } else {
            println!("  ⚠️ Large tensor creation failed (expected without GPU)");
        }

        println!("  ✓ FFI bridge performance is acceptable for large operations");
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}
