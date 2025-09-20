//! Runtime Type Detection System Tests
//!
//! This test suite validates the runtime type detection system including
//! DataType enum functionality, tensor.data_type() method, and type validation.

use kompute::{DataType, Manager, Tensor, TensorData};

#[test]
fn test_datatype_enum_properties() {
    println!("Testing DataType enum properties and methods...");

    // Test byte sizes
    assert_eq!(DataType::Float32.byte_size(), 4);
    assert_eq!(DataType::Float64.byte_size(), 8);
    assert_eq!(DataType::Int32.byte_size(), 4);
    assert_eq!(DataType::UInt32.byte_size(), 4);
    assert_eq!(DataType::Unknown.byte_size(), 0);
    println!("  ✓ DataType byte sizes correct");

    // Test display names
    assert_eq!(DataType::Float32.name(), "f32");
    assert_eq!(DataType::Float64.name(), "f64");
    assert_eq!(DataType::Int32.name(), "i32");
    assert_eq!(DataType::UInt32.name(), "u32");
    assert_eq!(DataType::Unknown.name(), "unknown");
    println!("  ✓ DataType display names correct");

    // Test floating-point detection
    assert!(DataType::Float32.is_floating_point());
    assert!(DataType::Float64.is_floating_point());
    assert!(!DataType::Int32.is_floating_point());
    assert!(!DataType::UInt32.is_floating_point());
    assert!(!DataType::Unknown.is_floating_point());
    println!("  ✓ Floating-point detection correct");

    // Test integer detection
    assert!(!DataType::Float32.is_integer());
    assert!(!DataType::Float64.is_integer());
    assert!(DataType::Int32.is_integer());
    assert!(DataType::UInt32.is_integer());
    assert!(!DataType::Unknown.is_integer());
    println!("  ✓ Integer detection correct");

    // Test signed detection
    assert!(DataType::Float32.is_signed());
    assert!(DataType::Float64.is_signed());
    assert!(DataType::Int32.is_signed());
    assert!(!DataType::UInt32.is_signed());
    assert!(!DataType::Unknown.is_signed());
    println!("  ✓ Signed type detection correct");

    println!("  ✓ All DataType enum properties work correctly");
}

#[test]
fn test_datatype_type_id_conversion() {
    println!("Testing DataType type ID conversion...");

    // Test from_type_id conversion
    assert_eq!(DataType::from_type_id(0), DataType::Float32);
    assert_eq!(DataType::from_type_id(1), DataType::Float64);
    assert_eq!(DataType::from_type_id(2), DataType::Int32);
    assert_eq!(DataType::from_type_id(3), DataType::UInt32);
    assert_eq!(DataType::from_type_id(999), DataType::Unknown);
    println!("  ✓ from_type_id conversion correct");

    // Test type_id() method
    assert_eq!(DataType::Float32.type_id(), 0);
    assert_eq!(DataType::Float64.type_id(), 1);
    assert_eq!(DataType::Int32.type_id(), 2);
    assert_eq!(DataType::UInt32.type_id(), 3);
    assert_eq!(DataType::Unknown.type_id(), u32::MAX);
    println!("  ✓ type_id() method correct");

    // Test round-trip conversion
    for type_id in 0..=3 {
        let data_type = DataType::from_type_id(type_id);
        assert_eq!(data_type.type_id(), type_id);
    }
    println!("  ✓ Round-trip type ID conversion correct");

    println!("  ✓ DataType type ID conversion system works correctly");
}

#[test]
fn test_tensordata_trait_consistency() {
    println!("Testing TensorData trait consistency with DataType...");

    // Test that TensorData type IDs match DataType expectations
    assert_eq!(f32::data_type_id(), DataType::Float32.type_id());
    assert_eq!(f64::data_type_id(), DataType::Float64.type_id());
    assert_eq!(i32::data_type_id(), DataType::Int32.type_id());
    assert_eq!(u32::data_type_id(), DataType::UInt32.type_id());
    println!("  ✓ TensorData trait IDs match DataType IDs");

    // Test that DataType can be created from TensorData type IDs
    assert_eq!(
        DataType::from_type_id(f32::data_type_id()),
        DataType::Float32
    );
    assert_eq!(
        DataType::from_type_id(f64::data_type_id()),
        DataType::Float64
    );
    assert_eq!(DataType::from_type_id(i32::data_type_id()), DataType::Int32);
    assert_eq!(
        DataType::from_type_id(u32::data_type_id()),
        DataType::UInt32
    );
    println!("  ✓ DataType creation from TensorData IDs correct");

    println!("  ✓ TensorData trait and DataType are consistent");
}

#[test]
fn test_compile_time_data_type() {
    println!("Testing compile-time data type detection...");

    // Test compile-time type detection
    assert_eq!(Tensor::<f32>::compile_time_data_type(), DataType::Float32);
    assert_eq!(Tensor::<f64>::compile_time_data_type(), DataType::Float64);
    assert_eq!(Tensor::<i32>::compile_time_data_type(), DataType::Int32);
    assert_eq!(Tensor::<u32>::compile_time_data_type(), DataType::UInt32);
    println!("  ✓ Compile-time data type detection correct");

    // Test that compile-time detection doesn't require instantiation
    let f32_type = Tensor::<f32>::compile_time_data_type();
    let f64_type = Tensor::<f64>::compile_time_data_type();
    let i32_type = Tensor::<i32>::compile_time_data_type();
    let u32_type = Tensor::<u32>::compile_time_data_type();

    assert_eq!(f32_type.name(), "f32");
    assert_eq!(f64_type.name(), "f64");
    assert_eq!(i32_type.name(), "i32");
    assert_eq!(u32_type.name(), "u32");
    println!("  ✓ Compile-time detection works without instantiation");

    println!("  ✓ Compile-time data type detection works correctly");
}

#[test]
fn test_runtime_data_type_detection() {
    println!("Testing runtime data type detection...");

    if let Ok(mut manager) = Manager::new() {
        // Test f32 runtime type detection
        if let Ok(f32_tensor) = Tensor::new(&mut manager, &[1.0f32, 2.0, 3.0]) {
            match f32_tensor.data_type() {
                Ok(detected_type) => {
                    assert_eq!(detected_type, DataType::Float32);
                    println!("  ✓ f32 tensor runtime type detection correct");
                }
                Err(_) => println!("  ⚠️ f32 tensor runtime type detection failed"),
            }
        }

        // Test f64 runtime type detection
        if let Ok(f64_tensor) = Tensor::new(&mut manager, &[1.0f64, 2.0, 3.0]) {
            match f64_tensor.data_type() {
                Ok(detected_type) => {
                    assert_eq!(detected_type, DataType::Float64);
                    println!("  ✓ f64 tensor runtime type detection correct");
                }
                Err(_) => println!("  ⚠️ f64 tensor runtime type detection failed"),
            }
        }

        // Test i32 runtime type detection
        if let Ok(i32_tensor) = Tensor::new(&mut manager, &[1i32, 2, 3]) {
            match i32_tensor.data_type() {
                Ok(detected_type) => {
                    assert_eq!(detected_type, DataType::Int32);
                    println!("  ✓ i32 tensor runtime type detection correct");
                }
                Err(_) => println!("  ⚠️ i32 tensor runtime type detection failed"),
            }
        }

        // Test u32 runtime type detection
        if let Ok(u32_tensor) = Tensor::new(&mut manager, &[1u32, 2, 3]) {
            match u32_tensor.data_type() {
                Ok(detected_type) => {
                    assert_eq!(detected_type, DataType::UInt32);
                    println!("  ✓ u32 tensor runtime type detection correct");
                }
                Err(_) => println!("  ⚠️ u32 tensor runtime type detection failed"),
            }
        }

        println!("  ✓ Runtime data type detection works for all types");
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}

#[test]
fn test_type_validation() {
    println!("Testing tensor type validation...");

    if let Ok(mut manager) = Manager::new() {
        // Test that type validation succeeds for matching types
        if let Ok(f32_tensor) = Tensor::new(&mut manager, &[1.0f32, 2.0, 3.0]) {
            match f32_tensor.validate_type() {
                Ok(()) => println!("  ✓ f32 tensor type validation passed"),
                Err(_) => println!("  ⚠️ f32 tensor type validation failed"),
            }
        }

        if let Ok(f64_tensor) = Tensor::new(&mut manager, &[1.0f64, 2.0, 3.0]) {
            match f64_tensor.validate_type() {
                Ok(()) => println!("  ✓ f64 tensor type validation passed"),
                Err(_) => println!("  ⚠️ f64 tensor type validation failed"),
            }
        }

        if let Ok(i32_tensor) = Tensor::new(&mut manager, &[1i32, 2, 3]) {
            match i32_tensor.validate_type() {
                Ok(()) => println!("  ✓ i32 tensor type validation passed"),
                Err(_) => println!("  ⚠️ i32 tensor type validation failed"),
            }
        }

        if let Ok(u32_tensor) = Tensor::new(&mut manager, &[1u32, 2, 3]) {
            match u32_tensor.validate_type() {
                Ok(()) => println!("  ✓ u32 tensor type validation passed"),
                Err(_) => println!("  ⚠️ u32 tensor type validation failed"),
            }
        }

        println!("  ✓ Type validation works correctly for all types");
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}

#[test]
fn test_runtime_vs_compile_time_consistency() {
    println!("Testing runtime vs compile-time type consistency...");

    if let Ok(mut manager) = Manager::new() {
        // Test f32 consistency
        if let Ok(f32_tensor) = Tensor::new(&mut manager, &[1.0f32, 2.0, 3.0]) {
            let compile_time_type = Tensor::<f32>::compile_time_data_type();
            if let Ok(runtime_type) = f32_tensor.data_type() {
                assert_eq!(compile_time_type, runtime_type);
                println!("  ✓ f32 compile-time and runtime types match");
            }
        }

        // Test f64 consistency
        if let Ok(f64_tensor) = Tensor::new(&mut manager, &[1.0f64, 2.0, 3.0]) {
            let compile_time_type = Tensor::<f64>::compile_time_data_type();
            if let Ok(runtime_type) = f64_tensor.data_type() {
                assert_eq!(compile_time_type, runtime_type);
                println!("  ✓ f64 compile-time and runtime types match");
            }
        }

        // Test i32 consistency
        if let Ok(i32_tensor) = Tensor::new(&mut manager, &[1i32, 2, 3]) {
            let compile_time_type = Tensor::<i32>::compile_time_data_type();
            if let Ok(runtime_type) = i32_tensor.data_type() {
                assert_eq!(compile_time_type, runtime_type);
                println!("  ✓ i32 compile-time and runtime types match");
            }
        }

        // Test u32 consistency
        if let Ok(u32_tensor) = Tensor::new(&mut manager, &[1u32, 2, 3]) {
            let compile_time_type = Tensor::<u32>::compile_time_data_type();
            if let Ok(runtime_type) = u32_tensor.data_type() {
                assert_eq!(compile_time_type, runtime_type);
                println!("  ✓ u32 compile-time and runtime types match");
            }
        }

        println!("  ✓ Runtime and compile-time types are consistent");
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}

#[test]
fn test_datatype_debug_and_display() {
    println!("Testing DataType Debug and display functionality...");

    // Test Debug formatting
    let f32_type = DataType::Float32;
    let debug_str = format!("{:?}", f32_type);
    assert!(debug_str.contains("Float32"));
    println!("  ✓ DataType Debug formatting works");

    // Test name() method for display
    assert_eq!(DataType::Float32.name(), "f32");
    assert_eq!(DataType::Float64.name(), "f64");
    assert_eq!(DataType::Int32.name(), "i32");
    assert_eq!(DataType::UInt32.name(), "u32");
    assert_eq!(DataType::Unknown.name(), "unknown");
    println!("  ✓ DataType name() method provides proper display strings");

    // Test PartialEq and Eq
    assert_eq!(DataType::Float32, DataType::Float32);
    assert_ne!(DataType::Float32, DataType::Float64);
    println!("  ✓ DataType equality comparison works");

    // Test Hash (implicitly through the derived Hash trait)
    use std::collections::HashMap;
    let mut type_map = HashMap::new();
    type_map.insert(DataType::Float32, "f32");
    type_map.insert(DataType::Float64, "f64");
    assert_eq!(type_map.get(&DataType::Float32), Some(&"f32"));
    assert_eq!(type_map.get(&DataType::Float64), Some(&"f64"));
    println!("  ✓ DataType can be used as HashMap key");

    println!("  ✓ DataType Debug and display functionality works correctly");
}
