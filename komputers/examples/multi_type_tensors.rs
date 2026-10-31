//! Multi-Type Tensor Operations Example
//!
//! This example demonstrates Komputers's comprehensive support for multiple data types
//! including f32, f64, i32, u32 tensors with runtime type detection and GPU operations.

use komputers::{DataType, Manager, Result, Tensor};

fn main() -> Result<()> {
    println!("🔢 Komputers Multi-Type Tensor Example");
    println!("=======================================");

    // 1. Initialize GPU manager
    println!("\n1. Creating GPU manager...");
    let mut manager = Manager::new()?;
    println!("   ✅ GPU manager initialized");

    // 2. Demonstrate f32 tensors (most common)
    println!("\n2. Creating f32 tensors...");
    let f32_data = vec![1.5f32, 2.7, 3.14159, 4.2, 5.0];
    let f32_tensor = Tensor::new(&mut manager, &f32_data)?;

    println!("   f32 data: {:?}", f32_data);
    println!("   Tensor size: {} elements", f32_tensor.size());
    println!("   Tensor bytes: {} bytes", f32_tensor.size_bytes());
    println!("   Data type: {:?}", f32_tensor.data_type()?);
    assert_eq!(f32_tensor.data_type()?, DataType::Float32);
    println!("   ✅ f32 tensor created and validated");

    // 3. Demonstrate f64 tensors (high precision)
    println!("\n3. Creating f64 tensors...");
    let f64_data = vec![
        std::f64::consts::PI,
        std::f64::consts::E,
        1.7976931348623157e+308, // Near f64::MAX
        2.2250738585072014e-308, // Near f64::MIN_POSITIVE
        -1.5e-100,
    ];
    let f64_tensor = Tensor::new(&mut manager, &f64_data)?;

    println!("   f64 data: {:?}", f64_data);
    println!("   Tensor size: {} elements", f64_tensor.size());
    println!("   Tensor bytes: {} bytes", f64_tensor.size_bytes());
    println!("   Data type: {:?}", f64_tensor.data_type()?);
    assert_eq!(f64_tensor.data_type()?, DataType::Float64);
    println!("   ✅ f64 tensor created with high precision values");

    // 4. Demonstrate i32 tensors (signed integers)
    println!("\n4. Creating i32 tensors...");
    let i32_data = vec![i32::MAX, i32::MIN, 0, -42, 1_000_000];
    let i32_tensor = Tensor::new(&mut manager, &i32_data)?;

    println!("   i32 data: {:?}", i32_data);
    println!("   Tensor size: {} elements", i32_tensor.size());
    println!("   Tensor bytes: {} bytes", i32_tensor.size_bytes());
    println!("   Data type: {:?}", i32_tensor.data_type()?);
    assert_eq!(i32_tensor.data_type()?, DataType::Int32);
    println!("   ✅ i32 tensor created with extreme values");

    // 5. Demonstrate u32 tensors (unsigned integers)
    println!("\n5. Creating u32 tensors...");
    let u32_data = vec![u32::MAX, u32::MIN, 42, 1_000_000, 0xDEADBEEF];
    let u32_tensor = Tensor::new(&mut manager, &u32_data)?;

    println!("   u32 data: {:?}", u32_data);
    println!("   Tensor size: {} elements", u32_tensor.size());
    println!("   Tensor bytes: {} bytes", u32_tensor.size_bytes());
    println!("   Data type: {:?}", u32_tensor.data_type()?);
    assert_eq!(u32_tensor.data_type()?, DataType::UInt32);
    println!("   ✅ u32 tensor created with large unsigned values");

    // 6. Demonstrate filled tensor creation
    println!("\n6. Creating filled tensors...");

    let zeros_f32: Tensor<f32> = manager.tensor_filled(1000, 0.0)?;
    let ones_i32: Tensor<i32> = manager.tensor_filled(500, 1)?;
    let max_u32: Tensor<u32> = manager.tensor_filled(100, u32::MAX)?;

    println!(
        "   Zeros (f32): {} elements filled with 0.0",
        zeros_f32.size()
    );
    println!("   Ones (i32): {} elements filled with 1", ones_i32.size());
    println!(
        "   Max (u32): {} elements filled with {}",
        max_u32.size(),
        u32::MAX
    );
    println!("   ✅ Filled tensors created efficiently");

    // 7. Demonstrate uninitialized tensor creation
    println!("\n7. Creating uninitialized tensors...");

    let uninit_f64: Tensor<f64> = Tensor::uninit(&mut manager, 2000)?;
    let uninit_i32: Tensor<i32> = Tensor::uninit(&mut manager, 1500)?;

    println!("   Uninit f64: {} elements allocated", uninit_f64.size());
    println!("   Uninit i32: {} elements allocated", uninit_i32.size());
    println!("   ✅ Uninitialized tensors allocated for performance");

    // 8. Demonstrate data retrieval and round-trip
    println!("\n8. Testing data retrieval round-trip...");

    // Test f64 precision preservation
    let retrieved_f64 = f64_tensor.to_vec()?;
    let precision_preserved = retrieved_f64
        .iter()
        .zip(f64_data.iter())
        .all(|(a, b)| (a - b).abs() < 1e-15);

    println!("   f64 precision preserved: {}", precision_preserved);
    assert!(precision_preserved, "f64 precision not preserved");

    // Test i32 value preservation
    let retrieved_i32 = i32_tensor.to_vec()?;
    let values_preserved = retrieved_i32 == i32_data;

    println!("   i32 values preserved: {}", values_preserved);
    assert!(values_preserved, "i32 values not preserved");

    // Test u32 value preservation
    let retrieved_u32 = u32_tensor.to_vec()?;
    let u32_preserved = retrieved_u32 == u32_data;

    println!("   u32 values preserved: {}", u32_preserved);
    assert!(u32_preserved, "u32 values not preserved");

    println!("   ✅ All data types preserved through GPU round-trip");

    // 9. Demonstrate type safety at compile time
    println!("\n9. Demonstrating type safety...");

    // The following would cause compile-time errors:
    // let wrong_type: Tensor<i32> = Tensor::new(&mut manager, &[1.0f32])?; // Error!
    // let bad_assign: Tensor<f64> = f32_tensor; // Error!

    println!("   ✅ Compile-time type safety enforced");

    // 10. Memory usage summary
    println!("\n10. Memory usage summary:");

    let total_elements = f32_tensor.size()
        + f64_tensor.size()
        + i32_tensor.size()
        + u32_tensor.size()
        + zeros_f32.size()
        + ones_i32.size()
        + max_u32.size()
        + uninit_f64.size()
        + uninit_i32.size();

    let total_bytes = f32_tensor.size_bytes()
        + f64_tensor.size_bytes()
        + i32_tensor.size_bytes()
        + u32_tensor.size_bytes()
        + zeros_f32.size_bytes()
        + ones_i32.size_bytes()
        + max_u32.size_bytes()
        + uninit_f64.size_bytes()
        + uninit_i32.size_bytes();

    println!("   Total elements: {}", total_elements);
    println!(
        "   Total GPU memory: {:.2} MB",
        total_bytes as f64 / 1024.0 / 1024.0
    );

    let device_count = Manager::device_count()?;
    println!("   Available devices: {}", device_count);

    // 6. Demonstrate advanced multi-type operations
    demonstrate_advanced_operations(&mut manager)?;

    println!("\n🎉 Multi-type tensor example completed successfully!");
    println!("   • Created tensors for f32, f64, i32, u32 data types");
    println!("   • Demonstrated runtime type detection");
    println!("   • Verified data precision preservation");
    println!("   • Showed efficient filled and uninitialized tensor creation");
    println!("   • Validated compile-time type safety");

    Ok(())
}

/// Demonstrates advanced multi-type tensor operations
fn demonstrate_advanced_operations(manager: &mut Manager) -> Result<()> {
    println!("\n🔬 Advanced Multi-Type Operations:");

    // Create mixed-type computation setup
    let scale_factors = vec![1.5f32, 2.0, 0.5, 3.14159];
    let indices = vec![0u32, 1, 2, 3];
    let flags = vec![1i32, 0, 1, 0];

    let scale_tensor = Tensor::new(manager, &scale_factors)?;
    let index_tensor = Tensor::new(manager, &indices)?;
    let flag_tensor = Tensor::new(manager, &flags)?;

    println!("   Created mixed-type tensor set:");
    println!("   - Scales (f32): {:?}", scale_factors);
    println!("   - Indices (u32): {:?}", indices);
    println!("   - Flags (i32): {:?}", flags);

    // Verify type detection works correctly
    assert_eq!(scale_tensor.data_type()?, DataType::Float32);
    assert_eq!(index_tensor.data_type()?, DataType::UInt32);
    assert_eq!(flag_tensor.data_type()?, DataType::Int32);

    println!("   ✅ Multi-type tensor set created and validated");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_type_tensors() {
        if let Err(e) = main() {
            println!(
                "Multi-type tensor test failed (expected without GPU): {}",
                e
            );
            if e.to_string().contains("device") || e.to_string().contains("Vulkan") {
                println!("Skipping test due to missing GPU/Vulkan support");
            } else {
                panic!("Unexpected error: {}", e);
            }
        }
    }

    #[test]
    fn test_data_type_enum() {
        // Test DataType enum values
        assert_ne!(DataType::Float32, DataType::Float64);
        assert_ne!(DataType::Int32, DataType::UInt32);

        // Test Debug formatting
        let types = [
            DataType::Float32,
            DataType::Float64,
            DataType::Int32,
            DataType::UInt32,
        ];
        for data_type in &types {
            let debug_str = format!("{:?}", data_type);
            assert!(!debug_str.is_empty());
        }
    }
}
