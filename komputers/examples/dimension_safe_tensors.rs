//! Dimension-Safe Tensor Operations Example
//!
//! This example demonstrates Komputers' revolutionary compile-time dimension safety
//! using const generics. Dimension mismatches are caught before your code even runs!

use komputers::{DimensionTensor, Manager, Result, Tensor};

fn main() -> Result<()> {
    println!("🔒 Komputers Dimension-Safe Tensors Example");
    println!("============================================");

    // Initialize GPU manager
    let mut manager = Manager::new()?;
    println!("✅ GPU manager initialized");

    // 1. Demonstrate compile-time dimension safety
    demonstrate_compile_time_safety(&mut manager)?;

    // 2. Demonstrate runtime validation for dynamic tensors
    demonstrate_runtime_validation(&mut manager)?;

    // 3. Demonstrate dimension-safe operations
    demonstrate_dimension_safe_operations(&mut manager)?;

    println!("\n🎉 Dimension-safe tensor example completed!");
    println!("   • Compile-time dimension validation prevents errors");
    println!("   • Runtime validation catches dynamic tensor issues");
    println!("   • Zero-cost abstractions for maximum safety");

    Ok(())
}

/// Demonstrate compile-time dimension safety with static tensors
fn demonstrate_compile_time_safety(manager: &mut Manager) -> Result<()> {
    println!("\n🛡️ Compile-Time Dimension Safety:");

    // ✅ Create dimension-safe tensors with compile-time size validation
    let tensor_4: DimensionTensor<f32, 4> =
        DimensionTensor::new_safe(manager, &[1.0, 2.0, 3.0, 4.0])?;

    let tensor_3: DimensionTensor<f32, 3> = DimensionTensor::new_safe(manager, &[1.0, 2.0, 3.0])?;

    println!(
        "   ✅ Created DimensionTensor<f32, 4>: {} elements",
        tensor_4.compile_time_size()
    );
    println!(
        "   ✅ Created DimensionTensor<f32, 3>: {} elements",
        tensor_3.compile_time_size()
    );

    // Validate compile-time properties
    assert!(tensor_4.is_dimension_safe());
    assert!(tensor_3.is_dimension_safe());
    assert_eq!(tensor_4.compile_time_size(), 4);
    assert_eq!(tensor_3.compile_time_size(), 3);

    println!("   🔒 Compile-time validation: PASSED");

    // ❌ These would NOT compile - dimension mismatches caught at compile time!
    // let bad_tensor: DimensionTensor<f32, 4> =
    //     DimensionTensor::new_safe(manager, &[1.0, 2.0])?; // COMPILE ERROR!

    Ok(())
}

/// Demonstrate runtime validation for dynamic tensors
fn demonstrate_runtime_validation(manager: &mut Manager) -> Result<()> {
    println!("\n⚡ Runtime Validation for Dynamic Tensors:");

    // Dynamic tensors (Tensor<T> = DimensionTensor<T, 0>) use runtime validation
    let dynamic_tensor = Tensor::new(manager, &[1.0f32, 2.0, 3.0, 4.0, 5.0])?;

    println!(
        "   ✅ Created dynamic Tensor<f32>: {} elements",
        dynamic_tensor.size()
    );
    println!(
        "   🔍 Compile-time size: {} (0 = dynamic)",
        dynamic_tensor.compile_time_size()
    );
    println!(
        "   📊 Is dimension-safe: {}",
        dynamic_tensor.is_dimension_safe()
    );

    // Demonstrate conversion from dynamic to static with validation
    match validate_and_convert_to_static(manager, &[1.0, 2.0, 3.0]) {
        Ok(static_tensor) => {
            println!("   ✅ Successfully converted to DimensionTensor<f32, 3>");
            println!(
                "      Compile-time size: {}",
                static_tensor.compile_time_size()
            );
        }
        Err(e) => {
            println!("   ❌ Conversion failed: {}", e);
        }
    }

    Ok(())
}

/// Helper function that validates dynamic data and creates a static tensor
fn validate_and_convert_to_static(
    manager: &mut Manager,
    data: &[f32],
) -> Result<DimensionTensor<f32, 3>> {
    if data.len() != 3 {
        return Err(komputers::KomputeError::tensor(format!(
            "Expected exactly 3 elements, got {}",
            data.len()
        )));
    }

    DimensionTensor::new(manager, data)
}

/// Demonstrate dimension-safe operations
fn demonstrate_dimension_safe_operations(manager: &mut Manager) -> Result<()> {
    println!("\n🚀 Dimension-Safe Operations:");

    // Create dimension-safe tensors for different operations
    let vector_3d: DimensionTensor<f32, 3> = DimensionTensor::zeros(manager, 3)?;
    let matrix_2x2: DimensionTensor<f32, 4> = DimensionTensor::ones(manager, 4)?;
    let custom_tensor: DimensionTensor<f32, 5> = DimensionTensor::filled(manager, 5, 3.14159)?;

    println!(
        "   ✅ 3D Vector: {} elements (compile-time validated)",
        vector_3d.compile_time_size()
    );
    println!(
        "   ✅ 2x2 Matrix: {} elements (compile-time validated)",
        matrix_2x2.compile_time_size()
    );
    println!(
        "   ✅ Custom Tensor: {} elements (compile-time validated)",
        custom_tensor.compile_time_size()
    );

    // Demonstrate compile-time size matching
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor_6: DimensionTensor<f32, 6> = DimensionTensor::new_safe(manager, &data)?;

    println!(
        "   🔍 Created tensor with exactly {} elements",
        tensor_6.compile_time_size()
    );

    // Show validation in action
    match DimensionTensor::<f32, 4>::new(manager, &[1.0, 2.0]) {
        Ok(_) => println!("   ❌ This should not happen!"),
        Err(e) => println!("   ✅ Runtime validation caught dimension mismatch: {}", e),
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_safety_compile_time() {
        // Test compile-time dimension validation
        assert!(DimensionTensor::<f32, 4>::validate_dimensions());
        assert_eq!(DimensionTensor::<f32, 4>::compile_time_size(), 4);
        assert!(DimensionTensor::<f32, 4>::is_dimension_safe());

        // Test dynamic tensor properties
        assert_eq!(Tensor::<f32>::compile_time_size(), 0);
        assert!(!Tensor::<f32>::is_dimension_safe());
    }

    #[test]
    #[ignore = "requires Vulkan-capable device"]
    fn test_dimension_safe_tensor_creation() {
        if let Ok(mut manager) = Manager::new() {
            // Test static tensor creation
            let data = [1.0f32, 2.0, 3.0, 4.0];
            let tensor: DimensionTensor<f32, 4> =
                DimensionTensor::new_safe(&mut manager, &data).unwrap();

            assert_eq!(tensor.size(), 4);
            assert_eq!(tensor.compile_time_size(), 4);
            assert!(tensor.is_dimension_safe());
        }
    }

    #[test]
    #[ignore = "requires Vulkan-capable device"]
    fn test_dimension_validation_error() {
        if let Ok(mut manager) = Manager::new() {
            // This should fail at runtime with clear error message
            let result = DimensionTensor::<f32, 4>::new(&mut manager, &[1.0, 2.0]);
            assert!(result.is_err());

            let error_msg = result.unwrap_err().to_string();
            assert!(error_msg.contains("Dimension mismatch"));
            assert!(error_msg.contains("expects 4 elements but got 2"));
        }
    }
}
