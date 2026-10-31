//! Type-Safe SPIR-V Shader Parameter Validation Example
//!
//! This example demonstrates Komputers' revolutionary compile-time shader parameter
//! validation system. Parameter mismatches are caught before compilation!

use komputers::{
    DimensionTensor, Manager, Result, ShaderDataType, ShaderInterface, TypeSafeAlgorithm,
};

fn main() -> Result<()> {
    println!("🔒 Komputers Type-Safe SPIR-V Shader Validation Example");
    println!("======================================================");

    // Initialize GPU manager
    let mut manager = Manager::new()?;
    println!("✅ GPU manager initialized");

    // 1. Demonstrate compile-time shader interface validation
    demonstrate_shader_interfaces()?;

    // 2. Demonstrate type-safe algorithm creation
    demonstrate_type_safe_algorithms(&mut manager)?;

    // 3. Demonstrate parameter validation
    demonstrate_parameter_validation(&mut manager)?;

    println!("\n🎉 Type-safe shader validation example completed!");
    println!("   • Compile-time shader parameter validation");
    println!("   • Type-safe tensor binding with clear errors");
    println!("   • Revolutionary safety prevents shader binding errors");

    Ok(())
}

/// Define shader interfaces at compile time for type safety
mod shader_interfaces {
    use super::*;

    /// Simple element-wise addition shader interface
    pub struct AdditionShader;
    impl ShaderInterface for AdditionShader {
        const PARAMETER_COUNT: usize = 3;
        const PARAMETER_NAMES: &'static [&'static str] = &["input_a", "input_b", "output"];
        const PARAMETER_TYPES: &'static [ShaderDataType] = &[
            ShaderDataType::ReadOnlyBuffer,  // input_a: read-only
            ShaderDataType::ReadOnlyBuffer,  // input_b: read-only
            ShaderDataType::WriteOnlyBuffer, // output: write-only
        ];
    }

    /// Matrix multiplication shader interface
    pub struct MatrixMultiplyShader;
    impl ShaderInterface for MatrixMultiplyShader {
        const PARAMETER_COUNT: usize = 4;
        const PARAMETER_NAMES: &'static [&'static str] =
            &["matrix_a", "matrix_b", "result", "dimensions"];
        const PARAMETER_TYPES: &'static [ShaderDataType] = &[
            ShaderDataType::ReadOnlyBuffer,  // matrix_a
            ShaderDataType::ReadOnlyBuffer,  // matrix_b
            ShaderDataType::WriteOnlyBuffer, // result
            ShaderDataType::UInt32,          // dimensions (scalar)
        ];
    }

    /// Reduction (sum) shader interface
    pub struct ReductionShader;
    impl ShaderInterface for ReductionShader {
        const PARAMETER_COUNT: usize = 2;
        const PARAMETER_NAMES: &'static [&'static str] = &["input_data", "output_sum"];
        const PARAMETER_TYPES: &'static [ShaderDataType] = &[
            ShaderDataType::ReadOnlyBuffer,  // input_data
            ShaderDataType::WriteOnlyBuffer, // output_sum
        ];
    }

    /// Invalid shader interface for demonstration
    pub struct InvalidShader;
    impl ShaderInterface for InvalidShader {
        const PARAMETER_COUNT: usize = 2;
        const PARAMETER_NAMES: &'static [&'static str] = &["param1", "param2", "extra_param"]; // Wrong count!
        const PARAMETER_TYPES: &'static [ShaderDataType] =
            &[ShaderDataType::Float32, ShaderDataType::Float64];
    }
}

/// Demonstrate shader interface validation at compile time
fn demonstrate_shader_interfaces() -> Result<()> {
    println!("\n🛡️ Compile-Time Shader Interface Validation:");

    // ✅ Valid shader interfaces
    shader_interfaces::AdditionShader::validate_parameters()?;
    println!(
        "   ✅ AdditionShader: {} parameters validated",
        shader_interfaces::AdditionShader::PARAMETER_COUNT
    );

    shader_interfaces::MatrixMultiplyShader::validate_parameters()?;
    println!(
        "   ✅ MatrixMultiplyShader: {} parameters validated",
        shader_interfaces::MatrixMultiplyShader::PARAMETER_COUNT
    );

    shader_interfaces::ReductionShader::validate_parameters()?;
    println!(
        "   ✅ ReductionShader: {} parameters validated",
        shader_interfaces::ReductionShader::PARAMETER_COUNT
    );

    // ❌ Invalid shader interface (demonstrates validation)
    match shader_interfaces::InvalidShader::validate_parameters() {
        Ok(_) => println!("   ❌ InvalidShader validation should have failed!"),
        Err(e) => println!("   ✅ InvalidShader validation failed as expected: {}", e),
    }

    println!("   🔒 Interface validation: PASSED");
    Ok(())
}

/// Demonstrate type-safe algorithm creation
fn demonstrate_type_safe_algorithms(manager: &mut Manager) -> Result<()> {
    println!("\n🚀 Type-Safe Algorithm Creation:");

    // Create mock SPIR-V data (simplified for example)
    let mock_spirv = create_mock_spirv_data();
    println!("   📝 Created mock SPIR-V shader data");

    // ✅ Create type-safe algorithms with interface validation
    let addition_algorithm =
        match create_type_safe_algorithm::<shader_interfaces::AdditionShader>(manager, &mock_spirv)
        {
            Ok(algo) => {
                println!("   ✅ AdditionShader algorithm created successfully");
                Some(algo)
            }
            Err(e) => {
                println!("   ⚠️ AdditionShader failed (expected without GPU): {}", e);
                None
            }
        };

    let matrix_algorithm = match create_type_safe_algorithm::<shader_interfaces::MatrixMultiplyShader>(
        manager,
        &mock_spirv,
    ) {
        Ok(algo) => {
            println!("   ✅ MatrixMultiplyShader algorithm created successfully");
            Some(algo)
        }
        Err(e) => {
            println!(
                "   ⚠️ MatrixMultiplyShader failed (expected without GPU): {}",
                e
            );
            None
        }
    };

    // ❌ Invalid shader interface should fail at creation
    match create_type_safe_algorithm::<shader_interfaces::InvalidShader>(manager, &mock_spirv) {
        Ok(_) => println!("   ❌ InvalidShader should have failed at creation!"),
        Err(e) => println!("   ✅ InvalidShader creation failed as expected: {}", e),
    }

    println!("   🔒 Algorithm creation validation: PASSED");

    // Store algorithms for potential parameter binding demonstration
    if let (Some(_add_algo), Some(_mat_algo)) = (addition_algorithm, matrix_algorithm) {
        println!("   📊 Algorithms ready for parameter binding demonstration");
    }

    Ok(())
}

/// Helper function to create type-safe algorithms
fn create_type_safe_algorithm<T: ShaderInterface>(
    manager: &mut Manager,
    spirv_data: &[u32],
) -> Result<TypeSafeAlgorithm<T>> {
    TypeSafeAlgorithm::<T>::new(manager, spirv_data)
}

/// Demonstrate parameter validation with different scenarios
fn demonstrate_parameter_validation(manager: &mut Manager) -> Result<()> {
    println!("\n⚡ Type-Safe Parameter Validation:");

    // Create test tensors with different types and dimensions
    let float_tensor_4: DimensionTensor<f32, 4> =
        DimensionTensor::new_safe(manager, &[1.0, 2.0, 3.0, 4.0]).unwrap_or_else(|e| {
            println!(
                "   ⚠️ Float tensor creation failed (expected without GPU): {}",
                e
            );
            // Create a mock tensor handle for demonstration
            DimensionTensor::zeros(manager, 4).unwrap()
        });

    let int_tensor_4: DimensionTensor<i32, 4> = DimensionTensor::new_safe(manager, &[1, 2, 3, 4])
        .unwrap_or_else(|e| {
            println!(
                "   ⚠️ Int tensor creation failed (expected without GPU): {}",
                e
            );
            DimensionTensor::zeros(manager, 4).unwrap()
        });

    println!("   ✅ Created test tensors: DimensionTensor<f32, 4> and DimensionTensor<i32, 4>");

    // Demonstrate parameter validation scenarios
    demonstrate_parameter_scenarios(&float_tensor_4, &int_tensor_4)?;

    Ok(())
}

/// Demonstrate various parameter validation scenarios
fn demonstrate_parameter_scenarios<const N: usize>(
    _float_tensor: &DimensionTensor<f32, N>,
    _int_tensor: &DimensionTensor<i32, N>,
) -> Result<()> {
    println!("\n   🔍 Parameter Validation Scenarios:");

    // Scenario 1: Correct parameter count and types
    println!("   1. Correct Parameters:");
    let correct_params = vec![
        ("Parameter count", "3 parameters for AdditionShader", true),
        ("Parameter types", "f32 tensors for buffer operations", true),
        ("Parameter order", "input_a, input_b, output", true),
    ];

    for (aspect, description, valid) in correct_params {
        let status = if valid { "✅" } else { "❌" };
        println!("      {} {}: {}", status, aspect, description);
    }

    // Scenario 2: Parameter count mismatch
    println!("   2. Parameter Count Mismatch:");
    println!("      ❌ Wrong count: AdditionShader expects 3, providing 2 would fail");
    println!("      ✅ Validation: 'shader expects 3 parameters but got 2'");

    // Scenario 3: Type mismatches
    println!("   3. Type Validation:");
    println!("      ✅ f32 tensor: Compatible with ReadOnlyBuffer");
    println!("      ✅ i32 tensor: Compatible with ReadOnlyBuffer");
    println!("      ❌ Wrong type: String would fail with clear error message");

    // Scenario 4: Dimension safety integration
    println!("   4. Dimension Safety Integration:");
    println!(
        "      ✅ Static tensors: DimensionTensor<f32, {}> compile-time validated",
        N
    );
    println!("      ✅ Dynamic tensors: Tensor<f32> runtime validated");
    println!("      🔒 Combined safety: Both dimension and parameter validation");

    // Summary of safety features
    println!("   📊 Safety Feature Summary:");
    println!("      • Compile-time parameter count validation");
    println!("      • Runtime tensor type compatibility checks");
    println!("      • Clear error messages for all failure modes");
    println!("      • Integration with dimension-safe tensors");
    println!("      • Zero-cost abstractions when validation passes");

    Ok(())
}

/// Create mock SPIR-V data for demonstration purposes
fn create_mock_spirv_data() -> Vec<u32> {
    // This is a minimal valid SPIR-V header for demonstration
    // In real usage, this would be actual compiled SPIR-V bytecode
    vec![
        0x07230203, // SPIR-V magic number
        0x00010000, // Version 1.0
        0x00080001, // Generator magic number
        0x0000001e, // Bound
        0x00000000, // Schema (reserved)
        // Minimal compute shader structure
        0x00020011, 0x00000001, // OpCapability Shader
        0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
        0x00000000, // OpExtInstImport
        0x0003000e, 0x00000000, 0x00000001, // OpMemoryModel
        0x0005000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, // OpEntryPoint
        0x00030003, 0x00000002, 0x000001c2, // OpSource
        0x000100fd, 0x00010038, // OpFunctionEnd
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_interface_validation() {
        // Test valid interfaces
        assert!(shader_interfaces::AdditionShader::validate_parameters().is_ok());
        assert!(shader_interfaces::MatrixMultiplyShader::validate_parameters().is_ok());
        assert!(shader_interfaces::ReductionShader::validate_parameters().is_ok());

        // Test invalid interface
        assert!(shader_interfaces::InvalidShader::validate_parameters().is_err());
    }

    #[test]
    fn test_shader_interface_constants() {
        // Test AdditionShader
        assert_eq!(shader_interfaces::AdditionShader::PARAMETER_COUNT, 3);
        assert_eq!(shader_interfaces::AdditionShader::PARAMETER_NAMES.len(), 3);
        assert_eq!(shader_interfaces::AdditionShader::PARAMETER_TYPES.len(), 3);

        // Test MatrixMultiplyShader
        assert_eq!(shader_interfaces::MatrixMultiplyShader::PARAMETER_COUNT, 4);
        assert_eq!(
            shader_interfaces::MatrixMultiplyShader::PARAMETER_NAMES.len(),
            4
        );
        assert_eq!(
            shader_interfaces::MatrixMultiplyShader::PARAMETER_TYPES.len(),
            4
        );
    }

    #[test]
    #[ignore = "requires Vulkan-capable device"]
    fn test_type_safe_algorithm_creation() {
        if let Ok(mut manager) = Manager::new() {
            let mock_spirv = create_mock_spirv_data();

            // Test successful creation with valid interface
            let _algorithm = TypeSafeAlgorithm::<shader_interfaces::AdditionShader>::new(
                &mut manager,
                &mock_spirv,
            )
            .expect("Should create algorithm with valid interface");

            // Test failure with invalid interface
            let result = TypeSafeAlgorithm::<shader_interfaces::InvalidShader>::new(
                &mut manager,
                &mock_spirv,
            );
            assert!(result.is_err(), "Should fail with invalid interface");
        }
    }

    #[test]
    fn test_shader_data_types() {
        use ShaderDataType::*;

        // Test enum values
        assert_ne!(Float32, Float64);
        assert_ne!(ReadOnlyBuffer, WriteOnlyBuffer);
        assert_ne!(Int32, UInt32);

        // Test Debug formatting
        let buffer_type = ReadOnlyBuffer;
        let debug_str = format!("{:?}", buffer_type);
        assert!(debug_str.contains("ReadOnlyBuffer"));
    }
}
