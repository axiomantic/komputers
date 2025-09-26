// Type-Safe GPU Computing Example
//
// This example demonstrates type-safe GPU computing patterns
// using the Komputers library with compile-time safety features.

use komputers::{Algorithm, DataType, Manager, Result, Tensor, VulkanConfig};

fn main() -> Result<()> {
    // Create GPU manager
    let config = VulkanConfig::auto_detect().map_err(|e| komputers::KomputeError::vulkan(&e))?;
    let mut manager = Manager::with_config(config)?;

    // Type-safe tensor creation
    basic_tensor_operations(&mut manager)?;

    // Algorithm binding with type safety
    algorithm_type_safety(&mut manager)?;

    // Multi-type tensor handling
    multi_type_example(&mut manager)?;

    Ok(())
}

/// Demonstrates basic type-safe tensor operations
fn basic_tensor_operations(manager: &mut Manager) -> Result<()> {
    println!("=== Basic Type-Safe Tensor Operations ===");

    // Create tensors with explicit types
    let f32_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let f32_tensor = Tensor::new(manager, &f32_data)?;

    // Type safety: These operations know the tensor type at compile time
    println!("F32 tensor size: {}", f32_tensor.size());
    println!("F32 tensor type: {:?}", f32_tensor.data_type()?);

    // Create uninitialized tensor with known type
    let result_tensor: Tensor<f32> = Tensor::uninit(manager, 4)?;

    // GPU memory operations with type safety - use consuming builder pattern
    let execution_result = manager
        .sequence()?
        .record_tensor_sync_device(f32_tensor.handle())?
        .record_tensor_sync_device(result_tensor.handle())?
        .execute()?;

    println!("Execution time: {:?}", execution_result.execution_time);

    Ok(())
}

/// Demonstrates algorithm binding with type safety
fn algorithm_type_safety(manager: &mut Manager) -> Result<()> {
    println!("\n=== Algorithm Type Safety ===");

    // Create typed tensors for algorithm
    let input_a = Tensor::new(manager, &[1.0f32, 2.0, 3.0, 4.0])?;
    let input_b = Tensor::new(manager, &[5.0f32, 6.0, 7.0, 8.0])?;
    let output: Tensor<f32> = Tensor::uninit(manager, 4)?;

    // Use built-in SPIR-V shader for element-wise addition
    use komputers::spirv_shaders::ELEMENTWISE_ADD_F32_SPIRV;

    // Create algorithm with tensor binding
    let algorithm = Algorithm::with_tensors(
        manager,
        ELEMENTWISE_ADD_F32_SPIRV,
        &[&input_a, &input_b, &output],
    )?;

    // Execute with type-safe operations using consuming builder pattern
    let _result = manager
        .sequence()?
        .record_algorithm_dispatch(algorithm.handle())?
        .execute()?;

    // Get results with type safety
    let output_data = output.to_vec()?;
    println!("Input A: {:?}", input_a.to_vec()?);
    println!("Input B: {:?}", input_b.to_vec()?);
    println!("Output (A + B): {:?}", output_data);

    // Verify results
    let data_a = [1.0f32, 2.0, 3.0, 4.0];
    let data_b = [5.0f32, 6.0, 7.0, 8.0];
    for i in 0..4 {
        let expected = data_a[i] + data_b[i];
        let actual = output_data[i];
        assert!(
            (expected - actual).abs() < 0.001,
            "Computation error at index {}",
            i
        );
    }

    println!("Computation verified successfully");

    Ok(())
}

/// Demonstrates multi-type tensor handling
fn multi_type_example(manager: &mut Manager) -> Result<()> {
    println!("\n=== Multi-Type Tensor Handling ===");

    // Create tensors of different types
    let f32_tensor = Tensor::new(manager, &[1.0f32, 2.0, 3.0])?;
    let f64_tensor = Tensor::new(manager, &[1.0f64, 2.0, 3.0])?;
    let i32_tensor = Tensor::new(manager, &[1i32, 2, 3])?;
    let u32_tensor = Tensor::new(manager, &[1u32, 2, 3])?;

    // Runtime type checking
    verify_tensor_type(&f32_tensor, DataType::Float32)?;
    verify_tensor_type(&f64_tensor, DataType::Float64)?;
    verify_tensor_type(&i32_tensor, DataType::Int32)?;
    verify_tensor_type(&u32_tensor, DataType::UInt32)?;

    println!("All tensor types verified successfully");

    Ok(())
}

/// Verify tensor has expected data type
fn verify_tensor_type<T: komputers::TensorData>(
    tensor: &Tensor<T>,
    expected_type: DataType,
) -> Result<()> {
    let actual_type = tensor.data_type()?;
    if actual_type == expected_type {
        println!("✓ Tensor type {:?} verified", expected_type);
        Ok(())
    } else {
        Err(komputers::KomputeError::TensorError(format!(
            "Expected {:?}, found {:?}",
            expected_type, actual_type
        )))
    }
}

// Helper functions have been removed - using inline data instead

/// Error handling patterns for type-safe operations
#[allow(dead_code)]
fn error_handling_patterns(manager: &mut Manager) -> Result<()> {
    println!("\n=== Error Handling Patterns ===");

    // Proper error handling with context
    match Tensor::new(manager, &[1.0f32]) {
        Ok(tensor) => {
            println!("Tensor created successfully: {} elements", tensor.size());

            // Type-safe operations with error handling
            match tensor.data_type() {
                Ok(data_type) => {
                    println!("Tensor data type: {:?}", data_type);
                }
                Err(e) => {
                    eprintln!("Failed to get tensor data type: {}", e);
                }
            }
        }
        Err(komputers::KomputeError::MemoryError(msg)) => {
            eprintln!("Memory allocation failed: {}", msg);
            // Handle memory errors specifically
        }
        Err(komputers::KomputeError::DeviceError(msg)) => {
            eprintln!("GPU device error: {}", msg);
            // Handle GPU setup errors
        }
        Err(other) => {
            eprintln!("Other error occurred: {}", other);
            // Handle other error types
        }
    }

    Ok(())
}
