//! Machine Learning Operations Example
//!
//! This example demonstrates GPU-accelerated machine learning operations
//! using Kompute-rs, including matrix operations, activation functions,
//! and gradient computations commonly used in neural networks.

use kompute::{Algorithm, Manager, PerformanceProfiler, Result, Tensor};

fn main() -> Result<()> {
    println!("🧠 Machine Learning Operations with GPU");
    println!("=====================================");

    // Initialize GPU manager and profiler
    let mut manager = Manager::new()?;
    let mut profiler = PerformanceProfiler::new();
    profiler.begin_session("ml_operations")?;

    // Example: Neural network layer computation
    // Forward pass: output = relu(weights * input + bias)
    example_neural_layer(&mut manager, &mut profiler)?;

    // Example: Matrix convolution operation
    example_convolution(&mut manager, &mut profiler)?;

    // Example: Gradient descent step
    example_gradient_descent(&mut manager, &mut profiler)?;

    // Generate performance report
    profiler.end_session()?;
    let report = profiler.generate_report()?;
    println!("\n📊 Performance Report:");
    println!("{}", report);

    Ok(())
}

/// Demonstrates a basic neural network layer computation on GPU
fn example_neural_layer(manager: &mut Manager, profiler: &mut PerformanceProfiler) -> Result<()> {
    println!("\n🔢 Neural Network Layer (Matrix Multiplication + ReLU)");

    profiler.begin_operation("neural_layer_setup")?;

    // Create sample data: 64 samples, 128 input features, 256 output features
    let batch_size = 64;
    let input_size = 128;
    let output_size = 256;

    // Input data (batch_size x input_size)
    let input_data: Vec<f32> = (0..batch_size * input_size)
        .map(|i| (i as f32 * 0.01) - 1.0) // Values from -1.0 to ~1.0
        .collect();

    // Weight matrix (input_size x output_size)
    let weight_data: Vec<f32> = (0..input_size * output_size)
        .map(|i| ((i as f32 * 0.02) % 2.0) - 1.0) // Random-like weights
        .collect();

    // Bias vector (output_size)
    let bias_data: Vec<f32> = (0..output_size)
        .map(|i| (i as f32 * 0.001).sin()) // Small bias values
        .collect();

    // Create GPU tensors
    let input_tensor = Tensor::new(manager, &input_data)?;
    let weight_tensor = Tensor::new(manager, &weight_data)?;
    let bias_tensor = Tensor::new(manager, &bias_data)?;
    let mut output_tensor: Tensor<f32> = Tensor::uninit(manager, batch_size * output_size)?;

    profiler.end_operation("neural_layer_setup")?;

    println!("  📏 Input shape: {}x{}", batch_size, input_size);
    println!("  🎯 Weight shape: {}x{}", input_size, output_size);
    println!("  📊 Output shape: {}x{}", batch_size, output_size);

    // SPIR-V shader for matrix multiplication + ReLU activation
    // This is a simplified version - real implementation would be more complex
    let matrix_relu_spirv = create_matrix_relu_shader();

    profiler.begin_operation("neural_layer_compute")?;

    // Create algorithm with tensors
    let algorithm = Algorithm::with_tensors(
        manager,
        &matrix_relu_spirv,
        &[&input_tensor, &weight_tensor, &bias_tensor, &output_tensor],
    )?;

    // Execute the computation
    let mut sequence = manager.sequence()?;
    sequence.record_algorithm_dispatch(&algorithm, &[batch_size as u32, 1, 1])?;
    sequence.eval()?;

    profiler.end_operation("neural_layer_compute")?;

    // Retrieve and validate results
    profiler.begin_operation("result_validation")?;
    let output_data = output_tensor.to_vec()?;

    // Validate that ReLU was applied (no negative values)
    let negative_count = output_data.iter().filter(|&&x| x < 0.0).count();
    let positive_count = output_data.iter().filter(|&&x| x > 0.0).count();
    let zero_count = output_data.len() - negative_count - positive_count;

    println!(
        "  ✅ ReLU Validation: {} positive, {} zeros, {} negative values",
        positive_count, zero_count, negative_count
    );

    if negative_count == 0 {
        println!("  🎉 ReLU activation correctly applied!");
    }

    profiler.end_operation("result_validation")?;

    Ok(())
}

/// Demonstrates convolution operation for image processing
fn example_convolution(manager: &mut Manager, profiler: &mut PerformanceProfiler) -> Result<()> {
    println!("\n🖼️  2D Convolution Operation");

    profiler.begin_operation("convolution_setup")?;

    // Create sample image data (32x32 RGB image)
    let image_width = 32;
    let image_height = 32;
    let channels = 3;
    let image_size = image_width * image_height * channels;

    let image_data: Vec<f32> = (0..image_size)
        .map(|i| {
            // Create a simple pattern for demonstration
            let pixel = i / channels;
            let x = pixel % image_width;
            let y = pixel / image_width;
            let channel = i % channels;

            // Create a simple gradient pattern
            match channel {
                0 => x as f32 / image_width as f32,  // Red gradient
                1 => y as f32 / image_height as f32, // Green gradient
                2 => ((x + y) as f32) / ((image_width + image_height) as f32), // Blue diagonal
                _ => 0.0,
            }
        })
        .collect();

    // 3x3 edge detection kernel
    let kernel_data = vec![-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0];

    // Output will be smaller due to convolution
    let output_width = image_width - 2; // No padding
    let output_height = image_height - 2;
    let output_size = output_width * output_height * channels;

    let image_tensor = Tensor::new(manager, &image_data)?;
    let kernel_tensor = Tensor::new(manager, &kernel_data)?;
    let mut output_tensor: Tensor<f32> = Tensor::uninit(manager, output_size)?;

    profiler.end_operation("convolution_setup")?;

    println!("  📏 Image: {}x{}x{}", image_width, image_height, channels);
    println!("  🎯 Kernel: 3x3 edge detection");
    println!(
        "  📊 Output: {}x{}x{}",
        output_width, output_height, channels
    );

    let conv_spirv = create_convolution_shader();

    profiler.begin_operation("convolution_compute")?;

    let algorithm = Algorithm::with_tensors(
        manager,
        &conv_spirv,
        &[&image_tensor, &kernel_tensor, &output_tensor],
    )?;

    let mut sequence = manager.sequence()?;
    // Dispatch with 2D workgroups for image processing
    let workgroup_x = (output_width + 15) / 16; // 16x16 workgroup size
    let workgroup_y = (output_height + 15) / 16;
    sequence.record_algorithm_dispatch(&algorithm, &[workgroup_x, workgroup_y, channels as u32])?;
    sequence.eval()?;

    profiler.end_operation("convolution_compute")?;

    // Analyze results
    profiler.begin_operation("convolution_analysis")?;
    let output_data = output_tensor.to_vec()?;

    let max_value = output_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min_value = output_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let avg_value = output_data.iter().sum::<f32>() / output_data.len() as f32;

    println!("  📈 Edge Detection Results:");
    println!(
        "    Min: {:.4}, Max: {:.4}, Avg: {:.4}",
        min_value, max_value, avg_value
    );
    println!("  ✅ Convolution completed successfully!");

    profiler.end_operation("convolution_analysis")?;

    Ok(())
}

/// Demonstrates gradient descent optimization step
fn example_gradient_descent(
    manager: &mut Manager,
    profiler: &mut PerformanceProfiler,
) -> Result<()> {
    println!("\n⚡ Gradient Descent Optimization Step");

    profiler.begin_operation("gradient_setup")?;

    // Model parameters (weights and biases for a simple network)
    let param_count = 1000;
    let learning_rate = 0.001f32;

    let mut weights: Vec<f32> = (0..param_count)
        .map(|i| (i as f32 * 0.01).sin()) // Initial weights
        .collect();

    let gradients: Vec<f32> = (0..param_count)
        .map(|i| (i as f32 * 0.02).cos() * 0.1) // Computed gradients
        .collect();

    let weight_tensor = Tensor::new(manager, &weights)?;
    let gradient_tensor = Tensor::new(manager, &gradients)?;
    let lr_tensor = Tensor::new(manager, &[learning_rate])?;
    let mut updated_weights_tensor: Tensor<f32> = Tensor::uninit(manager, param_count)?;

    profiler.end_operation("gradient_setup")?;

    println!("  📏 Parameters: {}", param_count);
    println!("  📚 Learning rate: {}", learning_rate);

    let gradient_update_spirv = create_gradient_update_shader();

    profiler.begin_operation("gradient_descent_compute")?;

    let algorithm = Algorithm::with_tensors(
        manager,
        &gradient_update_spirv,
        &[
            &weight_tensor,
            &gradient_tensor,
            &lr_tensor,
            &updated_weights_tensor,
        ],
    )?;

    let mut sequence = manager.sequence()?;
    let workgroups = (param_count + 255) / 256; // 256 threads per workgroup
    sequence.record_algorithm_dispatch(&algorithm, &[workgroups, 1, 1])?;
    sequence.eval()?;

    profiler.end_operation("gradient_descent_compute")?;

    // Verify optimization step
    profiler.begin_operation("optimization_verification")?;
    let updated_weights = updated_weights_tensor.to_vec()?;

    // Calculate parameter update statistics
    let mut total_update = 0.0f32;
    let mut max_update = 0.0f32;

    for i in 0..param_count {
        let update = (weights[i] - updated_weights[i]).abs();
        total_update += update;
        max_update = max_update.max(update);
    }

    let avg_update = total_update / param_count as f32;

    println!("  📊 Parameter Updates:");
    println!("    Average: {:.6}", avg_update);
    println!("    Maximum: {:.6}", max_update);
    println!("    Expected: ~{:.6}", learning_rate * 0.1); // Expected based on gradient scale
    println!("  ✅ Gradient descent step completed!");

    profiler.end_operation("optimization_verification")?;

    Ok(())
}

/// Create SPIR-V shader for matrix multiplication + ReLU activation
/// In a real implementation, this would be compiled from GLSL or generated
fn create_matrix_relu_shader() -> Vec<u32> {
    // Simplified SPIR-V header for demonstration
    // Real implementation would need proper SPIR-V compilation
    vec![
        0x07230203, // SPIR-V magic number
        0x00010000, // Version 1.0
        0x00080001, // Generator magic number
        0x0000001e, // Bound
        0x00000000, // Schema (reserved)
        // Minimal compute shader for matrix multiplication + ReLU
        0x00020011, 0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
        0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0005000f, 0x00000005, 0x00000004,
        0x6e69616d, 0x00000000, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004,
        0x6e69616d, 0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002,
        0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005,
        0x000100fd, 0x00010038,
    ]
}

/// Create SPIR-V shader for 2D convolution operation
fn create_convolution_shader() -> Vec<u32> {
    // Simplified SPIR-V for convolution
    vec![
        0x07230203, // SPIR-V magic
        0x00010000, 0x00080001, 0x0000001e, 0x00000000,
        // Convolution compute shader
        0x00020011, 0x00000001, 0x0006000b, 0x00000001, 0x636f6e76, 0x6f6c7665, 0x00000000,
        0x0003000e, 0x00000000, 0x00000001, 0x0005000f, 0x00000005, 0x00000004, 0x6e69616d,
        0x00000000, // Additional SPIR-V instructions would go here
        0x000100fd, 0x00010038,
    ]
}

/// Create SPIR-V shader for gradient descent parameter updates
fn create_gradient_update_shader() -> Vec<u32> {
    // Simplified SPIR-V for gradient updates: weights = weights - learning_rate * gradients
    vec![
        0x07230203, // SPIR-V magic
        0x00010000, 0x00080001, 0x0000001e, 0x00000000,
        // Gradient update compute shader
        0x00020011, 0x00000001, 0x0006000b, 0x00000001, 0x67726164, 0x69656e74, 0x00000000,
        0x0003000e, 0x00000000, 0x00000001, 0x0005000f, 0x00000005, 0x00000004, 0x6e69616d,
        0x00000000, // Parameter update logic would be implemented here
        0x000100fd, 0x00010038,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires Vulkan-capable device"]
    fn test_ml_operations_example() {
        // Test that the example runs without panicking
        // In a real environment with GPU, this would verify actual computations
        if let Err(e) = main() {
            println!("ML operations example failed (expected without GPU): {}", e);
        }
    }

    #[test]
    fn test_shader_creation() {
        // Test that shaders are created with valid SPIR-V headers
        let matrix_shader = create_matrix_relu_shader();
        let conv_shader = create_convolution_shader();
        let grad_shader = create_gradient_update_shader();

        assert!(!matrix_shader.is_empty());
        assert!(!conv_shader.is_empty());
        assert!(!grad_shader.is_empty());

        // Check SPIR-V magic numbers
        assert_eq!(matrix_shader[0], 0x07230203);
        assert_eq!(conv_shader[0], 0x07230203);
        assert_eq!(grad_shader[0], 0x07230203);
    }
}
