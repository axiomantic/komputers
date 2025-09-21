//! Basic GPU Computing Example for Kompute-rs
//!
//! This example demonstrates the most fundamental GPU computing operations:
//! - Creating tensors
//! - Performing basic GPU operations
//! - Retrieving results
//!
//! This serves as an introduction to GPU computing with Kompute.

use kompute::{Manager, Result, Tensor};

fn main() -> Result<()> {
    println!("🚀 Kompute Basic GPU Computing Example");

    // Initialize the GPU manager
    let mut manager = Manager::new()?;
    println!("✅ GPU Manager initialized successfully");

    // Create input tensors
    let data_a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let data_b = vec![2.0f32, 3.0, 4.0, 5.0, 6.0];

    let tensor_a: Tensor<f32> = Tensor::new(&mut manager, &data_a)?;
    let tensor_b: Tensor<f32> = Tensor::new(&mut manager, &data_b)?;
    let tensor_out: Tensor<f32> = Tensor::uninit(&mut manager, data_a.len())?;

    println!(
        "✅ Created {} tensors with {} elements each",
        3,
        data_a.len()
    );

    // Simple elementwise addition using the built-in SPIR-V shader
    use kompute::spirv_shaders::ELEMENTWISE_ADD_F32_SPIRV;

    // Create algorithm with tensors
    let algorithm = kompute::Algorithm::with_tensors(
        &mut manager,
        ELEMENTWISE_ADD_F32_SPIRV,
        &[&tensor_a, &tensor_b, &tensor_out],
    )?;

    let mut sequence = manager.sequence()?;
    sequence.record_algorithm_dispatch(algorithm.handle())?;
    sequence.eval()?;

    // Get the results
    let result_data = tensor_out.to_vec()?;

    println!("📊 GPU Computation Results:");
    println!("   Input A:  {:?}", data_a);
    println!("   Input B:  {:?}", data_b);
    println!("   Output:   {:?}", result_data);

    // Verify the results
    for i in 0..data_a.len() {
        let expected = data_a[i] + data_b[i];
        assert!(
            (result_data[i] - expected).abs() < 1e-6,
            "GPU computation error at index {}: expected {}, got {}",
            i,
            expected,
            result_data[i]
        );
    }

    println!("✅ GPU computation verified successfully!");
    println!("🎉 Basic GPU computing example completed");

    Ok(())
}
