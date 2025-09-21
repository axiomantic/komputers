//! Basic GPU Computing Example
//!
//! This example demonstrates the fundamental GPU computing capabilities of Komputers,
//! showcasing how to create tensors, execute high-level GPU operations, and validate results.

use kompute::{GpuOperations, Manager, Result, Tensor};

fn main() -> Result<()> {
    println!("🚀 Komputers Basic GPU Computing Example");
    println!("==========================================");

    // 1. Initialize GPU operations
    println!("\n1. Creating GPU operations context...");
    let mut ops = GpuOperations::new()?;
    println!("   ✅ GPU operations context created successfully");

    // 2. Create input tensors with data
    println!("\n2. Creating input tensors...");
    let input_a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let input_b_data = [10.0f32, 20.0, 30.0, 40.0, 50.0];

    let input_a = Tensor::new(ops.manager(), &input_a_data)?;
    let input_b = Tensor::new(ops.manager(), &input_b_data)?;
    let mut output: Tensor<f32> = Tensor::uninit(ops.manager(), 5)?;

    println!("   Input A: {:?}", input_a_data);
    println!("   Input B: {:?}", input_b_data);
    println!("   ✅ Input tensors created");

    // 3. Execute GPU computation
    println!("\n3. Executing GPU element-wise addition...");
    let start = std::time::Instant::now();
    ops.elementwise_add(&input_a, &input_b, &mut output)?;
    let duration = start.elapsed();

    println!("   ✅ GPU computation completed in {:?}", duration);

    // 4. Retrieve and display results
    println!("\n4. Retrieving results...");
    let results = output.to_vec()?;

    println!("   Results: {:?}", results);
    println!("   Expected: [11.0, 22.0, 33.0, 44.0, 55.0]");

    // Verify results
    let expected = [11.0f32, 22.0, 33.0, 44.0, 55.0];
    let success = results
        .iter()
        .zip(expected.iter())
        .all(|(a, b)| (a - b).abs() < 1e-6);

    if success {
        println!("   ✅ Results match expected values!");
    } else {
        println!("   ❌ Results don't match expected values");
        return Err(kompute::KomputeError::compute(
            "Computation verification failed".to_string(),
        ));
    }

    // 5. Display system information
    println!("\n5. System Information:");
    let device_count = Manager::device_count()?;
    println!("   Available GPU devices: {}", device_count);

    println!("\n🎉 Basic GPU computing example completed successfully!");
    println!("   • Created tensors on GPU memory");
    println!("   • Executed high-level GPU operations");
    println!("   • Verified computation results");
    println!(
        "   • Performance: {:?} for 5-element vector addition",
        duration
    );

    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_gpu_computing() {
        // This test may require GPU hardware
        if let Err(e) = main() {
            println!("GPU computing test failed (expected without GPU): {}", e);
            // Don't fail the test if GPU is not available
            if e.to_string().contains("device") || e.to_string().contains("Vulkan") {
                println!("Skipping test due to missing GPU/Vulkan support");
            } else {
                panic!("Unexpected error: {}", e);
            }
        }
    }
}
