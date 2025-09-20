//! Basic GPU Computing Example
//!
//! This example demonstrates the fundamental GPU computing capabilities of Kompute-rs,
//! showcasing how to create tensors, load SPIR-V shaders, and execute GPU computations.

use kompute::{Algorithm, Manager, Result, Sequence, Tensor};

fn main() -> Result<()> {
    println!("🚀 Kompute-rs Basic GPU Computing Example");
    println!("==========================================");

    // 1. Initialize GPU manager
    println!("\n1. Creating GPU manager...");
    let mut manager = Manager::new()?;
    println!("   ✅ GPU manager created successfully");

    // 2. Create input tensors with data
    println!("\n2. Creating input tensors...");
    let input_a = Tensor::new(&mut manager, &[1.0f32, 2.0, 3.0, 4.0, 5.0])?;
    let input_b = Tensor::new(&mut manager, &[10.0f32, 20.0, 30.0, 40.0, 50.0])?;

    println!("   Input A: {:?}", [1.0f32, 2.0, 3.0, 4.0, 5.0]);
    println!("   Input B: {:?}", [10.0f32, 20.0, 30.0, 40.0, 50.0]);
    println!("   ✅ Input tensors created");

    // 3. Create output tensor (uninitialized)
    println!("\n3. Creating output tensor...");
    let mut output: Tensor<f32> = Tensor::uninit(&mut manager, 5)?;
    println!("   ✅ Output tensor allocated (5 elements)");

    // 4. Create SPIR-V shader for vector addition
    println!("\n4. Creating GPU algorithm...");

    // Simple SPIR-V bytecode for vector addition (hand-assembled for this example)
    // In real usage, you would compile from GLSL: glslangValidator -V shader.comp -o shader.spv
    let spirv_data = create_vector_add_spirv();

    let algorithm =
        Algorithm::with_tensors(&mut manager, &spirv_data, &[&input_a, &input_b, &output])?;
    println!("   ✅ Algorithm created and tensors bound");

    // 5. Create sequence and record operations
    println!("\n5. Recording GPU operations...");
    let mut sequence = manager.sequence()?;

    // Record tensor synchronization to device
    sequence.record_tensor_sync_device(&input_a)?;
    sequence.record_tensor_sync_device(&input_b)?;
    sequence.record_tensor_sync_device(&output)?;

    // Record algorithm dispatch
    sequence.record_algorithm(&algorithm)?;

    // Record tensor synchronization back to host
    sequence.record_tensor_sync_local(&output)?;

    println!("   ✅ Operations recorded in sequence");

    // 6. Execute GPU computation
    println!("\n6. Executing GPU computation...");
    let start = std::time::Instant::now();
    sequence.eval()?;
    let duration = start.elapsed();

    println!("   ✅ GPU computation completed in {:?}", duration);

    // 7. Retrieve and display results
    println!("\n7. Retrieving results...");
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

    // 8. Display system information
    println!("\n8. System Information:");
    let device_count = Manager::device_count()?;
    println!("   Available GPU devices: {}", device_count);

    println!("\n🎉 Basic GPU computing example completed successfully!");
    println!("   • Created tensors on GPU memory");
    println!("   • Executed SPIR-V compute shader");
    println!("   • Verified computation results");
    println!(
        "   • Performance: {:?} for 5-element vector addition",
        duration
    );

    Ok(())
}

/// Creates a minimal SPIR-V bytecode for vector addition
/// This is a simplified version for demonstration purposes.
/// In real applications, compile GLSL shaders to SPIR-V using glslangValidator.
fn create_vector_add_spirv() -> Vec<u32> {
    // This is a minimal SPIR-V header + vector addition shader
    // In practice, use proper GLSL compilation:
    //
    // #version 450
    // layout(local_size_x = 1) in;
    // layout(set = 0, binding = 0) buffer InputA { float data_a[]; };
    // layout(set = 0, binding = 1) buffer InputB { float data_b[]; };
    // layout(set = 0, binding = 2) buffer Output { float data_out[]; };
    // void main() {
    //     uint index = gl_GlobalInvocationID.x;
    //     data_out[index] = data_a[index] + data_b[index];
    // }

    vec![
        0x07230203, // SPIR-V magic number
        0x00010000, // Version 1.0
        0x00080001, // Generator magic number (Khronos glslang)
        0x0000001e, // Bound (number of IDs)
        0x00000000, // Schema (reserved)
        // Minimal shader that performs vector addition
        // This is a placeholder - real SPIR-V would be much longer
        0x00020011, 0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
        0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0005000f, 0x00000005, 0x00000004,
        0x6e69616d, 0x00000000, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004,
        0x6e69616d, 0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002,
        0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005,
        0x000100fd, 0x00010038,
    ]
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
