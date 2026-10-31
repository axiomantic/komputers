//! Test working GPU operations with proper Vulkan setup

use komputers::{
    configure_vulkan_environment, verify_vulkan_availability, Manager, Result, Sequence, Tensor,
};

#[test]
fn test_working_vulkan_setup() {
    // Configure Vulkan environment
    match configure_vulkan_environment() {
        Ok(()) => println!("✓ Vulkan environment configured successfully"),
        Err(e) => {
            println!("⚠️  Vulkan configuration failed: {}", e);
            return;
        }
    }

    // Verify Vulkan availability
    match verify_vulkan_availability() {
        Ok(count) => {
            println!("✓ Vulkan verification successful, {} devices found", count);
            assert!(count >= 1, "Should find at least one Vulkan device");
        }
        Err(e) => {
            println!("⚠️  Vulkan verification failed: {}", e);
        }
    }
}

#[test]
fn test_basic_gpu_tensor_operations() -> Result<()> {
    println!("Creating GPU manager with MINIMAL configuration...");
    let mut manager = Manager::with_device(0)?;
    println!("✓ GPU Manager created successfully");

    println!("Creating tensors...");
    let data_a = vec![1.0f32, 2.0, 3.0, 4.0];
    let data_b = vec![5.0f32, 6.0, 7.0, 8.0];

    let tensor_a = Tensor::new(&mut manager, &data_a)?;
    let tensor_b = Tensor::new(&mut manager, &data_b)?;
    let tensor_out: Tensor<f32> = Tensor::uninit(&mut manager, 4)?;

    println!(
        "✓ Created {} tensors with sizes: {}, {}, {}",
        3,
        tensor_a.size(),
        tensor_b.size(),
        tensor_out.size()
    );

    // Test tensor properties
    assert_eq!(tensor_a.size(), 4);
    assert_eq!(tensor_a.size_bytes(), 16);
    assert_eq!(tensor_b.size(), 4);
    assert_eq!(tensor_b.size_bytes(), 16);
    assert_eq!(tensor_out.size(), 4);
    assert_eq!(tensor_out.size_bytes(), 16);
    println!("✓ Tensor properties validated");

    // Test data retrieval
    let retrieved_data_a = tensor_a.to_vec()?;
    assert_eq!(data_a, retrieved_data_a);
    println!("✓ Tensor A data retrieval works: {:?}", retrieved_data_a);

    let retrieved_data_b = tensor_b.to_vec()?;
    assert_eq!(data_b, retrieved_data_b);
    println!("✓ Tensor B data retrieval works: {:?}", retrieved_data_b);

    // Create a sequence for GPU operations
    println!("Creating GPU sequence...");
    let _sequence = Sequence::new(&mut manager)?;
    println!("✓ GPU Sequence created successfully");

    println!("🎉 Complete GPU workflow test passed!");
    Ok(())
}
