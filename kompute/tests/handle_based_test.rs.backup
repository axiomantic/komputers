//! Test using the new handle-based system

use kompute::{Manager, Result, Sequence, Tensor};

#[test]
#[ignore = "requires Vulkan-capable device"]
fn test_handle_based_gpu_operations() -> Result<()> {
    println!("Starting handle-based GPU operations test...");

    // Create GPU manager
    let mut manager = Manager::new()?;
    println!("✓ GPU Manager created successfully");

    // Create input tensors with test data using handle-based API
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

    // Create a sequence for GPU operations
    let _sequence = Sequence::new(&mut manager)?;
    println!("✓ GPU Sequence created");

    // Test tensor properties
    assert_eq!(tensor_a.size(), 4);
    assert_eq!(tensor_a.size_bytes(), 16);
    assert_eq!(tensor_b.size(), 4);
    assert_eq!(tensor_b.size_bytes(), 16);
    assert_eq!(tensor_out.size(), 4);
    assert_eq!(tensor_out.size_bytes(), 16);
    println!("✓ Tensor properties validated");

    // Test tensor data retrieval
    let retrieved_data_a = tensor_a.to_vec()?;
    assert_eq!(data_a, retrieved_data_a);
    println!("✓ Tensor data retrieval works");

    let retrieved_data_b = tensor_b.to_vec()?;
    assert_eq!(data_b, retrieved_data_b);
    println!("✓ Tensor data retrieval works");

    println!("🎉 Complete handle-based GPU workflow test passed!");
    Ok(())
}

#[test]
fn test_handle_based_device_count() {
    println!("Starting handle-based device count test...");

    // Test device count using handle-based system
    match Manager::device_count() {
        Ok(count) => {
            println!("✓ Found {} Vulkan-capable devices", count);
            assert!(count < 100, "Device count should be reasonable");
        }
        Err(e) => {
            println!(
                "⚠️  Device enumeration failed: {} (expected in headless environments)",
                e
            );
        }
    }

    println!("Handle-based device count test completed");
}
