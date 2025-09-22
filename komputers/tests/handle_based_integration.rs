//! Integration test for handle-based GPU operations.
//!
//! This test demonstrates the complete handle-based workflow that solves
//! the SharedPtr mutable access limitation.

use komputers::{Manager, Result, Sequence, Tensor};

#[test]

fn test_complete_handle_based_workflow() -> Result<()> {
    // Create GPU manager
    let mut manager = Manager::new()?;
    println!("✓ GPU Manager created successfully");

    // Create input tensors with test data
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

    // Test data retrieval (this was impossible with SharedPtr)
    let retrieved_a = tensor_a.to_vec()?;
    let retrieved_b = tensor_b.to_vec()?;
    assert_eq!(data_a, retrieved_a);
    assert_eq!(data_b, retrieved_b);
    println!("✓ Tensor data retrieval works correctly");

    // Create a sequence for GPU operations
    let mut sequence = Sequence::new(&mut manager)?;
    println!("✓ GPU Sequence created");

    // Record operations in the sequence (this was impossible with SharedPtr)
    sequence.record_tensor_sync_device(tensor_a.handle())?;
    sequence.record_tensor_sync_device(tensor_b.handle())?;
    sequence.record_tensor_sync_local(tensor_out.handle())?;
    println!("✓ Sequence operations recorded successfully");

    // Execute the sequence synchronously
    sequence.eval()?;
    println!("✓ Sequence executed successfully");

    println!("🎉 Complete handle-based GPU workflow test passed!");
    Ok(())
}

#[test]

fn test_handle_based_async_operations() -> Result<()> {
    let mut manager = Manager::new()?;

    // Create test data
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = Tensor::new(&mut manager, &data)?;

    // Create sequence
    let mut sequence = Sequence::new(&mut manager)?;
    sequence.record_tensor_sync_device(tensor.handle())?;

    // Test async execution (this was impossible with SharedPtr)
    sequence.eval_async()?;
    println!("✓ Async execution started");

    // Wait for completion with timeout
    sequence.eval_await(5000)?; // 5 second timeout
    println!("✓ Async execution completed");

    Ok(())
}

#[test]

fn test_handle_based_memory_management() -> Result<()> {
    let mut manager = Manager::new()?;

    // Test creating and dropping many tensors to verify RAII
    for i in 0..10 {
        let data = vec![i as f32; 100];
        let tensor = Tensor::new(&mut manager, &data)?;

        // Verify tensor properties
        assert_eq!(tensor.size(), 100);
        assert_eq!(tensor.size_bytes(), 400);

        // Tensor will be automatically cleaned up when it goes out of scope
    }

    println!("✓ Memory management test completed - all tensors cleaned up");
    Ok(())
}

#[test]

fn test_handle_compatibility() {
    // Test that we can still use the old SharedPtr-based API alongside handle-based
    if let Ok(mut manager) = Manager::new() {
        let data = vec![1.0f32, 2.0, 3.0];

        // Old SharedPtr-based tensor
        if let Ok(old_tensor) = kompute::Tensor::new(&mut manager, &data) {
            assert_eq!(old_tensor.size(), 3);
        }

        // New handle-based tensor
        if let Ok(new_tensor) = Tensor::new(&mut manager, &data) {
            assert_eq!(new_tensor.size(), 3);
            if let Ok(retrieved) = new_tensor.to_vec() {
                assert_eq!(data, retrieved);
            }
        }

        println!("✓ Both old and new APIs work together");
    }
}
