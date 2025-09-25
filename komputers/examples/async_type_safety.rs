// Async Type Safety Example
//
// This example demonstrates type-safe async GPU operations
// using Komputers with compile-time and runtime safety checks.

#[cfg(feature = "async")]
use komputers::{
    Manager, Tensor, DataType, Result,
    async_sequence::AsyncSequence
};
use std::time::Duration;

#[cfg(feature = "async")]
#[tokio::main(flavor = "current_thread")]
async fn main() -> komputers::Result<()> {
    println!("=== Async Type-Safe GPU Computing ===");

    // Basic async operations
    basic_async_operations().await?;

    // Error handling in async context
    async_error_handling().await?;

    // Multi-type async operations
    async_multi_type_operations().await?;

    Ok(())
}

#[cfg(not(feature = "async"))]
fn main() {
    println!("This example requires the 'async' feature to be enabled.");
    println!("Run with: cargo run --features async --example async_type_safety");
}

#[cfg(feature = "async")]
async fn basic_async_operations() -> komputers::Result<()> {
    println!("\n--- Basic Async Operations ---");

    // Create manager and tensors
    let mut manager = Manager::new()?;
    let input = Tensor::new(&mut manager, &[1.0f32, 2.0, 3.0, 4.0])?;
    let output: Tensor<f32> = Tensor::uninit(&mut manager, 4)?;

    // Create async sequence with consuming builder pattern
    let result = AsyncSequence::new(&mut manager)?
        .record_tensor_sync_device(input.handle())?
        .record_tensor_sync_device(output.handle())?
        .record_tensor_sync_local(output.handle())?
        .execute_async()
        .await?;

    println!("Async execution completed in {:?}", result.execution_time);
    println!("Operations count: {}", result.operation_count);
    println!("GPU wait time: {:?}", result.gpu_wait_time);
    println!("Async polls: {}", result.async_metrics.poll_count);

    Ok(())
}

#[cfg(feature = "async")]
async fn async_error_handling() -> komputers::Result<()> {
    println!("\n--- Async Error Handling ---");

    let mut manager = Manager::new()?;

    // Demonstrate different error scenarios in async context

    // 1. Tensor creation errors
    match async_tensor_operations(&mut manager).await {
        Ok(result) => {
            println!("Async tensor operations succeeded: {:?}", result);
        }
        Err(komputers::KomputeError::MemoryError(msg)) => {
            eprintln!("Memory error in async context: {}", msg);
        }
        Err(komputers::KomputeError::DeviceError(msg)) => {
            eprintln!("GPU device error: {}", msg);
        }
        Err(other) => {
            eprintln!("Other async error: {}", other);
        }
    }

    // 2. Timeout handling
    match async_operation_with_timeout(&mut manager).await {
        Ok(duration) => {
            println!("Operation completed within timeout: {:?}", duration);
        }
        Err(e) => {
            eprintln!("Operation failed or timed out: {}", e);
        }
    }

    Ok(())
}

#[cfg(feature = "async")]
async fn async_tensor_operations(manager: &mut Manager) -> komputers::Result<Vec<f32>> {
    // Create type-safe tensor
    let input = Tensor::new(manager, &[1.0f32, 2.0, 3.0, 4.0])?;

    // Verify type safety
    let data_type = input.data_type()?;
    assert_eq!(data_type, DataType::Float32);

    // Execute async operations with type safety
    let _result = AsyncSequence::new(manager)?
        .record_tensor_sync_device(input.handle())?
        .record_tensor_sync_local(input.handle())?
        .execute_async()
        .await?;

    // Return processed data
    input.to_vec()
}

#[cfg(feature = "async")]
async fn async_operation_with_timeout(manager: &mut Manager) -> komputers::Result<Duration> {
    let input = Tensor::new(manager, &[1.0f32; 1000])?; // Larger dataset

    let execution = AsyncSequence::new(manager)?
        .record_tensor_sync_device(input.handle())?
        .record_tensor_sync_local(input.handle())?
        .execute_async();

    // Add timeout to async operation
    match tokio::time::timeout(Duration::from_secs(10), execution).await {
        Ok(Ok(result)) => Ok(result.execution_time),
        Ok(Err(gpu_error)) => Err(gpu_error),
        Err(_timeout) => Err(komputers::KomputeError::compute("Async GPU operation timed out")),
    }
}

/// Demonstrates type-safe tensor handling with async operations
#[cfg(feature = "async")]
async fn async_multi_type_operations() -> komputers::Result<()> {
    println!("\n--- Async Multi-Type Operations ---");

    let mut manager = Manager::new()?;

    // Create tensors of different types
    let f32_tensor = Tensor::new(&mut manager, &[1.0f32, 2.0, 3.0])?;
    let i32_tensor = Tensor::new(&mut manager, &[1i32, 2, 3])?;

    // Process f32 data asynchronously
    let f32_result = AsyncSequence::new(&mut manager)?
        .record_tensor_sync_device(f32_tensor.handle())?
        .record_tensor_sync_local(f32_tensor.handle())?
        .execute_async()
        .await?;

    // Process i32 data asynchronously
    let i32_result = AsyncSequence::new(&mut manager)?
        .record_tensor_sync_device(i32_tensor.handle())?
        .record_tensor_sync_local(i32_tensor.handle())?
        .execute_async()
        .await?;

    // Verify type safety at runtime
    assert_eq!(f32_tensor.data_type()?, DataType::Float32);
    assert_eq!(i32_tensor.data_type()?, DataType::Int32);

    println!("F32 processing time: {:?}", f32_result.execution_time);
    println!("I32 processing time: {:?}", i32_result.execution_time);
    println!("F32 data: {:?}", f32_tensor.to_vec()?);
    println!("I32 data: {:?}", i32_tensor.to_vec()?);

    Ok(())
}

/// Utility function for handling async GPU errors with retry logic
#[cfg(feature = "async")]
async fn handle_async_gpu_error<F, T>(operation: F) -> komputers::Result<T>
where
    F: std::future::Future<Output = komputers::Result<T>>,
{
    match operation.await {
        Ok(result) => Ok(result),
        Err(komputers::KomputeError::DeviceError(msg)) => {
            eprintln!("GPU device error: {}", msg);
            // Could implement retry logic here
            Err(komputers::KomputeError::DeviceError(msg))
        }
        Err(komputers::KomputeError::MemoryError(msg)) => {
            eprintln!("GPU memory error: {}", msg);
            // Could implement memory cleanup here
            Err(komputers::KomputeError::MemoryError(msg))
        }
        Err(other) => {
            eprintln!("Other GPU error: {}", other);
            Err(other)
        }
    }
}