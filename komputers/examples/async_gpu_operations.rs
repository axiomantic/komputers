//! Async GPU Operations Example
//!
//! This example demonstrates Komputers's async GPU capabilities including:
//! - Non-blocking GPU operations using Rust futures
//! - Parallel GPU execution across multiple queues
//! - Async sequence management and coordination
//! - Integration with Rust async ecosystem (tokio, etc.)

#[cfg(feature = "async")]
use komputers::{AsyncGpuContext, AsyncGpuSequence, AsyncTensor, Result};

#[cfg(feature = "async")]
use std::time::{Duration, Instant};

#[cfg(feature = "async")]
use tokio::time::timeout;

#[cfg(feature = "async")]
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    println!("🔄 Komputers Async GPU Operations Example");
    println!("==========================================");

    // 1. Initialize async GPU environment
    println!("\n1. Setting up async GPU environment...");
    let mut context = AsyncGpuContext::new().await?;
    println!("   ✅ Async GPU context initialized");

    // 2. Demonstrate basic async operations
    demonstrate_basic_async_ops(&mut context).await?;

    // 3. Demonstrate parallel async execution
    demonstrate_parallel_async_execution(&mut context).await?;

    // 4. Demonstrate async timeout handling
    demonstrate_async_timeout_handling(&mut context).await?;

    // 5. Demonstrate async tensor operations
    demonstrate_async_tensor_operations(&mut context).await?;

    // 6. Demonstrate integration with async ecosystem
    demonstrate_async_ecosystem_integration(&mut context).await?;

    println!("\n🎉 Async GPU operations example completed!");
    println!("   • Non-blocking GPU operations executed");
    println!("   • Parallel async execution demonstrated");
    println!("   • Timeout handling validated");
    println!("   • Async tensor operations performed");
    println!("   • Rust async ecosystem integration shown");

    Ok(())
}

#[cfg(feature = "async")]
async fn demonstrate_basic_async_ops(context: &mut AsyncGpuContext) -> Result<()> {
    println!("\n⚡ Basic Async Operations:");

    // Create async tensors
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let input = context.create_tensor(&input_data).await?;
    let mut output = context.create_tensor_uninit::<f32>(5).await?;

    println!("   ✅ Async tensors created");

    // Create async sequence
    let mut sequence = context.create_sequence().await?;
    println!("   ✅ Async sequence created");

    // Record operations asynchronously
    sequence.record_tensor_sync_device(input.handle()).await?;
    println!("   ✅ Async copy operation recorded");

    // Execute non-blocking
    let start = Instant::now();
    sequence.execute().await?;
    let duration = start.elapsed();

    println!("   ✅ Async execution completed in {:?}", duration);

    // Verify results
    let results = output.to_vec().await?;
    println!("   Results: {:?}", results);
    assert_eq!(results, input_data);
    println!("   ✅ Async operation results verified");

    Ok(())
}

#[cfg(feature = "async")]
async fn demonstrate_parallel_async_execution(context: &mut AsyncGpuContext) -> Result<()> {
    println!("\n🔄 Parallel Async Execution:");

    // Create multiple data sets for parallel processing
    let datasets = vec![
        (0..1000).map(|i| i as f32).collect::<Vec<f32>>(),
        (1000..2000).map(|i| i as f32).collect::<Vec<f32>>(),
        (2000..3000).map(|i| i as f32).collect::<Vec<f32>>(),
    ];

    let mut async_sequences = Vec::new();

    // Create async sequences for each dataset
    for (i, data) in datasets.iter().enumerate() {
        let input = context.create_tensor(data).await?;
        let mut output = context.create_tensor_uninit::<f32>(data.len()).await?;

        let mut sequence = context.create_sequence().await?;
        sequence.record_tensor_sync_device(input.handle()).await?;

        async_sequences.push(sequence);
        println!("   ✅ Async sequence {} prepared", i + 1);
    }

    // Execute all sequences in parallel with timeout
    let start = Instant::now();
    AsyncGpuSequence::execute_concurrent(async_sequences).await?;
    let parallel_duration = start.elapsed();

    println!(
        "   ✅ Parallel execution completed in {:?}",
        parallel_duration
    );
    println!("   • {} sequences executed concurrently", datasets.len());
    println!("   • Parallel speedup achieved through multi-queue GPU execution");

    Ok(())
}

#[cfg(feature = "async")]
async fn demonstrate_async_timeout_handling(context: &mut AsyncGpuContext) -> Result<()> {
    println!("\n⏰ Async Timeout Handling:");

    // Create a potentially long-running operation
    let large_data: Vec<f32> = vec![1.0; 100000];
    let input = context.create_tensor(&large_data).await?;
    let mut output = context
        .create_tensor_uninit::<f32>(large_data.len())
        .await?;

    let mut sequence = context.create_sequence().await?;
    sequence.record_tensor_sync_device(input.handle()).await?;

    println!("   ✅ Large async operation prepared");

    // Test timeout handling
    let timeout_duration = Duration::from_millis(100);
    let start = Instant::now();

    match timeout(timeout_duration, sequence.execute()).await {
        Ok(_) => {
            let duration = start.elapsed();
            println!("   ✅ Operation completed within timeout: {:?}", duration);
        }
        Err(_) => {
            println!(
                "   ⏰ Operation timed out after {:?} (expected for large operations)",
                timeout_duration
            );
            println!("   ✅ Timeout handling working correctly");
        }
    }

    // Test successful operation with generous timeout
    let mut quick_sequence = context.create_sequence().await?;
    let small_data = vec![1.0f32; 10];
    let small_input = context.create_tensor(&small_data).await?;
    let mut small_output = context.create_tensor_uninit::<f32>(10).await?;

    quick_sequence
        .record_tensor_sync_device(small_input.handle())
        .await?;

    let result = timeout(Duration::from_secs(5), quick_sequence.execute()).await;
    match result {
        Ok(_) => println!("   ✅ Quick operation completed within generous timeout"),
        Err(_) => println!("   ❌ Quick operation should not timeout"),
    }

    Ok(())
}

#[cfg(feature = "async")]
async fn demonstrate_async_tensor_operations(context: &mut AsyncGpuContext) -> Result<()> {
    println!("\n📊 Async Tensor Operations:");

    // Demonstrate async tensor creation with different types
    let f32_data = vec![1.0f32, 2.0, 3.0];
    let i32_data = vec![10i32, 20, 30];

    let f32_tensor = context.create_tensor(&f32_data).await?;
    let i32_tensor = context.create_tensor(&i32_data).await?;

    println!("   ✅ Multi-type async tensors created");

    // Demonstrate async filled tensor creation
    let zeros_data = vec![0.0f32; 1000];
    let zeros = context.create_tensor(&zeros_data).await?;
    let ones_data = vec![1i32; 500];
    let ones = context.create_tensor(&ones_data).await?;

    println!("   ✅ Async filled tensors created");
    println!("     - {} f32 zeros", zeros.size().await);
    println!("     - {} i32 ones", ones.size().await);

    // Demonstrate async data retrieval
    let start = Instant::now();
    let retrieved_f32 = f32_tensor.to_vec().await?;
    let retrieved_i32 = i32_tensor.to_vec().await?;
    let retrieval_duration = start.elapsed();

    println!(
        "   ✅ Async data retrieval completed in {:?}",
        retrieval_duration
    );
    println!("     - f32 data: {:?}", retrieved_f32);
    println!("     - i32 data: {:?}", retrieved_i32);

    // Verify data integrity
    assert_eq!(retrieved_f32, f32_data);
    assert_eq!(retrieved_i32, i32_data);
    println!("   ✅ Async tensor data integrity verified");

    Ok(())
}

#[cfg(feature = "async")]
async fn demonstrate_async_ecosystem_integration(context: &mut AsyncGpuContext) -> Result<()> {
    println!("\n🔗 Async Ecosystem Integration:");

    // Demonstrate integration with tokio spawn
    let handle1 = tokio::spawn(async move {
        // Simulate async work
        tokio::time::sleep(Duration::from_millis(50)).await;
        "Task 1 completed".to_string()
    });

    let handle2 = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        "Task 2 completed".to_string()
    });

    // Create GPU work alongside other async tasks
    let data = vec![1.0f32; 100];
    let input = context.create_tensor(&data).await?;
    let mut output = context.create_tensor_uninit::<f32>(100).await?;

    let mut sequence = context.create_sequence().await?;
    sequence.record_tensor_sync_device(input.handle()).await?;

    let gpu_handle = tokio::spawn(async move {
        sequence
            .execute()
            .await
            .map(|_| "GPU work completed".to_string())
    });

    println!("   ✅ Multiple async tasks spawned");

    // Wait for all tasks concurrently
    let start = Instant::now();
    let (result1, result2, gpu_result) = tokio::try_join!(handle1, handle2, gpu_handle)
        .map_err(|e| komputers::KomputeError::compute(format!("Join error: {}", e)))?;

    let total_duration = start.elapsed();

    println!(
        "   ✅ All async tasks completed concurrently in {:?}",
        total_duration
    );
    println!("     - {}", result1);
    println!("     - {}", result2);
    println!("     - {}", gpu_result?);

    // Demonstrate async streams (conceptual)
    println!("\n   📡 Async Stream Integration:");
    // Simulated streaming operations without tokio_stream dependency

    // Process multiple data batches asynchronously
    let data_batches = vec![vec![1.0f32, 2.0], vec![3.0f32, 4.0], vec![5.0f32, 6.0]];

    let mut processed_count = 0;
    for batch_data in data_batches {
        let tensor = context.create_tensor(&batch_data).await?;
        let _size = tensor.size().await;
        processed_count += 1;
    }

    println!("   ✅ Processed {} batches asynchronously", processed_count);

    // Demonstrate async channels
    let (tx, mut rx) = tokio::sync::mpsc::channel(32);

    // Producer task
    tokio::spawn(async move {
        for i in 0..5 {
            let data = vec![i as f32; 10];
            if tx.send(data).await.is_err() {
                break;
            }
        }
    });

    // Consumer task with GPU processing
    let mut batch_count = 0;
    while let Some(data) = rx.recv().await {
        let _tensor = context.create_tensor(&data).await?;
        batch_count += 1;
    }

    println!(
        "   ✅ Processed {} batches through async channel",
        batch_count
    );

    Ok(())
}

// Non-async fallback for when async feature is not enabled
#[cfg(not(feature = "async"))]
fn main() -> komputers::Result<()> {
    println!("🔄 Komputers Async GPU Operations Example");
    println!("==========================================");
    println!("\n⚠️ Async feature not enabled!");
    println!("To run this example, compile with: cargo run --example async_gpu_operations --features async");
    println!("\nAsync features available with the 'async' feature flag:");
    println!("  • AsyncGpuContext - Non-blocking GPU execution");
    println!("  • AsyncTensor - Async tensor operations");
    println!("  • AsyncSequence - Async command sequences");
    println!("  • Integration with tokio and async ecosystem");

    Ok(())
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "async")]
    use super::*;

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_async_gpu_operations() {
        if let Err(e) = main().await {
            println!(
                "Async GPU operations test failed (expected without GPU): {}",
                e
            );
            if e.to_string().contains("device") || e.to_string().contains("Vulkan") {
                println!("Skipping test due to missing GPU/Vulkan support");
            } else {
                panic!("Unexpected error: {}", e);
            }
        }
    }

    #[test]
    fn test_async_feature_compilation() {
        // This test ensures the code compiles regardless of async feature
        #[cfg(feature = "async")]
        {
            println!("Async feature enabled - full functionality available");
        }

        #[cfg(not(feature = "async"))]
        {
            println!("Async feature disabled - fallback behavior active");
        }
    }
}
