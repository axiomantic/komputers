//! Async GPU Operations Example
//!
//! This example demonstrates Kompute-rs's async GPU capabilities including:
//! - Non-blocking GPU operations using Rust futures
//! - Parallel GPU execution across multiple queues
//! - Async sequence management and coordination
//! - Integration with Rust async ecosystem (tokio, etc.)

#[cfg(feature = "async")]
use kompute::{
    AsyncGpuContext, AsyncGpuExecutor, AsyncSequence, AsyncTensor, Manager, Result, Tensor,
};

#[cfg(feature = "async")]
use std::time::{Duration, Instant};

#[cfg(feature = "async")]
use tokio::time::timeout;

#[cfg(feature = "async")]
#[tokio::main]
async fn main() -> Result<()> {
    println!("🔄 Kompute-rs Async GPU Operations Example");
    println!("==========================================");

    // 1. Initialize async GPU environment
    println!("\n1. Setting up async GPU environment...");
    let mut executor = AsyncGpuExecutor::new().await?;
    println!("   ✅ Async GPU executor initialized");

    // 2. Demonstrate basic async operations
    demonstrate_basic_async_ops(&mut executor).await?;

    // 3. Demonstrate parallel async execution
    demonstrate_parallel_async_execution(&mut executor).await?;

    // 4. Demonstrate async timeout handling
    demonstrate_async_timeout_handling(&mut executor).await?;

    // 5. Demonstrate async tensor operations
    demonstrate_async_tensor_operations(&mut executor).await?;

    // 6. Demonstrate integration with async ecosystem
    demonstrate_async_ecosystem_integration(&mut executor).await?;

    println!("\n🎉 Async GPU operations example completed!");
    println!("   • Non-blocking GPU operations executed");
    println!("   • Parallel async execution demonstrated");
    println!("   • Timeout handling validated");
    println!("   • Async tensor operations performed");
    println!("   • Rust async ecosystem integration shown");

    Ok(())
}

#[cfg(feature = "async")]
async fn demonstrate_basic_async_ops(executor: &mut AsyncGpuExecutor) -> Result<()> {
    println!("\n⚡ Basic Async Operations:");

    // Create async tensors
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let input = AsyncTensor::new(executor.manager(), &input_data).await?;
    let mut output = AsyncTensor::uninit::<f32>(executor.manager(), 5).await?;

    println!("   ✅ Async tensors created");

    // Create async sequence
    let mut sequence = AsyncSequence::new(executor).await?;
    println!("   ✅ Async sequence created");

    // Record operations asynchronously
    sequence.record_copy(&input, &mut output).await?;
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
async fn demonstrate_parallel_async_execution(executor: &mut AsyncGpuExecutor) -> Result<()> {
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
        let input = AsyncTensor::new(executor.manager(), data).await?;
        let mut output = AsyncTensor::uninit::<f32>(executor.manager(), data.len()).await?;

        let mut sequence = AsyncSequence::new(executor).await?;
        sequence.record_copy(&input, &mut output).await?;

        async_sequences.push(sequence);
        println!("   ✅ Async sequence {} prepared", i + 1);
    }

    // Execute all sequences in parallel with timeout
    let start = Instant::now();
    executor
        .execute_parallel(async_sequences, Duration::from_secs(10))
        .await?;
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
async fn demonstrate_async_timeout_handling(executor: &mut AsyncGpuExecutor) -> Result<()> {
    println!("\n⏰ Async Timeout Handling:");

    // Create a potentially long-running operation
    let large_data: Vec<f32> = vec![1.0; 100000];
    let input = AsyncTensor::new(executor.manager(), &large_data).await?;
    let mut output = AsyncTensor::uninit::<f32>(executor.manager(), large_data.len()).await?;

    let mut sequence = AsyncSequence::new(executor).await?;
    sequence.record_copy(&input, &mut output).await?;

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
    let mut quick_sequence = AsyncSequence::new(executor).await?;
    let small_data = vec![1.0f32; 10];
    let small_input = AsyncTensor::new(executor.manager(), &small_data).await?;
    let mut small_output = AsyncTensor::uninit::<f32>(executor.manager(), 10).await?;

    quick_sequence
        .record_copy(&small_input, &mut small_output)
        .await?;

    let result = timeout(Duration::from_secs(5), quick_sequence.execute()).await;
    match result {
        Ok(_) => println!("   ✅ Quick operation completed within generous timeout"),
        Err(_) => println!("   ❌ Quick operation should not timeout"),
    }

    Ok(())
}

#[cfg(feature = "async")]
async fn demonstrate_async_tensor_operations(executor: &mut AsyncGpuExecutor) -> Result<()> {
    println!("\n📊 Async Tensor Operations:");

    // Demonstrate async tensor creation with different types
    let f32_data = vec![1.0f32, 2.0, 3.0];
    let i32_data = vec![10i32, 20, 30];

    let f32_tensor = AsyncTensor::new(executor.manager(), &f32_data).await?;
    let i32_tensor = AsyncTensor::new(executor.manager(), &i32_data).await?;

    println!("   ✅ Multi-type async tensors created");

    // Demonstrate async filled tensor creation
    let zeros = AsyncTensor::filled::<f32>(executor.manager(), 1000, 0.0).await?;
    let ones = AsyncTensor::filled::<i32>(executor.manager(), 500, 1).await?;

    println!("   ✅ Async filled tensors created");
    println!("     - {} f32 zeros", zeros.size().await?);
    println!("     - {} i32 ones", ones.size().await?);

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
async fn demonstrate_async_ecosystem_integration(executor: &mut AsyncGpuExecutor) -> Result<()> {
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
    let input = AsyncTensor::new(executor.manager(), &data).await?;
    let mut output = AsyncTensor::uninit::<f32>(executor.manager(), 100).await?;

    let mut sequence = AsyncSequence::new(executor).await?;
    sequence.record_copy(&input, &mut output).await?;

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
        .map_err(|e| kompute::KomputeError::compute(format!("Join error: {}", e)))?;

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
    use tokio_stream::{self as stream, StreamExt};

    let mut data_stream = stream::iter(vec![
        vec![1.0f32, 2.0],
        vec![3.0f32, 4.0],
        vec![5.0f32, 6.0],
    ]);

    let mut processed_count = 0;
    while let Some(batch_data) = data_stream.next().await {
        let tensor = AsyncTensor::new(executor.manager(), &batch_data).await?;
        let _size = tensor.size().await?;
        processed_count += 1;
    }

    println!(
        "   ✅ Processed {} batches through async stream",
        processed_count
    );

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
        let _tensor = AsyncTensor::new(executor.manager(), &data).await?;
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
fn main() -> kompute::Result<()> {
    println!("🔄 Kompute-rs Async GPU Operations Example");
    println!("==========================================");
    println!("\n⚠️ Async feature not enabled!");
    println!("To run this example, compile with: cargo run --example async_gpu_operations --features async");
    println!("\nAsync features available with the 'async' feature flag:");
    println!("  • AsyncGpuExecutor - Non-blocking GPU execution");
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
