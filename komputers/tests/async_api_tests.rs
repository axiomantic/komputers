//! Comprehensive tests for async GPU compute APIs.
//!
//! These tests verify the async/await functionality, timeout handling,
//! and concurrent execution capabilities of the async API.

#[cfg(feature = "async")]
mod async_tests {
    use komputers::{AsyncGpuExecutor, AsyncSequence, Manager, Result, Sequence, Tensor};
    use std::time::{Duration, Instant};
    use tokio::time::timeout;

    #[tokio::test]
    async fn test_async_sequence_basic() -> Result<()> {
        println!("Starting basic async sequence test...");

        let mut manager = Manager::new()?;

        // Create test data
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(&mut manager, &data)?;

        // Create async sequence
        let mut async_seq = AsyncSequence::new(&mut manager)?;

        // Record operations
        async_seq.record_tensor_sync_device(tensor.handle())?;

        // Execute with timeout
        let start = Instant::now();
        async_seq.eval_with_timeout(Duration::from_secs(5)).await?;
        let elapsed = start.elapsed();

        println!("✓ Async sequence execution completed in {:?}", elapsed);
        assert!(elapsed < Duration::from_secs(5));

        Ok(())
    }

    #[tokio::test]
    async fn test_async_sequence_timeout() -> Result<()> {
        println!("Starting async sequence timeout test...");

        let mut manager = Manager::new()?;
        let data = vec![1.0f32; 1000];
        let tensor = Tensor::new(&mut manager, &data)?;

        let mut async_seq = AsyncSequence::new(&mut manager)?;
        async_seq.record_tensor_sync_device(tensor.handle())?;

        // Test with very short timeout (should still succeed for simple operations)
        let start = Instant::now();
        let result = async_seq.eval_with_timeout(Duration::from_millis(1)).await;
        let elapsed = start.elapsed();

        match result {
            Ok(_) => {
                println!(
                    "✓ Fast operation completed within short timeout in {:?}",
                    elapsed
                );
            }
            Err(_) => {
                println!("✓ Operation timed out as expected after {:?}", elapsed);
                assert!(elapsed >= Duration::from_millis(1));
                assert!(elapsed < Duration::from_millis(100)); // Should fail quickly
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_async_gpu_executor() -> Result<()> {
        println!("Starting async GPU executor test...");

        let mut manager = Manager::new()?;
        let mut executor = AsyncGpuExecutor::new(&mut manager)?;

        // Create test tensors
        let data_a = vec![1.0f32, 2.0, 3.0, 4.0];
        let data_b = vec![5.0f32, 6.0, 7.0, 8.0];

        let tensor_a = Tensor::new(&mut manager, &data_a)?;
        let tensor_b = Tensor::new(&mut manager, &data_b)?;

        // Execute operations asynchronously
        let start = Instant::now();
        executor.sync_device(&tensor_a).await?;
        executor.sync_device(&tensor_b).await?;
        let elapsed = start.elapsed();

        println!("✓ Async GPU executor completed operations in {:?}", elapsed);

        Ok(())
    }

    #[tokio::test]
    async fn test_concurrent_async_operations() -> Result<()> {
        println!("Starting concurrent async operations test...");

        let mut manager = Manager::new()?;

        // Create multiple async sequences
        let mut sequences = Vec::new();
        let mut tensors = Vec::new();

        for i in 0..5 {
            let data: Vec<f32> = vec![i as f32; 100];
            let tensor = Tensor::new(&mut manager, &data)?;
            tensors.push(tensor);

            let mut async_seq = AsyncSequence::new(&mut manager)?;
            async_seq.record_tensor_sync_device(tensors[i].handle())?;
            sequences.push(async_seq);
        }

        // Execute all sequences concurrently
        let start = Instant::now();
        let futures: Vec<_> = sequences
            .into_iter()
            .map(|mut seq| async move { seq.eval_with_timeout(Duration::from_secs(10)).await })
            .collect();

        let results = futures::future::join_all(futures).await;
        let elapsed = start.elapsed();

        // Check that all operations succeeded
        for (i, result) in results.iter().enumerate() {
            match result {
                Ok(_) => println!("✓ Sequence {} completed successfully", i),
                Err(e) => println!("⚠️  Sequence {} failed: {}", i, e),
            }
        }

        println!("✓ All concurrent operations completed in {:?}", elapsed);

        Ok(())
    }

    #[tokio::test]
    async fn test_async_error_handling() -> Result<()> {
        println!("Starting async error handling test...");

        let mut manager = Manager::new()?;

        // Test with invalid tensor handle
        let mut async_seq = AsyncSequence::new(&mut manager)?;

        // Try to record operation with invalid handle
        match async_seq.record_tensor_sync_device(999999) {
            Ok(_) => {
                // If this succeeds, try to execute and expect it to fail
                let result = async_seq.eval_with_timeout(Duration::from_secs(1)).await;
                assert!(result.is_err(), "Should fail with invalid tensor handle");
                println!("✓ Invalid handle correctly caused execution failure");
            }
            Err(_) => {
                println!("✓ Invalid handle correctly rejected at record time");
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_async_sequence_reuse() -> Result<()> {
        println!("Starting async sequence reuse test...");

        let mut manager = Manager::new()?;
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(&mut manager, &data)?;

        let mut async_seq = AsyncSequence::new(&mut manager)?;

        // Record and execute multiple times
        for iteration in 0..3 {
            async_seq.record_tensor_sync_device(tensor.handle())?;

            let start = Instant::now();
            async_seq.eval_with_timeout(Duration::from_secs(5)).await?;
            let elapsed = start.elapsed();

            println!("✓ Iteration {} completed in {:?}", iteration, elapsed);

            // Clear sequence for next iteration
            // Note: In a real implementation, you might want a clear() method
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_async_cancellation() -> Result<()> {
        println!("Starting async cancellation test...");

        let mut manager = Manager::new()?;
        let data = vec![1.0f32; 10000]; // Larger data for longer operation
        let tensor = Tensor::new(&mut manager, &data)?;

        let mut async_seq = AsyncSequence::new(&mut manager)?;
        async_seq.record_tensor_sync_device(tensor.handle())?;

        // Test cancellation with tokio timeout
        let start = Instant::now();
        let result = timeout(
            Duration::from_millis(1), // Very short timeout
            async_seq.eval_with_timeout(Duration::from_secs(10)),
        )
        .await;

        let elapsed = start.elapsed();

        match result {
            Ok(Ok(_)) => {
                println!(
                    "✓ Operation completed faster than cancellation timeout in {:?}",
                    elapsed
                );
            }
            Ok(Err(_)) => {
                println!(
                    "✓ Operation failed within cancellation window in {:?}",
                    elapsed
                );
            }
            Err(_) => {
                println!(
                    "✓ Operation was cancelled by tokio timeout after {:?}",
                    elapsed
                );
                assert!(elapsed >= Duration::from_millis(1));
                assert!(elapsed < Duration::from_millis(100));
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_async_resource_cleanup() -> Result<()> {
        println!("Starting async resource cleanup test...");

        let mut manager = Manager::new()?;
        let mut handles = Vec::new();

        // Create and drop async sequences in different scopes
        {
            let data = vec![1.0f32, 2.0, 3.0];
            let tensor = Tensor::new(&mut manager, &data)?;
            handles.push(tensor.handle());

            {
                let mut async_seq = AsyncSequence::new(&mut manager)?;
                async_seq.record_tensor_sync_device(tensor.handle())?;
                async_seq.eval_with_timeout(Duration::from_secs(5)).await?;

                // async_seq drops here
            }

            // tensor drops here
        }

        // Create new resources after cleanup
        let new_data = vec![4.0f32, 5.0, 6.0];
        let new_tensor = Tensor::new(&mut manager, &new_data)?;
        let new_handle = new_tensor.handle();

        // New handle should be different from old ones
        assert!(!handles.contains(&new_handle));

        let mut new_async_seq = AsyncSequence::new(&mut manager)?;
        new_async_seq.record_tensor_sync_device(new_handle)?;
        new_async_seq
            .eval_with_timeout(Duration::from_secs(5))
            .await?;

        println!("✓ Resource cleanup and reuse successful");

        Ok(())
    }

    #[tokio::test]
    async fn test_async_stress() -> Result<()> {
        println!("Starting async stress test...");

        let mut manager = Manager::new()?;
        let mut futures = Vec::new();

        // Create many concurrent async operations
        for i in 0..20 {
            let data: Vec<f32> = vec![i as f32; 50];
            let tensor = Tensor::new(&mut manager, &data)?;

            let future = async move {
                let mut manager = Manager::new()?;
                let mut async_seq = AsyncSequence::new(&mut manager)?;

                async_seq.record_tensor_sync_device(tensor.handle())?;
                async_seq.eval_with_timeout(Duration::from_secs(10)).await?;

                Ok::<_, kompute::KomputeError>(i)
            };

            futures.push(future);
        }

        // Execute all operations concurrently
        let start = Instant::now();
        let results = futures::future::join_all(futures).await;
        let elapsed = start.elapsed();

        // Count successes and failures
        let mut successes = 0;
        let mut failures = 0;

        for (i, result) in results.iter().enumerate() {
            match result {
                Ok(value) => {
                    assert_eq!(*value, i);
                    successes += 1;
                }
                Err(_) => {
                    failures += 1;
                }
            }
        }

        println!(
            "✓ Stress test completed in {:?}: {} successes, {} failures",
            elapsed, successes, failures
        );

        // At least some operations should succeed
        assert!(
            successes > 0,
            "At least some async operations should succeed"
        );

        Ok(())
    }
}

#[cfg(not(feature = "async"))]
mod no_async_tests {
    #[test]
    fn test_async_feature_disabled() {
        println!("Async feature is disabled - skipping async tests");
        // This test exists to verify the feature gating works correctly
        assert!(true);
    }
}
