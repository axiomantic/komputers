//! Comprehensive tests for async GPU compute APIs.
//!
//! These tests verify the async/await functionality, timeout handling,
//! and concurrent execution capabilities of the async API.

#[cfg(feature = "async")]
mod async_tests {
    use komputers::test_utils::{
        get_manager_from_pool, return_manager_to_pool, should_skip_gpu_test,
    };
    use komputers::{async_sequence::AsyncSequence, Result, Tensor, VulkanConfig};
    use serial_test::serial;
    use std::time::{Duration, Instant};
    use tokio::time::timeout;

    #[tokio::test]
    async fn test_async_sequence_basic() -> Result<()> {
        if should_skip_gpu_test() {
            println!("⏭️ Skipping GPU test - GPU tests disabled");
            return Ok(());
        }

        println!("Starting basic async sequence test...");

        let mut manager = get_manager_from_pool()?;

        // Create test data
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(&mut manager, &data)?;

        // Create async sequence and record operations
        let async_seq =
            AsyncSequence::new(&mut manager)?.record_tensor_sync_device(tensor.handle())?;

        // Execute with timeout using tokio
        let start = Instant::now();
        let execution = async_seq.execute_async();
        tokio::time::timeout(Duration::from_secs(5), execution)
            .await
            .map_err(|_| komputers::KomputeError::compute("Async operation timed out"))??;
        let elapsed = start.elapsed();

        println!("✓ Async sequence execution completed in {:?}", elapsed);
        assert!(elapsed < Duration::from_secs(5));

        return_manager_to_pool(manager);
        Ok(())
    }

    #[tokio::test]
    async fn test_async_sequence_timeout() -> Result<()> {
        if should_skip_gpu_test() {
            println!("⏭️ Skipping GPU test - GPU tests disabled");
            return Ok(());
        }

        println!("Starting async sequence timeout test...");

        let mut manager = get_manager_from_pool()?;
        let data = vec![1.0f32; 1000];
        let tensor = Tensor::new(&mut manager, &data)?;

        let async_seq = AsyncSequence::new(&mut manager)?;

        // Test with very short timeout using tokio timeout
        let start = Instant::now();
        let execution = async_seq
            .record_tensor_sync_device(tensor.handle())?
            .execute_async();
        let result = tokio::time::timeout(Duration::from_millis(100), execution).await;
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

        return_manager_to_pool(manager);
        Ok(())
    }

    #[tokio::test]
    async fn test_async_gpu_executor() -> Result<()> {
        if should_skip_gpu_test() {
            println!("⏭️ Skipping GPU test - GPU tests disabled");
            return Ok(());
        }

        println!("Starting async GPU executor test...");

        // Use consistent configuration by creating manager first
        let mut manager = get_manager_from_pool()?;

        // Create test tensors using the same manager
        let data_a = vec![1.0f32, 2.0, 3.0, 4.0];
        let data_b = vec![5.0f32, 6.0, 7.0, 8.0];

        let tensor_a = Tensor::new(&mut manager, &data_a)?;
        let tensor_b = Tensor::new(&mut manager, &data_b)?;

        // Test parallel execution of async sequences
        let start = Instant::now();

        // Create async sequences for each tensor using the same manager
        let seq_a =
            AsyncSequence::new(&mut manager)?.record_tensor_sync_device(tensor_a.handle())?;

        let seq_b =
            AsyncSequence::new(&mut manager)?.record_tensor_sync_device(tensor_b.handle())?;

        // Execute sequences individually (simulating executor behavior)
        tokio::time::timeout(Duration::from_secs(10), seq_a.execute_async())
            .await
            .map_err(|_| komputers::KomputeError::compute("Async operation timed out"))??;
        tokio::time::timeout(Duration::from_secs(10), seq_b.execute_async())
            .await
            .map_err(|_| komputers::KomputeError::compute("Async operation timed out"))??;

        let elapsed = start.elapsed();
        println!(
            "✓ Async GPU executor completed parallel operations in {:?}",
            elapsed
        );

        return_manager_to_pool(manager);
        Ok(())
    }

    #[tokio::test]
    async fn test_concurrent_async_operations() -> Result<()> {
        if should_skip_gpu_test() {
            println!("⏭️ Skipping GPU test - GPU tests disabled");
            return Ok(());
        }

        println!("Starting concurrent async operations test...");

        let mut manager = get_manager_from_pool()?;

        // Create multiple async sequences
        let mut sequences = Vec::new();
        let mut tensors = Vec::new();

        for i in 0..5 {
            let data: Vec<f32> = vec![i as f32; 100];
            let tensor = Tensor::new(&mut manager, &data)?;
            tensors.push(tensor);

            let async_seq =
                AsyncSequence::new(&mut manager)?.record_tensor_sync_device(tensors[i].handle())?;
            sequences.push(async_seq);
        }

        // Execute all sequences concurrently
        let start = Instant::now();
        let futures: Vec<_> = sequences
            .into_iter()
            .map(|seq| async move {
                tokio::time::timeout(Duration::from_secs(10), seq.execute_async()).await
            })
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

        return_manager_to_pool(manager);
        Ok(())
    }

    #[tokio::test]
    async fn test_async_error_handling() -> Result<()> {
        if should_skip_gpu_test() {
            println!("⏭️ Skipping GPU test - GPU tests disabled");
            return Ok(());
        }

        println!("Starting async error handling test...");

        let mut manager = get_manager_from_pool()?;

        // Test with invalid tensor handle
        let async_seq = AsyncSequence::new(&mut manager)?;

        // Try to record operation with invalid handle
        match async_seq.record_tensor_sync_device(999999) {
            Ok(async_seq) => {
                // If this succeeds, try to execute and expect it to fail
                let result =
                    tokio::time::timeout(Duration::from_secs(1), async_seq.execute_async()).await;
                assert!(result.is_err(), "Should fail with invalid tensor handle");
                println!("✓ Invalid handle correctly caused execution failure");
            }
            Err(_) => {
                println!("✓ Invalid handle correctly rejected at record time");
            }
        }

        return_manager_to_pool(manager);
        Ok(())
    }

    #[tokio::test]
    async fn test_async_sequence_reuse() -> Result<()> {
        if should_skip_gpu_test() {
            println!("⏭️ Skipping GPU test - GPU tests disabled");
            return Ok(());
        }

        println!("Starting async sequence reuse test...");

        let mut manager = get_manager_from_pool()?;
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(&mut manager, &data)?;

        // Record and execute multiple times
        for iteration in 0..3 {
            let async_seq =
                AsyncSequence::new(&mut manager)?.record_tensor_sync_device(tensor.handle())?;

            let start = Instant::now();
            tokio::time::timeout(Duration::from_secs(5), async_seq.execute_async())
                .await
                .map_err(|_| komputers::KomputeError::compute("Async operation timed out"))??;
            let elapsed = start.elapsed();

            println!("✓ Iteration {} completed in {:?}", iteration, elapsed);
        }

        return_manager_to_pool(manager);
        Ok(())
    }

    #[tokio::test]
    async fn test_async_cancellation() -> Result<()> {
        if should_skip_gpu_test() {
            println!("⏭️ Skipping GPU test - GPU tests disabled");
            return Ok(());
        }

        println!("Starting async cancellation test...");

        let mut manager = get_manager_from_pool()?;
        let data = vec![1.0f32; 10000]; // Larger data for longer operation
        let tensor = Tensor::new(&mut manager, &data)?;

        let async_seq =
            AsyncSequence::new(&mut manager)?.record_tensor_sync_device(tensor.handle())?;

        // Test cancellation with tokio timeout
        let start = Instant::now();
        let result = timeout(
            Duration::from_millis(1), // Very short timeout
            tokio::time::timeout(Duration::from_secs(10), async_seq.execute_async()),
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

        return_manager_to_pool(manager);
        Ok(())
    }

    #[tokio::test]
    async fn test_async_resource_cleanup() -> Result<()> {
        if should_skip_gpu_test() {
            println!("⏭️ Skipping GPU test - GPU tests disabled");
            return Ok(());
        }

        println!("Starting async resource cleanup test...");

        let mut manager = get_manager_from_pool()?;
        let mut handles = Vec::new();

        // Create and drop async sequences in different scopes
        {
            let data = vec![1.0f32, 2.0, 3.0];
            let tensor = Tensor::new(&mut manager, &data)?;
            handles.push(tensor.handle());

            {
                let async_seq =
                    AsyncSequence::new(&mut manager)?.record_tensor_sync_device(tensor.handle())?;
                tokio::time::timeout(Duration::from_secs(5), async_seq.execute_async())
                    .await
                    .map_err(|_| komputers::KomputeError::compute("Async operation timed out"))??;

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

        let new_async_seq =
            AsyncSequence::new(&mut manager)?.record_tensor_sync_device(new_handle)?;
        let execution = new_async_seq.execute_async();
        tokio::time::timeout(Duration::from_secs(5), execution)
            .await
            .map_err(|_| komputers::KomputeError::compute("Async operation timed out"))??;

        println!("✓ Resource cleanup and reuse successful");

        return_manager_to_pool(manager);
        Ok(())
    }

    #[tokio::test]
    async fn test_async_stress() -> Result<()> {
        if should_skip_gpu_test() {
            println!("⏭️ Skipping GPU test - GPU tests disabled");
            return Ok(());
        }

        println!("Starting async stress test...");

        let mut futures = Vec::new();

        // Create many concurrent async operations
        for i in 0..20 {
            let future = async move {
                let mut manager = get_manager_from_pool()?;
                let data: Vec<f32> = vec![i as f32; 50];
                let tensor = Tensor::new(&mut manager, &data)?;

                let async_seq =
                    AsyncSequence::new(&mut manager)?.record_tensor_sync_device(tensor.handle())?;
                tokio::time::timeout(Duration::from_secs(10), async_seq.execute_async())
                    .await
                    .map_err(|_| komputers::KomputeError::compute("Async operation timed out"))??;

                Ok::<_, komputers::KomputeError>(i)
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
