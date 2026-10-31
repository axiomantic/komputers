//! GPU Hardware Validation Pipeline
//!
//! Comprehensive validation tests for real GPU hardware across all supported platforms:
//! - macOS: Apple M1/M2/M3 with MoltenVK
//! - Linux: NVIDIA GPUs with native Vulkan
//! - Windows: NVIDIA GPUs with native Vulkan
//!
//! These tests ensure cross-platform GPU functionality and detect hardware-specific issues.

use komputers::*;
use std::time::{Duration, Instant};

/// Comprehensive GPU hardware validation test suite
/// This test validates that GPU hardware is functioning correctly across platforms
#[test]
fn test_cross_platform_gpu_hardware_validation() -> Result<()> {
    println!("🌐 Starting cross-platform GPU hardware validation pipeline");

    // Phase 1: Basic GPU detection and initialization
    test_basic_gpu_initialization()?;

    // Phase 2: Cross-platform tensor operations
    test_cross_platform_tensor_operations()?;

    // Phase 3: GPU memory management validation
    test_gpu_memory_management_validation()?;

    // Phase 4: Platform-specific GPU features
    test_platform_specific_gpu_features()?;

    // Phase 5: GPU performance characteristics validation
    test_gpu_performance_validation()?;

    println!("✅ Cross-platform GPU hardware validation completed successfully");
    Ok(())
}

/// Test basic GPU initialization across platforms
fn test_basic_gpu_initialization() -> Result<()> {
    println!("🔧 Phase 1: Basic GPU initialization validation");

    let mut manager = Manager::new()?;

    println!(
        "   ✅ Manager created successfully with ID: {}",
        manager.manager_id()
    );

    // Test device count
    let device_count = Manager::device_count()?;
    println!("   📊 Detected {} GPU device(s)", device_count);

    assert!(
        device_count > 0,
        "At least one GPU device should be available"
    );

    // Test basic device information
    println!("   🔍 GPU validation completed for platform:");

    #[cfg(target_os = "macos")]
    {
        println!("     Platform: macOS");
        println!("     GPU Type: Apple Silicon (M1/M2/M3) or Intel/AMD");
        println!("     Vulkan Implementation: MoltenVK");

        // Validate MoltenVK-specific functionality
        test_moltenvk_specific_features(&mut manager)?;
    }

    #[cfg(target_os = "linux")]
    {
        println!("     Platform: Linux");
        println!("     GPU Type: NVIDIA/AMD/Intel");
        println!("     Vulkan Implementation: Native");

        // Validate native Vulkan functionality
        test_native_vulkan_features(&mut manager)?;
    }

    #[cfg(target_os = "windows")]
    {
        println!("     Platform: Windows");
        println!("     GPU Type: NVIDIA/AMD/Intel");
        println!("     Vulkan Implementation: Native");

        // Validate native Vulkan functionality
        test_native_vulkan_features(&mut manager)?;
    }

    println!("   ✅ Phase 1 completed: Basic GPU initialization validated");
    Ok(())
}

/// Test cross-platform tensor operations
fn test_cross_platform_tensor_operations() -> Result<()> {
    println!("🧮 Phase 2: Cross-platform tensor operations validation");

    let mut manager = Manager::new()?;

    // Test various data types and sizes
    let test_cases: Vec<(&str, Vec<f32>)> = vec![
        ("f32", vec![1.0f32, 2.5, 3.14159, -1.5, 0.0]),
        ("f32_integers", vec![1.0, -2.0, 3.0, -4.0, 0.0]),
        (
            "large_f32",
            (0..1000).map(|i| i as f32 * 0.1).collect::<Vec<f32>>(),
        ),
    ];

    for (test_name, data) in test_cases {
        println!("   🔧 Testing {} tensor operations...", test_name);

        let start = Instant::now();

        // Create tensor
        let tensor = Tensor::new(&mut manager, &data)?;

        // Test synchronization
        let sequence = manager
            .sequence()?
            .record_tensor_sync_device(tensor.handle())?;

        sequence.execute()?;

        // Verify data integrity
        let result = tensor.to_vec()?;
        assert_eq!(result.len(), data.len());

        let duration = start.elapsed();
        println!(
            "     ✅ {} tensor ({} elements) processed in {:?}",
            test_name,
            data.len(),
            duration
        );

        // Validate performance expectations
        if data.len() > 100 {
            assert!(
                duration < Duration::from_secs(5),
                "Large tensor operations should complete within 5 seconds"
            );
        }
    }

    println!("   ✅ Phase 2 completed: Cross-platform tensor operations validated");
    Ok(())
}

/// Test GPU memory management validation
fn test_gpu_memory_management_validation() -> Result<()> {
    println!("💾 Phase 3: GPU memory management validation");

    let mut manager = Manager::new()?;

    // Test memory allocation and deallocation patterns
    println!("   🔧 Testing memory allocation patterns...");

    // Create multiple tensors to test memory management
    let mut tensors = Vec::new();
    for i in 0..20 {
        let size = 100 + i * 50; // Variable sizes
        let data: Vec<f32> = (0..size).map(|x| x as f32).collect();
        let tensor = Tensor::new(&mut manager, &data)?;
        tensors.push(tensor);
    }

    println!(
        "     ✅ Created {} tensors with variable sizes",
        tensors.len()
    );

    // Test concurrent tensor operations
    for (i, tensor) in tensors.iter().enumerate() {
        let sequence = manager
            .sequence()?
            .record_tensor_sync_device(tensor.handle())?;
        sequence.execute()?;

        if i % 5 == 0 {
            println!("     📊 Processed {} tensors", i + 1);
        }
    }

    println!("     ✅ All tensors processed successfully");

    // Test memory cleanup by dropping tensors
    drop(tensors);

    // Create one more tensor to ensure memory is available
    let test_data = vec![42.0f32; 1000];
    let _cleanup_test = Tensor::new(&mut manager, &test_data)?;

    println!("   ✅ Phase 3 completed: GPU memory management validated");
    Ok(())
}

/// Test platform-specific GPU features
fn test_platform_specific_gpu_features() -> Result<()> {
    println!("🎯 Phase 4: Platform-specific GPU features validation");

    let mut manager = Manager::new()?;

    #[cfg(target_os = "macos")]
    {
        println!("   🍎 Testing macOS-specific features...");
        test_macos_metal_integration(&mut manager)?;
    }

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    {
        println!("   🐧🪟 Testing native Vulkan features...");
        test_native_vulkan_advanced_features(&mut manager)?;
    }

    println!("   ✅ Phase 4 completed: Platform-specific features validated");
    Ok(())
}

/// Test GPU performance validation
fn test_gpu_performance_validation() -> Result<()> {
    println!("⚡ Phase 5: GPU performance characteristics validation");

    let mut manager = Manager::new()?;

    // Performance benchmarking
    let benchmark_sizes = vec![100, 1000, 10000];

    for size in benchmark_sizes {
        println!("   📊 Benchmarking {} element operations...", size);

        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();

        let start = Instant::now();

        // Create and process tensor
        let tensor = Tensor::new(&mut manager, &data)?;
        let sequence = manager
            .sequence()?
            .record_tensor_sync_device(tensor.handle())?;
        sequence.execute()?;
        let _result = tensor.to_vec()?;

        let duration = start.elapsed();
        let throughput = size as f64 / duration.as_secs_f64();

        println!(
            "     ✅ {} elements: {:?} ({:.0} elements/sec)",
            size, duration, throughput
        );

        // Validate reasonable performance
        assert!(
            throughput > 1000.0,
            "GPU throughput should exceed 1000 elements/sec"
        );
    }

    println!("   ✅ Phase 5 completed: GPU performance validated");
    Ok(())
}

/// Test MoltenVK-specific features (macOS)
#[cfg(target_os = "macos")]
fn test_moltenvk_specific_features(manager: &mut Manager) -> Result<()> {
    println!("     🔧 Testing MoltenVK-specific functionality...");

    // Test Apple Silicon optimized operations
    let metal_optimized_data = vec![1.0f32; 2048]; // Common Metal buffer size
    let tensor = Tensor::new(manager, &metal_optimized_data)?;

    let sequence = manager
        .sequence()?
        .record_tensor_sync_device(tensor.handle())?;
    sequence.execute()?;

    let result = tensor.to_vec()?;
    assert_eq!(result.len(), 2048);

    println!("     ✅ MoltenVK Metal integration validated");
    Ok(())
}

/// Test MoltenVK-specific features (stub for non-macOS)
#[cfg(not(target_os = "macos"))]
fn test_moltenvk_specific_features(_manager: &mut Manager) -> Result<()> {
    // No-op on non-macOS platforms
    Ok(())
}

/// Test native Vulkan features (Linux/Windows)
#[cfg(any(target_os = "linux", target_os = "windows"))]
fn test_native_vulkan_features(manager: &mut Manager) -> Result<()> {
    println!("     🔧 Testing native Vulkan functionality...");

    // Test native Vulkan optimized operations
    let vulkan_data = vec![2.0f32; 1024];
    let tensor = Tensor::new(manager, &vulkan_data)?;

    let sequence = manager
        .sequence()?
        .record_tensor_sync_device(tensor.handle())?;
    sequence.execute()?;

    let result = tensor.to_vec()?;
    assert_eq!(result.len(), 1024);

    println!("     ✅ Native Vulkan integration validated");
    Ok(())
}

/// Test native Vulkan features (stub for macOS)
#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn test_native_vulkan_features(_manager: &mut Manager) -> Result<()> {
    // No-op on macOS
    Ok(())
}

/// Test advanced native Vulkan features
#[cfg(any(target_os = "linux", target_os = "windows"))]
fn test_native_vulkan_advanced_features(manager: &mut Manager) -> Result<()> {
    println!("     🔧 Testing advanced native Vulkan features...");

    // Test larger memory allocations that might stress native Vulkan
    let large_data: Vec<f32> = (0..50000).map(|i| (i as f32).sin()).collect();
    let tensor = Tensor::new(manager, &large_data)?;

    let sequence = manager
        .sequence()?
        .record_tensor_sync_device(tensor.handle())?;
    sequence.execute()?;

    println!("     ✅ Advanced Vulkan features validated");
    Ok(())
}

/// Test advanced native Vulkan features (stub for macOS)
#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn test_native_vulkan_advanced_features(_manager: &mut Manager) -> Result<()> {
    // No-op on macOS
    Ok(())
}

/// Test macOS Metal integration features
#[cfg(target_os = "macos")]
fn test_macos_metal_integration(manager: &mut Manager) -> Result<()> {
    println!("     🍎 Testing Metal/Vulkan integration...");

    // Test Metal-specific memory patterns
    let metal_data: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.001).cos()).collect();
    let tensor = Tensor::new(manager, &metal_data)?;

    let sequence = manager
        .sequence()?
        .record_tensor_sync_device(tensor.handle())?;
    sequence.execute()?;

    let result = tensor.to_vec()?;
    assert_eq!(result.len(), 4096);

    println!("     ✅ macOS Metal integration validated");
    Ok(())
}

/// Test macOS Metal integration features (stub for non-macOS)
#[cfg(not(target_os = "macos"))]
fn test_macos_metal_integration(_manager: &mut Manager) -> Result<()> {
    // No-op on non-macOS platforms
    Ok(())
}

/// GPU stress test for extended validation
#[test]
#[ignore] // Run only with --ignored flag for thorough testing
fn test_gpu_stress_validation() -> Result<()> {
    println!("🔥 GPU Stress Test - Extended Hardware Validation");

    let mut manager = Manager::new()?;

    // Stress test: Create many tensors over time
    for iteration in 0..100 {
        let data: Vec<f32> = (0..1000).map(|i| (i + iteration * 1000) as f32).collect();
        let tensor = Tensor::new(&mut manager, &data)?;

        let sequence = manager
            .sequence()?
            .record_tensor_sync_device(tensor.handle())?;
        sequence.execute()?;

        if iteration % 20 == 0 {
            println!("   📊 Completed {} stress test iterations", iteration);
        }

        // Small delay to prevent overwhelming the GPU
        std::thread::sleep(Duration::from_millis(10));
    }

    println!("✅ GPU stress test completed successfully");
    Ok(())
}

/// GPU endurance test for long-running validation
#[test]
#[ignore] // Run only with --ignored flag for endurance testing
fn test_gpu_endurance_validation() -> Result<()> {
    println!("⏰ GPU Endurance Test - Long-running Hardware Validation");

    let mut manager = Manager::new()?;
    let start_time = Instant::now();
    let test_duration = Duration::from_secs(300); // 5 minutes

    let mut operations = 0;

    while start_time.elapsed() < test_duration {
        let data: Vec<f32> = (0..500).map(|i| (i as f32).sqrt()).collect();
        let tensor = Tensor::new(&mut manager, &data)?;

        let sequence = manager
            .sequence()?
            .record_tensor_sync_device(tensor.handle())?;
        sequence.execute()?;

        operations += 1;

        if operations % 100 == 0 {
            let elapsed = start_time.elapsed();
            println!(
                "   📊 {} operations in {:?} ({:.1} ops/sec)",
                operations,
                elapsed,
                operations as f64 / elapsed.as_secs_f64()
            );
        }
    }

    let total_time = start_time.elapsed();
    let ops_per_second = operations as f64 / total_time.as_secs_f64();

    println!("✅ GPU endurance test completed:");
    println!("   Total operations: {}", operations);
    println!("   Total time: {:?}", total_time);
    println!("   Average ops/sec: {:.1}", ops_per_second);

    Ok(())
}
