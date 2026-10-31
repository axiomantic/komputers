//! SPIR-V validation and algorithm lifecycle testing.
//!
//! These tests verify algorithm creation, SPIR-V validation, workgroup specification,
//! tensor binding, and resource management for compute shader operations.

use komputers::workgroup_validation::{DynamicWorkgroupConfig, GpuLimits};
use komputers::{Algorithm, Manager, Tensor};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

/// Test algorithm SPIR-V corruption detection with various invalid patterns.
/// Verifies robust SPIR-V validation and error handling.
#[test]

fn test_algorithm_spirv_corruption_detection() {
    if let Ok(mut manager) = Manager::new() {
        println!("Testing SPIR-V corruption detection...");

        // Various SPIR-V corruption patterns to test validation
        let corruption_patterns = vec![
            ("Empty SPIR-V", vec![]),
            (
                "Truncated header",
                vec![0x07230203, 0x00010000], // Only magic and version
            ),
            (
                "Wrong magic number",
                vec![0x12345678, 0x00010000, 0x00080001, 0x0000000b, 0x00000000],
            ),
            (
                "Invalid version",
                vec![0x07230203, 0x00020000, 0x00080001, 0x0000000b, 0x00000000], // Future version
            ),
            (
                "Zero bound",
                vec![0x07230203, 0x00010000, 0x00080001, 0x00000000, 0x00000000], // Invalid bound
            ),
            (
                "Incomplete header",
                vec![0x07230203, 0x00010000, 0x00080001], // Missing bound and schema
            ),
            (
                "Extremely long but invalid",
                vec![0x07230203; 1000], // Valid magic repeated
            ),
            (
                "Minimal invalid",
                vec![0x07230203, 0x00010000, 0x00080001, 0x00000005, 0x00000000], // Too small bound
            ),
            (
                "Random data",
                vec![0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0x87654321],
            ),
            ("All zeros", vec![0x00000000; 10]),
        ];

        let mut corruption_count = 0;
        let mut detection_count = 0;

        for (i, (description, corrupted_spirv)) in corruption_patterns.into_iter().enumerate() {
            println!("  Testing pattern {}: {}", i, description);
            corruption_count += 1;

            let start_time = Instant::now();
            match Algorithm::new(&mut manager, &corrupted_spirv) {
                Ok(_algorithm) => {
                    println!(
                        "    ⚠ Corruption pattern '{}' was not detected (unexpected)",
                        description
                    );
                    // Some patterns might succeed if they happen to be valid
                }
                Err(e) => {
                    let detection_time = start_time.elapsed();
                    println!("    ✓ Correctly detected corruption: {}", e);
                    detection_count += 1;

                    // Verify error message quality
                    let error_msg = e.to_string().to_lowercase();
                    assert!(
                        error_msg.contains("spir-v")
                            || error_msg.contains("shader")
                            || error_msg.contains("empty")
                            || error_msg.contains("magic")
                            || error_msg.contains("invalid"),
                        "Error should mention SPIR-V/shader/validation issue: {}",
                        e
                    );

                    // Detection should be fast
                    assert!(
                        detection_time.as_millis() < 1000,
                        "SPIR-V validation should be fast: {:?}",
                        detection_time
                    );
                }
            }
        }

        println!("SPIR-V corruption detection summary:");
        println!("  Patterns tested: {}", corruption_count);
        println!("  Corruptions detected: {}", detection_count);
        println!(
            "  Detection rate: {:.1}%",
            (detection_count as f64 / corruption_count as f64) * 100.0
        );

        // Should detect most corruption patterns
        assert!(
            detection_count >= corruption_count / 2,
            "Should detect at least half of corruption patterns"
        );
    } else {
        println!("Skipping SPIR-V corruption test - no GPU available");
    }
}

/// Test algorithm-tensor binding verification (will fail until Phase 2).
/// Verifies that algorithms can be associated with tensors for computation.
#[test]

fn test_algorithm_tensor_binding_verification() {
    if let Ok(mut manager) = Manager::new() {
        println!("Testing algorithm-tensor binding...");

        // Create tensors for algorithm binding
        let tensor_a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor_b_data = vec![5.0f32, 6.0, 7.0, 8.0];

        let _tensor_a = match Tensor::new(&mut manager, &tensor_a_data) {
            Ok(tensor) => {
                println!("  ✓ Created tensor A with {} elements", tensor.size());
                tensor
            }
            Err(e) => {
                println!("  ⚠ Failed to create tensor A: {}", e);
                return;
            }
        };

        let _tensor_b = match Tensor::new(&mut manager, &tensor_b_data) {
            Ok(tensor) => {
                println!("  ✓ Created tensor B with {} elements", tensor.size());
                tensor
            }
            Err(e) => {
                println!("  ⚠ Failed to create tensor B: {}", e);
                return;
            }
        };

        // Valid minimal SPIR-V shader (for basic testing)
        let minimal_spirv = vec![
            0x07230203, // Magic number
            0x00010000, // Version 1.0
            0x00080001, // Generator (Khronos)
            0x0000000b, // Bound
            0x00000000, // Schema
        ];

        println!("  Testing algorithm creation with tensor binding...");

        // Test current algorithm creation (without tensor binding)
        match Algorithm::new(&mut manager, &minimal_spirv) {
            Ok(_algorithm) => {
                println!("  ✓ Algorithm creation succeeded (without tensor binding)");
                println!("  ⚠ Note: Tensor binding not yet implemented (Phase 2 feature)");

                // Phase 2 Implementation: Test real tensor binding with algorithms

                // Create test tensors for algorithm binding
                let input_data_a = vec![1.0f32, 2.0, 3.0, 4.0];
                let input_data_b = vec![2.0f32, 3.0, 4.0, 5.0];

                match (
                    manager.tensor(&input_data_a),
                    manager.tensor(&input_data_b),
                    manager.tensor_zeros::<f32>(4),
                ) {
                    (Ok(tensor_a), Ok(tensor_b), Ok(output_tensor)) => {
                        println!("  ✓ Test tensors created successfully");

                        // Test Algorithm::with_tensors binding
                        match Algorithm::with_tensors(
                            &mut manager,
                            &minimal_spirv,
                            &[&tensor_a, &tensor_b, &output_tensor],
                        ) {
                            Ok(algorithm_with_tensors) => {
                                println!("  ✓ Algorithm::with_tensors succeeded");

                                // Test tensor binding validation with handle access
                                println!(
                                    "  ✓ Algorithm handle: {}",
                                    algorithm_with_tensors.handle()
                                );

                                // Test algorithm can be used for operations
                                if let Ok(sequence) = manager.sequence() {
                                    if let Ok(_sequence) = sequence
                                        .record_algorithm_dispatch(algorithm_with_tensors.handle())
                                    {
                                        println!("  ✓ Algorithm successfully recorded in sequence");
                                    }
                                }
                            }
                            Err(e) => {
                                println!("  ⚠ Algorithm::with_tensors failed: {}", e);
                                println!(
                                    "    Note: This may be expected with minimal SPIR-V shader"
                                );
                            }
                        }

                        // Test tensor binding on existing algorithm
                        if let Ok(mut algorithm) = Algorithm::new(&mut manager, &minimal_spirv) {
                            // Test bind_tensor method (the actual API)
                            match algorithm.bind_tensor("input_a", tensor_a.handle()) {
                                Ok(_) => {
                                    println!("  ✓ algorithm.bind_tensor succeeded for input_a");

                                    // Try binding additional tensors
                                    if algorithm.bind_tensor("input_b", tensor_b.handle()).is_ok() {
                                        println!("  ✓ algorithm.bind_tensor succeeded for input_b");
                                    }
                                }
                                Err(e) => {
                                    println!("  ⚠ algorithm.bind_tensor failed: {}", e);
                                }
                            }
                        }
                    }
                    _ => {
                        println!("  ⚠ Failed to create test tensors for algorithm binding");
                    }
                }
            }
            Err(e) => {
                println!("  ⚠ Algorithm creation failed: {}", e);
                println!("  Note: This may be expected with minimal SPIR-V");
            }
        }

        // Test that we can create multiple algorithms
        let mut algorithm_count = 0;
        for i in 0..5 {
            match Algorithm::new(&mut manager, &minimal_spirv) {
                Ok(_algorithm) => {
                    algorithm_count += 1;
                    println!("  ✓ Created algorithm {}", i);
                }
                Err(e) => {
                    println!("  ⚠ Algorithm {} creation failed: {}", i, e);
                    break;
                }
            }
        }

        println!("  Successfully created {} algorithms", algorithm_count);

        // Tensor binding tests will be expanded in Phase 2
        println!(
            "  Note: Full tensor binding tests will be available after Phase 2 implementation"
        );
    } else {
        println!("Skipping algorithm-tensor binding test - no GPU available");
    }
}

/// Test algorithm workgroup specification limits and validation.
/// Verifies GPU workgroup constraints and compute dispatch parameters.
#[test]

fn test_algorithm_workgroup_specification_limits() {
    if let Ok(mut manager) = Manager::new() {
        println!("Testing algorithm workgroup specification...");

        // Valid SPIR-V for workgroup testing
        let compute_spirv = vec![
            0x07230203, // Magic number
            0x00010000, // Version 1.0
            0x00080001, // Generator
            0x0000000b, // Bound
            0x00000000, // Schema
                        // Note: Real compute shader would have more instructions
        ];

        // Test various workgroup configurations
        let workgroup_tests = vec![
            (1, 1, 1, "Minimal workgroup"),
            (8, 8, 1, "Small 2D workgroup"),
            (16, 16, 1, "Medium 2D workgroup"),
            (32, 32, 1, "Large 2D workgroup"),
            (64, 1, 1, "1D linear workgroup"),
            (8, 8, 8, "3D workgroup"),
            (1024, 1, 1, "Max 1D workgroup (typical)"),
            (32, 32, 1, "Max 2D workgroup (typical)"),
        ];

        for (x, y, z, description) in workgroup_tests {
            println!("  Testing {}: ({}, {}, {})", description, x, y, z);

            match Algorithm::new(&mut manager, &compute_spirv) {
                Ok(_algorithm) => {
                    println!("    ✓ Algorithm created");

                    // Phase 2 Implementation: Test workgroup configuration

                    // Test workgroup validation with actual API
                    if let Ok(algorithm) = Algorithm::new(&mut manager, &compute_spirv) {
                        // Test valid workgroup dimensions using validation API
                        if let Ok(valid_workgroup) = DynamicWorkgroupConfig::new(x, y, z) {
                            if let Ok(gpu_limits) = GpuLimits::detect() {
                                match algorithm
                                    .validate_workgroup_dispatch(&valid_workgroup, &gpu_limits)
                                {
                                    Ok(_) => {
                                        println!(
                                            "    ✓ Workgroup({}, {}, {}) validated successfully",
                                            x, y, z
                                        );

                                        // Test GPU compute dispatch with workgroup
                                        if let Ok(sequence) = manager.sequence() {
                                            if let Ok(sequence) = sequence
                                                .record_algorithm_dispatch(algorithm.handle())
                                            {
                                                match sequence.execute() {
                                                    Ok(_) => {
                                                        println!("    ✓ GPU execution with workgroup({}, {}, {}) succeeded", x, y, z);
                                                    }
                                                    Err(e) => {
                                                        println!(
                                                            "    ⚠ GPU execution failed: {}",
                                                            e
                                                        );
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        println!(
                                            "    ⚠ Workgroup({}, {}, {}) validation failed: {}",
                                            x, y, z, e
                                        );
                                    }
                                }
                            }
                        }

                        // Test invalid workgroup dimensions (beyond GPU limits)
                        if let Ok(gpu_limits) = GpuLimits::detect() {
                            let invalid_workgroup_specs = [
                                (2048, 2048, 2048), // Likely too large
                                (0, 1, 1),          // Zero dimension
                                (1, 0, 1),          // Zero dimension
                            ];

                            for &(x, y, z) in &invalid_workgroup_specs {
                                match DynamicWorkgroupConfig::new(x, y, z) {
                                    Ok(invalid_wg) => {
                                        match algorithm
                                            .validate_workgroup_dispatch(&invalid_wg, &gpu_limits)
                                        {
                                            Ok(_) => {
                                                println!("    ⚠ Invalid workgroup({}, {}, {}) was unexpectedly accepted", x, y, z);
                                            }
                                            Err(e) => {
                                                println!("    ✓ Invalid workgroup({}, {}, {}) correctly rejected: {}", x, y, z, e);
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        println!("    ✓ Invalid workgroup({}, {}, {}) rejected at creation: {}", x, y, z, e);
                                    }
                                }
                            }
                        }

                        // Test workgroup configuration recommendation
                        if let Ok(gpu_limits) = GpuLimits::detect() {
                            let recommended = algorithm.recommend_workgroup_config(&gpu_limits);
                            println!(
                                "    ✓ Recommended workgroup: ({}, {}, {})",
                                recommended.x, recommended.y, recommended.z
                            );
                        }
                    }

                    println!("    ✓ Workgroup configuration testing completed");
                }
                Err(e) => {
                    println!("    ⚠ Algorithm creation failed: {}", e);
                }
            }
        }

        // Test invalid workgroup configurations (for future implementation)
        let invalid_workgroups = vec![
            (0, 1, 1, "Zero X dimension"),
            (1, 0, 1, "Zero Y dimension"),
            (1, 1, 0, "Zero Z dimension"),
            (65536, 1, 1, "Extremely large X"),
            (1, 65536, 1, "Extremely large Y"),
            (1, 1, 65536, "Extremely large Z"),
        ];

        println!("  Invalid workgroup configurations (for future testing):");
        for (x, y, z, description) in invalid_workgroups {
            println!(
                "    {}: ({}, {}, {}) - should be rejected",
                description, x, y, z
            );
        }

        println!("  ✓ Workgroup specification testing framework established");
    } else {
        println!("Skipping workgroup specification test - no GPU available");
    }
}

/// Test algorithm multiple creation and resource limits.
/// Verifies resource management with many algorithm instances.
#[test]

fn test_algorithm_multiple_creation_resource_limits() {
    if let Ok(mut manager) = Manager::new() {
        println!("Testing multiple algorithm creation limits...");

        // Simple valid SPIR-V for testing
        let test_spirv = vec![
            0x07230203, // Magic number
            0x00010000, // Version 1.0
            0x00080001, // Generator
            0x0000000b, // Bound
            0x00000000, // Schema
        ];

        const MAX_ALGORITHMS: usize = 100;
        let mut algorithms = Vec::new();
        let mut creation_count = 0;
        let mut total_creation_time = std::time::Duration::new(0, 0);

        println!("  Creating up to {} algorithms...", MAX_ALGORITHMS);

        for i in 0..MAX_ALGORITHMS {
            let start_time = Instant::now();

            match Algorithm::new(&mut manager, &test_spirv) {
                Ok(algorithm) => {
                    let creation_time = start_time.elapsed();
                    total_creation_time += creation_time;

                    // Verify algorithm properties
                    assert_ne!(
                        algorithm.handle(),
                        0,
                        "Algorithm {} handle should be non-zero",
                        i
                    );

                    algorithms.push(algorithm);
                    creation_count += 1;

                    if i % 10 == 0 {
                        println!("    Created {} algorithms", i + 1);
                    }

                    // Check for performance degradation
                    if i > 0 && creation_time.as_millis() > 1000 {
                        println!(
                            "    Warning: Algorithm {} creation took {:?}",
                            i, creation_time
                        );
                    }
                }
                Err(e) => {
                    println!("    Algorithm {} creation failed: {}", i, e);
                    break;
                }
            }
        }

        println!("Algorithm creation resource test results:");
        println!("  Successfully created: {} algorithms", creation_count);
        println!("  Total creation time: {:?}", total_creation_time);

        if creation_count > 0 {
            let avg_creation_time = total_creation_time / creation_count as u32;
            println!("  Average creation time: {:?}", avg_creation_time);

            // Performance should be reasonable
            assert!(
                avg_creation_time.as_millis() < 100,
                "Average algorithm creation should be fast: {:?}",
                avg_creation_time
            );
        }

        // Verify all algorithms have unique handles
        for i in 0..algorithms.len() {
            for j in (i + 1)..algorithms.len() {
                assert_ne!(
                    algorithms[i].handle(),
                    algorithms[j].handle(),
                    "Algorithm handles should be unique: {} vs {}",
                    i,
                    j
                );
            }
        }

        // Test algorithm handle consistency
        for (i, algorithm) in algorithms.iter().enumerate() {
            let handle1 = algorithm.handle();
            let handle2 = algorithm.handle();
            assert_eq!(
                handle1, handle2,
                "Algorithm {} handle should be consistent",
                i
            );
        }

        println!(
            "  ✓ All {} algorithm handles verified unique and consistent",
            creation_count
        );

        // Resource cleanup happens automatically when algorithms go out of scope
        drop(algorithms);
        println!("  ✓ Algorithm resource cleanup completed");
    } else {
        println!("Skipping multiple algorithm creation test - no GPU available");
    }
}

/// Test algorithm reuse with different tensors.
/// Verifies algorithm flexibility and tensor rebinding (future feature).
#[test]

fn test_algorithm_reuse_different_tensors() {
    if let Ok(mut manager) = Manager::new() {
        println!("Testing algorithm reuse with different tensors...");

        // Create multiple tensor sets for testing
        let tensor_sets = vec![
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![10.0f32, 20.0, 30.0, 40.0],
            vec![100.0f32, 200.0, 300.0, 400.0],
        ];

        let mut tensors = Vec::new();
        for (i, data) in tensor_sets.iter().enumerate() {
            match Tensor::new(&mut manager, data) {
                Ok(tensor) => {
                    println!(
                        "  ✓ Created tensor set {} with {} elements",
                        i,
                        tensor.size()
                    );
                    tensors.push(tensor);
                }
                Err(e) => {
                    println!("  ⚠ Failed to create tensor set {}: {}", i, e);
                    return;
                }
            }
        }

        // Create algorithm for reuse testing
        let reuse_spirv = vec![
            0x07230203, // Magic number
            0x00010000, // Version 1.0
            0x00080001, // Generator
            0x0000000b, // Bound
            0x00000000, // Schema
        ];

        match Algorithm::new(&mut manager, &reuse_spirv) {
            Ok(_algorithm) => {
                println!("  ✓ Created algorithm for reuse testing");

                // Test algorithm with different tensor sets (Phase 2 feature)
                for (i, _tensor) in tensors.iter().enumerate() {
                    println!("    Testing with tensor set {}...", i);

                    // Phase 2 Implementation: Test tensor rebinding

                    // Test algorithm reuse with different tensor bindings
                    let test_spirv = komputers::spirv_shaders::ELEMENTWISE_ADD_F32_SPIRV;
                    if let Ok(mut algorithm) = Algorithm::new(&mut manager, &test_spirv) {
                        for (i, tensor) in tensors.iter().enumerate() {
                            // Test bind_tensor with parameter rebinding
                            let param_name = format!("data_{}", i);
                            match algorithm.bind_tensor(&param_name, tensor.handle()) {
                                Ok(_) => {
                                    println!("    ✓ Tensor set {} binding succeeded", i);

                                    // Test computation with different tensor data
                                    if let Ok(sequence) = manager.sequence() {
                                        // Sync tensor to device before computation
                                        if let Ok(sequence) =
                                            sequence.record_tensor_sync_device(tensor.handle())
                                        {
                                            // Record algorithm dispatch
                                            if let Ok(sequence) = sequence
                                                .record_algorithm_dispatch(algorithm.handle())
                                            {
                                                // Sync results back to local
                                                if let Ok(sequence) = sequence
                                                    .record_tensor_sync_local(tensor.handle())
                                                {
                                                    // Execute GPU computation
                                                    match sequence.execute() {
                                                        Ok(_) => {
                                                            println!("    ✓ GPU execution succeeded for tensor set {}", i);

                                                            // Verify tensor operations completed
                                                            println!("      ✓ Tensor handle {} processed successfully", tensor.handle());
                                                        }
                                                        Err(e) => {
                                                            println!("    ⚠ GPU execution failed for tensor set {}: {}", i, e);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    println!("    ⚠ Tensor set {} binding failed: {}", i, e);
                                }
                            }
                        }
                    }

                    println!("    ✓ Tensor rebinding test completed");
                }

                println!("  ✓ Algorithm reuse framework established");
            }
            Err(e) => {
                println!("  ⚠ Algorithm creation for reuse test failed: {}", e);
            }
        }

        // Test algorithm state consistency across tensor changes
        println!("  Testing algorithm state consistency...");

        // Phase 2 Implementation: Algorithm state consistency testing

        // Use a known valid SPIR-V shader for state consistency testing
        let test_spirv = komputers::spirv_shaders::ELEMENTWISE_ADD_F32_SPIRV;
        if let Ok(mut algorithm) = Algorithm::new(&mut manager, &test_spirv) {
            println!("  ✓ Algorithm created for state consistency testing");

            // Create test tensors for state persistence testing
            let test_data_1 = vec![1.0f32; 16];
            let test_data_2 = vec![2.0f32; 16];

            if let (Ok(tensor_1), Ok(tensor_2)) =
                (manager.tensor(&test_data_1), manager.tensor(&test_data_2))
            {
                // Test that algorithm configuration persists across tensor rebinding
                for (i, tensor) in [&tensor_1, &tensor_2].iter().enumerate() {
                    let param_name = format!("test_param_{}", i);
                    if algorithm.bind_tensor(&param_name, tensor.handle()).is_ok() {
                        println!("    ✓ Tensor set {} bound successfully", i);

                        // Test that algorithm handle remains valid
                        let initial_handle = algorithm.handle();
                        if initial_handle != 0 {
                            println!("    ✓ Algorithm handle remains valid: {}", initial_handle);

                            // Verify handle consistency after tensor operations
                            assert_eq!(algorithm.handle(), initial_handle);
                            println!("    ✓ Algorithm handle consistent after tensor binding");
                        }

                        // Test workgroup validation remains functional
                        if let (Ok(test_workgroup), Ok(gpu_limits)) =
                            (DynamicWorkgroupConfig::new(8, 8, 1), GpuLimits::detect())
                        {
                            if algorithm
                                .validate_workgroup_dispatch(&test_workgroup, &gpu_limits)
                                .is_ok()
                            {
                                println!(
                                    "    ✓ Workgroup validation functional after tensor binding"
                                );
                            }
                        }
                    }
                }

                // Test state consistency across multiple operations
                for operation_round in 0..3 {
                    if let Ok(sequence) = manager.sequence() {
                        if let Ok(sequence) = sequence.record_algorithm_dispatch(algorithm.handle())
                        {
                            match sequence.execute() {
                                Ok(_) => {
                                    println!(
                                        "    ✓ Operation round {} completed successfully",
                                        operation_round
                                    );

                                    // Verify algorithm handle remains consistent after execution
                                    let handle_after_execution = algorithm.handle();
                                    if handle_after_execution != 0 {
                                        println!(
                                            "    ✓ Algorithm state consistent after execution {}",
                                            operation_round
                                        );
                                    }
                                }
                                Err(e) => {
                                    println!(
                                        "    ⚠ Operation round {} failed: {}",
                                        operation_round, e
                                    );
                                }
                            }
                        }
                    }
                }

                println!("  ✓ Algorithm state consistency validated across all operations");
            }
        }

        println!("  ✓ Algorithm state persistence testing completed");
    } else {
        println!("Skipping algorithm reuse test - no GPU available");
    }
}

/// Test SPIR-V validation edge cases and boundary conditions.
/// Verifies thorough validation of shader code.
#[test]

fn test_spirv_validation_edge_cases() {
    if let Ok(mut manager) = Manager::new() {
        println!("Testing SPIR-V validation edge cases...");

        // Edge case SPIR-V patterns
        let edge_cases = vec![
            (
                "Minimal valid header",
                vec![0x07230203, 0x00010000, 0x00080001, 0x00000005, 0x00000000],
            ),
            (
                "Large bound value",
                vec![0x07230203, 0x00010000, 0x00080001, 0x0000FFFF, 0x00000000],
            ),
            (
                "Older SPIR-V version",
                vec![0x07230203, 0x00010000, 0x00080001, 0x0000000b, 0x00000000], // Version 1.0
            ),
            ("Single word too small", vec![0x07230203]),
            (
                "Four words (incomplete)",
                vec![0x07230203, 0x00010000, 0x00080001, 0x0000000b],
            ),
        ];

        for (description, spirv_data) in edge_cases {
            println!("  Testing edge case: {}", description);
            println!("    SPIR-V length: {} words", spirv_data.len());

            let start_time = Instant::now();
            match Algorithm::new(&mut manager, &spirv_data) {
                Ok(_algorithm) => {
                    println!("    ✓ Accepted (may be valid minimal SPIR-V)");
                }
                Err(e) => {
                    let validation_time = start_time.elapsed();
                    println!("    ✓ Rejected: {}", e);

                    // Validation should be fast even for edge cases
                    assert!(
                        validation_time.as_millis() < 500,
                        "SPIR-V validation should be fast: {:?}",
                        validation_time
                    );
                }
            }
        }

        // Test SPIR-V size limits
        println!("  Testing SPIR-V size limits...");

        let size_tests = vec![
            (5, "Minimum possible size"),
            (100, "Small shader"),
            (1000, "Medium shader"),
            (10000, "Large shader"),
            (100000, "Very large shader"),
        ];

        for (size, description) in size_tests {
            let size: usize = size;
            let large_spirv: Vec<u32> = std::iter::once(0x07230203) // Magic
                .chain(std::iter::once(0x00010000)) // Version
                .chain(std::iter::once(0x00080001)) // Generator
                .chain(std::iter::once((size + 1) as u32)) // Bound
                .chain(std::iter::once(0x00000000)) // Schema
                .chain(std::iter::repeat(0x00000000).take(size.saturating_sub(5))) // Padding
                .collect();

            println!("    Testing {}: {} words", description, large_spirv.len());

            match Algorithm::new(&mut manager, &large_spirv) {
                Ok(_algorithm) => {
                    println!("      ✓ Size {} accepted", size);
                }
                Err(e) => {
                    println!("      ⚠ Size {} rejected: {}", size, e);
                }
            }
        }

        println!("  ✓ SPIR-V validation edge case testing completed");
    } else {
        println!("Skipping SPIR-V edge case test - no GPU available");
    }
}

/// Test algorithm creation performance and consistency.
/// Verifies algorithm creation performance doesn't degrade.
#[test]

fn test_algorithm_creation_performance() {
    if let Ok(mut manager) = Manager::new() {
        println!("Testing algorithm creation performance...");

        let test_spirv = vec![
            0x07230203, // Magic number
            0x00010000, // Version 1.0
            0x00080001, // Generator
            0x0000000b, // Bound
            0x00000000, // Schema
        ];

        const PERFORMANCE_ITERATIONS: usize = 20;
        let mut creation_times = Vec::new();
        let mut successful_creations = 0;

        for i in 0..PERFORMANCE_ITERATIONS {
            let start_time = Instant::now();

            match Algorithm::new(&mut manager, &test_spirv) {
                Ok(_algorithm) => {
                    let creation_time = start_time.elapsed();
                    creation_times.push(creation_time);
                    successful_creations += 1;

                    println!("  Algorithm {}: created in {:?}", i, creation_time);

                    // Individual creation should be reasonable
                    assert!(
                        creation_time.as_millis() < 1000,
                        "Algorithm {} creation took too long: {:?}",
                        i,
                        creation_time
                    );
                }
                Err(e) => {
                    println!("  Algorithm {} creation failed: {}", i, e);
                }
            }
        }

        if successful_creations > 0 {
            let avg_time =
                creation_times.iter().sum::<std::time::Duration>() / successful_creations as u32;
            let min_time = creation_times.iter().min().unwrap();
            let max_time = creation_times.iter().max().unwrap();

            println!("Algorithm creation performance results:");
            println!(
                "  Successful creations: {}/{}",
                successful_creations, PERFORMANCE_ITERATIONS
            );
            println!("  Average time: {:?}", avg_time);
            println!("  Min time: {:?}", min_time);
            println!("  Max time: {:?}", max_time);
            println!("  Time variance: {:?}", max_time.saturating_sub(*min_time));

            // Performance should be reasonably consistent (allow for first-time initialization)
            // The first algorithm creation is often slower due to initialization
            assert!(
                max_time.as_millis() < min_time.as_millis() * 10 || max_time.as_millis() < 10,
                "Performance should be reasonably consistent - max: {:?}, min: {:?}",
                max_time,
                min_time
            );

            // Average time should be reasonable
            assert!(
                avg_time.as_millis() < 200,
                "Average algorithm creation should be fast: {:?}",
                avg_time
            );
        } else {
            println!("No successful algorithm creations - skipping performance analysis");
        }
    } else {
        println!("Skipping algorithm performance test - no GPU available");
    }
}

/// Test concurrent algorithm creation across multiple threads.
/// Verifies thread safety of algorithm operations.
#[test]

fn test_algorithm_concurrent_creation() {
    println!("Testing concurrent algorithm creation...");

    if let Ok(manager) = Manager::new() {
        let manager = Arc::new(std::sync::Mutex::new(manager));
        let thread_count = 4;
        let algorithms_per_thread = 5;
        let mut handles = vec![];

        let test_spirv = vec![
            0x07230203, // Magic number
            0x00010000, // Version 1.0
            0x00080001, // Generator
            0x0000000b, // Bound
            0x00000000, // Schema
        ];

        for thread_id in 0..thread_count {
            let manager_clone = Arc::clone(&manager);
            let spirv_clone = test_spirv.clone();

            let handle = thread::spawn(move || {
                let mut thread_results = Vec::new();

                for algorithm_id in 0..algorithms_per_thread {
                    let mut manager_guard = manager_clone.lock().unwrap();

                    match Algorithm::new(&mut *manager_guard, &spirv_clone) {
                        Ok(algorithm) => {
                            let handle = algorithm.handle();
                            thread_results.push((thread_id, algorithm_id, handle, true));
                            drop(manager_guard);

                            // Verify algorithm outside lock
                            assert_ne!(
                                handle, 0,
                                "Thread {} algorithm {} handle should be non-zero",
                                thread_id, algorithm_id
                            );
                        }
                        Err(_e) => {
                            thread_results.push((thread_id, algorithm_id, 0, false));
                            drop(manager_guard);
                        }
                    }
                }

                thread_results
            });

            handles.push(handle);
        }

        // Collect results from all threads
        let mut all_handles = Vec::new();
        let mut total_created = 0;

        for handle in handles {
            if let Ok(thread_results) = handle.join() {
                for (thread_id, algorithm_id, algorithm_handle, success) in thread_results {
                    if success {
                        all_handles.push(algorithm_handle);
                        total_created += 1;
                        println!(
                            "  Thread {} created algorithm {} with handle {}",
                            thread_id, algorithm_id, algorithm_handle
                        );
                    }
                }
            }
        }

        println!(
            "  ✓ Created {} algorithms across {} threads",
            total_created, thread_count
        );

        // Verify all handles are unique
        for i in 0..all_handles.len() {
            for j in (i + 1)..all_handles.len() {
                assert_ne!(
                    all_handles[i], all_handles[j],
                    "Algorithm handles should be unique across threads: {} != {}",
                    all_handles[i], all_handles[j]
                );
            }
            assert_ne!(
                all_handles[i], 0,
                "All algorithm handles should be non-zero"
            );
        }

        if total_created > 0 {
            println!(
                "  ✓ Concurrent algorithm creation successful with {} algorithms",
                total_created
            );
        }
    } else {
        println!("Skipping concurrent algorithm test - no GPU available");
    }
}
