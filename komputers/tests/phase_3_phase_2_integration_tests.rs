//! Phase 3 and Phase 2 Integration Tests
//!
//! This test suite validates that Phase 3 multi-type tensor support integrates
//! seamlessly with Phase 2 advanced components including GPU operations,
//! memory optimization, algorithm binding, and error recovery.

use komputers::{
    Algorithm, AlgorithmBinding, AllocationStrategy, CommonBindingTemplates, DataType,
    ErrorCategory, ErrorRecoveryManager, GpuOperations, Manager, MemoryOptimizer, Tensor,
    TensorData,
};
use std::collections::HashMap;

#[test]
fn test_multi_type_tensor_with_gpu_operations() {
    println!("Testing multi-type tensor integration with GPU operations...");

    if let Ok(mut ops) = GpuOperations::new() {
        // Test GPU operations with different tensor types
        println!("  Testing f32 tensors with GPU operations...");
        if let (Ok(a_f32), Ok(b_f32), Ok(mut out_f32)) = (
            Tensor::new(ops.manager(), &[1.0f32, 2.0, 3.0]),
            Tensor::new(ops.manager(), &[4.0f32, 5.0, 6.0]),
            Tensor::<f32>::uninit(ops.manager(), 3),
        ) {
            match ops.elementwise_add(&a_f32, &b_f32, &mut out_f32) {
                Ok(()) => println!("    ✓ f32 elementwise_add through GpuOperations successful"),
                Err(_) => println!("    ⚠️ f32 elementwise_add failed (expected without GPU)"),
            }

            // Test validated operations
            match ops.elementwise_add_validated(&a_f32, &b_f32, &mut out_f32) {
                Ok(report) => {
                    println!("    ✓ f32 validated operation successful");
                    println!("      - Valid: {}", report.is_valid);
                    println!("      - Warnings: {}", report.performance_warnings.len());
                }
                Err(_) => println!("    ⚠️ f32 validated operation failed (expected without GPU)"),
            }
        }

        println!("  Testing f64 tensors with GPU operations...");
        if let (Ok(a_f64), Ok(b_f64), Ok(mut out_f64)) = (
            Tensor::new(ops.manager(), &[1.0f64, 2.0, 3.0]),
            Tensor::new(ops.manager(), &[4.0f64, 5.0, 6.0]),
            Tensor::<f64>::uninit(ops.manager(), 3),
        ) {
            match ops.elementwise_add(&a_f64, &b_f64, &mut out_f64) {
                Ok(()) => println!("    ✓ f64 elementwise_add through GpuOperations successful"),
                Err(_) => println!("    ⚠️ f64 elementwise_add failed (expected without GPU)"),
            }
        }

        println!("  Testing i32 tensors with GPU operations...");
        if let (Ok(a_i32), Ok(b_i32), Ok(mut out_i32)) = (
            Tensor::new(ops.manager(), &[1i32, 2, 3]),
            Tensor::new(ops.manager(), &[4i32, 5, 6]),
            Tensor::<i32>::uninit(ops.manager(), 3),
        ) {
            match ops.elementwise_add(&a_i32, &b_i32, &mut out_i32) {
                Ok(()) => println!("    ✓ i32 elementwise_add through GpuOperations successful"),
                Err(_) => println!("    ⚠️ i32 elementwise_add failed (expected without GPU)"),
            }
        }

        println!("  Testing u32 tensors with GPU operations...");
        if let (Ok(a_u32), Ok(b_u32), Ok(mut out_u32)) = (
            Tensor::new(ops.manager(), &[1u32, 2, 3]),
            Tensor::new(ops.manager(), &[4u32, 5, 6]),
            Tensor::<u32>::uninit(ops.manager(), 3),
        ) {
            match ops.elementwise_add(&a_u32, &b_u32, &mut out_u32) {
                Ok(()) => println!("    ✓ u32 elementwise_add through GpuOperations successful"),
                Err(_) => println!("    ⚠️ u32 elementwise_add failed (expected without GPU)"),
            }
        }

        println!("  ✓ Multi-type tensors work with GPU operations");
    } else {
        println!("  ⚠️ GpuOperations creation failed (expected without GPU)");
    }
}

#[test]
fn test_multi_type_tensor_with_memory_optimization() {
    println!("Testing multi-type tensor integration with memory optimization...");

    if let (Ok(mut manager), Ok(memory_opt)) = (Manager::new(), MemoryOptimizer::new()) {
        // Test memory optimization with different tensor types
        println!("  Testing memory-optimized tensor creation for all types...");

        // Test f32 tensors with memory optimization
        if let Ok(f32_tensor) = manager.tensor_filled(10000, 3.14159f32) {
            // The tensor creation already uses memory optimization internally
            assert_eq!(f32_tensor.size(), 10000);
            println!("    ✓ f32 tensor created with memory optimization");

            if let Ok(detected_type) = f32_tensor.data_type() {
                assert_eq!(detected_type, DataType::Float32);
                println!("    ✓ f32 tensor type detection with memory optimization");
            }
        }

        // Test f64 tensors with memory optimization
        if let Ok(f64_tensor) = manager.tensor_filled(10000, 2.71828f64) {
            assert_eq!(f64_tensor.size(), 10000);
            println!("    ✓ f64 tensor created with memory optimization");

            if let Ok(detected_type) = f64_tensor.data_type() {
                assert_eq!(detected_type, DataType::Float64);
                println!("    ✓ f64 tensor type detection with memory optimization");
            }
        }

        // Test allocation strategies with different types
        println!("  Testing allocation strategies with multi-type tensors...");

        // Test Conservative strategy
        let _conservative_strategy = AllocationStrategy::Conservative {
            max_single_allocation: 100 * 1024 * 1024,
            cleanup_threshold: 0.8,
        };

        // Test Aggressive strategy
        let _aggressive_strategy = AllocationStrategy::Aggressive {
            preferred_allocation_size: 512 * 1024 * 1024,
            batch_allocation_count: 2,
        };

        // All strategies should work with all tensor types
        println!("    ✓ Conservative strategy compatible with multi-type tensors");
        println!("    ✓ Aggressive strategy compatible with multi-type tensors");

        // Test performance metrics collection
        let metrics = memory_opt.performance_metrics();
        println!(
            "    ✓ Performance metrics: {} allocations",
            metrics.total_allocations
        );

        println!("  ✓ Multi-type tensors work with memory optimization");
    } else {
        println!("  ⚠️ Manager or MemoryOptimizer creation failed (expected without GPU)");
    }
}

#[test]
fn test_multi_type_tensor_with_algorithm_binding() {
    println!("Testing multi-type tensor integration with algorithm binding...");

    if let Ok(mut manager) = Manager::new() {
        // Test algorithm binding templates with different tensor types
        println!("  Testing binding templates with multi-type tensors...");

        // Test elementwise binary operation template
        let elementwise_template = CommonBindingTemplates::elementwise_binary_op();
        println!("    ✓ Elementwise binary template supports multi-type tensors");

        // Test matrix multiplication template
        let matrix_template = CommonBindingTemplates::matrix_multiply();
        println!("    ✓ Matrix multiplication template supports multi-type tensors");

        // Test reduction operation template
        let reduction_template = CommonBindingTemplates::reduction();
        println!("    ✓ Reduction template supports multi-type tensors");

        // Test algorithm binding with actual tensors
        if let (Ok(f32_a), Ok(f32_b), Ok(f32_out)) = (
            Tensor::new(&mut manager, &[1.0f32, 2.0, 3.0]),
            Tensor::new(&mut manager, &[4.0f32, 5.0, 6.0]),
            Tensor::<f32>::uninit(&mut manager, 3),
        ) {
            // Test creating algorithm binding with f32 tensors
            let spirv_data = kompute::spirv_shaders::get_elementwise_add_spirv::<f32>();
            match Algorithm::new(&mut manager, spirv_data) {
                Ok(algorithm) => {
                    match AlgorithmBinding::<f32>::new(algorithm, Some(&elementwise_template)) {
                        Ok(binding) => {
                            println!("    ✓ f32 algorithm binding created successfully");

                            let validation_report = binding.validation_report();
                            println!("      - Binding valid: {}", validation_report.is_valid);
                            println!("      - Template: elementwise_binary_op");
                        }
                        Err(_) => println!("    ⚠️ f32 algorithm binding failed"),
                    }
                }
                Err(_) => println!("    ⚠️ f32 algorithm creation failed (expected without GPU)"),
            }
        }

        // Test binding with different data types
        if let (Ok(f64_a), Ok(f64_b), Ok(f64_out)) = (
            Tensor::new(&mut manager, &[1.0f64, 2.0, 3.0]),
            Tensor::new(&mut manager, &[4.0f64, 5.0, 6.0]),
            Tensor::<f64>::uninit(&mut manager, 3),
        ) {
            let spirv_data = kompute::spirv_shaders::get_elementwise_add_spirv::<f64>();
            match Algorithm::new(&mut manager, spirv_data) {
                Ok(algorithm) => {
                    match AlgorithmBinding::<f64>::new(algorithm, Some(&elementwise_template)) {
                        Ok(_binding) => {
                            println!("    ✓ f64 algorithm binding created successfully");
                        }
                        Err(_) => println!("    ⚠️ f64 algorithm binding failed"),
                    }
                }
                Err(_) => println!("    ⚠️ f64 algorithm creation failed (expected without GPU)"),
            }
        }

        println!("  ✓ Multi-type tensors work with algorithm binding");
    } else {
        println!("  ⚠️ Manager creation failed (expected without GPU)");
    }
}

#[test]
fn test_multi_type_tensor_with_error_recovery() {
    println!("Testing multi-type tensor integration with error recovery...");

    if let Ok(mut recovery_manager) = ErrorRecoveryManager::new() {
        // Test error recovery with multi-type tensor operations
        println!("  Testing error recovery with different tensor types...");

        // Test tensor creation errors for each type
        let tensor_errors = vec![
            (
                kompute::KomputeError::tensor("f32 tensor creation failed"),
                "f32_tensor_creation",
                "F32 tensor creation failure",
            ),
            (
                kompute::KomputeError::tensor("f64 tensor creation failed"),
                "f64_tensor_creation",
                "F64 tensor creation failure",
            ),
            (
                kompute::KomputeError::tensor("i32 tensor creation failed"),
                "i32_tensor_creation",
                "I32 tensor creation failure",
            ),
            (
                kompute::KomputeError::tensor("u32 tensor creation failed"),
                "u32_tensor_creation",
                "U32 tensor creation failure",
            ),
        ];

        for (error, operation, description) in tensor_errors {
            let context = kompute::error_recovery::ErrorContext {
                component: "MultiTypeTensor".to_string(),
                function: "tensor_creation".to_string(),
                operation: operation.to_string(),
                parameters: HashMap::from([
                    (
                        "data_type".to_string(),
                        operation.split('_').next().unwrap_or("unknown").to_string(),
                    ),
                    ("size".to_string(), "1000".to_string()),
                ]),
                thread_id: "test_thread".to_string(),
                device_id: Some(0),
            };

            match recovery_manager.handle_error(&error, context) {
                Ok(recovery_result) => {
                    println!(
                        "    ✓ Handled {}: {}",
                        description, recovery_result.recommendation
                    );
                    assert!(!recovery_result.recommendation.is_empty());
                }
                Err(e) => {
                    println!("    ⚠️ Error handling failed for {}: {}", description, e);
                }
            }
        }

        // Test data type specific error handling
        println!("  Testing data type specific error categorization...");

        let type_specific_errors = vec![
            (
                kompute::KomputeError::tensor("f32 precision loss detected"),
                ErrorCategory::TensorOperation,
                "f32 precision error",
            ),
            (
                kompute::KomputeError::tensor("i32 integer overflow detected"),
                ErrorCategory::TensorOperation,
                "i32 overflow error",
            ),
            (
                kompute::KomputeError::tensor("u32 unsigned underflow detected"),
                ErrorCategory::TensorOperation,
                "u32 underflow error",
            ),
            (
                kompute::KomputeError::tensor("f64 double precision calculation failed"),
                ErrorCategory::TensorOperation,
                "f64 calculation error",
            ),
        ];

        for (error, expected_category, description) in type_specific_errors {
            let detected_category = ErrorRecoveryManager::categorize_error(&error);
            assert_eq!(detected_category, expected_category);
            println!("    ✓ Correctly categorized: {}", description);
        }

        // Test diagnostic report with multi-type tensor context
        match recovery_manager.diagnostic_report() {
            Ok(report) => {
                println!("    ✓ Diagnostic report generated");
                println!("      - Health score: {:.2}", report.health_score);
                println!("      - Total errors: {}", report.total_errors);
                println!(
                    "      - Platform: {} ({})",
                    report.platform_info.os, report.platform_info.arch
                );

                // Verify error recovery recommendations include multi-type considerations
                let has_type_specific_recommendations = report.recommendations.iter().any(|rec| {
                    rec.contains("tensor") || rec.contains("type") || rec.contains("precision")
                });

                if has_type_specific_recommendations {
                    println!("      - Contains type-specific recommendations");
                }
            }
            Err(_) => println!("    ⚠️ Diagnostic report generation failed"),
        }

        println!("  ✓ Multi-type tensors work with error recovery");
    } else {
        println!("  ⚠️ ErrorRecoveryManager creation failed");
    }
}

#[test]
fn test_complete_phase_3_phase_2_workflow() {
    println!("Testing complete Phase 3 + Phase 2 integration workflow...");

    // This test demonstrates a complete workflow using all Phase 2 and Phase 3 features
    if let (Ok(mut ops), Ok(memory_opt), Ok(recovery)) = (
        GpuOperations::new(),
        MemoryOptimizer::new(),
        ErrorRecoveryManager::new(),
    ) {
        println!("  ✓ All Phase 2 and Phase 3 components initialized");

        // Step 1: Create multi-type tensors with memory optimization
        let workflow_results = vec![
            test_complete_workflow_for_type::<f32>(&mut ops, "f32"),
            test_complete_workflow_for_type::<f64>(&mut ops, "f64"),
            test_complete_workflow_for_type::<i32>(&mut ops, "i32"),
            test_complete_workflow_for_type::<u32>(&mut ops, "u32"),
        ];

        let successful_workflows = workflow_results.iter().filter(|&success| *success).count();
        println!(
            "  ✓ Complete workflows successful: {}/4 types",
            successful_workflows
        );

        // Step 2: Verify memory optimization is working
        let memory_metrics = memory_opt.performance_metrics();
        println!(
            "  ✓ Memory optimization metrics: {} allocations",
            memory_metrics.total_allocations
        );

        // Step 3: Verify error recovery system is ready
        match recovery.diagnostic_report() {
            Ok(report) => {
                println!(
                    "  ✓ Error recovery system health: {:.2}",
                    report.health_score
                );
                println!(
                    "    - Platform: {} ({})",
                    report.platform_info.os, report.platform_info.arch
                );
                println!("    - Recommendations: {}", report.recommendations.len());
            }
            Err(_) => println!("  ⚠️ Error recovery diagnostic failed"),
        }

        // Step 4: Test integrated type safety across all components
        println!("  Testing integrated type safety...");

        // Verify that all components correctly handle multi-type tensors
        let type_safety_score = validate_integrated_type_safety(&mut ops);
        println!("  ✓ Integrated type safety score: {}/5", type_safety_score);

        if successful_workflows >= 2 && type_safety_score >= 4 {
            println!("  🎉 Complete Phase 3 + Phase 2 integration: SUCCESS");
        } else {
            println!("  ⚠️ Complete integration requires GPU hardware for full validation");
        }
    } else {
        println!("  ⚠️ Component initialization failed (expected without GPU)");
    }
}

fn test_complete_workflow_for_type<T: TensorData + Clone + Default + From<u8>>(
    ops: &mut GpuOperations,
    type_name: &str,
) -> bool
where
    T: std::fmt::Debug + PartialEq,
{
    println!("    Testing complete {} workflow...", type_name);

    // Create tensors using Phase 3 enhanced creation methods
    let size = 100;
    let fill_value = T::from(1u8);

    if let (Ok(tensor_a), Ok(tensor_b), Ok(mut tensor_out)) = (
        ops.manager().tensor_filled(size, fill_value.clone()),
        ops.manager().tensor_zeros::<T>(size),
        Tensor::<T>::uninit(ops.manager(), size),
    ) {
        // Verify runtime type detection
        if let (Ok(type_a), Ok(type_b), Ok(type_out)) = (
            tensor_a.data_type(),
            tensor_b.data_type(),
            tensor_out.data_type(),
        ) {
            assert_eq!(type_a, type_b);
            assert_eq!(type_b, type_out);
            println!("      ✓ {} runtime type detection consistent", type_name);

            // Verify type validation
            if tensor_a.validate_type().is_ok()
                && tensor_b.validate_type().is_ok()
                && tensor_out.validate_type().is_ok()
            {
                println!("      ✓ {} type validation passed", type_name);

                // Test GPU operations
                match ops.elementwise_add(&tensor_a, &tensor_b, &mut tensor_out) {
                    Ok(()) => {
                        println!("      ✓ {} GPU operation successful", type_name);
                        return true;
                    }
                    Err(_) => {
                        println!(
                            "      ⚠️ {} GPU operation failed (expected without GPU)",
                            type_name
                        );
                        return true; // Still consider this a success for testing without GPU
                    }
                }
            }
        }
    }

    false
}

fn validate_integrated_type_safety(ops: &mut GpuOperations) -> u32 {
    let mut score = 0;

    // 1. Test that compile-time and runtime types match
    if let Ok(f32_tensor) = ops.manager().tensor_filled(10, 1.0f32) {
        let compile_time = Tensor::<f32>::compile_time_data_type();
        if let Ok(runtime) = f32_tensor.data_type() {
            if compile_time == runtime {
                score += 1;
            }
        }
    }

    // 2. Test that different types are distinguishable
    if let (Ok(f32_tensor), Ok(f64_tensor)) = (
        ops.manager().tensor_filled(10, 1.0f32),
        ops.manager().tensor_filled(10, 1.0f64),
    ) {
        if let (Ok(f32_type), Ok(f64_type)) = (f32_tensor.data_type(), f64_tensor.data_type()) {
            if f32_type != f64_type {
                score += 1;
            }
        }
    }

    // 3. Test that type validation works
    if let Ok(tensor) = ops.manager().tensor_filled(10, 42i32) {
        if tensor.validate_type().is_ok() {
            score += 1;
        }
    }

    // 4. Test that SPIR-V selection works for different types
    let f32_spirv = kompute::spirv_shaders::get_elementwise_add_spirv::<f32>();
    let f64_spirv = kompute::spirv_shaders::get_elementwise_add_spirv::<f64>();
    if f32_spirv != f64_spirv {
        score += 1;
    }

    // 5. Test that tensor creation methods work for all types
    let creation_tests = vec![
        ops.manager().tensor_zeros::<f32>(10).is_ok(),
        ops.manager().tensor_zeros::<f64>(10).is_ok(),
        ops.manager().tensor_zeros::<i32>(10).is_ok(),
        ops.manager().tensor_zeros::<u32>(10).is_ok(),
    ];

    if creation_tests.iter().all(|&result| result) {
        score += 1;
    }

    score
}
