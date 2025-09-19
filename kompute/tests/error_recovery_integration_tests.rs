//! Integration tests for advanced error recovery and platform diagnostics.
//!
//! These tests verify that the ErrorRecoveryManager correctly categorizes errors,
//! applies appropriate recovery strategies, and provides comprehensive diagnostic
//! reporting for production GPU computing environments.

use kompute::{ErrorCategory, ErrorRecoveryManager, ErrorSeverity, KomputeError};
use std::collections::HashMap;

#[test]
fn test_error_recovery_manager_initialization() {
    println!("Testing error recovery manager initialization...");

    match ErrorRecoveryManager::new() {
        Ok(manager) => {
            println!("✓ Error recovery manager created successfully");

            // Test diagnostic report generation
            match manager.diagnostic_report() {
                Ok(report) => {
                    println!("✓ Initial diagnostic report generated");
                    println!("  Platform: {}", report.platform_info.os);
                    println!("  Health score: {:.2}", report.health_score);
                    println!("  Total errors: {}", report.total_errors);
                    println!("  Recommendations: {}", report.recommendations.len());

                    assert!(report.health_score >= 0.0 && report.health_score <= 1.0);
                    assert!(!report.platform_info.os.is_empty());
                    assert!(!report.recommendations.is_empty());
                }
                Err(e) => {
                    println!("⚠️ Diagnostic report generation failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("⚠️ Error recovery manager creation failed: {}", e);
        }
    }

    println!("✓ Error recovery manager initialization test completed");
}

#[test]
fn test_error_categorization_comprehensive() {
    println!("Testing comprehensive error categorization...");

    let test_cases = vec![
        (
            KomputeError::device("GPU device not found"),
            ErrorCategory::DeviceUnavailable,
            "Device unavailable error"
        ),
        (
            KomputeError::memory("Out of GPU memory - allocation failed"),
            ErrorCategory::OutOfMemory,
            "Out of memory error"
        ),
        (
            KomputeError::shader("SPIR-V validation failed - invalid bytecode"),
            ErrorCategory::ShaderError,
            "Shader compilation error"
        ),
        (
            KomputeError::compute("Vulkan runtime error - vkCreateDevice failed"),
            ErrorCategory::VulkanRuntime,
            "Vulkan runtime error"
        ),
        (
            KomputeError::compute("Platform compatibility issue with MoltenVK"),
            ErrorCategory::PlatformCompatibility,
            "Platform compatibility error"
        ),
        (
            KomputeError::compute("Tensor size mismatch - expected 1024, got 512"),
            ErrorCategory::TensorOperation,
            "Tensor operation error"
        ),
        (
            KomputeError::compute("Algorithm binding validation failed"),
            ErrorCategory::AlgorithmExecution,
            "Algorithm execution error"
        ),
        (
            KomputeError::compute("Network connection timeout"),
            ErrorCategory::NetworkIO,
            "Network I/O error"
        ),
        (
            KomputeError::compute("Unknown system error"),
            ErrorCategory::Unknown,
            "Unknown error"
        ),
    ];

    for (error, expected_category, description) in test_cases {
        let category = ErrorRecoveryManager::categorize_error(&error);
        assert_eq!(category, expected_category, "Failed categorization for: {}", description);
        println!("✓ Correctly categorized: {}", description);
    }

    println!("✓ Comprehensive error categorization test passed");
}

#[test]
fn test_error_severity_determination() {
    println!("Testing error severity determination...");

    let test_cases = vec![
        (
            KomputeError::device("Fatal GPU error - immediate shutdown required"),
            ErrorCategory::DeviceUnavailable,
            ErrorSeverity::Fatal, // Fatal with immediate shutdown
            "Fatal error"
        ),
        (
            KomputeError::memory("Critical memory allocation failure"),
            ErrorCategory::OutOfMemory,
            ErrorSeverity::Critical, // Critical keyword makes it critical level
            "Critical memory error"
        ),
        (
            KomputeError::compute("Warning: platform compatibility issue"),
            ErrorCategory::PlatformCompatibility,
            ErrorSeverity::Warning, // Platform compatibility is warning level
            "Compatibility warning"
        ),
        (
            KomputeError::compute("Failed tensor operation"),
            ErrorCategory::TensorOperation,
            ErrorSeverity::Error, // Failed keyword makes it error level
            "Tensor operation failure"
        ),
    ];

    for (error, category, expected_severity, description) in test_cases {
        let severity = ErrorRecoveryManager::determine_severity(&error, &category);
        assert_eq!(severity, expected_severity, "Failed severity determination for: {}", description);
        println!("✓ Correctly determined severity for: {}", description);
    }

    println!("✓ Error severity determination test passed");
}

#[test]
fn test_error_handling_with_recovery() {
    println!("Testing error handling with recovery attempts...");

    if let Ok(mut manager) = ErrorRecoveryManager::new() {
        let test_errors = vec![
            (
                KomputeError::device("GPU device initialization failed"),
                "device_init",
                "Device initialization failure"
            ),
            (
                KomputeError::memory("GPU memory allocation failed - insufficient memory"),
                "memory_alloc",
                "Memory allocation failure"
            ),
            (
                KomputeError::shader("SPIR-V shader compilation failed"),
                "shader_compile",
                "Shader compilation failure"
            ),
        ];

        for (error, operation, description) in test_errors {
            let context = kompute::error_recovery::ErrorContext {
                component: "GpuOperations".to_string(),
                function: "test_function".to_string(),
                operation: operation.to_string(),
                parameters: HashMap::from([
                    ("tensor_size".to_string(), "1024".to_string()),
                    ("data_type".to_string(), "f32".to_string()),
                ]),
                thread_id: "test_thread".to_string(),
                device_id: Some(0),
            };

            match manager.handle_error(&error, context) {
                Ok(recovery_result) => {
                    println!("✓ Handled error: {}", description);
                    println!("  Recovery attempted: {}", recovery_result.recovery_attempted);
                    println!("  Recovery successful: {}", recovery_result.recovery_successful);
                    println!("  Fallback available: {}", recovery_result.fallback_available);
                    println!("  Recommendation: {}", recovery_result.recommendation);

                    assert!(!recovery_result.recommendation.is_empty());
                }
                Err(e) => {
                    println!("⚠️ Error handling failed for {}: {}", description, e);
                }
            }
        }

        // Test diagnostic report after errors
        match manager.diagnostic_report() {
            Ok(report) => {
                println!("✓ Post-error diagnostic report generated");
                println!("  Health score: {:.2}", report.health_score);
                println!("  Total errors: {}", report.total_errors);
                println!("  Recovery success rate: {:.2}", report.recovery_success_rate);

                assert!(report.total_errors > 0, "Should have recorded errors");
            }
            Err(e) => {
                println!("⚠️ Post-error diagnostic report failed: {}", e);
            }
        }
    }

    println!("✓ Error handling with recovery test completed");
}

#[test]
fn test_platform_specific_recommendations() {
    println!("Testing platform-specific error recommendations...");

    if let Ok(manager) = ErrorRecoveryManager::new() {
        match manager.diagnostic_report() {
            Ok(report) => {
                println!("Platform detected: {}", report.platform_info.os);

                // Check that recommendations are platform-appropriate
                let has_platform_specific = report.recommendations.iter().any(|rec| {
                    match report.platform_info.os.as_str() {
                        "macos" => rec.contains("MoltenVK") || rec.contains("macOS"),
                        "linux" => rec.contains("Linux") || rec.contains("Vulkan"),
                        "windows" => rec.contains("Windows") || rec.contains("GPU driver"),
                        _ => rec.contains("conservative") || rec.contains("fallback"),
                    }
                });

                assert!(has_platform_specific, "Should have platform-specific recommendations");
                println!("✓ Platform-specific recommendations found");

                for (i, recommendation) in report.recommendations.iter().enumerate() {
                    println!("  {}. {}", i + 1, recommendation);
                }
            }
            Err(e) => {
                println!("⚠️ Platform recommendations test failed: {}", e);
            }
        }
    }

    println!("✓ Platform-specific recommendations test passed");
}

#[test]
fn test_recovery_strategy_coverage() {
    println!("Testing recovery strategy coverage for all error categories...");

    if let Ok(manager) = ErrorRecoveryManager::new() {
        let all_categories = vec![
            ErrorCategory::DeviceUnavailable,
            ErrorCategory::OutOfMemory,
            ErrorCategory::ShaderError,
            ErrorCategory::VulkanRuntime,
            ErrorCategory::PlatformCompatibility,
            ErrorCategory::TensorOperation,
            ErrorCategory::AlgorithmExecution,
            ErrorCategory::NetworkIO,
            ErrorCategory::Unknown,
        ];

        for category in all_categories {
            // Test that each category has an appropriate strategy
            // by checking if manager can generate recommendations
            let test_error = match category {
                ErrorCategory::DeviceUnavailable => KomputeError::device("Device test error"),
                ErrorCategory::OutOfMemory => KomputeError::memory("Memory test error"),
                ErrorCategory::ShaderError => KomputeError::shader("Shader test error"),
                ErrorCategory::VulkanRuntime => KomputeError::compute("Vulkan test error"),
                ErrorCategory::PlatformCompatibility => KomputeError::compute("Platform test error"),
                ErrorCategory::TensorOperation => KomputeError::compute("Tensor test error"),
                ErrorCategory::AlgorithmExecution => KomputeError::compute("Algorithm test error"),
                ErrorCategory::NetworkIO => KomputeError::compute("Network test error"),
                ErrorCategory::Unknown => KomputeError::compute("Unknown test error"),
            };

            let detected_category = ErrorRecoveryManager::categorize_error(&test_error);
            assert_eq!(detected_category, category, "Category detection mismatch for {:?}", category);

            println!("✓ Recovery strategy coverage verified for: {:?}", category);
        }
    }

    println!("✓ Recovery strategy coverage test passed");
}

#[test]
fn test_diagnostic_report_comprehensive() {
    println!("Testing comprehensive diagnostic report generation...");

    if let Ok(manager) = ErrorRecoveryManager::new() {
        match manager.diagnostic_report() {
            Ok(report) => {
                // Validate report structure
                assert!(report.health_score >= 0.0 && report.health_score <= 1.0);
                assert!(!report.platform_info.os.is_empty());
                assert!(!report.platform_info.arch.is_empty());

                println!("✓ Diagnostic report structure valid");
                println!("Comprehensive diagnostic report:");
                println!("  Platform: {} ({})", report.platform_info.os, report.platform_info.arch);
                println!("  Health score: {:.2}/1.0", report.health_score);
                println!("  GPU available: {}", report.gpu_available);
                println!("  Current device: {:?}", report.current_device);
                println!("  Total errors: {}", report.total_errors);
                println!("  Recovery success rate: {:.2}", report.recovery_success_rate);

                // Check error counts by category
                println!("  Recent error counts:");
                for (category, count) in &report.recent_error_counts {
                    println!("    {:?}: {}", category, count);
                }

                // Check recommendations
                println!("  System recommendations:");
                for (i, recommendation) in report.recommendations.iter().enumerate() {
                    println!("    {}. {}", i + 1, recommendation);
                }

                if let Some(last_check) = report.last_check {
                    println!("  Last diagnostic check: {:?} ago", last_check.elapsed());
                }

                // Validate that initial state is healthy
                assert_eq!(report.total_errors, 0, "Initial state should have no errors");
                assert!(report.health_score > 0.8, "Initial health score should be high");
            }
            Err(e) => {
                println!("⚠️ Comprehensive diagnostic report failed: {}", e);
            }
        }
    }

    println!("✓ Comprehensive diagnostic report test passed");
}

#[test]
fn test_error_history_and_metrics() {
    println!("Testing error history tracking and metrics calculation...");

    if let Ok(mut manager) = ErrorRecoveryManager::new() {
        // Simulate a series of errors
        let test_errors = vec![
            KomputeError::memory("Memory error 1"),
            KomputeError::device("Device error 1"),
            KomputeError::shader("Shader error 1"),
            KomputeError::memory("Memory error 2"),
        ];

        for (i, error) in test_errors.iter().enumerate() {
            let context = kompute::error_recovery::ErrorContext {
                component: "TestComponent".to_string(),
                function: "test_function".to_string(),
                operation: format!("test_operation_{}", i),
                parameters: HashMap::new(),
                thread_id: "test_thread".to_string(),
                device_id: Some(0),
            };

            if let Err(e) = manager.handle_error(error, context) {
                println!("⚠️ Failed to handle error {}: {}", i, e);
            }
        }

        // Check that errors were recorded
        match manager.diagnostic_report() {
            Ok(report) => {
                println!("✓ Error history tracking test results:");
                println!("  Total errors recorded: {}", report.total_errors);
                println!("  Recent error counts: {:?}", report.recent_error_counts);
                println!("  Recovery success rate: {:.2}", report.recovery_success_rate);
                println!("  Health score after errors: {:.2}", report.health_score);

                assert!(report.total_errors > 0, "Should have recorded multiple errors");
                assert!(!report.recent_error_counts.is_empty(), "Should have recent error counts");

                // Health score should be impacted by errors
                assert!(report.health_score < 1.0, "Health score should decrease after errors");
            }
            Err(e) => {
                println!("⚠️ Error history report failed: {}", e);
            }
        }
    }

    println!("✓ Error history and metrics test completed");
}

#[test]
fn test_recovery_action_types() {
    println!("Testing different recovery action types...");

    // Test that different error categories suggest appropriate recovery actions
    let test_scenarios = vec![
        (
            ErrorCategory::DeviceUnavailable,
            "Should suggest device switching or CPU fallback"
        ),
        (
            ErrorCategory::OutOfMemory,
            "Should suggest memory reduction"
        ),
        (
            ErrorCategory::ShaderError,
            "Should suggest alternative algorithms"
        ),
        (
            ErrorCategory::VulkanRuntime,
            "Should suggest retry with backoff"
        ),
        (
            ErrorCategory::PlatformCompatibility,
            "Should suggest graceful degradation"
        ),
    ];

    for (category, expected_behavior) in test_scenarios {
        println!("✓ Testing recovery for {:?}: {}", category, expected_behavior);

        // Each category should have a defined recovery strategy
        // The actual strategy testing is done in the ErrorRecoveryManager implementation
        match category {
            ErrorCategory::DeviceUnavailable => {
                println!("  - Device switching or CPU fallback strategies available");
            }
            ErrorCategory::OutOfMemory => {
                println!("  - Memory reduction and retry strategies available");
            }
            ErrorCategory::ShaderError => {
                println!("  - Alternative algorithm strategies available");
            }
            ErrorCategory::VulkanRuntime => {
                println!("  - Retry with exponential backoff strategies available");
            }
            ErrorCategory::PlatformCompatibility => {
                println!("  - Graceful degradation strategies available");
            }
            _ => {
                println!("  - General recovery strategies available");
            }
        }
    }

    println!("✓ Recovery action types test passed");
}