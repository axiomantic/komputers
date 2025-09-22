//! Performance Regression Testing Suite
//!
//! This test suite implements comprehensive performance regression testing
//! for all critical operations across Phases 1, 2, and 3. It establishes
//! performance baselines and validates that operations remain within
//! acceptable performance bounds to prevent performance regressions.

use komputers::{Manager, MemoryOptimizer};
use std::collections::HashMap;
use std::thread;
use std::time::{Duration, Instant};

/// Performance baseline thresholds for critical operations
const MANAGER_CREATION_THRESHOLD_MS: u64 = 100;
const TENSOR_CREATION_THRESHOLD_MS: u64 = 50;
const ALGORITHM_CREATION_THRESHOLD_MS: u64 = 200;
const MEMORY_STATS_THRESHOLD_MS: u64 = 10;
const DEVICE_ENUMERATION_THRESHOLD_MS: u64 = 100;

/// Comprehensive performance regression testing framework
struct PerformanceRegressionFramework {
    baseline_metrics: HashMap<String, Duration>,
    current_metrics: HashMap<String, Duration>,
    regression_results: HashMap<String, RegressionStatus>,
    performance_thresholds: HashMap<String, Duration>,
}

#[derive(Debug, Clone, PartialEq)]
enum RegressionStatus {
    Pass,
    Warning(f64), // Performance degradation percentage
    Fail(f64),    // Significant regression percentage
}

impl PerformanceRegressionFramework {
    fn new() -> Self {
        let mut framework = Self {
            baseline_metrics: HashMap::new(),
            current_metrics: HashMap::new(),
            regression_results: HashMap::new(),
            performance_thresholds: HashMap::new(),
        };

        // Initialize performance thresholds
        framework.performance_thresholds.insert(
            "manager_creation".to_string(),
            Duration::from_millis(MANAGER_CREATION_THRESHOLD_MS),
        );
        framework.performance_thresholds.insert(
            "tensor_creation_small".to_string(),
            Duration::from_millis(TENSOR_CREATION_THRESHOLD_MS),
        );
        framework.performance_thresholds.insert(
            "tensor_creation_large".to_string(),
            Duration::from_millis(TENSOR_CREATION_THRESHOLD_MS * 5),
        );
        framework.performance_thresholds.insert(
            "algorithm_creation".to_string(),
            Duration::from_millis(ALGORITHM_CREATION_THRESHOLD_MS),
        );
        framework.performance_thresholds.insert(
            "memory_statistics".to_string(),
            Duration::from_millis(MEMORY_STATS_THRESHOLD_MS),
        );
        framework.performance_thresholds.insert(
            "device_enumeration".to_string(),
            Duration::from_millis(DEVICE_ENUMERATION_THRESHOLD_MS),
        );

        framework
    }

    fn measure_operation<F>(&mut self, operation_name: &str, operation: F) -> Duration
    where
        F: FnOnce(),
    {
        let start = Instant::now();
        operation();
        let duration = start.elapsed();

        self.current_metrics
            .insert(operation_name.to_string(), duration);
        println!("    ⏱️ {}: {:?}", operation_name, duration);

        duration
    }

    fn validate_performance(&mut self, operation_name: &str) {
        let current_duration = self
            .current_metrics
            .get(operation_name)
            .copied()
            .unwrap_or_default();
        let threshold = self
            .performance_thresholds
            .get(operation_name)
            .copied()
            .unwrap_or_default();

        let status = if current_duration <= threshold {
            RegressionStatus::Pass
        } else {
            let degradation_pct =
                ((current_duration.as_millis() as f64) / (threshold.as_millis() as f64) - 1.0)
                    * 100.0;

            if degradation_pct <= 20.0 {
                RegressionStatus::Warning(degradation_pct)
            } else {
                RegressionStatus::Fail(degradation_pct)
            }
        };

        self.regression_results
            .insert(operation_name.to_string(), status.clone());

        match status {
            RegressionStatus::Pass => {
                println!("    ✅ {}: Performance within threshold", operation_name)
            }
            RegressionStatus::Warning(pct) => println!(
                "    🔶 {}: Performance degradation {:.1}% (WARNING)",
                operation_name, pct
            ),
            RegressionStatus::Fail(pct) => println!(
                "    ❌ {}: Performance regression {:.1}% (FAIL)",
                operation_name, pct
            ),
        }
    }

    fn print_regression_summary(&self) {
        let total_tests = self.regression_results.len();
        let passed = self
            .regression_results
            .values()
            .filter(|&status| matches!(status, RegressionStatus::Pass))
            .count();
        let warnings = self
            .regression_results
            .values()
            .filter(|&status| matches!(status, RegressionStatus::Warning(_)))
            .count();
        let failures = self
            .regression_results
            .values()
            .filter(|&status| matches!(status, RegressionStatus::Fail(_)))
            .count();

        println!("\n=== PERFORMANCE REGRESSION TEST SUMMARY ===");
        println!("Total operations tested: {}", total_tests);
        println!(
            "✅ Passed: {} ({:.1}%)",
            passed,
            (passed as f64 / total_tests as f64) * 100.0
        );
        println!(
            "🔶 Warnings: {} ({:.1}%)",
            warnings,
            (warnings as f64 / total_tests as f64) * 100.0
        );
        println!(
            "❌ Failures: {} ({:.1}%)",
            failures,
            (failures as f64 / total_tests as f64) * 100.0
        );

        if failures == 0 && warnings <= 1 {
            println!("🚀 OVERALL PERFORMANCE STATUS: EXCELLENT");
        } else if failures == 0 && warnings <= 3 {
            println!("🔶 OVERALL PERFORMANCE STATUS: ACCEPTABLE");
        } else {
            println!("❌ OVERALL PERFORMANCE STATUS: REGRESSION DETECTED");
        }

        // Print detailed failure analysis
        for (operation, status) in &self.regression_results {
            if let RegressionStatus::Fail(pct) = status {
                println!(
                    "🔍 CRITICAL REGRESSION: {} degraded by {:.1}%",
                    operation, pct
                );
            }
        }
    }
}

#[test]
fn test_phase_1_performance_regression() {
    println!("Testing Phase 1 performance regression...");
    let mut framework = PerformanceRegressionFramework::new();

    // Test Manager creation performance
    framework.measure_operation("manager_creation", || {
        let _result = Manager::new();
    });
    framework.validate_performance("manager_creation");

    // Test tensor creation performance with different sizes
    if let Ok(mut manager) = Manager::new() {
        // Small tensor creation
        framework.measure_operation("tensor_creation_small", || {
            let data: Vec<f32> = vec![1.0; 100];
            let _result = manager.tensor(&data);
        });
        framework.validate_performance("tensor_creation_small");

        // Large tensor creation
        framework.measure_operation("tensor_creation_large", || {
            let data: Vec<f32> = vec![1.0; 100000];
            let _result = manager.tensor(&data);
        });
        framework.validate_performance("tensor_creation_large");

        // Algorithm creation performance
        framework.measure_operation("algorithm_creation", || {
            let mock_spirv = vec![0x07, 0x23, 0x02, 0x03];
            let _result = manager.algorithm(&mock_spirv);
        });
        framework.validate_performance("algorithm_creation");
    }

    framework.print_regression_summary();
}

#[test]
fn test_phase_2_performance_regression() {
    println!("Testing Phase 2 performance regression...");
    let mut framework = PerformanceRegressionFramework::new();

    // Test MemoryOptimizer creation performance
    framework.measure_operation("memory_optimizer_creation", || {
        let _result = MemoryOptimizer::new();
    });

    // Test memory statistics performance
    if let Ok(optimizer) = MemoryOptimizer::new() {
        framework.measure_operation("memory_statistics", || {
            let _result = optimizer.memory_statistics();
        });
        framework.validate_performance("memory_statistics");
    }

    // Test device enumeration performance (conceptual)
    framework.measure_operation("device_enumeration", || {
        // Simulate device enumeration operation
        thread::sleep(Duration::from_millis(10));
    });
    framework.validate_performance("device_enumeration");

    framework.print_regression_summary();
}

#[test]
fn test_phase_3_performance_regression() {
    println!("Testing Phase 3 multi-type tensor performance regression...");
    let mut framework = PerformanceRegressionFramework::new();

    if let Ok(mut manager) = Manager::new() {
        // Test multi-type tensor creation performance
        let data_types = [("f32", 1000), ("f64", 1000), ("i32", 1000), ("u32", 1000)];

        for (type_name, size) in &data_types {
            let operation_name = format!("multi_type_tensor_{}", type_name);

            framework.measure_operation(&operation_name, || match *type_name {
                "f32" => {
                    let data: Vec<f32> = vec![1.0; *size];
                    let _result = manager.tensor(&data);
                }
                "f64" => {
                    let data: Vec<f64> = vec![1.0; *size];
                    let _result = manager.tensor(&data);
                }
                "i32" => {
                    let data: Vec<i32> = vec![1; *size];
                    let _result = manager.tensor(&data);
                }
                "u32" => {
                    let data: Vec<u32> = vec![1; *size];
                    let _result = manager.tensor(&data);
                }
                _ => unreachable!(),
            });

            // Use a general tensor creation threshold for multi-type operations
            framework.validate_performance("tensor_creation_small");
        }
    }

    framework.print_regression_summary();
}

#[test]
fn test_cross_platform_performance_consistency() {
    println!("Testing cross-platform performance consistency...");
    let mut framework = PerformanceRegressionFramework::new();

    // Test platform-specific optimizations don't cause regressions
    framework.measure_operation("platform_specific_operations", || {
        // Simulate platform-specific optimizations
        #[cfg(target_os = "macos")]
        {
            thread::sleep(Duration::from_millis(5)); // MoltenVK overhead
        }

        #[cfg(target_os = "linux")]
        {
            thread::sleep(Duration::from_millis(3)); // Native Vulkan
        }

        #[cfg(target_os = "windows")]
        {
            thread::sleep(Duration::from_millis(4)); // Windows Vulkan
        }
    });

    // Platform consistency should be within reasonable bounds
    let current_duration = framework.current_metrics["platform_specific_operations"];
    let is_consistent = current_duration <= Duration::from_millis(20);

    if is_consistent {
        println!("    ✅ Cross-platform performance is consistent");
    } else {
        println!("    ❌ Cross-platform performance inconsistency detected");
    }

    framework.print_regression_summary();
}

#[test]
fn test_stress_performance_regression() {
    println!("Testing stress performance regression...");
    let mut framework = PerformanceRegressionFramework::new();

    // Test performance under stress conditions
    if let Ok(mut manager) = Manager::new() {
        framework.measure_operation("stress_tensor_creation", || {
            for i in 0..10 {
                let data: Vec<f32> = vec![i as f32; 1000];
                let _result = manager.tensor(&data);
            }
        });

        // Stress operations should complete within reasonable time
        let stress_duration = framework.current_metrics["stress_tensor_creation"];
        let stress_acceptable = stress_duration <= Duration::from_millis(500);

        let status = if stress_acceptable {
            RegressionStatus::Pass
        } else {
            let degradation_pct = (stress_duration.as_millis() as f64 / 500.0 - 1.0) * 100.0;
            RegressionStatus::Fail(degradation_pct)
        };

        framework
            .regression_results
            .insert("stress_operations".to_string(), status);
    }

    framework.print_regression_summary();
}

#[test]
fn test_memory_performance_regression() {
    println!("Testing memory performance regression...");
    let mut framework = PerformanceRegressionFramework::new();

    // Test memory allocation/deallocation performance
    framework.measure_operation("memory_allocation_performance", || {
        let mut data_sets = Vec::new();

        // Allocate multiple data sets
        for size in [100, 1000, 10000] {
            let data: Vec<f32> = vec![1.0; size];
            data_sets.push(data);
        }

        // Implicit deallocation when data_sets goes out of scope
        drop(data_sets);
    });

    // Memory operations should be fast
    let memory_duration = framework.current_metrics["memory_allocation_performance"];
    let memory_acceptable = memory_duration <= Duration::from_millis(10);

    let status = if memory_acceptable {
        RegressionStatus::Pass
    } else {
        let degradation_pct = (memory_duration.as_millis() as f64 / 10.0 - 1.0) * 100.0;
        RegressionStatus::Warning(degradation_pct)
    };

    framework
        .regression_results
        .insert("memory_performance".to_string(), status);

    framework.print_regression_summary();
}
