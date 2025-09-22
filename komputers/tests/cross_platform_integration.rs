//! Cross-platform reliability and integration tests.
//!
//! These tests verify that Komputers works reliably across different platforms,
//! handles environment validation, and provides appropriate error recovery.

use komputers::{EnvironmentValidator, GpuSupport, Manager, PlatformInfo, RuntimeUtils};

#[test]
fn test_platform_detection_comprehensive() {
    let platform = PlatformInfo::detect();

    // Basic platform information should always be available
    assert!(!platform.os.is_empty(), "OS detection should work");
    assert!(
        !platform.arch.is_empty(),
        "Architecture detection should work"
    );

    // Platform should be one of the supported types
    assert!(
        matches!(platform.os.as_str(), "macos" | "linux" | "windows"),
        "Platform should be one of the major supported OS types, got: {}",
        platform.os
    );

    // GPU support should be classified
    println!("Platform: {} ({})", platform.os, platform.arch);
    println!("GPU Support: {:?}", platform.gpu_support);
    println!(
        "Vulkan SDK: {}",
        if platform.vulkan_sdk_available {
            "Available"
        } else {
            "Not Available"
        }
    );
    println!(
        "Vulkan Runtime: {}",
        if platform.vulkan_runtime_available {
            "Available"
        } else {
            "Not Available"
        }
    );

    // Print platform notes for debugging
    for note in &platform.notes {
        println!("Note: {}", note);
    }
}

#[test]
fn test_environment_validation_comprehensive() {
    match EnvironmentValidator::validate() {
        Ok(report) => {
            println!("Environment Validation Report:");
            println!("{}", report.summary());

            // Report should contain platform information
            assert!(!report.platform.os.is_empty());

            // Validation report should be self-consistent
            if report.is_valid() {
                println!("✅ Environment is ready for GPU computing!");
                assert!(
                    report.issues.is_empty(),
                    "Valid environment should have no critical issues"
                );
            } else {
                println!("❌ Environment has issues that need attention");
                if !report.recommendations.is_empty() {
                    println!("Recommendations provided for fixing issues");
                }
            }
        }
        Err(e) => {
            println!("⚠️ Environment validation failed: {}", e);
            // This is not necessarily a test failure - depends on the environment
        }
    }
}

#[test]
fn test_platform_specific_optimizations() {
    let platform = PlatformInfo::detect();

    // Test that platform optimizations can be applied
    match RuntimeUtils::optimize_for_platform() {
        Ok(()) => {
            println!(
                "✅ Platform optimizations applied successfully for {}",
                platform.os
            );
        }
        Err(e) => {
            println!("⚠️ Platform optimization failed: {}", e);
            // This might be expected in some environments
        }
    }

    // Test memory limits are reasonable
    let (min_mem, max_mem) = RuntimeUtils::memory_limits();
    assert!(min_mem > 0, "Minimum memory limit should be positive");
    assert!(
        max_mem > min_mem,
        "Maximum memory should be greater than minimum"
    );
    assert!(
        min_mem >= 256 * 1024 * 1024,
        "Minimum memory should be at least 256MB"
    );

    println!(
        "Memory limits for {}: {} - {} bytes",
        platform.os, min_mem, max_mem
    );
}

#[test]
fn test_platform_troubleshooting_guidance() {
    let platform = PlatformInfo::detect();
    let guide = platform.troubleshooting_guide();

    // Should always provide some guidance
    assert!(
        !guide.is_empty(),
        "Troubleshooting guide should not be empty"
    );

    // Guidance should be platform-appropriate
    match platform.os.as_str() {
        "macos" => {
            assert!(
                guide
                    .iter()
                    .any(|g| g.contains("MoltenVK") || g.contains("macOS")),
                "macOS guidance should mention MoltenVK"
            );
        }
        "linux" => {
            assert!(
                guide.iter().any(|g| g.contains("apt")
                    || g.contains("vulkan-tools")
                    || g.contains("drivers")),
                "Linux guidance should mention package management or drivers"
            );
        }
        "windows" => {
            assert!(
                guide
                    .iter()
                    .any(|g| g.contains("drivers") || g.contains("SDK")),
                "Windows guidance should mention drivers or SDK"
            );
        }
        _ => {}
    }

    println!("Troubleshooting guidance for {}:", platform.os);
    for (i, item) in guide.iter().enumerate() {
        println!("  {}. {}", i + 1, item);
    }
}

#[test]
fn test_environment_configuration_recommendations() {
    let platform = PlatformInfo::detect();
    let env_vars = platform.recommended_environment();

    println!("Recommended environment variables for {}:", platform.os);
    for (key, value) in &env_vars {
        println!("  {}={}", key, value);
    }

    // Platform-specific validation
    match platform.os.as_str() {
        "macos" => {
            if platform.vulkan_sdk_available {
                assert!(
                    env_vars.iter().any(|(k, _)| k == "VK_ICD_FILENAMES"),
                    "macOS with Vulkan SDK should set VK_ICD_FILENAMES"
                );
            }
        }
        "linux" | "windows" => {
            // These platforms may or may not need explicit environment variables
            // depending on system configuration
        }
        _ => {}
    }
}

#[test]
fn test_cross_platform_device_detection() {
    let platform = PlatformInfo::detect();

    // Try to detect devices using the Manager
    match Manager::device_count() {
        Ok(count) => {
            println!("✅ Found {} GPU device(s) on {}", count, platform.os);
            assert!(count > 0, "Should find at least one device if test passes");

            // Test device enumeration on each platform
            for device_id in 0..count.min(3) {
                // Test up to 3 devices
                match Manager::with_device(device_id) {
                    Ok(_manager) => {
                        println!("  ✅ Device {} initialized successfully", device_id);
                    }
                    Err(e) => {
                        println!("  ⚠️ Device {} initialization failed: {}", device_id, e);
                        // Print recovery suggestions
                        let suggestions = e.recovery_suggestions();
                        for suggestion in suggestions {
                            println!("    💡 {}", suggestion);
                        }
                    }
                }
            }
        }
        Err(e) => {
            println!("❌ Device detection failed on {}: {}", platform.os, e);

            // Print detailed error analysis
            println!("Error analysis:");
            println!("  Recoverable: {}", e.is_recoverable());
            println!(
                "  Requires environment setup: {}",
                e.requires_environment_setup()
            );

            println!("Recovery suggestions:");
            for suggestion in e.recovery_suggestions() {
                println!("  💡 {}", suggestion);
            }

            // This test can fail in environments without GPU support
            // The important thing is that we get meaningful error information
        }
    }
}

#[test]
fn test_gpu_support_classification() {
    let platform = PlatformInfo::detect();

    // Verify GPU support classification makes sense
    match platform.os.as_str() {
        "macos" => {
            assert_eq!(
                platform.gpu_support,
                GpuSupport::Translated,
                "macOS should use translated GPU support (MoltenVK)"
            );
        }
        "linux" | "windows" => {
            assert_eq!(
                platform.gpu_support,
                GpuSupport::Native,
                "Linux and Windows should have native GPU support"
            );
        }
        _ => {
            // Unknown platforms might have unknown support
            assert!(
                matches!(platform.gpu_support, GpuSupport::Unknown),
                "Unknown platforms should have unknown GPU support"
            );
        }
    }
}

#[test]
fn test_platform_requirements_checking() {
    let platform = PlatformInfo::detect();

    // Test requirement checking logic
    let meets_requirements = platform.meets_requirements();

    println!("Platform meets requirements: {}", meets_requirements);
    println!(
        "Vulkan runtime available: {}",
        platform.vulkan_runtime_available
    );
    println!("GPU support level: {:?}", platform.gpu_support);

    // If requirements are met, both Vulkan runtime and appropriate GPU support should be available
    if meets_requirements {
        assert!(
            platform.vulkan_runtime_available,
            "Requirements met implies Vulkan runtime available"
        );
        assert!(
            matches!(
                platform.gpu_support,
                GpuSupport::Native | GpuSupport::Translated
            ),
            "Requirements met implies native or translated GPU support"
        );
    }

    // If requirements are not met, troubleshooting should provide guidance
    if !meets_requirements {
        let guide = platform.troubleshooting_guide();
        assert!(
            !guide.is_empty(),
            "Unmet requirements should provide troubleshooting guidance"
        );
    }
}

#[test]
fn test_validation_report_consistency() {
    match EnvironmentValidator::validate() {
        Ok(report) => {
            // Test that validation report is internally consistent

            // If valid, should have no critical issues
            if report.is_valid() {
                assert!(
                    report.issues.is_empty(),
                    "Valid report should have no issues"
                );
            }

            // If invalid, should have issues or requirements not met
            if !report.is_valid() {
                assert!(
                    !report.issues.is_empty() || !report.platform.meets_requirements(),
                    "Invalid report should have issues or unmet requirements"
                );
            }

            // All reports should have platform information
            assert!(!report.platform.os.is_empty());
            assert!(!report.platform.arch.is_empty());

            // Reports with issues should provide recommendations
            if !report.issues.is_empty() || !report.warnings.is_empty() {
                assert!(
                    !report.recommendations.is_empty(),
                    "Reports with issues/warnings should provide recommendations"
                );
            }
        }
        Err(e) => {
            println!("Environment validation error: {}", e);
            // Error in validation itself is not necessarily a test failure
        }
    }
}

#[test]
fn test_platform_summary_formatting() {
    let platform = PlatformInfo::detect();
    let summary = platform.summary();

    // Summary should contain key information
    assert!(
        summary.contains(&platform.os),
        "Summary should contain OS name"
    );
    assert!(
        summary.contains(&platform.arch),
        "Summary should contain architecture"
    );
    assert!(summary.contains("Vulkan"), "Summary should mention Vulkan");
    assert!(
        summary.contains("GPU Support"),
        "Summary should mention GPU support"
    );
    assert!(
        summary.contains("Requirements Met"),
        "Summary should indicate if requirements are met"
    );

    println!("Platform Summary:");
    println!("{}", summary);
}

#[test]
fn test_error_recovery_integration() {
    use komputers::KomputeError;

    // Test different error types and their recovery suggestions
    let test_errors = vec![
        KomputeError::vulkan("Test Vulkan error"),
        KomputeError::hardware("Test hardware error"),
        KomputeError::device("Test device error"),
        KomputeError::memory("Test memory error"),
        KomputeError::platform("Test platform error"),
        KomputeError::environment("Test environment error"),
    ];

    for error in test_errors {
        println!("Error: {}", error);
        println!("  Recoverable: {}", error.is_recoverable());
        println!(
            "  Requires env setup: {}",
            error.requires_environment_setup()
        );

        let suggestions = error.recovery_suggestions();
        assert!(
            !suggestions.is_empty(),
            "All errors should provide recovery suggestions"
        );

        for suggestion in suggestions {
            println!("  💡 {}", suggestion);
        }
        println!();
    }
}
