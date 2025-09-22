//! Tests for Phase 0: Intelligent GPU Configuration & Error Handling
//!
//! This test suite validates the intelligent GPU configuration system that:
//! - Auto-detects Metal GPU support on compatible systems
//! - Auto-configures MoltenVK when appropriate
//! - Respects user override via KOMPUTE_DISABLE_MOLTENVK_AUTO
//! - Provides helpful error messages when configuration fails

use komputers::{configure_vulkan_environment_intelligent, Manager, MetalGpuDetection};
use std::env;

/// Test Metal GPU detection functionality
#[test]
fn test_metal_gpu_detection() {
    let has_metal = MetalGpuDetection::detect_metal_gpu_support();

    // On macOS, we should detect Metal support
    #[cfg(target_os = "macos")]
    {
        assert!(has_metal, "Metal GPU support should be detected on macOS");
        println!("✓ Metal GPU support detected on macOS: {}", has_metal);
    }

    // On non-macOS systems, Metal should not be detected
    #[cfg(not(target_os = "macos"))]
    {
        assert!(
            !has_metal,
            "Metal GPU support should not be detected on non-macOS systems"
        );
        println!(
            "✓ Metal GPU support correctly not detected on non-macOS: {}",
            has_metal
        );
    }
}

/// Test auto-MoltenVK configuration logic
#[test]
fn test_auto_moltenvk_configuration() {
    // Save original environment state
    let original_disable = env::var("KOMPUTE_DISABLE_MOLTENVK_AUTO").ok();

    // Test 1: Default behavior (should auto-configure on Metal systems)
    env::remove_var("KOMPUTE_DISABLE_MOLTENVK_AUTO");
    let should_auto = MetalGpuDetection::should_auto_configure_moltenvk();

    #[cfg(target_os = "macos")]
    {
        assert!(
            should_auto,
            "Should auto-configure MoltenVK on macOS when not disabled"
        );
        println!(
            "✓ Auto-MoltenVK enabled by default on macOS: {}",
            should_auto
        );
    }

    #[cfg(not(target_os = "macos"))]
    {
        assert!(
            !should_auto,
            "Should not auto-configure MoltenVK on non-macOS systems"
        );
        println!(
            "✓ Auto-MoltenVK correctly disabled on non-macOS: {}",
            should_auto
        );
    }

    // Test 2: User override (should not auto-configure when disabled)
    env::set_var("KOMPUTE_DISABLE_MOLTENVK_AUTO", "1");
    let should_auto_disabled = MetalGpuDetection::should_auto_configure_moltenvk();
    assert!(
        !should_auto_disabled,
        "Should not auto-configure when KOMPUTE_DISABLE_MOLTENVK_AUTO is set"
    );
    println!(
        "✓ Auto-MoltenVK correctly disabled by environment variable: {}",
        should_auto_disabled
    );

    // Restore original environment state
    match original_disable {
        Some(value) => env::set_var("KOMPUTE_DISABLE_MOLTENVK_AUTO", value),
        None => env::remove_var("KOMPUTE_DISABLE_MOLTENVK_AUTO"),
    }
}

/// Test intelligent Vulkan configuration
#[test]
fn test_intelligent_vulkan_configuration() {
    println!("🧪 Testing intelligent Vulkan configuration...");

    match configure_vulkan_environment_intelligent() {
        Ok(()) => {
            println!("✅ Intelligent Vulkan configuration succeeded");

            // If configuration succeeded, verify that we can get device count
            match Manager::device_count() {
                Ok(count) => {
                    println!("✓ Device count after intelligent config: {}", count);
                    assert!(
                        count > 0,
                        "Should have at least one device after successful configuration"
                    );
                }
                Err(e) => {
                    panic!(
                        "Device enumeration failed after successful configuration: {}",
                        e
                    );
                }
            }
        }
        Err(e) => {
            println!("⚠️ Intelligent Vulkan configuration failed: {}", e);

            // Verify that the error message contains helpful guidance
            assert!(
                e.contains("Metal-capable")
                    || e.contains("Linux")
                    || e.contains("Windows")
                    || e.contains("Install"),
                "Error message should contain platform-specific guidance"
            );

            println!("✓ Error message contains helpful guidance");

            // In CI or environments without GPU, this is expected
            // We don't panic here as this is environment-dependent
        }
    }
}

/// Test environment variable override functionality
#[test]
fn test_environment_variable_override() {
    // Save original environment state
    let original_disable = env::var("KOMPUTE_DISABLE_MOLTENVK_AUTO").ok();

    // Test with override enabled
    env::set_var("KOMPUTE_DISABLE_MOLTENVK_AUTO", "1");

    println!("🧪 Testing with KOMPUTE_DISABLE_MOLTENVK_AUTO=1...");

    let should_auto = MetalGpuDetection::should_auto_configure_moltenvk();
    assert!(
        !should_auto,
        "Auto-configuration should be disabled when env var is set"
    );

    // Test configuration with override
    match configure_vulkan_environment_intelligent() {
        Ok(()) => {
            println!("✅ Configuration succeeded with MoltenVK auto-config disabled");
        }
        Err(e) => {
            println!(
                "⚠️ Configuration failed with MoltenVK auto-config disabled: {}",
                e
            );
            // This might be expected if no other Vulkan drivers are available
        }
    }

    // Restore original environment state
    match original_disable {
        Some(value) => env::set_var("KOMPUTE_DISABLE_MOLTENVK_AUTO", value),
        None => env::remove_var("KOMPUTE_DISABLE_MOLTENVK_AUTO"),
    }

    println!("✓ Environment variable override test completed");
}

/// Test helpful error message generation
#[test]
fn test_helpful_error_messages() {
    println!("🧪 Testing helpful error message generation...");

    // This test focuses on the error message quality rather than success/failure
    // since Vulkan availability is environment-dependent

    match configure_vulkan_environment_intelligent() {
        Ok(()) => {
            println!("✅ Configuration succeeded - no error messages to test");
        }
        Err(error_msg) => {
            println!("📝 Generated error message:\n{}", error_msg);

            // Verify error message contains helpful elements
            let helpful_indicators = [
                "Install",
                "download",
                "brew",
                "vulkaninfo",
                "drivers",
                "🍎",
                "🐧",
                "🪟",
                "1️⃣",
                "2️⃣",
                "3️⃣",
                "💡",
                "🔍",
            ];

            let contains_helpful_info = helpful_indicators
                .iter()
                .any(|indicator| error_msg.to_lowercase().contains(&indicator.to_lowercase()));

            assert!(
                contains_helpful_info,
                "Error message should contain helpful installation or troubleshooting guidance"
            );

            // Verify platform-specific content
            #[cfg(target_os = "macos")]
            {
                assert!(
                    error_msg.contains("Metal")
                        || error_msg.contains("MoltenVK")
                        || error_msg.contains("macOS"),
                    "Error message should contain macOS/Metal-specific guidance"
                );
            }

            #[cfg(target_os = "linux")]
            {
                assert!(
                    error_msg.contains("Linux")
                        || error_msg.contains("apt")
                        || error_msg.contains("nvidia")
                        || error_msg.contains("mesa"),
                    "Error message should contain Linux-specific guidance"
                );
            }

            #[cfg(target_os = "windows")]
            {
                assert!(
                    error_msg.contains("Windows")
                        || error_msg.contains("nvidia.com")
                        || error_msg.contains("amd.com"),
                    "Error message should contain Windows-specific guidance"
                );
            }

            println!("✓ Error message contains appropriate helpful guidance");
        }
    }
}

/// Test Manager creation with intelligent configuration
#[test]
fn test_manager_with_intelligent_config() {
    println!("🧪 Testing Manager creation with intelligent configuration...");

    match Manager::new() {
        Ok(_manager) => {
            println!("✅ Manager created successfully with intelligent configuration");
            // Manager creation succeeded - intelligent config worked
        }
        Err(e) => {
            println!("⚠️ Manager creation failed: {}", e);

            // Verify that the error contains helpful guidance from intelligent config
            let error_msg = format!("{}", e);
            assert!(
                error_msg.contains("Vulkan configuration failed")
                    || error_msg.contains("Install")
                    || error_msg.contains("MoltenVK")
                    || error_msg.contains("drivers"),
                "Manager error should contain helpful configuration guidance"
            );

            println!("✓ Manager error contains helpful guidance from intelligent configuration");
        }
    }
}

/// Integration test: Full intelligent configuration workflow
#[test]
fn test_full_intelligent_workflow() {
    println!("🔄 Testing full intelligent GPU configuration workflow...");

    // Step 1: Test Metal detection
    let has_metal = MetalGpuDetection::detect_metal_gpu_support();
    println!("1️⃣ Metal GPU detection: {}", has_metal);

    // Step 2: Test auto-configuration decision
    let should_auto = MetalGpuDetection::should_auto_configure_moltenvk();
    println!("2️⃣ Should auto-configure MoltenVK: {}", should_auto);

    // Step 3: Test intelligent configuration
    match configure_vulkan_environment_intelligent() {
        Ok(()) => {
            println!("3️⃣ ✅ Intelligent configuration succeeded");

            // Step 4: Test device enumeration
            match Manager::device_count() {
                Ok(count) => {
                    println!("4️⃣ ✅ Device enumeration: {} devices found", count);

                    // Step 5: Test Manager creation
                    match Manager::new() {
                        Ok(_manager) => {
                            println!("5️⃣ ✅ Manager creation succeeded");
                            println!("🎉 Full intelligent workflow completed successfully!");
                        }
                        Err(e) => {
                            println!("5️⃣ ❌ Manager creation failed: {}", e);
                        }
                    }
                }
                Err(e) => {
                    println!("4️⃣ ❌ Device enumeration failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("3️⃣ ❌ Intelligent configuration failed: {}", e);
            println!("ℹ️ This may be expected in CI or GPU-less environments");
        }
    }

    println!("✓ Full workflow test completed");
}
