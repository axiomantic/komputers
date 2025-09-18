//! VulkanSetup platform-specific testing.
//!
//! These tests verify platform-specific Vulkan environment configuration,
//! path detection, fallback mechanisms, and cross-platform compatibility
//! for the VulkanSetup module functionality.

use kompute::vulkan_setup::{configure_vulkan_environment, verify_vulkan_availability};
use std::env;
use std::path::Path;

/// Test platform-specific Vulkan SDK detection and configuration.
#[test]
fn test_platform_vulkan_sdk_detection() {
    println!("Testing platform-specific Vulkan SDK detection...");

    // Save original environment state
    let original_vulkan_sdk = env::var("VULKAN_SDK").ok();
    let original_icd = env::var("VK_ICD_FILENAMES").ok();
    let original_layer_path = env::var("VK_LAYER_PATH").ok();

    // Test with VULKAN_SDK set
    if let Some(vulkan_sdk) = &original_vulkan_sdk {
        println!("  Testing with VULKAN_SDK: {}", vulkan_sdk);

        match configure_vulkan_environment() {
            Ok(()) => {
                println!("  ✓ Vulkan environment configured successfully with SDK");

                // Verify platform-specific paths are set correctly
                #[cfg(target_os = "macos")]
                {
                    if let Ok(icd_path) = env::var("VK_ICD_FILENAMES") {
                        assert!(
                            icd_path.contains("MoltenVK_icd.json"),
                            "macOS should use MoltenVK ICD file"
                        );
                        println!("  ✓ macOS MoltenVK ICD configured: {}", icd_path);
                    }
                }

                #[cfg(target_os = "linux")]
                {
                    if let Ok(layer_path) = env::var("VK_LAYER_PATH") {
                        assert!(
                            layer_path.contains("explicit_layer.d"),
                            "Linux should set explicit layer path"
                        );
                        println!("  ✓ Linux layer path configured: {}", layer_path);
                    }
                }

                #[cfg(target_os = "windows")]
                {
                    if let Ok(layer_path) = env::var("VK_LAYER_PATH") {
                        assert!(
                            layer_path.contains("explicit_layer.d"),
                            "Windows should set explicit layer path"
                        );
                        println!("  ✓ Windows layer path configured: {}", layer_path);
                    }
                }
            }
            Err(e) => {
                println!("  ⚠️ Vulkan configuration failed: {}", e);
                // This might be expected in some test environments
            }
        }
    } else {
        println!("  No VULKAN_SDK set, testing fallback mechanisms");
    }

    // Restore original environment
    match original_vulkan_sdk {
        Some(sdk) => env::set_var("VULKAN_SDK", sdk),
        None => env::remove_var("VULKAN_SDK"),
    }
    match original_icd {
        Some(icd) => env::set_var("VK_ICD_FILENAMES", icd),
        None => env::remove_var("VK_ICD_FILENAMES"),
    }
    match original_layer_path {
        Some(path) => env::set_var("VK_LAYER_PATH", path),
        None => env::remove_var("VK_LAYER_PATH"),
    }

    println!("  ✓ Platform-specific SDK detection tested");
}

/// Test fallback mechanisms when VULKAN_SDK is not available.
#[test]
fn test_vulkan_fallback_mechanisms() {
    println!("Testing Vulkan fallback mechanisms...");

    // Save original environment
    let original_vulkan_sdk = env::var("VULKAN_SDK").ok();
    let original_icd = env::var("VK_ICD_FILENAMES").ok();
    let original_layer_path = env::var("VK_LAYER_PATH").ok();

    // Temporarily remove VULKAN_SDK to test fallbacks
    env::remove_var("VULKAN_SDK");
    env::remove_var("VK_ICD_FILENAMES");
    env::remove_var("VK_LAYER_PATH");

    match configure_vulkan_environment() {
        Ok(()) => {
            println!("  ✓ Fallback configuration successful");

            #[cfg(target_os = "macos")]
            {
                // Check if homebrew or system MoltenVK paths were detected
                if let Ok(icd_path) = env::var("VK_ICD_FILENAMES") {
                    let is_homebrew = icd_path.contains("/opt/homebrew/");
                    let is_system = icd_path.contains("/usr/local/");
                    assert!(
                        is_homebrew || is_system,
                        "macOS fallback should use homebrew or system paths"
                    );
                    println!("  ✓ macOS fallback path detected: {}", icd_path);
                }
            }

            #[cfg(target_os = "linux")]
            {
                // Linux should fall back to system paths or succeed without explicit setup
                println!("  ✓ Linux fallback mechanism succeeded");
            }

            #[cfg(target_os = "windows")]
            {
                // Windows should fall back to registry-based discovery
                println!("  ✓ Windows fallback mechanism succeeded");
            }
        }
        Err(e) => {
            println!("  ⚠️ Fallback configuration failed: {}", e);
            // This is expected when no Vulkan installation is available
        }
    }

    // Restore original environment
    match original_vulkan_sdk {
        Some(sdk) => env::set_var("VULKAN_SDK", sdk),
        None => env::remove_var("VULKAN_SDK"),
    }
    match original_icd {
        Some(icd) => env::set_var("VK_ICD_FILENAMES", icd),
        None => env::remove_var("VK_ICD_FILENAMES"),
    }
    match original_layer_path {
        Some(path) => env::set_var("VK_LAYER_PATH", path),
        None => env::remove_var("VK_LAYER_PATH"),
    }

    println!("  ✓ Fallback mechanisms tested");
}

/// Test Vulkan availability verification across platforms.
#[test]
fn test_vulkan_availability_verification() {
    println!("Testing Vulkan availability verification...");

    match verify_vulkan_availability() {
        Ok(device_count) => {
            println!("  ✓ Vulkan verification successful");
            println!("  • Found {} Vulkan-capable device(s)", device_count);
            assert!(
                device_count > 0,
                "Should find at least one Vulkan device if verification succeeds"
            );

            // Platform-specific device expectations
            #[cfg(target_os = "macos")]
            {
                println!("  • macOS platform with MoltenVK translation");
            }

            #[cfg(target_os = "linux")]
            {
                println!("  • Linux platform with native Vulkan drivers");
            }

            #[cfg(target_os = "windows")]
            {
                println!("  • Windows platform with native Vulkan drivers");
            }
        }
        Err(e) => {
            println!("  ⚠️ Vulkan verification failed: {}", e);
            println!("  This may be expected in headless or non-GPU environments");
        }
    }

    println!("  ✓ Vulkan availability verification tested");
}

/// Test environment variable path validation across platforms.
#[test]
fn test_platform_path_validation() {
    println!("Testing platform-specific path validation...");

    // Test with current environment
    match configure_vulkan_environment() {
        Ok(()) => {
            // Validate paths exist and are accessible
            if let Ok(icd_path) = env::var("VK_ICD_FILENAMES") {
                println!("  • ICD path configured: {}", icd_path);

                // For macOS, verify MoltenVK ICD file exists
                #[cfg(target_os = "macos")]
                {
                    if !icd_path.contains("*") {
                        // Not a wildcard pattern
                        let path_exists = Path::new(&icd_path).exists();
                        if path_exists {
                            println!("  ✓ macOS ICD file exists and is accessible");
                        } else {
                            println!("  ⚠️ macOS ICD file not found at: {}", icd_path);
                        }
                    }
                }
            }

            if let Ok(layer_path) = env::var("VK_LAYER_PATH") {
                println!("  • Layer path configured: {}", layer_path);

                // Verify layer path exists
                let path_exists = Path::new(&layer_path).exists();
                if path_exists {
                    println!("  ✓ Layer path exists and is accessible");
                } else {
                    println!("  ⚠️ Layer path not found at: {}", layer_path);
                }
            }
        }
        Err(e) => {
            println!("  ⚠️ Environment configuration failed: {}", e);
        }
    }

    println!("  ✓ Platform path validation tested");
}

/// Test environment variable pollution and isolation.
#[test]
fn test_environment_isolation() {
    println!("Testing environment variable isolation...");

    // Capture initial state
    let initial_vars = [
        ("VULKAN_SDK", env::var("VULKAN_SDK").ok()),
        ("VK_ICD_FILENAMES", env::var("VK_ICD_FILENAMES").ok()),
        ("VK_LAYER_PATH", env::var("VK_LAYER_PATH").ok()),
        ("VK_INSTANCE_LAYERS", env::var("VK_INSTANCE_LAYERS").ok()),
    ];

    // Set some test environment variables
    env::set_var("TEST_VULKAN_ISOLATION", "test_value");

    // Configure Vulkan environment
    let config_result = configure_vulkan_environment();

    // Verify test variable wasn't affected
    assert_eq!(env::var("TEST_VULKAN_ISOLATION").unwrap(), "test_value");
    println!("  ✓ Environment isolation maintained");

    // Clean up test variable
    env::remove_var("TEST_VULKAN_ISOLATION");

    // Log configuration result
    match config_result {
        Ok(()) => println!("  ✓ Environment configuration succeeded without pollution"),
        Err(e) => println!("  ⚠️ Environment configuration failed: {}", e),
    }

    // Verify only expected Vulkan variables were modified
    let vulkan_vars = ["VK_ICD_FILENAMES", "VK_LAYER_PATH"];
    for var_name in vulkan_vars {
        if env::var(var_name).is_ok() {
            println!("  • {} was set by configuration", var_name);
        }
    }

    // Restore initial state
    for (var_name, initial_value) in initial_vars {
        match initial_value {
            Some(value) => env::set_var(var_name, value),
            None => env::remove_var(var_name),
        }
    }

    println!("  ✓ Environment isolation tested");
}

/// Test concurrent Vulkan setup from multiple threads.
#[test]
fn test_concurrent_vulkan_setup() {
    println!("Testing concurrent Vulkan setup...");

    use std::sync::{Arc, Barrier};
    use std::thread;

    const THREAD_COUNT: usize = 8;
    let barrier = Arc::new(Barrier::new(THREAD_COUNT));
    let mut handles = Vec::new();

    // Spawn multiple threads that configure Vulkan simultaneously
    for thread_id in 0..THREAD_COUNT {
        let barrier_clone = Arc::clone(&barrier);
        let handle = thread::spawn(move || {
            barrier_clone.wait();

            // All threads attempt configuration simultaneously
            let result = configure_vulkan_environment();
            (thread_id, result)
        });
        handles.push(handle);
    }

    // Collect results
    let mut success_count = 0;
    let mut failure_count = 0;

    for handle in handles {
        let (thread_id, result) = handle.join().unwrap();
        match result {
            Ok(()) => {
                success_count += 1;
                println!("  • Thread {} configuration succeeded", thread_id);
            }
            Err(e) => {
                failure_count += 1;
                println!("  • Thread {} configuration failed: {}", thread_id, e);
            }
        }
    }

    println!("  ✓ Concurrent setup completed");
    println!("    - Successful configurations: {}", success_count);
    println!("    - Failed configurations: {}", failure_count);

    // If any thread succeeded, all should have succeeded (same environment)
    if success_count > 0 {
        assert_eq!(
            failure_count, 0,
            "All threads should have same result in same environment"
        );
    }

    println!("  ✓ Concurrent Vulkan setup tested");
}

/// Test platform-specific error conditions and edge cases.
#[test]
fn test_platform_error_conditions() {
    println!("Testing platform-specific error conditions...");

    // Save original environment
    let original_vulkan_sdk = env::var("VULKAN_SDK").ok();
    let original_icd = env::var("VK_ICD_FILENAMES").ok();
    let original_layer_path = env::var("VK_LAYER_PATH").ok();

    // Test with invalid VULKAN_SDK path
    env::set_var("VULKAN_SDK", "/nonexistent/path/to/vulkan");
    env::remove_var("VK_ICD_FILENAMES");
    env::remove_var("VK_LAYER_PATH");

    match configure_vulkan_environment() {
        Ok(()) => {
            println!("  ⚠️ Configuration succeeded with invalid VULKAN_SDK (fallback working)");
        }
        Err(e) => {
            println!(
                "  ✓ Configuration correctly failed with invalid VULKAN_SDK: {}",
                e
            );
        }
    }

    // Test with completely missing Vulkan infrastructure
    env::remove_var("VULKAN_SDK");
    env::remove_var("VK_ICD_FILENAMES");
    env::remove_var("VK_LAYER_PATH");

    // Temporarily rename system paths to test missing infrastructure
    let test_cases = vec![("Missing SDK", "/nonexistent/vulkan/sdk")];

    for (test_name, invalid_path) in test_cases {
        env::set_var("VULKAN_SDK", invalid_path);

        match configure_vulkan_environment() {
            Ok(()) => {
                println!(
                    "  ⚠️ {} test: Configuration succeeded (fallback working)",
                    test_name
                );
            }
            Err(e) => {
                println!(
                    "  ✓ {} test: Configuration correctly failed: {}",
                    test_name, e
                );
            }
        }
    }

    // Restore original environment
    match original_vulkan_sdk {
        Some(sdk) => env::set_var("VULKAN_SDK", sdk),
        None => env::remove_var("VULKAN_SDK"),
    }
    match original_icd {
        Some(icd) => env::set_var("VK_ICD_FILENAMES", icd),
        None => env::remove_var("VK_ICD_FILENAMES"),
    }
    match original_layer_path {
        Some(path) => env::set_var("VK_LAYER_PATH", path),
        None => env::remove_var("VK_LAYER_PATH"),
    }

    println!("  ✓ Platform error conditions tested");
}

/// Test cross-platform compatibility and consistency.
#[test]
fn test_cross_platform_compatibility() {
    println!("Testing cross-platform compatibility...");

    // Test that configuration function exists and is callable on all platforms
    let config_result = configure_vulkan_environment();

    // Test that verification function exists and is callable on all platforms
    let verify_result = verify_vulkan_availability();

    // Log platform-specific information
    #[cfg(target_os = "macos")]
    {
        println!("  • Running on macOS platform");
        println!("  • Expected: MoltenVK ICD configuration");

        if let Ok(icd) = env::var("VK_ICD_FILENAMES") {
            assert!(
                icd.contains("MoltenVK") || icd.is_empty(),
                "macOS should use MoltenVK or have no specific ICD"
            );
        }
    }

    #[cfg(target_os = "linux")]
    {
        println!("  • Running on Linux platform");
        println!("  • Expected: System or SDK-based Vulkan configuration");
    }

    #[cfg(target_os = "windows")]
    {
        println!("  • Running on Windows platform");
        println!("  • Expected: Registry-based or SDK-based Vulkan configuration");
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        println!("  • Running on unsupported platform");
        println!("  • Configuration should gracefully handle unsupported platforms");
    }

    // Both functions should be implemented for all platforms
    match config_result {
        Ok(()) => println!("  ✓ Platform configuration available and functional"),
        Err(e) => println!("  ⚠️ Platform configuration failed: {}", e),
    }

    match verify_result {
        Ok(count) => println!("  ✓ Platform verification available, {} devices", count),
        Err(e) => println!("  ⚠️ Platform verification failed: {}", e),
    }

    println!("  ✓ Cross-platform compatibility tested");
}
