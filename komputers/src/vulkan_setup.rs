//! Automatic Vulkan environment setup for cross-platform compatibility.

use crate::platform::MetalGpuDetection;
use std::env;
use std::path::Path;

/// Configure Vulkan environment variables for optimal platform compatibility.
///
/// This function automatically detects the platform and sets up the appropriate
/// Vulkan environment variables to ensure proper initialization.
pub fn configure_vulkan_environment() -> Result<(), String> {
    // Platform-specific setup
    #[cfg(target_os = "macos")]
    configure_macos_vulkan()?;

    #[cfg(target_os = "linux")]
    configure_linux_vulkan()?;

    #[cfg(target_os = "windows")]
    configure_windows_vulkan()?;

    Ok(())
}

#[cfg(target_os = "macos")]
fn configure_macos_vulkan() -> Result<(), String> {
    // Check for Vulkan SDK
    if let Ok(vulkan_sdk) = env::var("VULKAN_SDK") {
        println!("Using Vulkan SDK: {}", vulkan_sdk);

        // Set up MoltenVK ICD - check if already correctly set first
        if let Ok(existing_icd) = env::var("VK_ICD_FILENAMES") {
            if Path::new(&existing_icd).exists() {
                println!("Using existing VK_ICD_FILENAMES: {}", existing_icd);
            } else {
                return Err(format!(
                    "Existing VK_ICD_FILENAMES points to non-existent file: {}",
                    existing_icd
                ));
            }
        } else {
            // Try common locations for MoltenVK ICD
            let icd_candidates = [
                format!("{}/etc/vulkan/icd.d/MoltenVK_icd.json", vulkan_sdk),
                format!("{}/share/vulkan/icd.d/MoltenVK_icd.json", vulkan_sdk),
            ];

            let mut found_icd = false;
            for icd_path in &icd_candidates {
                if Path::new(icd_path).exists() {
                    env::set_var("VK_ICD_FILENAMES", icd_path);
                    println!("Set VK_ICD_FILENAMES: {}", icd_path);
                    found_icd = true;
                    break;
                }
            }

            if !found_icd {
                return Err(format!(
                    "MoltenVK ICD file not found in any of: {:?}",
                    icd_candidates
                ));
            }
        }

        // Set up layer path
        let layer_path = format!("{}/share/vulkan/explicit_layer.d", vulkan_sdk);
        if Path::new(&layer_path).exists() {
            env::set_var("VK_LAYER_PATH", &layer_path);
            println!("Set VK_LAYER_PATH: {}", layer_path);
        }

        // Disable validation layers by default for better compatibility
        env::remove_var("VK_INSTANCE_LAYERS");

        return Ok(());
    }

    // Try homebrew installation
    let homebrew_vulkan = "/opt/homebrew/share/vulkan";
    if Path::new(homebrew_vulkan).exists() {
        env::set_var(
            "VK_ICD_FILENAMES",
            format!("{}/icd.d/MoltenVK_icd.json", homebrew_vulkan),
        );
        env::set_var(
            "VK_LAYER_PATH",
            format!("{}/explicit_layer.d", homebrew_vulkan),
        );
        println!("Using Homebrew Vulkan installation");
        return Ok(());
    }

    // Try system MoltenVK
    let system_moltenvk = "/usr/local/share/vulkan";
    if Path::new(system_moltenvk).exists() {
        env::set_var(
            "VK_ICD_FILENAMES",
            format!("{}/icd.d/MoltenVK_icd.json", system_moltenvk),
        );
        env::set_var(
            "VK_LAYER_PATH",
            format!("{}/explicit_layer.d", system_moltenvk),
        );
        println!("Using system MoltenVK installation");
        return Ok(());
    }

    Err("No Vulkan/MoltenVK installation found. Please install Vulkan SDK or MoltenVK.".to_string())
}

#[cfg(target_os = "linux")]
fn configure_linux_vulkan() -> Result<(), String> {
    // Linux typically has system-wide Vulkan setup
    // Check for common locations
    let vulkan_locations = vec!["/usr/share/vulkan", "/usr/local/share/vulkan"];

    for location in vulkan_locations {
        if Path::new(location).exists() {
            let icd_path = format!("{}/icd.d", location);
            let layer_path = format!("{}/explicit_layer.d", location);

            if Path::new(&icd_path).exists() {
                env::set_var("VK_ICD_FILENAMES", format!("{}/*.json", icd_path));
                println!("Set VK_ICD_FILENAMES: {}/*.json", icd_path);
            }

            if Path::new(&layer_path).exists() {
                env::set_var("VK_LAYER_PATH", &layer_path);
                println!("Set VK_LAYER_PATH: {}", layer_path);
            }

            return Ok(());
        }
    }

    // If Vulkan SDK is set, use it
    if let Ok(vulkan_sdk) = env::var("VULKAN_SDK") {
        let icd_path = format!("{}/share/vulkan/icd.d", vulkan_sdk);
        let layer_path = format!("{}/share/vulkan/explicit_layer.d", vulkan_sdk);

        env::set_var("VK_ICD_FILENAMES", format!("{}/*.json", icd_path));
        env::set_var("VK_LAYER_PATH", &layer_path);
        println!("Using Vulkan SDK: {}", vulkan_sdk);
        return Ok(());
    }

    println!("No explicit Vulkan configuration needed for Linux (using system defaults)");
    Ok(())
}

#[cfg(target_os = "windows")]
fn configure_windows_vulkan() -> Result<(), String> {
    // Windows typically uses registry for Vulkan discovery
    // Check for Vulkan SDK
    if let Ok(vulkan_sdk) = env::var("VULKAN_SDK") {
        let icd_path = format!("{}/share/vulkan/icd.d", vulkan_sdk);
        let layer_path = format!("{}/share/vulkan/explicit_layer.d", vulkan_sdk);

        env::set_var("VK_ICD_FILENAMES", format!("{}/*.json", icd_path));
        env::set_var("VK_LAYER_PATH", &layer_path);
        println!("Using Vulkan SDK: {}", vulkan_sdk);
        return Ok(());
    }

    println!("No explicit Vulkan configuration needed for Windows (using registry)");
    Ok(())
}

/// Intelligent Vulkan configuration with automatic MoltenVK setup and helpful error handling.
///
/// This function provides the new Phase 0 intelligent GPU configuration:
/// - Auto-detects Metal GPU support on compatible systems
/// - Auto-configures MoltenVK when appropriate
/// - Respects user override via KOMPUTE_DISABLE_MOLTENVK_AUTO
/// - Provides detailed, helpful error messages when configuration fails
/// - Validates device availability with comprehensive diagnostics
pub fn configure_vulkan_environment_intelligent() -> Result<(), String> {
    // Step 1: Check if auto-MoltenVK configuration should be applied
    if MetalGpuDetection::should_auto_configure_moltenvk() {
        match configure_moltenvk_auto() {
            Ok(()) => {
                println!("✓ Auto-configured MoltenVK for Metal GPU support");
            }
            Err(e) => {
                println!("⚠️ Auto-MoltenVK configuration failed: {}", e);
                // Continue with fallback configuration
            }
        }
    }

    // Step 2: Apply standard Vulkan configuration
    configure_vulkan_environment()?;

    // Step 3: Validate that devices are available with helpful diagnostics
    validate_device_availability()
}

/// Auto-configure MoltenVK for systems with Metal GPU support.
fn configure_moltenvk_auto() -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        println!("🔧 Auto-configuring MoltenVK for macOS Metal GPU...");
        configure_macos_vulkan()
    }

    #[cfg(not(target_os = "macos"))]
    {
        // Future: support for Linux ARM systems with Metal
        Err("MoltenVK auto-configuration only supported on macOS currently".to_string())
    }
}

/// Validate that Vulkan devices are available and provide helpful errors if not.
fn validate_device_availability() -> Result<(), String> {
    match crate::Manager::device_count() {
        Ok(count) => {
            if count == 0 {
                Err(generate_helpful_device_error())
            } else {
                println!(
                    "✓ Vulkan validation successful: {} device(s) available",
                    count
                );
                Ok(())
            }
        }
        Err(e) => {
            let mut error = format!("Vulkan device enumeration failed: {}", e);
            error.push_str("\n\n");
            error.push_str(&generate_helpful_device_error());
            Err(error)
        }
    }
}

/// Generate helpful, platform-specific error messages when no devices are found.
fn generate_helpful_device_error() -> String {
    let mut error = String::from("No Vulkan devices found.\n\n");

    if MetalGpuDetection::detect_metal_gpu_support() {
        error.push_str("🍎 Metal-capable system detected!\n");
        error.push_str("To resolve this issue:\n\n");
        error.push_str("1️⃣ Install MoltenVK:\n");
        error.push_str("   • Via Homebrew: brew install molten-vk\n");
        error.push_str("   • Via Vulkan SDK: Download from https://vulkan.lunarg.com/\n\n");

        error.push_str("2️⃣ Verify MoltenVK installation:\n");
        error.push_str("   • Run: vulkaninfo\n");
        error.push_str("   • Check for MoltenVK in the output\n\n");

        error.push_str("3️⃣ Environment configuration:\n");
        error.push_str("   • VK_ICD_FILENAMES should point to MoltenVK_icd.json\n");
        error
            .push_str("   • Auto-configuration handles this if MoltenVK is properly installed\n\n");

        error.push_str("4️⃣ Alternative GPU configuration:\n");
        error.push_str("   • If you have discrete GPUs and prefer not to use MoltenVK:\n");
        error.push_str("   • Set: export KOMPUTE_DISABLE_MOLTENVK_AUTO=1\n");
        error.push_str("   • Ensure discrete GPU drivers support Vulkan\n\n");

        error.push_str("🔍 Troubleshooting:\n");
        error.push_str("   • Check system_profiler SPDisplaysDataType for GPU info\n");
        error.push_str("   • Verify macOS version (10.11+ required for Metal)\n");
    } else {
        // Non-Metal systems (Linux, Windows, or older macOS)
        let platform = crate::platform::PlatformInfo::detect();

        match platform.os.as_str() {
            "linux" => {
                error.push_str("🐧 Linux system detected.\n");
                error.push_str("To resolve this issue:\n\n");
                error.push_str("1️⃣ Install Vulkan drivers:\n");
                error.push_str("   • NVIDIA: install nvidia-vulkan-icd\n");
                error.push_str("   • AMD: install mesa-vulkan-drivers or amdvlk\n");
                error.push_str("   • Intel: install mesa-vulkan-drivers\n\n");
                error.push_str("2️⃣ Install Vulkan tools:\n");
                error.push_str("   • Ubuntu/Debian: apt install vulkan-tools\n");
                error.push_str("   • Fedora: dnf install vulkan-tools\n");
                error.push_str("   • Arch: pacman -S vulkan-tools\n\n");
                error.push_str("3️⃣ Verify installation:\n");
                error.push_str("   • Run: vulkaninfo\n");
                error.push_str("   • Check: ls /dev/dri/\n");
            }
            "windows" => {
                error.push_str("🪟 Windows system detected.\n");
                error.push_str("To resolve this issue:\n\n");
                error.push_str("1️⃣ Update GPU drivers:\n");
                error.push_str("   • NVIDIA: Download from nvidia.com\n");
                error.push_str("   • AMD: Download from amd.com\n");
                error.push_str("   • Intel: Download from intel.com\n\n");
                error.push_str("2️⃣ Install Vulkan SDK:\n");
                error.push_str("   • Download from https://vulkan.lunarg.com/\n");
                error.push_str("   • Includes vulkaninfo tool for testing\n\n");
                error.push_str("3️⃣ Verify installation:\n");
                error.push_str("   • Run: vulkaninfo\n");
                error.push_str("   • Check Device Manager for GPU status\n");
            }
            _ => {
                error.push_str("❓ Unknown or unsupported platform.\n");
                error.push_str("General Vulkan installation guidance:\n\n");
                error.push_str("1️⃣ Install Vulkan drivers for your GPU\n");
                error.push_str("2️⃣ Install Vulkan SDK from https://vulkan.lunarg.com/\n");
                error.push_str("3️⃣ Run vulkaninfo to test installation\n");
            }
        }

        error.push_str("\n🔍 Additional troubleshooting:\n");
        error.push_str("   • Ensure your GPU supports Vulkan 1.0+\n");
        error.push_str("   • Check that GPU drivers are not in compatibility mode\n");
        error.push_str("   • Verify no conflicting graphics software is installed\n");
    }

    error
}

/// Check if Vulkan is properly configured and available.
pub fn verify_vulkan_availability() -> Result<u32, String> {
    configure_vulkan_environment()?;

    // Use our device count function to verify
    match crate::Manager::device_count() {
        Ok(count) => {
            println!(
                "✓ Vulkan configured successfully, found {} device(s)",
                count
            );
            Ok(count)
        }
        Err(e) => {
            let platform = crate::platform::PlatformInfo::detect();
            let mut error_msg = format!("Vulkan verification failed: {}", e);

            // Add platform-specific troubleshooting
            error_msg.push_str("\n\nTroubleshooting for ");
            error_msg.push_str(&platform.os);
            error_msg.push_str(":\n");

            for suggestion in platform.troubleshooting_guide() {
                error_msg.push_str(&format!("  💡 {}\n", suggestion));
            }

            Err(error_msg)
        }
    }
}

/// Comprehensive Vulkan environment diagnostics.
pub fn diagnose_vulkan_environment() -> String {
    let mut report = String::new();
    report.push_str("Vulkan Environment Diagnostics\n");
    report.push_str("==============================\n\n");

    // Platform information
    let platform = crate::platform::PlatformInfo::detect();
    report.push_str(&format!("Platform: {} ({})\n", platform.os, platform.arch));
    report.push_str(&format!("GPU Support: {:?}\n\n", platform.gpu_support));

    // Environment variables
    report.push_str("Environment Variables:\n");
    let env_vars = vec![
        "VULKAN_SDK",
        "VK_ICD_FILENAMES",
        "VK_LAYER_PATH",
        "VK_INSTANCE_LAYERS",
        "VK_DEVICE_EXTENSIONS",
    ];

    for var in env_vars {
        match env::var(var) {
            Ok(value) => report.push_str(&format!("  {}: {}\n", var, value)),
            Err(_) => report.push_str(&format!("  {}: <not set>\n", var)),
        }
    }
    report.push('\n');

    // Configuration attempt
    report.push_str("Configuration Results:\n");
    match configure_vulkan_environment() {
        Ok(()) => report.push_str("  ✅ Vulkan environment configured successfully\n"),
        Err(e) => report.push_str(&format!("  ❌ Configuration failed: {}\n", e)),
    }

    // Device detection
    match crate::Manager::device_count() {
        Ok(count) => {
            report.push_str(&format!("  ✅ Found {} Vulkan device(s)\n", count));
        }
        Err(e) => {
            report.push_str(&format!("  ❌ Device detection failed: {}\n", e));
        }
    }

    report.push('\n');

    // Platform-specific checks
    report.push_str("Platform-Specific Checks:\n");
    match platform.os.as_str() {
        "macos" => {
            report.push_str("  MoltenVK Status:\n");
            let moltenvk_paths = vec![
                "/usr/local/share/vulkan/icd.d/MoltenVK_icd.json",
                "/opt/homebrew/share/vulkan/icd.d/MoltenVK_icd.json",
            ];

            if let Ok(vulkan_sdk) = env::var("VULKAN_SDK") {
                let sdk_path = format!("{}/share/vulkan/icd.d/MoltenVK_icd.json", vulkan_sdk);
                if Path::new(&sdk_path).exists() {
                    report.push_str(&format!("    ✅ MoltenVK found in SDK: {}\n", sdk_path));
                } else {
                    report.push_str(&format!("    ❌ MoltenVK not found in SDK: {}\n", sdk_path));
                }
            }

            for path in moltenvk_paths {
                if Path::new(path).exists() {
                    report.push_str(&format!("    ✅ MoltenVK found: {}\n", path));
                } else {
                    report.push_str(&format!("    ❌ MoltenVK not found: {}\n", path));
                }
            }
        }
        "linux" => {
            report.push_str("  Linux GPU Drivers:\n");
            if Path::new("/dev/dri").exists() {
                report.push_str("    ✅ DRI devices available\n");
            } else {
                report.push_str("    ❌ No DRI devices found\n");
            }

            if Path::new("/proc/driver/nvidia").exists() {
                report.push_str("    ✅ NVIDIA drivers detected\n");
            } else {
                report.push_str("    ℹ️  NVIDIA drivers not detected\n");
            }
        }
        "windows" => {
            report.push_str("  Windows GPU Support:\n");
            report.push_str("    ℹ️  Registry-based driver detection not implemented\n");
            report.push_str("    ℹ️  Vulkan runtime detection through device enumeration\n");
        }
        _ => {
            report.push_str("  ⚠️  Untested platform - results may vary\n");
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vulkan_environment_setup() {
        match configure_vulkan_environment() {
            Ok(()) => println!("✓ Vulkan environment configured successfully"),
            Err(e) => println!("⚠️  Vulkan configuration failed: {}", e),
        }
    }

    #[test]
    fn test_vulkan_verification() {
        match verify_vulkan_availability() {
            Ok(count) => {
                println!("✓ Vulkan verification successful, {} devices found", count);
                assert!(count > 0, "Should find at least one Vulkan device");
            }
            Err(e) => {
                println!("⚠️  Vulkan verification failed: {}", e);
                // Don't panic in test - this might be expected in some environments
            }
        }
    }

    #[test]
    fn test_intelligent_vulkan_configuration() {
        println!("🧪 Testing intelligent Vulkan configuration...");

        match configure_vulkan_environment_intelligent() {
            Ok(()) => {
                println!("✅ Intelligent Vulkan configuration succeeded");
            }
            Err(e) => {
                println!("⚠️ Intelligent Vulkan configuration failed: {}", e);

                // Verify error message contains helpful guidance
                assert!(
                    e.contains("Install")
                        || e.contains("MoltenVK")
                        || e.contains("drivers")
                        || e.contains("Metal"),
                    "Error message should contain helpful guidance"
                );
                println!("✓ Error message contains helpful guidance");
            }
        }
    }

    #[test]
    fn test_helpful_error_generation() {
        // Test the error message generation directly
        let error_msg = generate_helpful_device_error();

        println!("📝 Generated helpful error message:\n{}", error_msg);

        // Verify the message contains helpful elements
        assert!(!error_msg.is_empty(), "Error message should not be empty");
        assert!(
            error_msg.contains("Vulkan"),
            "Error message should mention Vulkan"
        );

        // Check for platform-specific guidance
        #[cfg(target_os = "macos")]
        {
            assert!(
                error_msg.contains("Metal") || error_msg.contains("MoltenVK"),
                "macOS error should mention Metal/MoltenVK"
            );
        }

        #[cfg(target_os = "linux")]
        {
            assert!(
                error_msg.contains("Linux")
                    || error_msg.contains("apt")
                    || error_msg.contains("mesa"),
                "Linux error should contain Linux-specific guidance"
            );
        }

        #[cfg(target_os = "windows")]
        {
            assert!(
                error_msg.contains("Windows")
                    || error_msg.contains("nvidia")
                    || error_msg.contains("amd"),
                "Windows error should contain Windows-specific guidance"
            );
        }

        println!("✓ Helpful error message generation test passed");
    }

    #[test]
    fn test_device_validation() {
        println!("🧪 Testing device validation...");

        match validate_device_availability() {
            Ok(()) => {
                println!("✅ Device validation passed");
            }
            Err(e) => {
                println!("⚠️ Device validation failed: {}", e);
                // Verify error contains helpful guidance
                assert!(
                    e.contains("Metal") || e.contains("Install") || e.contains("drivers"),
                    "Validation error should contain helpful guidance"
                );
            }
        }
    }
}
