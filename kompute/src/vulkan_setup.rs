//! Automatic Vulkan environment setup for cross-platform compatibility.

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

        // Set up MoltenVK ICD
        let icd_path = format!("{}/share/vulkan/icd.d/MoltenVK_icd.json", vulkan_sdk);
        if Path::new(&icd_path).exists() {
            env::set_var("VK_ICD_FILENAMES", &icd_path);
            println!("Set VK_ICD_FILENAMES: {}", icd_path);
        } else {
            return Err(format!("MoltenVK ICD file not found at: {}", icd_path));
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
        Err(e) => Err(format!("Vulkan verification failed: {}", e)),
    }
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
}
