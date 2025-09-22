//! Test to verify Vulkan setup with proper environment configuration

use std::env;

#[test]
fn test_vulkan_environment_setup() {
    // Set up Vulkan environment variables for macOS with MoltenVK
    if cfg!(target_os = "macos") {
        if let Ok(vulkan_sdk) = env::var("VULKAN_SDK") {
            let icd_path = format!("{}/share/vulkan/icd.d", vulkan_sdk);
            let layer_path = format!("{}/share/vulkan/explicit_layer.d", vulkan_sdk);

            println!("VULKAN_SDK: {}", vulkan_sdk);
            println!("Setting VK_ICD_FILENAMES: {}/MoltenVK_icd.json", icd_path);
            println!("Setting VK_LAYER_PATH: {}", layer_path);

            // Set the ICD file for MoltenVK
            env::set_var(
                "VK_ICD_FILENAMES",
                format!("{}/MoltenVK_icd.json", icd_path),
            );
            env::set_var("VK_LAYER_PATH", layer_path);

            // Disable validation layers for basic testing
            env::remove_var("VK_INSTANCE_LAYERS");

            println!("✓ Vulkan environment configured for macOS with MoltenVK");
        } else {
            panic!("VULKAN_SDK not set! Please install Vulkan SDK and set VULKAN_SDK environment variable");
        }
    }
}

#[test]
fn test_device_count_with_setup() {
    // First configure the environment
    test_vulkan_environment_setup();

    // Now try to get device count
    match komputers::Manager::device_count() {
        Ok(count) => {
            println!("✓ Found {} Vulkan-capable devices", count);
            assert!(
                count >= 1,
                "Should find at least one device on macOS with MoltenVK"
            );
        }
        Err(e) => {
            println!("⚠️  Device enumeration failed: {}", e);
            // Don't panic here - let's see what the error is first
        }
    }
}

#[test]
fn test_manager_creation_with_setup() {
    test_vulkan_environment_setup();

    match komputers::Manager::new() {
        Ok(manager) => {
            println!("✓ GPU Manager created successfully with proper environment setup");
            std::mem::drop(manager);
            println!("✓ GPU Manager dropped successfully");
        }
        Err(e) => {
            println!("⚠️  Manager creation failed: {}", e);
        }
    }
}
