//! Test utilities for mandatory Vulkan/MoltenVK availability
//!
//! CRITICAL: All tests REQUIRE Vulkan to be available.
//! Missing Vulkan is a CATASTROPHIC FAILURE, not something to skip.

#[cfg(test)]
use crate::error::Result;

/// CRITICAL: Vulkan MUST be available for ALL tests
/// We NEVER skip tests - absence of Vulkan is a FAILURE
#[cfg(test)]
pub fn require_vulkan() -> Result<()> {
    // Use the new VulkanConfig system for testing
    let config = crate::VulkanConfig::for_testing();

    // Test device enumeration using explicit configuration
    let result = crate::Manager::device_count_with_config(&config);
    match result {
        Ok(count) => {
            if count > 0 {
                println!(
                    "✅ Vulkan configured successfully with explicit config, found {} device(s)",
                    count
                );
                Ok(())
            } else {
                eprintln!("❌ No Vulkan devices found with explicit configuration");
                eprintln!("Configuration used: {}", config.configuration_reasoning);
                eprintln!("💡 Try installing MoltenVK: brew install molten-vk");
                Err(crate::KomputeError::vulkan(
                    "No Vulkan devices found with explicit config",
                ))
            }
        }
        Err(e) => {
            eprintln!(
                "❌ Failed to enumerate devices using explicit configuration: {}",
                e
            );
            eprintln!("Configuration used: {}", config.configuration_reasoning);

            // Fall back to auto-detection for troubleshooting
            match crate::VulkanConfig::auto_detect() {
                Ok(auto_config) => {
                    eprintln!(
                        "Auto-detection succeeded: {}",
                        auto_config.configuration_reasoning
                    );
                    eprintln!("But explicit testing config failed. This suggests a configuration mismatch.");
                }
                Err(config_error) => {
                    eprintln!("Auto-detection also failed: {}", config_error);
                }
            }

            eprintln!("💡 Try installing MoltenVK: brew install molten-vk");

            eprintln!("=== CRITICAL VULKAN FAILURE ===");
            eprintln!("Vulkan/MoltenVK is REQUIRED but not available!");
            eprintln!("This is a CATASTROPHIC TEST FAILURE.");
            eprintln!("All tests REQUIRE a Vulkan device.");

            Err(e)
        }
    }
}

/// Ensure Vulkan is available at test module initialization
#[cfg(test)]
pub fn init_test_environment() {
    // This will panic and fail the entire test suite if Vulkan is not available
    require_vulkan().expect("Test environment initialization failed - Vulkan REQUIRED");
}

/// Helper to check if we can create a Manager (which requires Vulkan)
#[cfg(test)]
pub fn can_create_manager() -> bool {
    // First ensure Vulkan is available (will panic if not)
    require_vulkan().unwrap();

    // Then try to create a manager with explicit configuration
    let config = crate::VulkanConfig::for_testing();
    match crate::Manager::with_config(config) {
        Ok(_) => true,
        Err(e) => {
            eprintln!(
                "Manager creation failed even though Vulkan is available: {}",
                e
            );
            false
        }
    }
}
