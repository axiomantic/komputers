//! Test utilities for mandatory Vulkan/MoltenVK availability
//!
//! CRITICAL: All tests REQUIRE Vulkan to be available.
//! Missing Vulkan is a CATASTROPHIC FAILURE, not something to skip.

#[cfg(test)]
use crate::error::{KomputeError, Result};
#[cfg(test)]
use crate::vulkan_setup;

/// CRITICAL: Vulkan MUST be available for ALL tests
/// We NEVER skip tests - absence of Vulkan is a FAILURE
#[cfg(test)]
pub fn require_vulkan() -> Result<()> {
    // First, ensure environment is configured for MoltenVK on macOS
    #[cfg(target_os = "macos")]
    {
        // Step 1: Find or set VULKAN_SDK
        let vulkan_sdk = if let Ok(sdk) = std::env::var("VULKAN_SDK") {
            std::path::PathBuf::from(sdk)
        } else {
            // Common SDK locations
            let sdk_candidates = ["/opt/homebrew", "/usr/local"];
            let mut found_sdk = None;

            for candidate in &sdk_candidates {
                let path = std::path::Path::new(candidate);
                if path.join("include/vulkan/vulkan.h").exists() {
                    found_sdk = Some(path.to_path_buf());
                    std::env::set_var("VULKAN_SDK", candidate);
                    break;
                }
            }

            found_sdk.unwrap_or_else(|| std::path::PathBuf::from("/opt/homebrew"))
        };

        // Step 2: Search for MoltenVK ICD file starting from VULKAN_SDK and its parent
        // Always override if the current path doesn't exist
        let need_icd_search = if let Ok(current_path) = std::env::var("VK_ICD_FILENAMES") {
            let exists = std::path::Path::new(&current_path).exists();
            !exists
        } else {
            true
        };

        if need_icd_search {
            fn find_file_recursive(
                dir: &std::path::Path,
                filename: &str,
            ) -> Option<std::path::PathBuf> {
                if let Ok(entries) = std::fs::read_dir(dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.is_file() && path.file_name().map_or(false, |name| name == filename)
                        {
                            return Some(path);
                        } else if path.is_dir() {
                            if let Some(found) = find_file_recursive(&path, filename) {
                                return Some(found);
                            }
                        }
                    }
                }
                None
            }

            // Search in VULKAN_SDK and its parent directory
            let search_dirs = [
                vulkan_sdk.clone(),
                vulkan_sdk.parent().unwrap_or(&vulkan_sdk).to_path_buf(),
            ];

            for search_dir in &search_dirs {
                if let Some(icd_path) = find_file_recursive(search_dir, "MoltenVK_icd.json") {
                    std::env::set_var("VK_ICD_FILENAMES", &icd_path);
                    break;
                }
            }
        }

        // Step 3: Search for Vulkan layer path
        if std::env::var("VK_LAYER_PATH").map_or(true, |path| !std::path::Path::new(&path).exists())
        {
            let layer_candidates = [
                vulkan_sdk.join("share/vulkan/explicit_layer.d"),
                vulkan_sdk
                    .parent()
                    .unwrap_or(&vulkan_sdk)
                    .join("share/vulkan/explicit_layer.d"),
            ];

            for candidate in &layer_candidates {
                if candidate.exists() {
                    std::env::set_var("VK_LAYER_PATH", candidate);
                    break;
                }
            }
        }
    }

    match vulkan_setup::verify_vulkan_availability() {
        Ok(_) => Ok(()),
        Err(e) => {
            // Try to provide better diagnostics
            eprintln!("=== CRITICAL VULKAN FAILURE ===");
            eprintln!("Vulkan/MoltenVK is REQUIRED but not available!");
            eprintln!("Error: {}", e);
            eprintln!("");
            eprintln!("Environment:");
            eprintln!("  VULKAN_SDK: {:?}", std::env::var("VULKAN_SDK"));
            eprintln!(
                "  VK_ICD_FILENAMES: {:?}",
                std::env::var("VK_ICD_FILENAMES")
            );
            eprintln!("  VK_LAYER_PATH: {:?}", std::env::var("VK_LAYER_PATH"));
            eprintln!("");

            #[cfg(target_os = "macos")]
            eprintln!("On macOS: Ensure MoltenVK is installed via Homebrew:");
            #[cfg(target_os = "macos")]
            eprintln!("  brew install molten-vk");

            eprintln!("");
            eprintln!("This is a CATASTROPHIC TEST FAILURE.");
            eprintln!("All tests REQUIRE a Vulkan device.");

            panic!(
                "CRITICAL FAILURE: Vulkan/MoltenVK not available! This is a complete test infrastructure failure. Error: {}",
                e
            );
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

    // Then try to create a manager
    match crate::Manager::new() {
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
