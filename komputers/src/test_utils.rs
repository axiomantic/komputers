//! Test utilities for mandatory Vulkan/MoltenVK availability
//!
//! CRITICAL: All tests REQUIRE Vulkan to be available.
//! Missing Vulkan is a CATASTROPHIC FAILURE, not something to skip.

#[cfg(any(test, feature = "test-utils"))]
use crate::error::Result;

/// CRITICAL: Vulkan MUST be available for ALL tests
/// We NEVER skip tests - absence of Vulkan is a FAILURE
#[cfg(any(test, feature = "test-utils"))]
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
#[cfg(any(test, feature = "test-utils"))]
pub fn init_test_environment() {
    // This will panic and fail the entire test suite if Vulkan is not available
    require_vulkan().expect("Test environment initialization failed - Vulkan REQUIRED");
}

/// Helper to check if we can create a Manager (which requires Vulkan)
#[cfg(any(test, feature = "test-utils"))]
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

// =============================================================================
// GPU Manager Pool for Test Isolation
// =============================================================================

#[cfg(any(test, feature = "test-utils"))]
use std::collections::VecDeque;
#[cfg(any(test, feature = "test-utils"))]
use std::sync::{Arc, Mutex, OnceLock};
#[cfg(any(test, feature = "test-utils"))]
use std::time::Duration;

/// Thread-safe pool of Manager instances for test isolation
#[cfg(any(test, feature = "test-utils"))]
static MANAGER_POOL: OnceLock<Arc<Mutex<VecDeque<crate::Manager>>>> = OnceLock::new();

/// Shared Manager instance for tests that need the same manager throughout their execution
#[cfg(any(test, feature = "test-utils"))]
static SHARED_MANAGER: OnceLock<Arc<Mutex<crate::Manager>>> = OnceLock::new();

/// Check if we're running in single-threaded test mode (resource contention likely)
#[cfg(any(test, feature = "test-utils"))]
pub fn is_single_threaded_test_mode() -> bool {
    std::env::var("CARGO_TEST_THREADS").map_or(false, |v| v == "1")
}

/// Skip GPU tests when running in problematic environments
#[cfg(any(test, feature = "test-utils"))]
pub fn should_skip_gpu_test() -> bool {
    // Skip if explicitly requested
    if std::env::var("SKIP_GPU_TESTS").map_or(false, |v| v == "1" || v.to_lowercase() == "true") {
        return true;
    }

    // For now, don't skip any tests - but this is where we'd add logic
    // for environments where GPU tests are problematic
    false
}

/// Get a Manager instance from the pool with retry logic
#[cfg(any(test, feature = "test-utils"))]
pub fn get_manager_from_pool() -> Result<crate::Manager> {
    if should_skip_gpu_test() {
        return Err(crate::KomputeError::compute(
            "GPU tests skipped in this environment",
        ));
    }

    // Try to get from pool first
    let pool = MANAGER_POOL.get_or_init(|| {
        let mut managers = VecDeque::new();

        // Try to create 2-3 managers for the pool
        for i in 0..3 {
            match create_manager_with_retry() {
                Ok(manager) => {
                    println!("✅ Created pooled Manager {}", i + 1);
                    managers.push_back(manager);
                }
                Err(e) => {
                    println!("⚠️ Failed to create pooled Manager {}: {}", i + 1, e);
                    // Continue with fewer managers in pool
                    break;
                }
            }
        }

        if managers.is_empty() {
            println!("❌ No managers could be created for pool - tests will create on-demand");
        } else {
            println!(
                "🏊 Manager pool initialized with {} instances",
                managers.len()
            );
        }

        Arc::new(Mutex::new(managers))
    });

    // Try to get from pool
    if let Ok(mut pool_guard) = pool.lock() {
        if let Some(manager) = pool_guard.pop_front() {
            println!("♻️ Reusing Manager from pool");
            return Ok(manager);
        }
    }

    // Pool empty or locked, create new manager
    println!("🆕 Creating new Manager (pool empty)");
    create_manager_with_retry()
}

/// Return a Manager instance to the pool for reuse
#[cfg(any(test, feature = "test-utils"))]
pub fn return_manager_to_pool(manager: crate::Manager) {
    if let Some(pool) = MANAGER_POOL.get() {
        if let Ok(mut pool_guard) = pool.lock() {
            // Only keep up to 3 managers in pool to avoid resource waste
            if pool_guard.len() < 3 {
                pool_guard.push_back(manager);
                println!("♻️ Returned Manager to pool");
                return;
            }
        }
    }

    // Pool full or unavailable, just drop the manager
    drop(manager);
    println!("🗑️ Dropped Manager (pool full)");
}

/// Get a shared Manager instance that persists across multiple test calls
#[cfg(any(test, feature = "test-utils"))]
pub fn get_shared_manager() -> Arc<Mutex<crate::Manager>> {
    SHARED_MANAGER
        .get_or_init(|| match create_manager_with_retry() {
            Ok(manager) => {
                println!("✅ Created shared Manager for test module");
                Arc::new(Mutex::new(manager))
            }
            Err(e) => {
                panic!("Failed to create shared Manager: {}", e);
            }
        })
        .clone()
}

/// Create a Manager with retry logic and error recovery
#[cfg(any(test, feature = "test-utils"))]
fn create_manager_with_retry() -> Result<crate::Manager> {
    // Ensure Vulkan is available first
    require_vulkan()?;

    let config = crate::VulkanConfig::for_testing();

    for attempt in 0..3 {
        match crate::Manager::with_config(config.clone()) {
            Ok(manager) => {
                if attempt > 0 {
                    println!("✅ Manager created successfully on attempt {}", attempt + 1);
                }
                return Ok(manager);
            }
            Err(e) if attempt < 2 => {
                println!("⚠️ Manager creation attempt {} failed: {}", attempt + 1, e);
                println!("🔄 Retrying in {}ms...", 50 * (attempt + 1));

                // Exponential backoff
                std::thread::sleep(Duration::from_millis(50 * (attempt + 1) as u64));
                continue;
            }
            Err(e) => {
                eprintln!("❌ All Manager creation attempts failed. Last error: {}", e);
                eprintln!("💡 This might indicate:");
                eprintln!("   - GPU resource exhaustion");
                eprintln!("   - Driver issues");
                eprintln!("   - Multiple competing processes");
                return Err(e);
            }
        }
    }

    unreachable!()
}

/// Clean up the manager pool (useful for test teardown)
#[cfg(any(test, feature = "test-utils"))]
pub fn cleanup_manager_pool() {
    if let Some(pool) = MANAGER_POOL.get() {
        if let Ok(mut pool_guard) = pool.lock() {
            let count = pool_guard.len();
            pool_guard.clear();
            if count > 0 {
                println!("🧹 Cleaned up {} managers from pool", count);
            }
        }
    }
}

/// Macro for GPU tests that automatically handles Manager lifecycle
#[cfg(any(test, feature = "test-utils"))]
#[macro_export]
macro_rules! gpu_test {
    ($test_name:ident, $test_body:expr) => {
        #[test]
        #[serial_test::serial(gpu)]
        fn $test_name() -> $crate::Result<()> {
            use $crate::test_utils::{
                get_manager_from_pool, return_manager_to_pool, should_skip_gpu_test,
            };

            if should_skip_gpu_test() {
                println!(
                    "⏭️ Skipping GPU test {} - GPU tests disabled",
                    stringify!($test_name)
                );
                return Ok(());
            }

            let manager = get_manager_from_pool()?;
            let result = (|| $test_body)(manager);

            match result {
                Ok(manager) => {
                    return_manager_to_pool(manager);
                    Ok(())
                }
                Err(e) => {
                    eprintln!("❌ GPU test {} failed: {}", stringify!($test_name), e);
                    Err(e)
                }
            }
        }
    };
}

/// Macro for GPU tests that need a mutable manager reference
#[cfg(any(test, feature = "test-utils"))]
#[macro_export]
macro_rules! gpu_test_mut {
    ($test_name:ident, $test_body:expr) => {
        #[test]
        #[serial_test::serial(gpu)]
        fn $test_name() -> $crate::Result<()> {
            use $crate::test_utils::{
                get_manager_from_pool, return_manager_to_pool, should_skip_gpu_test,
            };

            if should_skip_gpu_test() {
                println!(
                    "⏭️ Skipping GPU test {} - GPU tests disabled",
                    stringify!($test_name)
                );
                return Ok(());
            }

            let mut manager = get_manager_from_pool()?;
            let result = (|| $test_body)(&mut manager);

            match result {
                Ok(()) => {
                    return_manager_to_pool(manager);
                    Ok(())
                }
                Err(e) => {
                    eprintln!("❌ GPU test {} failed: {}", stringify!($test_name), e);
                    // Still return manager to pool even on test failure
                    return_manager_to_pool(manager);
                    Err(e)
                }
            }
        }
    };
}
